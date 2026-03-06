"""
plan.py — Walk through the full planning pipeline for a prompt and show what happens at each step.

Shows:
  - Environment scan (files available, what the planner sees)
  - Classification decision (why the task is simple/complex/ambiguous)
  - Decomposition / clarification / spec generation (if triggered)
  - Output contract generation (what each task produces)
  - Critique pass (issues found, approval status, any revisions)
  - Final plan summary (tasks, agents, contracts, dependency chain)

Usage:
    python plan.py "Write a fibonacci function"
    python plan.py "Research chicken thigh macros and write a calculator, save as calc.py"
    python plan.py "Build a note-taking app with save, load, and list"
"""
import sys
import logging

# Windows cp1252 safety: force UTF-8 output so BOM/special chars in filenames don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --- Custom log capture -------------------------------------------------
# Intercept the hierarchical planner's INFO logs to surface internal
# classification/decomposition decisions in a human-readable way.

class _PlannerLogCapture(logging.Handler):
    """Captures structured log records from the hierarchical planner."""
    def __init__(self):
        super().__init__()
        self.events: list[tuple[str, str]] = []  # (logger_name, message)

    def emit(self, record: logging.LogRecord):
        self.events.append((record.name, record.getMessage()))


_capture = _PlannerLogCapture()
_capture.setLevel(logging.DEBUG)

# get_logger("X") returns "ai_intern.X" — not "ai_intern.planning.X"
for _name in ("ai_intern.hierarchical", "ai_intern.classifier",
              "ai_intern.spec_generator", "ai_intern.environment"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.DEBUG)
    _lg.addHandler(_capture)
    _lg.propagate = False

# Silence all other ai_intern output (routing, validation, etc.)
logging.getLogger("ai_intern").setLevel(logging.WARNING)
# -----------------------------------------------------------------------


from ai_intern.schemas import RequestSchema, OutputContract
from ai_intern.planning.hierarchical import HierarchicalPlanner
from ai_intern.planning.environment import EnvironmentScanner
from ai_intern.planning.red_team.council import RedTeamCouncil


# --- Display helpers ----------------------------------------------------

W = 72

def _sep(char="="):
    print(char * W)

def _section(title: str):
    print()
    print(f"  [ {title} ]")
    print("  " + "-" * (W - 2))


def _extract_decisions(events: list[tuple[str, str]]) -> dict:
    """
    Pull key planning decisions from captured log messages.
    Returns a dict with parsed fields for each decision type.
    """
    d = {
        "verdict": None,
        "app_scale": False,
        "reasoning": None,
        "spec_summary": None,
        "spec_components": [],
        "subtasks_logged": [],
        "overrides": [],
        "guards": [],
        "clarified_to": None,
    }
    for _, msg in events:
        m = msg.strip()

        if m.startswith("Verdict:"):
            parts = m.split("|")
            d["verdict"] = parts[0].replace("Verdict:", "").strip()
            d["app_scale"] = "app_scale=True" in m

        elif m.startswith("Reasoning:"):
            d["reasoning"] = m.replace("Reasoning:", "").strip()

        elif m.startswith("Spec:"):
            d["spec_summary"] = m.replace("Spec:", "").strip()

        elif m.startswith("Components:"):
            d["spec_components"] = [c.strip() for c in m.replace("Components:", "").split(",")]

        elif m.startswith("Subtask ") and ":" in m:
            d["subtasks_logged"].append(m)

        elif "overridden" in m.lower():
            d["overrides"].append(m)

        elif "obviously simple" in m.lower() or "max depth" in m.lower():
            d["guards"].append(m)

        elif m.startswith("Clarified:"):
            d["clarified_to"] = m.replace("Clarified:", "").strip()

    return d

# -----------------------------------------------------------------------


VERDICT_LABELS = {
    "execute":                "EXECUTE  (simple + clear -> single-shot execution, no decomposition)",
    "decompose":              "DECOMPOSE  (complex + clear -> break into subtasks)",
    "clarify":                "CLARIFY  (simple + ambiguous -> rewrite goal, then execute)",
    "clarify_then_decompose": "CLARIFY_THEN_DECOMPOSE  (complex + ambiguous -> clarify, then decompose)",
}

prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ").strip()

print()
_sep()
print(f'  Planning: "{prompt}"')
_sep()


# -----------------------------------------------------------------------
# ENVIRONMENT SCAN
# Shows what context is injected into every LLM call during planning.
# -----------------------------------------------------------------------
_section("Environment Scan  (injected into every planner LLM call)")

env = EnvironmentScanner.scan()
print(f"  Date/time     : {env.current_day} {env.current_date} at {env.current_time}")

if env.available_files:
    print(f"  user_data/    : {len(env.available_files)} file(s)")
    for f in env.available_files:
        meta = f"{f.size_bytes:,}B"
        if f.row_count is not None:
            meta += f"  {f.row_count} rows"
        if f.columns:
            cols = ", ".join(f.columns[:5]) + (f"  +{len(f.columns)-5}" if len(f.columns) > 5 else "")
            meta += f"  cols=[{cols}]"
        print(f"    {f.filename:<32} {meta}")
else:
    print("  user_data/    : (empty)")


# -----------------------------------------------------------------------
# PLANNING
# Runs the full HierarchicalPlanner and captures internal log events
# so we can show each decision as it happened.
# -----------------------------------------------------------------------
_section("Stage 1 — Planning  (HierarchicalPlanner)")
_capture.events.clear()

# Capture stdout during planning so we control the display order.
# The planner prints inline during _generate_contracts — we replay those
# ourselves in the right place rather than letting them appear mid-section.
import io as _io
_stdout_buf = _io.StringIO()
_real_stdout = sys.stdout
sys.stdout = _stdout_buf

req = RequestSchema(content=prompt)
plan = HierarchicalPlanner.create_plan(req)

sys.stdout = _real_stdout
_planner_stdout = _stdout_buf.getvalue()

D = _extract_decisions(_capture.events)

# -- Classification verdict --
verdict_raw = (D["verdict"] or "unknown").lower()
print(f"  Verdict       : {VERDICT_LABELS.get(verdict_raw, verdict_raw.upper())}")

if D["app_scale"]:
    print(f"  App-scale     : YES -> SpecGenerator triggered to produce a multi-file spec")

for ov in D["overrides"]:
    print(f"  Override      : {ov}")

if D["guards"]:
    print(f"  Depth guards  : {len(D['guards'])} subtask(s) bypassed LLM classification (obviously simple)")

if D["reasoning"]:
    print(f"  LLM reasoning : {D['reasoning']}")

# -- Clarification --
if D["clarified_to"]:
    print()
    print(f"  Clarified to  : {D['clarified_to']}")
    print(f"  (The LLM rewrote the goal to make implicit details explicit before decomposing.)")

# -- AppSpec --
if plan.app_spec:
    spec = plan.app_spec
    print()
    print(f"  AppSpec:")
    print(f"    Summary    : {spec.get('summary', '')}")
    feats = spec.get("features", [])
    if feats:
        print(f"    Features   :")
        for feat in feats:
            print(f"      - {feat}")
    comps = spec.get("components", [])
    if comps:
        print(f"    Components :")
        for c in comps:
            deps = c.get("depends_on", [])
            dep_str = f"  imports: {', '.join(deps)}" if deps else ""
            iface = c.get("public_interface", "")
            print(f"      [{c['name']:<20}]  {c['output_file']:<28}  {iface}{dep_str}")
    criteria = spec.get("done_criteria", [])
    if criteria:
        print(f"    Done when  :")
        for cr in criteria:
            print(f"      * {cr}")

# -- Output contracts --
contracts = [t for t in plan.tasks if t.output_contract is not None]
if contracts:
    print()
    print(f"  Output contracts  (generated after decomposition, {len(contracts)} task(s)):")
    print(f"  {'Task':<6} {'Type':<17} Format / Consumers")
    print("  " + "-" * 70)
    for t in plan.tasks:
        if isinstance(t.output_contract, OutputContract):
            oc = t.output_contract
            consumed = str(oc.required_by_tasks) if oc.required_by_tasks else "none"
            print(f"  [{t.task_order}]    {oc.output_type:<17} {oc.output_format}  ->  consumed by: {consumed}")
        elif isinstance(t.output_contract, dict):
            oc = t.output_contract
            print(f"  [{t.task_order}]    spec-component    {oc.get('output_file', '?')}")

print()
print(f"  Tasks generated : {len(plan.tasks)}")
print(f"  Tokens used     : {plan.token_usage:,}")


# -----------------------------------------------------------------------
# RED TEAM COUNCIL
# Adversarial multi-round review: Executor, Integrator, Minimalist,
# Cross-Examination, Blue Team Defense, Synthesis.
#
# Rounds:
#   R1 - Three independent agents each find issues (no communication)
#   R2 - Each agent reviews the others' findings (global indices stable)
#   R3 - Planner defends plan against surviving findings
#   R4 - Deterministic scoring (no LLM); high-confidence findings trigger R5
#   R5 - Constraint manifest built; re-plan injected (conditional)
# -----------------------------------------------------------------------

def _subsection(title: str, width: int = 68):
    print(f"  {'   ' + title}")
    print(f"  {'  ' + '-' * (width - 2)}")

def _finding_header(f, prefix=""):
    sev_tag = "[block]" if f.severity == "block" else "[warn ]"
    task_ref = f"Task {f.task_index:2d}" if f.task_index >= 0 else "Plan   "
    print(f"  {prefix}[{f.original_index}] {sev_tag}  {task_ref}  {f.agent}/{f.finding_type}")

def _wrap(text: str, indent: int, width: int = 66) -> list[str]:
    """Word-wrap text to width, prefixed by indent spaces."""
    prefix = " " * indent
    words = text.split()
    lines, current = [], []
    for w in words:
        if sum(len(x) + 1 for x in current) + len(w) > width - indent:
            if current:
                lines.append(prefix + " ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(prefix + " ".join(current))
    return lines

def _print_wrapped(text: str, indent: int):
    for line in _wrap(text, indent):
        print(line)

_section("Stage 1.5 -- Red Team Council  (Adversarial Review)")

if len(plan.tasks) < 2:
    print(f"  Skipped: {len(plan.tasks)} task -- council only runs on 2+ task plans")
else:
    print(f"  Reviewing {len(plan.tasks)}-task plan...")
    print(f"  Each round's data flows into the next via global finding indices.")
    print()

    # Suppress the council's live [R1]/[R2]/... print output during execution.
    # We replay per-round detail ourselves below in structured form.
    _stdout_buf2 = _io.StringIO()
    sys.stdout = _stdout_buf2
    council_verdict, council_tokens = RedTeamCouncil.review(plan, req)
    sys.stdout = _real_stdout
    plan.token_usage += council_tokens

    all_findings      = council_verdict.all_findings
    cross_responses   = council_verdict.cross_exam_responses
    blue_responses    = council_verdict.blue_team_responses
    confidence_scores = council_verdict.confidence_scores
    rounds_used       = council_verdict.rounds_used

    # surviving set = findings that made it past R2 filtering (have a score)
    survived_indices = set(confidence_scores.keys())

    # -----------------------------------------------------------------
    # ROUND 1: Independent Red Team
    # -----------------------------------------------------------------
    print(f"  Round 1 -- Independent Red Team  ({len(all_findings)} finding(s))")
    print(f"  " + "-" * 68)
    print(f"  Three agents run independently. No shared state.")
    print(f"  Each assigns a global index (immutable from this point on).")
    print()

    for agent_name in ("executor", "integrator", "minimalist"):
        agent_label = {
            "executor":   "Executor   -- traces data flow task-by-task",
            "integrator": "Integrator -- reasons about the full artifact graph",
            "minimalist": "Minimalist -- counterfactual simplicity check",
        }[agent_name]
        agent_findings = [f for f in all_findings if f.agent == agent_name]
        print(f"    {agent_label}")
        if not agent_findings:
            print(f"      (no findings)")
        else:
            for f in agent_findings:
                sev_tag = "[block]" if f.severity == "block" else "[warn ]"
                task_ref = f"Task {f.task_index:2d}" if f.task_index >= 0 else "Plan   "
                print(f"      [{f.original_index}] {sev_tag}  {task_ref}  {f.finding_type}")
                _print_wrapped(f.description, 10)
                _print_wrapped(f"Evidence: {f.evidence}", 10)
        print()

    if rounds_used == 1:
        print(f"  No findings -- short-circuited after R1.")
        print()

    # -----------------------------------------------------------------
    # ROUND 2: Cross-Examination  (only if R2 ran)
    # -----------------------------------------------------------------
    if rounds_used >= 2 and all_findings:
        print(f"  Round 2 -- Cross-Examination  ({len(cross_responses)} response(s))")
        print(f"  " + "-" * 68)
        print(f"  Each agent reviews findings NOT from itself.")
        print(f"  Global indices passed as-is -- no re-numbering.")
        print(f"  A single DISPUTE removes a finding from R3+.")
        print()

        for f in all_findings:
            relevant = [r for r in cross_responses if r.finding_index == f.original_index]
            if not relevant:
                continue

            sev_tag  = "[block]" if f.severity == "block" else "[warn ]"
            task_ref = f"Task {f.task_index:2d}" if f.task_index >= 0 else "Plan   "
            survived_tag = "survived" if f.original_index in survived_indices else "FILTERED"
            print(f"    [{f.original_index}] {sev_tag}  {task_ref}  {f.agent}/{f.finding_type}  -> {survived_tag}")

            confirms  = [r for r in relevant if r.verdict == "confirm"]
            disputes  = [r for r in relevant if r.verdict == "dispute"]
            extends   = [r for r in relevant if r.verdict == "extend"]

            verdict_summary = []
            if confirms:
                verdict_summary.append(f"{len(confirms)} confirm")
            if disputes:
                verdict_summary.append(f"{len(disputes)} dispute")
            if extends:
                verdict_summary.append(f"{len(extends)} extend")
            print(f"      votes: {', '.join(verdict_summary) if verdict_summary else 'none'}")

            for r in relevant:
                tag = {"confirm": "CONFIRM", "dispute": "DISPUTE", "extend": "EXTEND "}[r.verdict]
                _print_wrapped(f"  [{tag}]  {r.reasoning}", 8)
                if r.additional_evidence:
                    _print_wrapped(f"  + evidence: {r.additional_evidence}", 10)
            print()

        n_survived = len(survived_indices)
        n_filtered = len(all_findings) - n_survived
        print(f"  Filter result: {n_survived}/{len(all_findings)} survived  ({n_filtered} removed by dispute)")
        print()

        if rounds_used == 2:
            print(f"  All surviving findings disputed -- short-circuited after R2.")
            print()

    # -----------------------------------------------------------------
    # ROUND 3: Blue Team Defense  (only if R3 ran)
    # -----------------------------------------------------------------
    if rounds_used >= 3 and blue_responses:
        print(f"  Round 3 -- Blue Team Defense  ({len(blue_responses)} response(s))")
        print(f"  " + "-" * 68)
        print(f"  Planner (as skeptical self-critic) defends against surviving findings.")
        print(f"  Conceding a finding adds +0.25 to its confidence score.")
        print()

        for f in all_findings:
            if f.original_index not in survived_indices:
                continue
            relevant_blue = [r for r in blue_responses if r.finding_index == f.original_index]
            if not relevant_blue:
                continue

            sev_tag  = "[block]" if f.severity == "block" else "[warn ]"
            task_ref = f"Task {f.task_index:2d}" if f.task_index >= 0 else "Plan   "
            r = relevant_blue[0]
            outcome = "REBUT  " if r.can_rebut else "CONCEDE"
            print(f"    [{f.original_index}] {sev_tag}  {task_ref}  {f.agent}/{f.finding_type}")
            print(f"      Planner: {outcome}")
            if r.rebuttal:
                _print_wrapped(r.rebuttal, 8)
            print()

    # -----------------------------------------------------------------
    # ROUND 4: Synthesis  (deterministic, no LLM)
    # -----------------------------------------------------------------
    if rounds_used >= 4 and survived_indices:
        from ai_intern.planning.red_team.synthesis import BLOCK_THRESHOLD, WARN_SCORE_CAP

        print(f"  Round 4 -- Synthesis  (deterministic, zero LLM cost)")
        print(f"  " + "-" * 68)
        print(f"  Scoring breakdown per surviving finding:")
        print(f"    base               +0.20  (found in R1)")
        print(f"    corroborated       +0.10 per similar finding (same type+task, max +0.20)")
        print(f"    confirmed in R2    +0.25  (>= 1 confirm)")
        print(f"    not disputed       +0.10  (zero disputes)")
        print(f"    blue team concede  +0.25  (can_rebut=false)")
        print(f"    warn severity cap  capped at {WARN_SCORE_CAP:.2f}  (cannot trigger re-plan)")
        print(f"    block threshold    {BLOCK_THRESHOLD:.2f}  -> triggers revision if exceeded")
        print()

        for f in all_findings:
            if f.original_index not in survived_indices:
                continue

            score = confidence_scores[f.original_index]
            sev_tag  = "[block]" if f.severity == "block" else "[warn ]"
            task_ref = f"Task {f.task_index:2d}" if f.task_index >= 0 else "Plan   "
            filled = int(score * 10)
            bar = "X" * filled + "." * (10 - filled)
            block_tag = "  => [BLOCK]" if score >= BLOCK_THRESHOLD else ""
            print(f"    [{f.original_index}] {sev_tag}  {task_ref}  {f.agent}/{f.finding_type}")

            # Reconstruct breakdown
            idx = f.original_index
            similar = sum(
                1 for x in all_findings
                if x.original_index != idx
                and x.finding_type == f.finding_type
                and x.task_index == f.task_index
            )
            relevant_cross = [r for r in cross_responses if r.finding_index == idx]
            confirms = sum(1 for r in relevant_cross if r.verdict == "confirm")
            disputes = sum(1 for r in relevant_cross if r.verdict == "dispute")
            relevant_blue = [r for r in blue_responses if r.finding_index == idx]
            blue_concede = relevant_blue and not relevant_blue[0].can_rebut

            running = 0.20
            print(f"      base             +0.20  -> {running:.2f}")
            if similar:
                gain = min(similar * 0.10, 0.20)
                running += gain
                print(f"      corroborated     +{gain:.2f}  -> {running:.2f}  ({similar} similar finding(s))")
            if confirms >= 1:
                running += 0.25
                print(f"      confirmed R2     +0.25  -> {running:.2f}  ({confirms} confirm(s))")
            if disputes == 0:
                running += 0.10
                print(f"      not disputed     +0.10  -> {running:.2f}")
            if blue_concede:
                running += 0.25
                print(f"      blue concede     +0.25  -> {running:.2f}")
            if f.severity == "warn" and min(running, 1.0) > WARN_SCORE_CAP:
                print(f"      warn cap         capped {min(running,1.0):.2f} -> {WARN_SCORE_CAP:.2f}")
            print(f"      final score      [{bar}] {score:.2f}{block_tag}")
            print()

    # -----------------------------------------------------------------
    # ROUND 5: Constraint Manifest  (conditional)
    # -----------------------------------------------------------------
    if council_verdict.constraint_manifest:
        m = council_verdict.constraint_manifest
        total_constraints = (
            len(m.task_constraints) + len(m.structural_constraints)
            + len(m.must_include_tasks) + len(m.must_not_combine)
        )
        print(f"  Round 5 -- Constraint Manifest  ({total_constraints} constraint(s))")
        print(f"  " + "-" * 68)
        print(f"  High-confidence block findings triggered a re-plan.")
        print(f"  These constraints are injected as HARD RULES into the planner.")
        print()
        for tidx, c in m.task_constraints.items():
            _print_wrapped(f"  Task {tidx}: {c}", 4)
        for c in m.structural_constraints:
            _print_wrapped(f"  STRUCTURAL: {c}", 4)
        for g in m.must_include_tasks:
            _print_wrapped(f"  MUST INCLUDE: {g}", 4)
        for pair in m.must_not_combine:
            if len(pair) == 2:
                print(f"    NEVER combine [{pair[0]}] and [{pair[1]}] in one task")
        print()

    # -----------------------------------------------------------------
    # Summary line
    # -----------------------------------------------------------------
    status = "APPROVED" if council_verdict.approved else "REJECTED (re-plan triggered)"
    n_block = len(council_verdict.high_confidence_findings)
    print(f"  Verdict     : {status}")
    print(f"  Rounds used : {rounds_used} / 5  (short-circuits when no issues survive)")
    print(f"  R1 findings : {len(all_findings)}  "
          f"survived R2: {len(survived_indices)}  "
          f"block threshold: {n_block}")
    print(f"  Tokens      : {council_tokens:,}")

    # -----------------------------------------------------------------
    # REPLAN LOOP (hardened — deterministic constraint verification)
    # Mirrors the Stage 1.5 loop in orchestrator.py.
    # -----------------------------------------------------------------
    if not council_verdict.approved and council_verdict.constraint_manifest:
        from ai_intern.planning.red_team.constraint_verifier import ConstraintVerifier
        from ai_intern.orchestration.orchestrator import Orchestrator

        _orch = Orchestrator()
        manifest = council_verdict.constraint_manifest
        MAX_REPLAN_CYCLES = 3

        print()
        for replan_cycle in range(1, MAX_REPLAN_CYCLES + 1):
            n_issues = len(council_verdict.high_confidence_findings)
            print(f"   Cycle {replan_cycle}/{MAX_REPLAN_CYCLES}: {n_issues} blocking issue(s) -- replanning...")

            plan = _orch._replan_with_manifest(req, manifest)

            # Deterministic check before burning another council review
            violations = ConstraintVerifier.check(plan, manifest)
            if violations:
                print(f"   Post-replan verifier: {len(violations)} constraint(s) still unmet after replan:")
                for v in violations:
                    print(f"     - {v}")
            else:
                print(f"   Post-replan verifier: all {len(manifest.must_include_tasks)} required tasks present \u2713")

            # Second council review
            _stdout_buf_r = _io.StringIO()
            sys.stdout = _stdout_buf_r
            council_verdict2, council_tokens2 = RedTeamCouncil.review(plan, req, cycle=replan_cycle)
            sys.stdout = _real_stdout
            plan.token_usage += council_tokens2

            if council_verdict2.approved:
                print(f"   Plan approved after {replan_cycle} replan cycle(s)")
                break

            # Still rejected
            council_verdict = council_verdict2
            if council_verdict2.constraint_manifest:
                manifest = council_verdict2.constraint_manifest

            if replan_cycle >= MAX_REPLAN_CYCLES:
                final_violations = ConstraintVerifier.check(plan, manifest)
                print(f"   Max replan cycles ({MAX_REPLAN_CYCLES}) reached.")
                print(f"   {len(final_violations)} constraint(s) remain unmet:")
                for v in final_violations:
                    print(f"     - {v}")
                print(f"   Proceeding with best available plan.")


# -----------------------------------------------------------------------
# FINAL PLAN SUMMARY
# Shows the complete ordered task list with agents, contracts, dependencies.
# -----------------------------------------------------------------------
_section("Final Plan")

print(f"  {len(plan.tasks)} task(s)   {plan.token_usage:,} total tokens")
print()

for t in plan.tasks:
    agent = t.declared_agent or t.suggested_agent or "?"
    goal_lines = t.goal.replace("\n", " | ").split(" | ")
    first_line = goal_lines[0]

    print(f"  [{t.task_order}] [{agent:<8}] {first_line}")
    for extra in goal_lines[1:]:
        if extra.strip():
            print(f"              {extra}")

    if isinstance(t.output_contract, OutputContract):
        oc = t.output_contract
        consumed = f"consumed by task(s) {oc.required_by_tasks}" if oc.required_by_tasks else "terminal output"
        print(f"          -> {oc.output_type:<17}  {oc.output_format}")
        print(f"             {consumed}")
    elif isinstance(t.output_contract, dict):
        oc = t.output_contract
        print(f"          -> spec-component    {oc.get('output_file', '?')}")
        iface = oc.get("public_interface", "")
        if iface:
            print(f"             interface: {iface}")

    if t.depends_on:
        deps = ", ".join(f"[{d}]" for d in t.depends_on)
        print(f"          depends on: {deps}")

    print()

# Execution flow chain
if len(plan.tasks) > 1:
    chain = "  ->  ".join(f"[{t.task_order}] {t.declared_agent or t.suggested_agent or '?'}" for t in plan.tasks)
    print(f"  Flow: {chain}")

if plan.app_spec:
    print()
    print(f"  Workspace: outputs/{{plan_id}}/")
    for t in plan.tasks:
        if isinstance(t.output_contract, dict):
            print(f"    - {t.output_contract.get('output_file', '?')}")

print()
_sep()
