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
from ai_intern.planning.critic import CritiqueAgent


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
# CRITIQUE PASS
# Shows what the CritiqueAgent found — deterministic checks (zero cost)
# followed by a full LLM review of the plan's internal consistency.
# -----------------------------------------------------------------------
_section("Stage 1.5 — Plan Critique  (CritiqueAgent)")

if len(plan.tasks) < 3:
    print(f"  Skipped: {len(plan.tasks)} task(s) — critique only runs on 3+ task plans")
    critique = None
    critique_tokens = 0
else:
    print(f"  Reviewing {len(plan.tasks)}-task plan...")
    print()
    print(f"  Step 1: Deterministic checks (zero LLM cost)")
    print(f"    Simulates TaskRouter keyword scoring for each task.")
    print(f"    Checks: contract type vs agent type, context format mismatches.")
    print()
    print(f"  Step 2: LLM critique pass")
    print(f"    Looks for: missing tasks, wrong agent routing, over-decomposition,")
    print(f"    context mismatches, vague contracts.")
    print()

    critique, critique_tokens = CritiqueAgent.critique_plan(plan)
    plan.token_usage += critique_tokens

    blocking = [i for i in critique.issues if i.severity == "blocking"]
    warnings  = [i for i in critique.issues if i.severity == "warning"]

    print()
    status_str = "APPROVED" if critique.approved else "REJECTED (revision triggered)"
    print(f"  Verdict         : {status_str}")
    print(f"  Blocking issues : {len(blocking)}")
    print(f"  Warnings        : {len(warnings)}")
    print(f"  Tokens used     : {critique_tokens:,}")

    if critique.issues:
        print()
        print(f"  Issues:")
        for issue in critique.issues:
            sev = "BLOCKING" if issue.severity == "blocking" else "warning "
            print(f"    [{sev}]  Task {issue.task_order:2d}  [{issue.issue_type}]")
            print(f"              {issue.description}")
            print(f"              Fix: {issue.suggested_fix}")
            print()

    if critique.revised_goals:
        print(f"  Revisions applied ({len(critique.revised_goals)}):")
        for task_order, new_goal in critique.revised_goals.items():
            for t in plan.tasks:
                if t.task_order == task_order:
                    print(f"    Task {task_order} before: {t.goal}")
                    print(f"    Task {task_order} after : {new_goal}")

    if critique.critique_reasoning:
        print()
        print(f"  LLM reasoning   : {critique.critique_reasoning}")


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
