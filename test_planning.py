"""
Planning diagnostic tool -- walks through every stage of the planning pipeline.

Usage:
    python test_planning.py                        # Run 5 default cases
    python test_planning.py "your goal here"       # Run a single custom goal
    python test_planning.py --quick "goal"         # Skip Step 3 (saves tokens)
    python test_planning.py --quick                # Run all 5 cases without Step 3

Stages shown per case:
  Step 1: TaskClassifier    -- 3-axis verdict: complexity x ambiguity x app-scale
  Step 2: Planning walk-through -- each inner step one at a time (clarify / spec-gen / decompose)
  Step 3: Full hierarchical plan -- complete task list with agent + contract details
"""

import sys
import io
import time
import textwrap
import traceback

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from ai_intern.schemas import RequestSchema, TaskSchema, PlanSchema
from ai_intern.planning.classifier import TaskClassifier, TaskVerdict
from ai_intern.planning.hierarchical import HierarchicalPlanner
from ai_intern.planning.spec_generator import SpecGenerator
from ai_intern.planning.environment import EnvironmentScanner
from ai_intern.planning.context import PlanningContext
from ai_intern.config import settings

# ─────────────────────────────────────────────────────────
# ANSI color helpers (ASCII-safe box drawing only)
# ─────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"


def c(text, *codes):
    return "".join(codes) + str(text) + RESET


def header(text):
    bar = "+" + "=" * (len(text) + 4) + "+"
    print(f"\n{c(bar, BOLD, CYAN)}")
    print(f"{c('|  ' + text + '  |', BOLD, CYAN)}")
    print(f"{c(bar, BOLD, CYAN)}")


def step_header(num: int, text: str):
    label = f"Step {num}: {text}"
    print(f"\n{c(label, BOLD, BLUE)}")
    print(f"  {c('-' * len(label), DIM, BLUE)}")


def sub(text: str):
    print(f"\n    {c('[ ' + text + ' ]', BOLD, YELLOW)}")


def field(label: str, value, indent: int = 6):
    print(f"{' ' * indent}{c(label + ':', BOLD)} {value}")


def dimfield(label: str, value, indent: int = 6):
    print(f"{' ' * indent}{c(label + ':', BOLD)} {c(str(value), DIM)}")


def wrapped_field(label: str, value: str, indent: int = 6):
    """Print a field whose value may be long -- wrap at ~100 chars."""
    prefix_plain = " " * indent + label + ": "
    prefix_colored = " " * indent + c(label + ":", BOLD) + " "
    avail = 100 - len(prefix_plain)
    lines = textwrap.wrap(str(value), max(avail, 30))
    print(prefix_colored + c(lines[0] if lines else "", DIM))
    for line in lines[1:]:
        print(" " * (len(prefix_plain)) + c(line, DIM))


def bullet(text: str, indent: int = 8):
    print(f"{' ' * indent}{c('-', CYAN)} {c(text, DIM)}")


def err_box(label: str, exc: Exception):
    print(f"\n  {c('ERROR [' + label + ']:', BOLD, RED)}")
    print(f"  {c(str(exc), RED)}")
    for line in traceback.format_exc().strip().splitlines()[-5:]:
        print(f"    {c(line, DIM)}")


def verdict_badge(v: TaskVerdict) -> str:
    vc = {
        TaskVerdict.EXECUTE:                GREEN,
        TaskVerdict.DECOMPOSE:              YELLOW,
        TaskVerdict.CLARIFY:               MAGENTA,
        TaskVerdict.CLARIFY_THEN_DECOMPOSE: RED,
    }
    return c(f"[{v.value.upper()}]", BOLD, vc.get(v, RESET))


def agent_badge(agent: str | None, source: str = "") -> str:
    ac = {"code": CYAN, "research": BLUE, "file": YELLOW, "analysis": MAGENTA}
    a = (agent or "?").lower()
    badge = c(f"[{a}]", BOLD, ac.get(a, DIM))
    if source:
        badge += c(f" ({source})", DIM)
    return badge


def task_agent_badge(task: TaskSchema) -> str:
    """Show declared_agent (from SubtaskSpec) or suggested_agent with source label."""
    if task.declared_agent:
        return agent_badge(task.declared_agent, "declared")
    if task.suggested_agent:
        return agent_badge(task.suggested_agent, "suggested")
    return agent_badge(None)


def yn(val: bool) -> str:
    return c("Yes", BOLD, GREEN) if val else c("No", DIM)


# ─────────────────────────────────────────────────────────
# Step 1 -- Classification
# ─────────────────────────────────────────────────────────

def run_classify(
    goal: str,
    env_prompt: str = "",
    planning_context: "PlanningContext | None" = None,
) -> "tuple | None":
    """
    Calls TaskClassifier once and shows all 3 axes of the result:
      - is_complex, is_ambiguous  (map to verdict)
      - is_app_scale  (from LLM + keyword safety-net)
    Returns (verdict, reasoning, tokens, is_app_scale_final) or None.
    """
    step_header(1, "TaskClassifier  (complexity x ambiguity x app-scale)")
    task = TaskSchema(plan_id="test", task_order=0, goal=goal)
    field("Goal", c(goal, DIM))
    if planning_context and planning_context.available_files:
        files_str = ", ".join(planning_context.available_files[:5])
        field("Files visible to classifier", c(files_str, DIM))
    print(f"      {c('Calling Ollama...', DIM)}", end="", flush=True)

    try:
        t0 = time.perf_counter()
        verdict, reasoning, tokens, is_app_scale_llm = TaskClassifier.classify(
            task,
            planning_context=planning_context,
            context={"environment": env_prompt},
        )
        elapsed = time.perf_counter() - t0
        print()

        # Keyword safety-net that also runs inside _decompose_recursive
        is_app_scale_kw    = HierarchicalPlanner._keyword_is_app_scale(goal)
        is_app_scale_final = is_app_scale_llm or is_app_scale_kw

        is_complex   = verdict in (TaskVerdict.DECOMPOSE, TaskVerdict.CLARIFY_THEN_DECOMPOSE)
        is_ambiguous = verdict in (TaskVerdict.CLARIFY,   TaskVerdict.CLARIFY_THEN_DECOMPOSE)

        print()
        field("Verdict",           verdict_badge(verdict))
        field("  complex?",        yn(is_complex))
        field("  ambiguous?",      yn(is_ambiguous))
        print()
        field("  is_app_scale (LLM)",     yn(is_app_scale_llm))

        kw_note = ""
        if is_app_scale_kw:
            matched = [kw for kw in HierarchicalPlanner._APP_SCALE_KEYWORDS if kw in goal.lower()]
            kw_note = c(f"  matched: {matched}", DIM)
        field("  is_app_scale (keyword)", yn(is_app_scale_kw) + kw_note)

        if is_app_scale_final and not is_app_scale_llm and is_complex:
            field("  OVERRIDE applied",  c("EXECUTE -> DECOMPOSE  (keyword safety net)", YELLOW))
        if is_app_scale_final and not is_app_scale_llm and not is_complex:
            field("  OVERRIDE applied",  c("CLARIFY -> CTD  (keyword safety net)", YELLOW))
        field("  final app-scale?",   yn(is_app_scale_final))

        print()
        wrapped_field("Reasoning", reasoning)
        field("Tokens / Time", f"{tokens} tok  |  {elapsed:.2f}s")

        return verdict, reasoning, tokens, is_app_scale_final

    except Exception as e:
        print()
        err_box("TaskClassifier", e)
        return None


# ─────────────────────────────────────────────────────────
# Step 2 -- Planning walk-through
# ─────────────────────────────────────────────────────────

def run_walkthrough(
    goal: str,
    verdict: TaskVerdict,
    is_app_scale: bool,
    env_prompt: str = "",
    planning_context: "PlanningContext | None" = None,
) -> int:
    """
    Manually invokes each inner planning step based on verdict + is_app_scale.
    Each LLM call is shown separately so the reasoning at each stage is visible.

    Paths covered:
      EXECUTE                    -> leaf-node analysis only (no LLM)
      CLARIFY                    -> [A] clarify
      DECOMPOSE (non-app-scale)  -> [B] decompose (LLM)  -> [C] subtask analysis
      DECOMPOSE (app-scale)      -> [B] spec-gen (LLM)   -> [C] spec-to-tasks (no LLM)
      CTD (non-app-scale)        -> [A] clarify (LLM)    -> [B] decompose (LLM)   -> [C] subtask analysis
      CTD (app-scale)            -> [B] spec-gen (LLM)   -> [C] spec-to-tasks (no LLM)
                                    (clarification skipped -- spec replaces it)

    Returns total tokens used.
    """
    step_header(2, "Planning walk-through  (inner steps, one at a time)")
    task    = TaskSchema(plan_id="test", task_order=0, goal=goal)
    ctx     = {"parent_goal": None, "environment": env_prompt}
    total   = 0

    # ──────────────────────────────────────────────────────────────────
    # EXECUTE path: nothing to plan -- just show the leaf-node analysis
    # ──────────────────────────────────────────────────────────────────
    if verdict == TaskVerdict.EXECUTE and not is_app_scale:
        print(f"\n      {c('Verdict is EXECUTE -- task is simple and clear.', DIM)}")
        print(f"      {c('No clarification or decomposition needed.', DIM)}")

        sub("Leaf-node analysis  (no LLM)")
        is_simple = HierarchicalPlanner._is_obviously_simple(goal)
        inferred  = HierarchicalPlanner._infer_agent_from_goal(goal)
        field("_is_obviously_simple", yn(is_simple),
              indent=6)
        field("Inferred agent",       agent_badge(inferred))
        return 0

    # ──────────────────────────────────────────────────────────────────
    # A: Clarification
    # Runs for: CLARIFY (single action, vague)
    #           CTD + non-app-scale (complex, vague, needs clarify before decompose)
    # Skipped for: CTD + app-scale (SpecGenerator replaces clarification)
    # ──────────────────────────────────────────────────────────────────
    clarified_task = task  # updated if clarification runs
    do_clarify = (
        verdict in (TaskVerdict.CLARIFY, TaskVerdict.CLARIFY_THEN_DECOMPOSE)
        and not is_app_scale
    )
    if do_clarify:
        sub("A: Clarification  (LLM call)")
        field("Original goal", c(goal, DIM))
        print(f"      {c('Calling Ollama (clarify)...', DIM)}", end="", flush=True)
        try:
            t0 = time.perf_counter()
            clarified_task, cl_tokens = HierarchicalPlanner._clarify_task(task, ctx)
            elapsed = time.perf_counter() - t0
            total += cl_tokens
            print()
            field("Clarified goal", c(clarified_task.goal, GREEN))
            field("Tokens / Time",  f"{cl_tokens} tok  |  {elapsed:.2f}s")
        except Exception as e:
            print()
            err_box("_clarify_task", e)

        # CLARIFY-only path ends here
        if verdict == TaskVerdict.CLARIFY:
            inferred = HierarchicalPlanner._infer_agent_from_goal(clarified_task.goal)
            field("Inferred agent", agent_badge(inferred))
            return total

    # ──────────────────────────────────────────────────────────────────
    # B: App-scale note  (for DECOMPOSE / CTD paths)
    # ──────────────────────────────────────────────────────────────────
    if is_app_scale:
        sub("B: App-scale detected -- SpecGenerator path")
        print(f"      {c('The planner skips standard LLM decomposition.', DIM)}")
        print(f"      {c('SpecGenerator produces a structured AppSpec instead.', DIM)}")
        if verdict == TaskVerdict.CLARIFY_THEN_DECOMPOSE:
            print(f"      {c('(CTD + app-scale: clarification is also skipped -- spec replaces it)', DIM)}")
    else:
        sub("B: Standard decomposition -- LLM decompose path")
        print(f"      {c('No app-spec needed. Decompose into 2-3 subtasks directly.', DIM)}")

    # ──────────────────────────────────────────────────────────────────
    # C: SpecGenerator  (app-scale only)
    # ──────────────────────────────────────────────────────────────────
    spec = None
    if is_app_scale:
        sub("C: SpecGenerator  (LLM call)")
        print(f"      {c('Calling Ollama (spec generator)...', DIM)}", end="", flush=True)
        try:
            t0 = time.perf_counter()
            spec, spec_tokens = SpecGenerator.generate(task, ctx)
            elapsed = time.perf_counter() - t0
            total += spec_tokens
            print()

            field("Summary",    c(spec.summary, GREEN))

            print()
            field("Features",   f"({len(spec.features)} items)")
            for feat in spec.features:
                bullet(feat)

            print()
            field("Components", f"({len(spec.components)} -- each should be implementable in <=40 lines)")
            for i, comp in enumerate(spec.components):
                print()
                print(f"        {c('[' + str(i) + '] ' + comp.name, BOLD, CYAN)}")
                dimfield("          Responsibility", comp.responsibility)
                dimfield("          Output file",   comp.output_file)
                field(   "          Interface",     c(comp.public_interface, YELLOW))
                deps = ", ".join(comp.depends_on) if comp.depends_on else c("(none)", DIM)
                field("          Depends on",    deps)

            print()
            field("Done criteria", f"({len(spec.done_criteria)} items)")
            for crit in spec.done_criteria:
                bullet(crit)

            print()
            field("Tokens / Time", f"{spec_tokens} tok  |  {elapsed:.2f}s")

        except Exception as e:
            print()
            err_box("SpecGenerator", e)
            spec = None

    # ──────────────────────────────────────────────────────────────────
    # D: Task creation from spec  (app-scale, no LLM)
    # ──────────────────────────────────────────────────────────────────
    if is_app_scale and spec is not None:
        sub("D: Spec-to-tasks  (deterministic, no LLM)")
        print(f"      {c('One task per component. No Ollama call. Token cost: 0.', DIM)}")
        subtasks, _ = HierarchicalPlanner._decompose_task_from_spec(task, spec)
        for i, st in enumerate(subtasks):
            print()
            print(f"        {c('[Task ' + str(i) + ']', BOLD, CYAN)} {agent_badge(st.suggested_agent)}")
            goal_lines = st.goal.split("\n")
            for j, line in enumerate(goal_lines):
                if j == 0:
                    field("          Goal", c(line, DIM))
                else:
                    print(f"          {' ' * 6}{c(line, DIM)}")
            if st.output_contract:
                oc = st.output_contract
                print(f"          {c('output_contract:', BOLD)}")
                dimfield("            component",  oc.get("component_name", ""))
                dimfield("            output_file", oc.get("output_file", ""))
                dimfield("            interface",   oc.get("public_interface", ""))

        return total

    # ──────────────────────────────────────────────────────────────────
    # D: Standard decomposition  (non-app-scale, LLM call)
    # ──────────────────────────────────────────────────────────────────
    sub("D: Standard decomposition  (LLM call)")
    decompose_target = clarified_task
    field("Task being decomposed", c(decompose_target.goal[:80], DIM))
    print(f"      {c('Calling Ollama (decompose)...', DIM)}", end="", flush=True)
    subtasks = []
    try:
        t0 = time.perf_counter()
        subtasks, decomp_tokens = HierarchicalPlanner._decompose_task(
            decompose_target, ctx, planning_context
        )
        elapsed = time.perf_counter() - t0
        total += decomp_tokens
        print()
        field("Subtasks returned", f"{len(subtasks)} in {elapsed:.2f}s  |  {decomp_tokens} tok")
        print(f"      {c('(The LLM may return up to 4; hallucination filter may remove some)', DIM)}")
        for i, st in enumerate(subtasks):
            print()
            print(f"        {c('[' + str(i) + ']', BOLD, CYAN)} {task_agent_badge(st)}")
            field("          Goal", c(st.goal, DIM))
            if st.declared_agent:
                field("          declared_agent", c(st.declared_agent, GREEN))

    except Exception as e:
        print()
        err_box("_decompose_task", e)

    # ──────────────────────────────────────────────────────────────────
    # E: Subtask analysis  (non-app-scale, static checks -- no LLM)
    # ──────────────────────────────────────────────────────────────────
    if subtasks:
        sub("E: Subtask analysis  (static, no LLM)")
        print(f"      {c('_is_obviously_simple() decides if a subtask skips the classifier.', DIM)}")
        print(f"      {c('If True -> treated as leaf node immediately (no Ollama call).', DIM)}")
        print(f"      {c('If False -> classifier would run again on this subtask.', DIM)}")
        for i, st in enumerate(subtasks):
            is_simple = HierarchicalPlanner._is_obviously_simple(st.goal)
            inferred  = HierarchicalPlanner._infer_agent_from_goal(st.goal)
            fate      = c("LEAF -- skip classifier", DIM) if is_simple else c("CLASSIFY -- LLM call", YELLOW)
            print()
            print(f"        {c('[' + str(i) + ']', BOLD, CYAN)}")
            dimfield("          Goal",                  st.goal[:70])
            if st.declared_agent:
                field("          declared_agent",        c(st.declared_agent, GREEN) + c("  (from SubtaskSpec -- routing is deterministic)", DIM))
            else:
                field("          Inferred agent",        agent_badge(inferred))
            field(   "          _is_obviously_simple", f"{yn(is_simple)}  ->  {fate}")

    return total


# ─────────────────────────────────────────────────────────
# Step 3 -- Full hierarchical plan
# ─────────────────────────────────────────────────────────

def run_full_plan(request: RequestSchema) -> "PlanSchema | None":
    """
    Calls HierarchicalPlanner.create_plan() and prints the complete integrated result.
    Shows: plan metadata, app_spec (if any), and each task with its full goal,
    suggested_agent, and output_contract.
    """
    step_header(3, "Full hierarchical plan  (complete recursive result)")
    field("Input",  c(request.content, DIM))
    dimfield("Config", f"MAX_DECOMPOSITION_DEPTH={settings.MAX_DECOMPOSITION_DEPTH}")
    print(f"      {c('Calling Ollama (full plan)...', DIM)}", end="", flush=True)

    try:
        t0 = time.perf_counter()
        plan = HierarchicalPlanner.create_plan(request)
        elapsed = time.perf_counter() - t0
        print()

        field("Plan ID", c(plan.plan_id, DIM))
        field("Tasks",   str(len(plan.tasks)))
        field("Tokens",  str(plan.token_usage))
        field("Time",    f"{elapsed:.2f}s")

        # App spec summary
        if plan.app_spec:
            sub("App spec (attached to plan by SpecGenerator)")
            sd = plan.app_spec
            field("Summary",    c(sd.get("summary", ""), GREEN))
            comps = sd.get("components", [])
            field("Components", f"({len(comps)})")
            for comp in comps:
                print()
                print(f"        {c(comp['name'], BOLD, CYAN)}")
                dimfield("          output_file", comp.get("output_file", ""))
                field(   "          interface",   c(comp.get("public_interface", ""), YELLOW))
                deps = comp.get("depends_on", [])
                field("          depends_on",  ", ".join(deps) if deps else c("(none)", DIM))

        # Final task list -- this is what the Orchestrator will execute
        sub("Executable task list  (DFS order -- what the Orchestrator runs)")
        print(f"      {c('Each task is a single agent call. Spec-driven tasks have output_contract.', DIM)}")

        for task in plan.tasks:
            print()
            badge = c(f"[{task.task_order}]", BOLD, CYAN)
            print(f"      {badge}  {task_agent_badge(task)}")

            # Show full multi-line goal (spec-driven goals span 4-5 lines)
            goal_lines = task.goal.split("\n")
            for j, line in enumerate(goal_lines):
                if j == 0:
                    field("        Goal", c(line, DIM))
                else:
                    print(f"               {c(line, DIM)}")

            # Agent routing source
            if task.declared_agent:
                field("        Routing", c(f"declared_agent={task.declared_agent}", GREEN) + c("  (from SubtaskSpec)", DIM))
            elif task.suggested_agent:
                field("        Routing", c(f"suggested_agent={task.suggested_agent}", YELLOW) + c("  (heuristic)", DIM))
            else:
                field("        Routing", c("keyword scoring (fallback)", DIM))

            # output_contract (only on spec-driven component tasks)
            if task.output_contract:
                oc = task.output_contract
                field("        Contract",   c(oc.get("component_name", ""), BOLD, CYAN))
                dimfield("          file",      oc.get("output_file", ""))
                dimfield("          interface", oc.get("public_interface", ""))

        return plan

    except Exception as e:
        print()
        err_box("HierarchicalPlanner.create_plan", e)
        return None


# ─────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────

DEFAULT_TEST_CASES = [
    # (label, goal, expected_verdict)
    #
    # Covers all 4 verdict paths + app-scale vs non-app-scale.
    # Expected verdict is used for a PASS/FAIL check at Step 1.
    # Set to None when verdict is legitimately non-deterministic.

    (
        "EXECUTE path  [simple + clear]",
        "Write a fibonacci function",
        TaskVerdict.EXECUTE,
    ),
    (
        "DECOMPOSE path  [research + code + file, non-app-scale]",
        "Research chicken thigh macros and write a macro calculator, save as macros.py",
        TaskVerdict.DECOMPOSE,
    ),
    (
        "CLARIFY path  [simple + ambiguous]",
        "Make it better",
        TaskVerdict.CLARIFY,
    ),
    (
        "DECOMPOSE path  [app-scale: note-taking app]",
        "Build a note-taking app with save, load, and list",
        TaskVerdict.DECOMPOSE,
    ),
    (
        "CTD path  [app-scale: vague app request]",
        "Build an app for my business",
        TaskVerdict.CLARIFY_THEN_DECOMPOSE,
    ),
]


# ─────────────────────────────────────────────────────────
# Per-case runner
# ─────────────────────────────────────────────────────────

def run_case(
    label: str,
    goal: str,
    expected: "TaskVerdict | None",
    quick: bool = False,
) -> "dict | None":
    header(label)

    # Scan environment once; share across all steps
    env_ctx    = EnvironmentScanner.scan()
    env_prompt = env_ctx.to_prompt_block()
    planning_ctx = PlanningContext.build()
    if env_ctx.available_files:
        files_str = ", ".join(f.filename for f in env_ctx.available_files[:5])
        print(f"  {c('Files in user_data/:', BOLD)} {c(files_str, DIM)}")

    total_tokens = 0

    # Step 1 -- classify
    cls_result = run_classify(goal, env_prompt, planning_context=planning_ctx)
    if cls_result is None:
        print(f"\n  {c('Classification failed -- skipping remaining steps.', RED)}")
        return None

    verdict, _reasoning, cls_tokens, is_app_scale = cls_result
    total_tokens += cls_tokens

    if expected is not None:
        match = (verdict == expected)
        status = c("PASS", BOLD, GREEN) if match else c("FAIL", BOLD, RED)
        print(f"\n      Expected: {verdict_badge(expected)}  ->  {status}")

    # Step 2 -- walk-through
    walk_tokens = run_walkthrough(goal, verdict, is_app_scale, env_prompt, planning_context=planning_ctx)
    total_tokens += walk_tokens

    # Step 3 -- full plan (skipped in --quick mode)
    plan = None
    if not quick:
        print(f"\n  {c('(Step 3 re-runs the planner end-to-end -- LLM calls may vary slightly.)', DIM)}")
        plan = run_full_plan(RequestSchema(content=goal))
        if plan:
            total_tokens += plan.token_usage

    print(f"\n  {c('Case total tokens:', BOLD)} {total_tokens}")

    return {
        "label":     label,
        "verdict":   verdict.value,
        "expected":  expected.value if expected else "--",
        "match":     (verdict == expected) if expected else None,
        "app_scale": is_app_scale,
        "tasks":     len(plan.tasks) if plan else "--",
        "tokens":    total_tokens,
    }


# ─────────────────────────────────────────────────────────
# Suite runner + summary
# ─────────────────────────────────────────────────────────

def run_suite(custom_goals: "list[str] | None", quick: bool):
    if custom_goals:
        cases = [(f"Custom: {g[:55]}", g, None) for g in custom_goals]
    else:
        cases = DEFAULT_TEST_CASES

    results   = []
    total_tok = 0

    for label, goal, expected in cases:
        result = run_case(label, goal, expected, quick=quick)
        if result:
            results.append(result)
            total_tok += result["tokens"]

    # Summary table
    header("Summary")
    col_w = [45, 22, 22, 7, 11, 7, 7]
    hdrs  = ["Label", "Verdict", "Expected", "Match", "App-scale", "Tasks", "Tokens"]
    print("  " + "  ".join(f"{h:<{w}}" for h, w in zip(hdrs, col_w)))
    print("  " + "  ".join("-" * w for w in col_w))

    passes = fails = 0
    for r in results:
        if r["match"] is True:
            match_str = c("PASS", GREEN)
            passes += 1
        elif r["match"] is False:
            match_str = c("FAIL", RED)
            fails += 1
        else:
            match_str = c("--", DIM)

        print(
            f"  {r['label']:<{col_w[0]}}"
            f"  {r['verdict']:<{col_w[1]}}"
            f"  {r['expected']:<{col_w[2]}}"
            f"  {match_str:<{col_w[3]}}"
            f"  {str(r['app_scale']):<{col_w[4]}}"
            f"  {str(r['tasks']):<{col_w[5]}}"
            f"  {r['tokens']:>{col_w[6]}}"
        )

    expected_count = passes + fails
    print(f"\n  Verdict accuracy:   {passes}/{expected_count}")
    print(f"  Grand total tokens: {total_tok}")

    if quick:
        print(f"  {c('(--quick mode: Step 3 was skipped for all cases)', DIM)}")
    else:
        print(f"  {c('Tokens: Step 1 (classify) + Step 2 (walk-through) + Step 3 (full plan)', DIM)}")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = sys.argv[1:]
    quick  = "--quick" in args
    goals  = [a for a in args if a != "--quick"]

    if goals:
        print(c(f"\nRunning {len(goals)} custom goal(s)...", BOLD))
        if quick:
            print(c("  (--quick: Step 3 skipped)", DIM))
    else:
        print(c("\nPlanning diagnostic  --  5 test cases", BOLD))
        print(c("  Covers: EXECUTE / DECOMPOSE / CLARIFY / CTD paths + app-scale", DIM))
        print(c("  Add --quick to skip Step 3 and save ~50% of tokens.", DIM))
        print(c('  Example: python test_planning.py "Build a REST API"', DIM))
        print(c('  Example: python test_planning.py --quick', DIM))

    run_suite(goals if goals else None, quick=quick)
