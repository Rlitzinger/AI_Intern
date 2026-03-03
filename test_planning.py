"""
Planning diagnostic tool - runs only the planning stage and prints rich detail.

Usage:
    python test_planning.py
    python test_planning.py "your custom task here"

Tests the full planning pipeline in isolation:
  - TaskClassifier (2x2 complexity/ambiguity verdict)
  - Intermediate walk-through (clarify / decompose steps individually)
  - HierarchicalPlanner (full recursive decomposition)
  - Flat PlanningAgent (fallback mode, for comparison)
"""

import sys
import io
import time
import traceback

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from ai_intern.schemas import RequestSchema, TaskSchema
from ai_intern.planning.classifier import TaskClassifier, TaskVerdict
from ai_intern.planning.hierarchical import HierarchicalPlanner
from ai_intern.planning.planner import PlanningAgent
from ai_intern.config import settings

# ─────────────────────────────────────────────
# ANSI color helpers
# ─────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"

def color(text, *codes):
    return "".join(codes) + str(text) + RESET

def header(text):
    bar = "═" * (len(text) + 4)
    print(f"\n{color(f'╔{bar}╗', BOLD, CYAN)}")
    print(f"{color(f'║  {text}  ║', BOLD, CYAN)}")
    print(f"{color(f'╚{bar}╝', BOLD, CYAN)}")

def section(text):
    print(f"\n{color('┌─ ' + text, BOLD, BLUE)}")

def subsection(text):
    print(f"  {color('│  ' + text, BOLD, YELLOW)}")

def field(label, value, indent=4):
    pad = " " * indent
    print(f"{pad}{color(label + ':', BOLD)} {value}")

def dimfield(label, value, indent=4):
    pad = " " * indent
    print(f"{pad}{color(label + ':', BOLD)} {color(value, DIM)}")

def error_box(label: str, err: Exception):
    print(f"\n  {color('ERROR in ' + label + ':', BOLD, RED)}")
    print(f"  {color(type(err).__name__ + ': ' + str(err), RED)}")
    for line in traceback.format_exc().strip().splitlines()[-6:]:
        print(f"    {color(line, DIM)}")

def verdict_color(verdict: TaskVerdict) -> str:
    colors = {
        TaskVerdict.EXECUTE:                GREEN,
        TaskVerdict.DECOMPOSE:              YELLOW,
        TaskVerdict.CLARIFY:               MAGENTA,
        TaskVerdict.CLARIFY_THEN_DECOMPOSE: RED,
    }
    return color(f"[{verdict.value.upper()}]", BOLD, colors.get(verdict, RESET))

def arrow(label):
    print(f"    {color('→ ' + label, CYAN)}")

# ─────────────────────────────────────────────
# Step 1: Isolated classifier
# ─────────────────────────────────────────────
def run_classifier(goal: str):
    """Returns (verdict, reasoning, tokens) or None on failure."""
    section("Step 1 — TaskClassifier")
    task = TaskSchema(plan_id="test", task_order=0, goal=goal)

    field("Goal", goal)
    print(f"  {color('Calling Ollama...', DIM)}", end="", flush=True)

    try:
        t0 = time.perf_counter()
        verdict, reasoning, tokens, is_app_scale = TaskClassifier.classify(task)
        elapsed = time.perf_counter() - t0

        print(f"\r  {color('Result:', BOLD)}")
        field("Verdict",   verdict_color(verdict))
        field("Complex?",  color("Yes", YELLOW) if verdict in (TaskVerdict.DECOMPOSE, TaskVerdict.CLARIFY_THEN_DECOMPOSE) else color("No", GREEN))
        field("Ambiguous?",color("Yes", MAGENTA) if verdict in (TaskVerdict.CLARIFY, TaskVerdict.CLARIFY_THEN_DECOMPOSE) else color("No", GREEN))
        field("App-scale?",color("Yes", YELLOW) if is_app_scale else color("No", GREEN))
        dimfield("Reasoning", reasoning)
        field("Tokens", f"{tokens}  |  Time: {elapsed:.2f}s")
        return verdict, reasoning, tokens, is_app_scale

    except Exception as e:
        print()
        error_box("TaskClassifier", e)
        return None

# ─────────────────────────────────────────────
# Step 2: Intermediate walk-through
# Manually calls _clarify_task / _decompose_task
# to expose what the LLM produces at each inner step.
# ─────────────────────────────────────────────
def run_intermediate(goal: str, verdict: TaskVerdict):
    """
    Manually invoke the internal planner methods based on the verdict.
    Shows clarified goal, extracted_details, and raw subtask list.
    Returns total tokens used.
    """
    section("Step 2 — Intermediate Walk-through")
    task = TaskSchema(plan_id="test", task_order=0, goal=goal)
    context = {"parent_goal": None}
    total_tokens = 0

    # ── EXECUTE: nothing to walk through ──────
    if verdict == TaskVerdict.EXECUTE:
        print(f"  {color('Verdict is EXECUTE — no intermediate steps needed.', DIM)}")
        print(f"  {color('Task is clear and simple enough to run directly.', DIM)}")
        return 0

    # ── CLARIFY (with or without decompose) ───
    if verdict in (TaskVerdict.CLARIFY, TaskVerdict.CLARIFY_THEN_DECOMPOSE):
        subsection("Clarification step")
        print(f"  {color('Calling Ollama...', DIM)}", end="", flush=True)
        try:
            t0 = time.perf_counter()
            clarified_task, tokens = HierarchicalPlanner._clarify_task(task, context)
            elapsed = time.perf_counter() - t0
            total_tokens += tokens

            print(f"\r")
            field("Original goal",   color(goal, DIM))
            arrow("Clarified goal")
            field("Clarified goal",  color(clarified_task.goal, GREEN))

            # Re-run classifier on the clarified goal so we can see if it changed
            print(f"\n    {color('Re-classifying clarified goal...', DIM)}", end="", flush=True)
            clarified_schema = TaskSchema(plan_id="test", task_order=0, goal=clarified_task.goal)
            new_verdict, new_reasoning, cls_tokens = TaskClassifier.classify(clarified_schema)
            total_tokens += cls_tokens
            print(f"\r    {color('Clarified verdict:', BOLD)} {verdict_color(new_verdict)}")
            dimfield("Reasoning", new_reasoning, indent=4)
            field("Tokens", f"{tokens} (clarify) + {cls_tokens} (re-classify)  |  Time: {elapsed:.2f}s")

            # Use clarified task for decomposition step below
            task = clarified_task

        except Exception as e:
            print()
            error_box("_clarify_task", e)
            # Fall through with original task

    # ── DECOMPOSE (with or without prior clarify) ──
    if verdict in (TaskVerdict.DECOMPOSE, TaskVerdict.CLARIFY_THEN_DECOMPOSE):
        subsection("Decomposition step")
        field("Task to decompose", color(task.goal, DIM))
        print(f"  {color('Calling Ollama...', DIM)}", end="", flush=True)
        try:
            t0 = time.perf_counter()
            subtasks, tokens = HierarchicalPlanner._decompose_task(task, context)
            elapsed = time.perf_counter() - t0
            total_tokens += tokens

            print(f"\r  {color('Raw subtasks generated:', BOLD)} ({len(subtasks)} tasks in {elapsed:.2f}s, {tokens} tok)")
            for i, st in enumerate(subtasks):
                badge = color(f"[{i}]", BOLD, CYAN)
                print(f"    {badge} {st.goal}")

            # For each subtask, run the classifier to show what it would do next
            print(f"\n    {color('Subtask classification preview:', BOLD)}")
            for i, st in enumerate(subtasks):
                if HierarchicalPlanner._is_obviously_simple(st.goal):
                    badge = color(f"[{i}]", BOLD, CYAN)
                    print(f"    {badge} {color('[obviously simple — skips classifier]', DIM)} {st.goal[:60]}")
                else:
                    print(f"    {color('Calling Ollama...', DIM)}", end="", flush=True)
                    st_schema = TaskSchema(plan_id="test", task_order=i, goal=st.goal)
                    sub_verdict, sub_reasoning, sub_tokens = TaskClassifier.classify(st_schema, {"parent_goal": task.goal})
                    total_tokens += sub_tokens
                    badge = color(f"[{i}]", BOLD, CYAN)
                    print(f"\r    {badge} {verdict_color(sub_verdict)} {st.goal[:60]}")
                    dimfield("Reasoning", sub_reasoning[:100], indent=8)

        except Exception as e:
            print()
            error_box("_decompose_task", e)

    return total_tokens

# ─────────────────────────────────────────────
# Step 3: Full hierarchical plan
# ─────────────────────────────────────────────
def run_hierarchical(request: RequestSchema):
    """Returns plan or None on failure."""
    section("Step 3 — HierarchicalPlanner (full recursive result)")
    field("Input", request.content)
    dimfield("Config", f"MAX_DECOMPOSITION_DEPTH={settings.MAX_DECOMPOSITION_DEPTH}")
    print(f"  {color('Calling Ollama...', DIM)}", end="", flush=True)

    try:
        t0 = time.perf_counter()
        plan = HierarchicalPlanner.create_plan(request)
        elapsed = time.perf_counter() - t0

        print(f"\r  {color('Final plan:', BOLD)} {color(plan.plan_id, DIM)}")
        field("Tasks",  str(len(plan.tasks)))
        field("Tokens", str(plan.token_usage))
        field("Time",   f"{elapsed:.2f}s")
        print(f"\n  {color('Executable task list (DFS order):', BOLD)}")
        for task in plan.tasks:
            badge = color(f"[{task.task_order}]", BOLD, CYAN)
            print(f"    {badge} {task.goal}")
        return plan

    except Exception as e:
        print()
        error_box("HierarchicalPlanner", e)
        return None

# ─────────────────────────────────────────────
# Step 4: Flat planning agent (for comparison)
# ─────────────────────────────────────────────
def run_flat_planner(request: RequestSchema):
    """Returns plan or None on failure."""
    section("Step 4 — Flat PlanningAgent (legacy, for comparison)")
    field("Input", request.content)
    print(f"  {color('Calling Ollama...', DIM)}", end="", flush=True)

    try:
        t0 = time.perf_counter()
        plan = PlanningAgent.create_plan(request)
        elapsed = time.perf_counter() - t0

        print(f"\r  {color('Plan:', BOLD)} {color(plan.plan_id, DIM)}")
        field("Tasks",  str(len(plan.tasks)))
        field("Tokens", str(plan.token_usage))
        field("Time",   f"{elapsed:.2f}s")
        print(f"\n  {color('Task list:', BOLD)}")
        for task in plan.tasks:
            badge = color(f"[{task.task_order}]", BOLD, CYAN)
            print(f"    {badge} {task.goal}")
        return plan

    except Exception as e:
        print()
        error_box("PlanningAgent", e)
        return None

# ─────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────
DEFAULT_TEST_CASES = [
    # (label, input, expected_verdict)
    ("Simple + Clear",
     "Write a fibonacci function",
     TaskVerdict.EXECUTE),

    ("Complex + Clear",
     "Research chicken thigh macros and write a calculator function, save as calc.py",
     TaskVerdict.DECOMPOSE),

    ("Simple + Ambiguous",
     "Make it better",
     TaskVerdict.CLARIFY),

    ("Complex + Ambiguous",
     "Build a web scraper with storage and make it good",
     TaskVerdict.CLARIFY_THEN_DECOMPOSE),

    ("File read + analysis",
     "Read Workouts.csv and calculate average calories, save summary",
     TaskVerdict.DECOMPOSE),
]


def run_suite(task_inputs: list[str] | None = None):
    if task_inputs:
        cases = [(f"Custom: {t[:40]}", t, None) for t in task_inputs]
    else:
        cases = DEFAULT_TEST_CASES

    total_tokens = 0
    results = []

    for label, task_str, expected in cases:
        header(label)

        # ── Step 1: Classifier ───────────────────
        cls_result = run_classifier(task_str)
        verdict = None
        cls_tokens = 0

        if cls_result is not None:
            verdict, _, cls_tokens, _is_app_scale = cls_result
            total_tokens += cls_tokens

            if expected is not None:
                match = verdict == expected
                status = color("PASS", BOLD, GREEN) if match else color("FAIL", BOLD, RED)
                print(f"\n  Expected verdict {color(expected.value.upper(), DIM)} → {status}")

        # ── Step 2: Intermediate walk-through ────
        if verdict is not None:
            inter_tokens = run_intermediate(task_str, verdict)
            total_tokens += inter_tokens
        else:
            inter_tokens = 0

        # ── Step 3: Full hierarchical plan ───────
        request = RequestSchema(content=task_str)
        hier_plan = run_hierarchical(request)
        hier_tokens = hier_plan.token_usage if hier_plan else 0
        total_tokens += hier_tokens

        # ── Step 4: Flat planner (comparison) ────
        flat_plan = run_flat_planner(request)
        flat_tokens = flat_plan.token_usage if flat_plan else 0
        total_tokens += flat_tokens

        # ── Side-by-side comparison ───────────────
        if hier_plan is not None and flat_plan is not None:
            section("Comparison: Hierarchical vs Flat")
            hier_goals = [t.goal for t in hier_plan.tasks]
            flat_goals = [t.goal for t in flat_plan.tasks]
            max_len = max(len(hier_goals), len(flat_goals))

            print(f"  {'Hierarchical':<50}  {'Flat'}")
            print(f"  {'─'*48}  {'─'*48}")
            for i in range(max_len):
                h  = f"[{i}] {hier_goals[i]}" if i < len(hier_goals) else color("(no task)", DIM)
                f_ = f"[{i}] {flat_goals[i]}" if i < len(flat_goals) else color("(no task)", DIM)
                print(f"  {h[:48]:<50}  {f_[:48]}")

        # ── Record for summary ────────────────────
        results.append({
            "label":      label,
            "verdict":    verdict.value if verdict else color("ERROR", RED),
            "expected":   expected.value if expected else "—",
            "match":      (verdict == expected) if (verdict and expected) else None,
            "hier_tasks": len(hier_plan.tasks) if hier_plan else color("ERR", RED),
            "flat_tasks": len(flat_plan.tasks) if flat_plan else color("ERR", RED),
            "tokens":     cls_tokens + inter_tokens + hier_tokens + flat_tokens,
            "hier_error": hier_plan is None,
            "flat_error": flat_plan is None,
            "cls_error":  cls_result is None,
        })

    # ── Summary table ────────────────────────────
    header("Summary")
    print(f"  {'Label':<40} {'Verdict':<28} {'Expected':<28} {'Match':<6} {'H':>4} {'F':>4} {'Tok':>6}")
    print(f"  {'─'*40} {'─'*27} {'─'*27} {'─'*6} {'─':>4} {'─':>4} {'─':>6}")

    for r in results:
        match_str = (
            color("✓", GREEN) if r["match"] is True
            else color("✗", RED) if r["match"] is False
            else color("─", DIM)
        )
        print(f"  {r['label']:<40} {r['verdict']:<28} {r['expected']:<28} {match_str:<6} "
              f"{str(r['hier_tasks']):>4} {str(r['flat_tasks']):>4} {r['tokens']:>6}")

    errors = [r for r in results if r["cls_error"] or r["hier_error"] or r["flat_error"]]
    if errors:
        print(f"\n  {color('Errors encountered:', BOLD, RED)}")
        for r in errors:
            parts = []
            if r["cls_error"]:  parts.append("Classifier")
            if r["hier_error"]: parts.append("HierarchicalPlanner")
            if r["flat_error"]: parts.append("PlanningAgent")
            print(f"    {r['label']}: {', '.join(parts)} failed — see output above for details")

    print(f"\n  {color('Total tokens used:', BOLD)} {total_tokens}")
    print(f"  H = hierarchical task count  |  F = flat task count\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    custom_tasks = sys.argv[1:] if len(sys.argv) > 1 else None

    if custom_tasks:
        print(color(f"\nRunning {len(custom_tasks)} custom task(s)...", BOLD))
    else:
        print(color("\nRunning default test suite (5 cases)...", BOLD))
        print(color("  Pass a task as argument to test a custom input instead.", DIM))
        print(color('  Example: python test_planning.py "Research X and write a script"', DIM))

    run_suite(custom_tasks)
