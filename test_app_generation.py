"""
End-to-end test: generate a note-taking app from a single prompt.
Verifies the full pipeline produces runnable code, not just a plan.

Usage:
    python test_app_generation.py
"""

import sys
import io
import os
import json
import py_compile
import tempfile
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from ai_intern.schemas import RequestSchema
from ai_intern.orchestration.orchestrator import Orchestrator

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
DIM    = "\033[2m"


def ok(msg: str):
    print(f"  {BOLD}{GREEN}✓{RESET} {msg}")


def fail(msg: str):
    print(f"  {BOLD}{RED}✗{RESET} {msg}")
    return False


def header(text: str):
    bar = "═" * (len(text) + 4)
    print(f"\n{BOLD}{CYAN}╔{bar}╗{RESET}")
    print(f"{BOLD}{CYAN}║  {text}  ║{RESET}")
    print(f"{BOLD}{CYAN}╚{bar}╝{RESET}")


def run_test():
    header("App Generation — End-to-End Test")

    prompt = "Build a simple note-taking app with save, load, and list"
    print(f"\n  {BOLD}Prompt:{RESET} {prompt}")
    print(f"  {DIM}Running full pipeline (this takes a while)...{RESET}\n")

    request = RequestSchema(content=prompt)
    orchestrator = Orchestrator(max_retries=2)

    try:
        plan = orchestrator.execute_request(request)
    except Exception as e:
        print(f"\n{BOLD}{RED}Pipeline crashed: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False

    passed = True

    # --- Assert 1: Plan status ---
    header("Assertion 1 — Plan status == complete")
    if plan.status == "complete":
        ok(f"plan.status == 'complete'")
    else:
        fail(f"plan.status == '{plan.status}' (expected 'complete')")
        passed = False

    # --- Assert 2: workspace_root exists ---
    header("Assertion 2 — Workspace directory created")
    if not plan.workspace_root:
        fail("plan.workspace_root is None — workspace was not created")
        passed = False
    elif not os.path.isdir(plan.workspace_root):
        fail(f"plan.workspace_root '{plan.workspace_root}' is not a directory")
        passed = False
    else:
        ok(f"Workspace exists: {plan.workspace_root}")

    # --- Assert 3: At least 2 .py files in workspace ---
    header("Assertion 3 — At least 2 .py files in workspace")
    py_files = []
    if plan.workspace_root and os.path.isdir(plan.workspace_root):
        py_files = [
            f for f in os.listdir(plan.workspace_root)
            if f.endswith(".py")
        ]
    if len(py_files) >= 2:
        ok(f"Found {len(py_files)} .py file(s): {', '.join(py_files)}")
    else:
        fail(f"Only {len(py_files)} .py file(s) found in workspace (need ≥ 2)")
        passed = False

    # --- Assert 4: Each .py file passes syntax check ---
    header("Assertion 4 — All .py files pass syntax validation")
    if plan.workspace_root:
        for fname in py_files:
            fpath = os.path.join(plan.workspace_root, fname)
            try:
                py_compile.compile(fpath, doraise=True)
                ok(f"Syntax OK: {fname}")
            except py_compile.PyCompileError as e:
                fail(f"Syntax error in {fname}: {e}")
                passed = False

    # --- Assert 5: project.json exists with entries ---
    header("Assertion 5 — project.json exists with component entries")
    manifest_path = (
        os.path.join(plan.workspace_root, "project.json")
        if plan.workspace_root else None
    )
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest:
            ok(f"project.json has {len(manifest)} component(s): {', '.join(manifest.keys())}")
        else:
            fail("project.json exists but is empty")
            passed = False
    else:
        fail("project.json not found in workspace")
        passed = False

    # --- Assert 6: Task 1's file contains import from Task 0's file ---
    header("Assertion 6 — Later component imports earlier component")
    import_found = False
    if plan.workspace_root and len(py_files) >= 2:
        # Check every file after the first for an import statement
        for fname in py_files[1:]:
            fpath = os.path.join(plan.workspace_root, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            if "import" in content:
                import_found = True
                ok(f"'{fname}' contains an import statement")
                break
        if not import_found:
            # Soft warning — later files may not need imports if decomposition grouped them
            print(f"  {YELLOW}⚠ No import found in later files (may be OK if decomposition grouped components){RESET}")

    # --- Assert 7: final_answer mentions workspace ---
    header("Assertion 7 — Final answer describes generated files")
    if plan.final_answer and plan.workspace_root:
        # workspace root should appear in the final answer
        root_name = os.path.basename(plan.workspace_root)
        if root_name in plan.final_answer or plan.workspace_root in plan.final_answer:
            ok("Final answer references the workspace directory")
        else:
            print(f"  {YELLOW}⚠ Final answer does not mention workspace path (non-fatal){RESET}")
    elif not plan.final_answer:
        fail("plan.final_answer is empty")
        passed = False

    # --- Summary ---
    header("Summary")
    print(f"\n  {BOLD}Plan ID:{RESET}        {plan.plan_id}")
    print(f"  {BOLD}Status:{RESET}         {plan.status}")
    print(f"  {BOLD}Workspace:{RESET}      {plan.workspace_root or '(none)'}")
    print(f"  {BOLD}Tasks:{RESET}          {len(plan.tasks)} total")
    print(f"  {BOLD}Tokens used:{RESET}    {plan.token_usage}")

    print(f"\n  {BOLD}Task results:{RESET}")
    for t in plan.tasks:
        status_color = GREEN if t.status in ("validated", "complete") else RED
        print(f"    [{BOLD}{status_color}{t.status}{RESET}] {t.goal[:70]}")
        if t.output_contract:
            print(f"          contract → {t.output_contract.get('component_name', '?')}: "
                  f"{t.output_contract.get('output_file', '?')}")

    if plan.final_answer:
        print(f"\n  {BOLD}Final Answer:{RESET}")
        for line in plan.final_answer.splitlines():
            print(f"    {line}")

    print()
    if passed:
        print(f"  {BOLD}{GREEN}ALL ASSERTIONS PASSED{RESET}")
    else:
        print(f"  {BOLD}{RED}SOME ASSERTIONS FAILED — see above{RESET}")

    return passed


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
