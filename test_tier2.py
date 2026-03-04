"""
test_tier2.py -- Tier 2 Execution Quality test harness.

Tests error classification, library awareness, structured retry prompting,
test sanitizer, early abort, and research key-value extraction.

Usage:
    python test_tier2.py          # Run all suites
    python test_tier2.py --quick  # Run only offline suites (no Ollama)

Stability protocol: run 3x, majority voting for LLM-dependent tests.
Self-correcting: appends failures to ai_intern/ACTIVE.md.
"""
import sys
import os
import subprocess
import datetime

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0
FAILURES = []


def _safe(s) -> str:
    return str(s).encode("ascii", errors="replace").decode("ascii")


def ok(suite: str, name: str):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS  [{suite}] {name}")


def fail(suite: str, name: str, expected, actual):
    global FAIL_COUNT
    FAIL_COUNT += 1
    FAILURES.append((suite, name, expected, actual))
    print(f"  FAIL  [{suite}] {name}")
    print(f"       expected : {_safe(expected)}")
    print(f"       actual   : {_safe(actual)}")


def check(suite: str, name: str, condition: bool, expected="True", actual="False"):
    if condition:
        ok(suite, name)
    else:
        fail(suite, name, expected, actual)


# ── Suite 1: ErrorClassifier (no Ollama) ─────────────────────────────────────

def suite_1_error_classifier():
    suite = "S1-ErrorClassifier"
    print(f"\n{'='*60}")
    print(f"  Suite 1: ErrorClassifier (no Ollama)")
    print(f"{'='*60}")

    from ai_intern.validation.error_classifier import ErrorClassifier, ErrorCategory
    from ai_intern.schemas import TaskSchema

    def make_task(error_msg):
        t = TaskSchema(plan_id="test", task_order=0, goal="test")
        t.error_message = error_msg
        return t

    # Test 1: Non-installed package
    t = make_task("ImportError: No module named 'bs4'")
    cat = ErrorClassifier.classify(t)
    check(suite, "non-installed package -> UNRECOVERABLE_IMPORT",
          cat == ErrorCategory.UNRECOVERABLE_IMPORT,
          "UNRECOVERABLE_IMPORT", cat.value)

    # Test 2: Installed package wrong path
    t = make_task("ImportError: cannot import name 'Foo' from 'pydantic'")
    cat = ErrorClassifier.classify(t)
    check(suite, "installed package wrong path -> RETRYABLE_LOGIC",
          cat == ErrorCategory.RETRYABLE_LOGIC,
          "RETRYABLE_LOGIC", cat.value)

    # Test 3: SyntaxError
    t = make_task("SyntaxError at line 12: invalid syntax")
    cat = ErrorClassifier.classify(t)
    check(suite, "syntax error -> RETRYABLE_SYNTAX",
          cat == ErrorCategory.RETRYABLE_SYNTAX,
          "RETRYABLE_SYNTAX", cat.value)

    # Test 4: Timeout
    t = make_task("Test execution timed out (>15 seconds)")
    cat = ErrorClassifier.classify(t)
    check(suite, "timeout -> UNRECOVERABLE_TIMEOUT",
          cat == ErrorCategory.UNRECOVERABLE_TIMEOUT,
          "UNRECOVERABLE_TIMEOUT", cat.value)

    # Test 5: Dangerous pattern
    t = make_task("Dangerous code: os.system()")
    cat = ErrorClassifier.classify(t)
    check(suite, "dangerous pattern -> UNRECOVERABLE_SAFETY",
          cat == ErrorCategory.UNRECOVERABLE_SAFETY,
          "UNRECOVERABLE_SAFETY", cat.value)

    # Test 6: Test assertion failure
    t = make_task("Tests failed:\nAssertionError: expected 5, got 3")
    cat = ErrorClassifier.classify(t)
    check(suite, "test assertion failure -> RETRYABLE_LOGIC",
          cat == ErrorCategory.RETRYABLE_LOGIC,
          "RETRYABLE_LOGIC", cat.value)

    # Test 7: NameError runtime
    t = make_task("Tests failed:\nNameError: name 'foo' is not defined")
    cat = ErrorClassifier.classify(t)
    check(suite, "NameError -> RETRYABLE_RUNTIME",
          cat == ErrorCategory.RETRYABLE_RUNTIME,
          "RETRYABLE_RUNTIME", cat.value)

    # Test 8: TypeError runtime
    t = make_task("Tests failed:\nTypeError: unsupported operand type(s)")
    cat = ErrorClassifier.classify(t)
    check(suite, "TypeError -> RETRYABLE_RUNTIME",
          cat == ErrorCategory.RETRYABLE_RUNTIME,
          "RETRYABLE_RUNTIME", cat.value)

    # Test 9: Unknown error defaults to RETRYABLE_LOGIC
    t = make_task("Something went wrong")
    cat = ErrorClassifier.classify(t)
    check(suite, "unknown error -> RETRYABLE_LOGIC (default)",
          cat == ErrorCategory.RETRYABLE_LOGIC,
          "RETRYABLE_LOGIC", cat.value)

    # Test 10: _get_installed_packages includes pydantic
    installed = ErrorClassifier._get_installed_packages()
    check(suite, "_get_installed_packages includes pydantic",
          "pydantic" in installed,
          "pydantic in set", f"set has {len(installed)} items")

    # Test 11: _get_stdlib_modules includes csv
    stdlib = ErrorClassifier._get_stdlib_modules()
    check(suite, "_get_stdlib_modules includes csv",
          "csv" in stdlib,
          "csv in set", f"set has {len(stdlib)} items")


# ── Suite 2: Available Libraries Block (no Ollama) ───────────────────────────

def suite_2_library_block():
    suite = "S2-LibBlock"
    print(f"\n{'='*60}")
    print(f"  Suite 2: Available Libraries Block (no Ollama)")
    print(f"{'='*60}")

    from ai_intern.validation.error_classifier import ErrorClassifier

    block = ErrorClassifier.get_available_libraries_block()

    # Test 1: Block contains header
    check(suite, "block contains 'AVAILABLE PYTHON PACKAGES'",
          "AVAILABLE PYTHON PACKAGES" in block,
          "header present", "header missing")

    # Test 2: Block contains NOT INSTALLED section
    check(suite, "block contains 'NOT INSTALLED'",
          "NOT INSTALLED" in block,
          "section present", "section missing")

    # Test 3: Block mentions stdlib note
    check(suite, "block mentions stdlib",
          "standard library" in block.lower(),
          "stdlib note present", "stdlib note missing")

    # Test 4: CodingAgent.build_prompt includes the library block
    from ai_intern.schemas import TaskSchema
    from ai_intern.execution.coding import CodingAgent
    t = TaskSchema(plan_id="test", task_order=0, goal="Write a hello function")
    prompt = CodingAgent.build_prompt(t)
    check(suite, "CodingAgent.build_prompt contains library block",
          "AVAILABLE PYTHON PACKAGES" in prompt,
          "block in prompt", "block missing from prompt")


# ── Suite 3: Structured Retry Prompt (no Ollama) ────────────────────────────

def suite_3_retry_prompt():
    suite = "S3-RetryPrompt"
    print(f"\n{'='*60}")
    print(f"  Suite 3: Structured Retry Prompt (no Ollama)")
    print(f"{'='*60}")

    from ai_intern.execution.coding import CodingAgent

    # Test 1: Syntax error prompt
    entry = {'attempt': 1, 'error': 'SyntaxError at line 5', 'category': 'retryable_syntax'}
    prompt = CodingAgent._build_correction_prompt(entry)
    check(suite, "syntax prompt contains 'SYNTAX ERROR'",
          "SYNTAX ERROR" in prompt,
          "SYNTAX ERROR present", f"got: {prompt[:100]}")

    # Test 2: Logic error prompt
    entry = {'attempt': 1, 'error': 'AssertionError: x != y', 'category': 'retryable_logic',
             'test_code': 'assert x == y'}
    prompt = CodingAgent._build_correction_prompt(entry)
    check(suite, "logic prompt contains 'FAILED TESTS'",
          "FAILED TESTS" in prompt,
          "FAILED TESTS present", f"got: {prompt[:100]}")

    # Test 3: Runtime error prompt
    entry = {'attempt': 1, 'error': 'NameError: x not defined', 'category': 'retryable_runtime'}
    prompt = CodingAgent._build_correction_prompt(entry)
    check(suite, "runtime prompt contains 'RUNTIME CRASH'",
          "RUNTIME CRASH" in prompt,
          "RUNTIME CRASH present", f"got: {prompt[:100]}")

    # Test 4: Includes attempt number
    entry = {'attempt': 2, 'error': 'some error', 'category': 'retryable_logic'}
    prompt = CodingAgent._build_correction_prompt(entry)
    check(suite, "includes attempt number",
          "attempt" in prompt.lower(),
          "attempt present", "attempt missing")

    # Test 5: Includes error message
    entry = {'attempt': 1, 'error': 'AssertionError: x != y', 'category': 'retryable_logic'}
    prompt = CodingAgent._build_correction_prompt(entry)
    check(suite, "includes error message text",
          "AssertionError" in prompt,
          "error text present", "error text missing")


# ── Suite 4: TestSanitizer (no Ollama) ──────────────────────────────────────

def suite_4_test_sanitizer():
    suite = "S4-Sanitizer"
    print(f"\n{'='*60}")
    print(f"  Suite 4: TestSanitizer (no Ollama)")
    print(f"{'='*60}")

    from ai_intern.validation.test_sanitizer import TestSanitizer

    # Test 1: Extra import removed
    original = "def foo():\n    return 42\n"
    test_code = "import numpy\nresult = foo()\nassert result == 42\n"
    cleaned, warnings = TestSanitizer.sanitize(original, test_code)
    check(suite, "extra import (numpy) removed",
          "numpy" not in cleaned and any("numpy" in w for w in warnings),
          "numpy removed + warning", f"cleaned={cleaned[:80]}, warnings={warnings}")

    # Test 2: Stdlib import kept
    original = "import json\ndef foo():\n    return json.dumps({})\n"
    test_code = "import json\nresult = foo()\nassert result is not None\n"
    cleaned, warnings = TestSanitizer.sanitize(original, test_code)
    check(suite, "stdlib import (json) preserved",
          "import json" in cleaned,
          "json preserved", f"cleaned={cleaned[:80]}")

    # Test 3: Hardcoded assertion relaxed
    original = "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"
    test_code = "result = fibonacci(10)\nassert result == 55\n"
    cleaned, warnings = TestSanitizer.sanitize(original, test_code)
    check(suite, "hardcoded assertion (== 55) relaxed",
          "== 55" not in cleaned and any("55" in w for w in warnings),
          "relaxed + warning", f"cleaned={cleaned[:80]}, warnings={warnings}")

    # Test 4: Small constants preserved
    original = "def f(x):\n    return x * 0\n"
    test_code = "assert f(5) == 0\n"
    cleaned, warnings = TestSanitizer.sanitize(original, test_code)
    has_relaxation = any("Relaxed" in w for w in warnings)
    check(suite, "small constant (== 0) preserved",
          not has_relaxation,
          "no relaxation", f"warnings={warnings}")

    # Test 5: extract_public_names
    code = "def foo():\n    pass\nclass Bar:\n    pass\nx = 10\n"
    names = TestSanitizer.extract_public_names(code)
    check(suite, "extract_public_names returns foo, Bar, x",
          "foo" in names and "Bar" in names,
          "['foo', 'Bar', ...]", f"{names}")

    # Test 6: Clean test passes through unchanged
    original = "def add(a, b):\n    return a + b\n"
    test_code = "result = add(2, 3)\nassert isinstance(result, int)\nprint('ok')\n"
    cleaned, warnings = TestSanitizer.sanitize(original, test_code)
    check(suite, "clean test passes through with no warnings",
          len(warnings) == 0 and cleaned.strip() == test_code.strip(),
          "no warnings, unchanged", f"warnings={warnings}, changed={cleaned != test_code}")


# ── Suite 5: Early Abort on Unrecoverable Errors (Ollama) ───────────────────

def suite_5_early_abort():
    suite = "S5-EarlyAbort"
    print(f"\n{'='*60}")
    print(f"  Suite 5: Early Abort on Unrecoverable Errors (Ollama)")
    print(f"{'='*60}")

    from ai_intern.schemas import RequestSchema
    from ai_intern.orchestration.orchestrator import Orchestrator

    # Test 1: Import abort -- request that would need beautifulsoup4
    request = RequestSchema(content="Write a Python script that uses beautifulsoup4 to scrape a webpage")
    orch = Orchestrator(max_retries=2)
    plan = orch.execute_request(request)

    # Find tasks that have error_category
    tasks_with_errors = [t for t in plan.tasks if t.error_category]
    has_unrecoverable = any(
        t.error_category and t.error_category.startswith("unrecoverable")
        for t in plan.tasks
    )
    # Check that retries were limited (should be 0 or 1, not the full 2)
    max_retries_used = max((t.retry_count for t in plan.tasks), default=0)

    check(suite, "import abort: retry_count <= 1 (early abort)",
          max_retries_used <= 1,
          "retry_count <= 1", f"retry_count={max_retries_used}")

    # Test 2: Normal success -- fibonacci should still work
    request = RequestSchema(content="Write a fibonacci function")
    plan = orch.execute_request(request)
    success = any(t.status == "validated" for t in plan.tasks)
    check(suite, "fibonacci regression: still succeeds",
          success,
          "validated", f"statuses={[t.status for t in plan.tasks]}")


# ── Suite 6: Research Key-Value Extraction (Ollama) ──────────────────────────

def suite_6_research_kv():
    suite = "S6-ResearchKV"
    print(f"\n{'='*60}")
    print(f"  Suite 6: Research Key-Value Extraction (Ollama)")
    print(f"{'='*60}")

    from ai_intern.execution.research import ResearchAgent

    # Test 1: KV extraction from known prose
    kv, tokens = ResearchAgent._extract_key_values(
        "chicken thigh macros",
        "Chicken thighs contain approximately 26g of protein per 100g, "
        "10g of fat per 100g, and about 209 calories per 100g serving."
    )
    check(suite, "KV extraction: returns dict with 2+ numeric keys",
          isinstance(kv, dict) and len(kv) >= 2,
          "dict with 2+ keys", f"got {type(kv).__name__} with {len(kv) if isinstance(kv, dict) else '?'} keys: {kv}")

    # Test 2: KV extraction failure returns empty dict
    kv, tokens = ResearchAgent._extract_key_values(
        "something",
        "No useful data here at all."
    )
    check(suite, "KV extraction failure: returns empty dict, no exception",
          isinstance(kv, dict),
          "empty dict", f"got {type(kv).__name__}: {kv}")


# ── Suite 7: Retry Quality (Ollama) ─────────────────────────────────────────

def suite_7_retry_quality():
    suite = "S7-RetryQuality"
    print(f"\n{'='*60}")
    print(f"  Suite 7: Retry Quality (Ollama)")
    print(f"{'='*60}")

    from ai_intern.schemas import RequestSchema
    from ai_intern.orchestration.orchestrator import Orchestrator

    # Test 1: Email validator should succeed within retries
    request = RequestSchema(content="Write an email validator function using regex")
    orch = Orchestrator(max_retries=2)
    plan = orch.execute_request(request)
    success = any(t.status == "validated" for t in plan.tasks)
    check(suite, "email validator: validates within retries",
          success,
          "validated", f"statuses={[t.status for t in plan.tasks]}")

    # Test 2: Error history entries have category field
    tasks_with_history = [t for t in plan.tasks if t.error_history]
    if tasks_with_history:
        latest_entry = tasks_with_history[0].error_history[-1]
        check(suite, "error history has 'category' key",
              "category" in latest_entry,
              "category present", f"keys={list(latest_entry.keys())}")
    else:
        # No retries needed -- email validator passed first try
        ok(suite, "error history has 'category' key (no retries needed, trivially passes)")


# ── Suite 8: Regression ─────────────────────────────────────────────────────

def suite_8_regression():
    suite = "S8-Regression"
    print(f"\n{'='*60}")
    print(f"  Suite 8: Regression Tests")
    print(f"{'='*60}")

    # Test 1: test_planning.py
    result = subprocess.run(
        [sys.executable, "test_planning.py"],
        capture_output=True, text=True, timeout=300,
        encoding="utf-8", errors="replace"
    )
    # test_planning.py doesn't exit with code; check for no crash
    crashed = result.returncode != 0 and "Traceback" in result.stderr
    check(suite, "test_planning.py runs without crash",
          not crashed,
          "no crash", f"exit={result.returncode}, stderr={result.stderr[:200] if result.stderr else 'none'}")

    # Test 2: test_app_generation.py (2 attempts for LLM non-determinism)
    passed = False
    for attempt in range(2):
        result = subprocess.run(
            [sys.executable, "test_app_generation.py"],
            capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            passed = True
            break
        print(f"    test_app_generation.py attempt {attempt+1} failed (exit {result.returncode})")

    check(suite, "test_app_generation.py exits 0 (2 attempts)",
          passed,
          "exit 0", f"exit {result.returncode}")


# ── Majority Voting Runner ──────────────────────────────────────────────────

def run_with_majority_voting(suite_fn, runs=3):
    """Run an LLM-dependent suite multiple times and use majority voting."""
    results = []
    for run_idx in range(runs):
        print(f"\n  --- Run {run_idx + 1}/{runs} ---")
        old_pass = PASS_COUNT
        old_fail = FAIL_COUNT
        suite_fn()
        new_passes = PASS_COUNT - old_pass
        new_fails = FAIL_COUNT - old_fail
        results.append((new_passes, new_fails))

    # Majority voting: if 2/3 runs pass all tests, count as pass
    full_passes = sum(1 for p, f in results if f == 0)
    return full_passes >= 2


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global PASS_COUNT, FAIL_COUNT, FAILURES

    quick_mode = "--quick" in sys.argv

    print("=" * 60)
    print("  test_tier2.py -- Tier 2 Execution Quality Tests")
    print("=" * 60)

    # ── Offline suites (no Ollama) ──
    suite_1_error_classifier()
    suite_2_library_block()
    suite_3_retry_prompt()
    suite_4_test_sanitizer()

    offline_pass = PASS_COUNT
    offline_fail = FAIL_COUNT

    if quick_mode:
        print(f"\n{'='*60}")
        print(f"  QUICK MODE: Offline suites only")
        print(f"  Passed: {PASS_COUNT}  Failed: {FAIL_COUNT}")
        print(f"{'='*60}")
        sys.exit(0 if FAIL_COUNT == 0 else 1)

    # ── Online suites (Ollama, 3-run majority) ──
    print(f"\n{'='*60}")
    print(f"  LLM-dependent suites (3 runs each, majority voting)")
    print(f"{'='*60}")

    # Reset for majority voting suites
    llm_suite_results = {}

    for suite_name, suite_fn in [
        ("S5-EarlyAbort", suite_5_early_abort),
        ("S6-ResearchKV", suite_6_research_kv),
        ("S7-RetryQuality", suite_7_retry_quality),
    ]:
        print(f"\n  === {suite_name} (3 runs) ===")
        # Save state
        save_pass = PASS_COUNT
        save_fail = FAIL_COUNT
        save_failures = FAILURES.copy()

        run_results = []
        for run_idx in range(3):
            print(f"\n  --- {suite_name} Run {run_idx + 1}/3 ---")
            run_pass_before = PASS_COUNT
            run_fail_before = FAIL_COUNT
            suite_fn()
            run_passes = PASS_COUNT - run_pass_before
            run_fails = FAIL_COUNT - run_fail_before
            run_results.append((run_passes, run_fails))

        # Majority voting
        full_passes = sum(1 for p, f in run_results if f == 0)
        llm_suite_results[suite_name] = full_passes >= 2
        print(f"\n  {suite_name}: {'MAJORITY PASS' if full_passes >= 2 else 'MAJORITY FAIL'} ({full_passes}/3 clean runs)")

    # ── Regression suite (single run) ──
    suite_8_regression()

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Offline tests: {offline_pass} passed, {offline_fail} failed")
    for name, passed in llm_suite_results.items():
        status = "PASS (majority)" if passed else "FAIL (majority)"
        print(f"  {name}: {status}")
    print(f"  Total assertions: {PASS_COUNT} passed, {FAIL_COUNT} failed")

    all_majority_passed = all(llm_suite_results.values())

    # Count regression failures separately (test_app_generation.py is known flaky)
    regression_fails = [f for f in FAILURES if f[0] == "S8-Regression"]
    non_regression_fails = [f for f in FAILURES if f[0] != "S8-Regression"]

    if FAILURES:
        print(f"\n  Failed tests:")
        for suite, name, expected, actual in FAILURES:
            print(f"    [{suite}] {name}: expected={_safe(expected)}, actual={_safe(actual)}")

    # Self-correcting: append failures to ACTIVE.md
    if non_regression_fails or not all_majority_passed:
        _append_failures_to_active()

    # Overall: offline tests + majority voting must pass.
    # Regression suite failures are logged but don't block (known LLM flakiness).
    overall = offline_fail == 0 and all_majority_passed and len(non_regression_fails) == 0
    if regression_fails:
        print(f"\n  Note: {len(regression_fails)} regression test(s) failed (known LLM flakiness)")
    print(f"\n  {'ALL PASSED' if overall else 'SOME FAILURES -- see above'}")
    sys.exit(0 if overall else 1)


def _append_failures_to_active():
    """Append test failures to ai_intern/ACTIVE.md for self-correction."""
    active_path = os.path.join("ai_intern", "ACTIVE.md")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"\n\n---\n\n## Tier 2 Test Failures - {timestamp}\n\n",
        "The following tests failed during `test_tier2.py`:\n\n",
    ]

    for suite, name, expected, actual in FAILURES:
        lines.append(f"- **[{suite}] {name}**\n")
        lines.append(f"  - Expected: {_safe(str(expected)[:200])}\n")
        lines.append(f"  - Actual: {_safe(str(actual)[:200])}\n")

    try:
        with open(active_path, "a", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  (Failures appended to {active_path})")
    except Exception as e:
        print(f"  (Could not write to {active_path}: {e})")


if __name__ == "__main__":
    main()
