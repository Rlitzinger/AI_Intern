"""
Validation tests for llm.py Outlines migration.
Run from project root: python test_llm_migration.py
"""
import time
import sys


def test_imports():
    """Verify all required packages import correctly."""
    from llama_cpp import Llama
    from outlines.models.llamacpp import LlamaCpp
    from outlines.generator import Generator
    import json
    print("PASS: all imports OK")


def test_no_bad_imports():
    """Verify the wrong import path is not used anywhere in llm.py."""
    with open("ai_intern/llm.py", "r") as f:
        content = f.read()
    assert "from outlines import generate" not in content, \
        "FAIL: llm.py still uses 'from outlines import generate' — this module does not exist"
    assert "generate.json(" not in content, \
        "FAIL: llm.py still uses generate.json() — wrong API"
    print("PASS: no bad imports in llm.py")


def test_structured_call_returns_pydantic():
    """call_ollama_structured must return a real Pydantic instance, not a string."""
    from ai_intern.llm import call_ollama_structured
    from ai_intern.planning.classifier import ClassificationResult

    result, tokens = call_ollama_structured(
        model="instruct",
        prompt="Task: Write a fibonacci function",
        system="You are a task classifier. Evaluate complexity and ambiguity.",
        response_schema=ClassificationResult,
        temperature=0.1,
    )

    assert isinstance(result, ClassificationResult), \
        f"FAIL: Expected ClassificationResult instance, got {type(result)}: {result!r}"
    assert isinstance(result.is_complex, bool), \
        f"FAIL: is_complex should be bool, got {type(result.is_complex)}"
    assert isinstance(result.is_ambiguous, bool), \
        f"FAIL: is_ambiguous should be bool, got {type(result.is_ambiguous)}"
    assert isinstance(result.reasoning, str) and len(result.reasoning) > 0, \
        f"FAIL: reasoning should be non-empty string"
    assert tokens > 0, "FAIL: tokens_used should be > 0"

    print(f"PASS: structured call returned ClassificationResult")
    print(f"      is_complex={result.is_complex}, is_ambiguous={result.is_ambiguous}")
    print(f"      reasoning={result.reasoning[:80]}...")
    print(f"      tokens={tokens}")


def test_code_call_returns_function():
    """call_ollama_code must return a string containing a Python function."""
    from ai_intern.llm import call_ollama_code

    result, tokens = call_ollama_code(
        model="coder",
        prompt="Write a Python function that returns the sum of two numbers.",
        system="You are a coding assistant. Return only Python code, no explanation.",
        temperature=0.1,
    )

    assert isinstance(result, str), \
        f"FAIL: Expected str, got {type(result)}"
    assert "def " in result, \
        f"FAIL: Expected a function definition in output. Got: {result[:300]}"
    assert tokens > 0, "FAIL: tokens_used should be > 0"

    print(f"PASS: code call returned function definition")
    print(f"      tokens={tokens}")
    print(f"      preview: {result[:150].strip()}...")


def test_singleton_no_reload():
    """Second call must reuse loaded model — must be significantly faster than first."""
    from ai_intern.llm import call_ollama_structured
    from ai_intern.planning.classifier import ClassificationResult

    # First call loads model + builds generator FSM
    t0 = time.time()
    call_ollama_structured(
        model="instruct",
        prompt="Task: Read a CSV file",
        system="You are a task classifier.",
        response_schema=ClassificationResult,
        temperature=0.1,
    )
    first_duration = time.time() - t0

    # Second call should reuse everything
    t0 = time.time()
    call_ollama_structured(
        model="instruct",
        prompt="Task: Write a sorting algorithm",
        system="You are a task classifier.",
        response_schema=ClassificationResult,
        temperature=0.1,
    )
    second_duration = time.time() - t0

    print(f"      First call:  {first_duration:.2f}s (includes model load)")
    print(f"      Second call: {second_duration:.2f}s (should reuse singleton)")

    assert second_duration < first_duration * 0.5, (
        f"FAIL: Second call ({second_duration:.2f}s) should be much faster than "
        f"first ({first_duration:.2f}s). Singleton may not be working."
    )
    print("PASS: singleton confirmed — second call significantly faster")


def test_generator_cached():
    """Generator FSM should be built once, not on every call."""
    from ai_intern import llm as llm_module
    from ai_intern.planning.classifier import ClassificationResult
    from ai_intern.llm import call_ollama_structured

    # Make two calls with the same schema
    call_ollama_structured(
        model="instruct",
        prompt="Task: Parse JSON",
        system="You are a task classifier.",
        response_schema=ClassificationResult,
        temperature=0.1,
    )
    call_ollama_structured(
        model="instruct",
        prompt="Task: Sort a list",
        system="You are a task classifier.",
        response_schema=ClassificationResult,
        temperature=0.1,
    )

    # The generator cache should have exactly one entry for this schema
    cache = llm_module._instruct_generators
    assert ClassificationResult in cache, \
        "FAIL: Generator was not cached — _instruct_generators is empty"
    assert len(cache) == 1, \
        f"FAIL: Expected 1 cached generator, found {len(cache)}"
    print(f"PASS: generator cached correctly ({len(cache)} entry in _instruct_generators)")


if __name__ == "__main__":
    print("=" * 55)
    print("  LLM Migration Validation Tests")
    print("=" * 55)
    print()

    tests = [
        test_imports,
        test_no_bad_imports,
        test_structured_call_returns_pydantic,
        test_code_call_returns_function,
        test_singleton_no_reload,
        test_generator_cached,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print(f"--- {test.__name__} ---")
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(str(e))
            failed += 1
        except Exception as e:
            print(f"FAIL: Unexpected error — {type(e).__name__}: {e}")
            failed += 1
        print()

    print("=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55)

    sys.exit(0 if failed == 0 else 1)
