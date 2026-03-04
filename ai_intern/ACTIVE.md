# ACTIVE: Fix llm.py — Outlines API Correction

## Status
llm.py exists and is mostly correct. One critical import is wrong.
config.py is complete and does not need changes.
This task is a targeted fix, not a rewrite.

---

## What Is Already Working (DO NOT CHANGE)

- `llama-cpp-python 0.3.8` compiled from source with CUDA — GPU confirmed working
- `outlines 1.2.12` + `outlines-core` + `numba` installed and verified
- All 29/29 model layers offload to RTX 3070 (4168 MiB VRAM)
- `config.py` — complete, correct, do not modify
- `call_ollama_code()` — correct implementation using raw llama_cpp, do not change
- `call_ollama()` — compatibility shim, do not change
- Singleton pattern for `_instruct_model` / `_coder_model` — correct, do not change
- `_format_prompt()` with Qwen2.5 chat template — correct, do not change

---

## The Problem

`llm.py` currently imports:
```python
from outlines import generate
```

And uses it as:
```python
generator = generate.json(outlines_model, response_schema)
result = generator(...)
# assumes result is a Pydantic object
```

**Both of these are wrong.**

`outlines.generate` does not exist as a module in outlines 1.2.12.
Verified by running `import pkgutil; list(pkgutil.walk_packages(outlines.__path__))` —
the submodule is `outlines.generator` (with an 'r'), not `outlines.generate`.

`Generator` also returns a **string**, not a Pydantic object.
The string is guaranteed to be valid JSON by the constrained decoding,
but you must parse it manually.

Additionally, `OutlinesLlamaCpp(llm)` is constructed on every call to
`call_ollama_structured`. This is expensive — the Outlines wrapper and
generator FSM should be built once and cached.

---

## Verified Correct API

These imports and call patterns were tested and confirmed working on this machine:

```python
# CORRECT imports
from outlines.models.llamacpp import LlamaCpp as OutlinesLlamaCpp
from outlines.generator import Generator  # note: generator not generate
import json

# CORRECT: build Outlines wrapper once (singleton)
outlines_model = OutlinesLlamaCpp(llm)  # llm is the Llama singleton

# CORRECT: build generator once per schema type (cached)
generator = Generator(outlines_model, MyPydanticSchema)

# CORRECT: call generator — returns a STRING, not a Pydantic object
raw: str = generator(formatted_prompt, temperature=0.1, max_tokens=1024,
                     stop=["<|im_end|>", "<|endoftext|>"])

# CORRECT: parse manually
result = MyPydanticSchema.model_validate(json.loads(raw))
```

---

## Exact Changes Required in llm.py

### Change 1: Fix the import (line ~6)

REMOVE:
```python
from outlines import generate
```

ADD:
```python
from outlines.generator import Generator as OutlinesGenerator
import json
```

### Change 2: Add Outlines model singletons (after the Llama singletons)

After the existing `_instruct_model` and `_coder_model` variables, add:

```python
# Outlines wrapper singletons (built once from the Llama singletons)
_instruct_outlines_model: OutlinesLlamaCpp | None = None
_coder_outlines_model: OutlinesLlamaCpp | None = None

# Generator cache: schema class → Generator instance
# Keyed separately per model so instruct and coder don't share generators
_instruct_generators: dict[type, OutlinesGenerator] = {}
_coder_generators: dict[type, OutlinesGenerator] = {}


def _get_instruct_outlines_model() -> OutlinesLlamaCpp:
    global _instruct_outlines_model
    if _instruct_outlines_model is None:
        _instruct_outlines_model = OutlinesLlamaCpp(_get_instruct_model())
    return _instruct_outlines_model


def _get_coder_outlines_model() -> OutlinesLlamaCpp:
    global _coder_outlines_model
    if _coder_outlines_model is None:
        _coder_outlines_model = OutlinesLlamaCpp(_get_coder_model())
    return _coder_outlines_model


def _get_generator(model_name: str, schema: type) -> OutlinesGenerator:
    """Return cached Generator for this model+schema pair, building if needed."""
    if "coder" in model_name.lower():
        cache = _coder_generators
        outlines_model = _get_coder_outlines_model()
    else:
        cache = _instruct_generators
        outlines_model = _get_instruct_outlines_model()

    if schema not in cache:
        logger.info(f"Building generator FSM for schema: {schema.__name__}")
        cache[schema] = OutlinesGenerator(outlines_model, schema)

    return cache[schema]
```

### Change 3: Rewrite call_ollama_structured body only

Keep the function signature exactly as-is. Replace only the body:

```python
def call_ollama_structured(
    model: str,
    prompt: str,
    system: str,
    response_schema: Type[BaseModel],
    temperature: float = 0.7,
    max_retries: int = 1,
) -> tuple[BaseModel, int]:
    """
    Generate structured output constrained to response_schema.

    Uses Outlines constrained decoding — output is guaranteed valid JSON.
    No retry loop needed. max_retries kept for signature compatibility only.

    Returns (validated_pydantic_object, estimated_tokens_used).
    """
    formatted_prompt = _format_prompt(system, prompt)
    generator = _get_generator(model, response_schema)

    try:
        raw: str = generator(
            formatted_prompt,
            temperature=temperature,
            max_tokens=1024,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        logger.debug(f"Raw constrained output: {raw}")

        result = response_schema.model_validate(json.loads(raw))

        tokens_used = _estimate_tokens(formatted_prompt + raw)
        logger.debug(
            f"Structured call | Schema: {response_schema.__name__} "
            f"| Model: {model} | ~{tokens_used} tokens"
        )
        return result, tokens_used

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed for {response_schema.__name__}. Raw: {raw!r}")
        raise ValueError(f"Constrained decoding produced invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Constrained generation failed for {response_schema.__name__}: {e}")
        raise ValueError(f"Failed to generate valid {response_schema.__name__}: {e}")
```

---

## config.py — NO CHANGES NEEDED

The file already contains all required settings including:
- `LLAMA_MODEL_PATH`, `LLAMA_CODER_MODEL_PATH`
- `LLAMA_N_GPU_LAYERS`, `LLAMA_N_CTX`, `LLAMA_N_THREADS`, `LLAMA_VERBOSE`
- `PLANNING_MODEL`, `CODING_MODEL`, `VALIDATION_MODEL` etc.

Do not touch config.py.

---

## Validation: Create and Run test_llm_migration.py

Place this file in the project root (same level as ai_intern/ package).
Run it with: `python test_llm_migration.py`
All tests must pass before this task is complete.

```python
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
```

---

## What NOT to Do

- Do NOT use `from outlines import generate` — does not exist in outlines 1.2.12
- Do NOT use `generate.json(...)` — wrong API
- Do NOT expect `Generator(...)()` to return a Pydantic object — it returns a string
- Do NOT construct `OutlinesLlamaCpp(llm)` inside `call_ollama_structured` — build once
- Do NOT construct `Generator(...)` inside `call_ollama_structured` — cache it
- Do NOT modify `call_ollama_code` — it is correct as-is
- Do NOT modify `config.py` — it is complete and correct
- Do NOT change any function signatures — callers depend on them

---

## Success Criteria

All 6 tests in `test_llm_migration.py` pass with exit code 0.