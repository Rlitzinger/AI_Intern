# ACTIVE: Migrate llm.py from Ollama to llama-cpp-python + Outlines

## Context
Multi-agent AI orchestration platform. Local hardware:
- Windows 10, RTX 3070 (8GB VRAM), 32GB RAM
- Models at E:\models\
  - Qwen2.5-7B-Instruct-Q4_K_M.gguf
  - Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf

Goal: Replace Ollama HTTP calls in llm.py with in-process llama-cpp-python +
Outlines constrained decoding. Invalid JSON/schema outputs become structurally
impossible. All caller signatures stay identical — zero changes outside llm.py
and config.py.

---

## Step 1: Install Dependencies

```bash
pip install outlines
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

If CUDA wheel fails, install CPU version and set LLAMA_N_GPU_LAYERS=0 in config:
```bash
pip install llama-cpp-python
```

Verify before proceeding:
```python
from llama_cpp import Llama
import outlines
print("Dependencies OK")
```

---

## Step 2: Update config.py

Add these settings. Do NOT remove existing OLLAMA_ settings yet:

```python
# llama-cpp-python settings (replacing Ollama)
LLAMA_MODEL_PATH: str = r"E:\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
LLAMA_CODER_MODEL_PATH: str = r"E:\models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
LLAMA_N_GPU_LAYERS: int = 99    # Offload all layers to GPU
LLAMA_N_CTX: int = 4096         # Context window
LLAMA_N_THREADS: int = 8        # CPU threads for non-GPU ops
LLAMA_VERBOSE: bool = False     # Suppress llama.cpp startup noise
```

---

## Step 3: Rewrite llm.py

Save original as llm_ollama_backup.py first. Then rewrite llm.py as follows.
ALL function signatures must be preserved exactly.

### Imports
```python
import time
from typing import Any, Type
from pydantic import BaseModel
from llama_cpp import Llama
import outlines
from .config import settings
from .logging_config import get_logger

logger = get_logger("llm")
```

### Exception Classes (keep for caller compatibility)
```python
class OllamaConnectionError(Exception):
    """Kept for compatibility. Raised when model cannot load."""
    pass

class OllamaTimeoutError(Exception):
    """Kept for compatibility."""
    pass

class LlamaLoadError(Exception):
    """Raised when llama-cpp-python model fails to load."""
    pass
```

### Lazy Model Singletons
```python
_instruct_model = None
_coder_model = None

def _get_instruct_model() -> Llama:
    global _instruct_model
    if _instruct_model is None:
        logger.info(f"Loading instruct model: {settings.LLAMA_MODEL_PATH}")
        _instruct_model = Llama(
            model_path=settings.LLAMA_MODEL_PATH,
            n_gpu_layers=settings.LLAMA_N_GPU_LAYERS,
            n_ctx=settings.LLAMA_N_CTX,
            n_threads=settings.LLAMA_N_THREADS,
            verbose=settings.LLAMA_VERBOSE
        )
        logger.info("Instruct model loaded successfully")
    return _instruct_model

def _get_coder_model() -> Llama:
    global _coder_model
    if _coder_model is None:
        logger.info(f"Loading coder model: {settings.LLAMA_CODER_MODEL_PATH}")
        _coder_model = Llama(
            model_path=settings.LLAMA_CODER_MODEL_PATH,
            n_gpu_layers=settings.LLAMA_N_GPU_LAYERS,
            n_ctx=settings.LLAMA_N_CTX,
            n_threads=settings.LLAMA_N_THREADS,
            verbose=settings.LLAMA_VERBOSE
        )
        logger.info("Coder model loaded successfully")
    return _coder_model

def _get_model(model_name: str) -> Llama:
    """Route model name string to the correct loaded model."""
    if "coder" in model_name.lower():
        return _get_coder_model()
    return _get_instruct_model()
```

### Prompt Formatting
```python
def _format_prompt(system: str, user_prompt: str) -> str:
    """Format using Qwen2.5 chat template (im_start/im_end tokens)."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
```

### Public Utility Functions (keep signatures identical)
```python
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 3 chars for English text."""
    return len(text) // 3

def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = max_tokens * 3
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"

def _estimate_tokens(text: str) -> int:
    """Internal alias for token estimation."""
    return estimate_tokens(text)
```

### check_ollama_available (compatibility shim)
```python
def check_ollama_available() -> bool:
    """
    Check if models are loadable. Kept for interface compatibility.
    Now checks llama-cpp-python model load instead of Ollama HTTP ping.
    """
    try:
        _get_instruct_model()
        return True
    except Exception as e:
        logger.error(f"Model load check failed: {e}")
        return False
```

### call_ollama_structured (CRITICAL — constrained decoding)
```python
def call_ollama_structured(
    model: str,
    prompt: str,
    system: str,
    response_schema: Type[BaseModel],
    temperature: float = 0.7,
    max_retries: int = 1  # NOTE: kept for signature compatibility, no longer used.
                          # Outlines makes retries unnecessary — invalid output
                          # is impossible at the token sampling level.
) -> tuple[BaseModel, int]:
    """
    Generate structured output constrained to response_schema.

    Uses Outlines to enforce schema at token sampling level.
    Output is guaranteed valid — no JSON parsing, no retry logic needed.
    Returns (validated_pydantic_object, estimated_tokens_used).
    """
    llm = _get_model(model)
    formatted_prompt = _format_prompt(system, prompt)

    # Outlines builds a token mask from the Pydantic schema.
    # At each generation step, only tokens that could lead to a valid
    # completion are allowed — invalid structure is impossible.
    generator = outlines.generate.json(llm, response_schema)

    try:
        result = generator(formatted_prompt, temperature=temperature)
        # result is already a validated Pydantic instance

        tokens_used = _estimate_tokens(formatted_prompt + str(result))
        logger.debug(
            f"Structured call | Schema: {response_schema.__name__} "
            f"| Model: {model} | ~{tokens_used} tokens"
        )
        return result, tokens_used

    except Exception as e:
        logger.error(f"Constrained generation failed for {response_schema.__name__}: {e}")
        raise ValueError(
            f"Failed to generate valid {response_schema.__name__}: {e}"
        )
```

### call_ollama_code (plain text, no constraint)
```python
def call_ollama_code(
    model: str,
    prompt: str,
    system: str,
    temperature: float = 0.1
) -> tuple[str, int]:
    """
    Generate code as plain text (unconstrained).
    Returns (code_string, tokens_used).
    """
    llm = _get_model(model)
    formatted_prompt = _format_prompt(system, prompt)

    response = llm(
        formatted_prompt,
        temperature=temperature,
        max_tokens=2048,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    code = response["choices"][0]["text"].strip()
    tokens_used = response["usage"]["total_tokens"]

    logger.debug(
        f"Code gen | Model: {model} | Temp: {temperature} "
        f"| Tokens: {tokens_used} | Length: {len(code)} chars"
    )
    logger.debug(f"Code preview: {code[:200]}...")
    return code, tokens_used
```

### call_ollama (keep for any direct callers)
```python
def call_ollama(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    format: str = "json"
) -> dict:
    """
    Compatibility shim. Prefer call_ollama_structured for typed output.
    Returns dict mimicking old Ollama response structure.
    """
    llm = _get_model(model)
    formatted_prompt = _format_prompt(system, prompt)

    response = llm(
        formatted_prompt,
        temperature=temperature,
        max_tokens=1024,
        stop=["<|im_end|>", "<|endoftext|>"]
    )

    text = response["choices"][0]["text"].strip()
    tokens_used = response["usage"]["total_tokens"]

    # Mimic Ollama response structure for compatibility
    return {
        "response": text,
        "prompt_eval_count": response["usage"].get("prompt_tokens", 0),
        "eval_count": response["usage"].get("completion_tokens", 0),
    }
```

---

## Step 4: Create Test File

Create test_llm_migration.py in project root:

```python
"""Validates llm.py migration to llama-cpp-python + Outlines."""
import sys
sys.path.insert(0, ".")

from ai_intern.llm import (
    call_ollama_structured,
    call_ollama_code,
    check_ollama_available
)
from ai_intern.planning.classifier import ClassificationResult

def test_availability():
    result = check_ollama_available()
    assert result, "Model failed to load — check LLAMA_MODEL_PATH in config"
    print("OK: Model loads successfully")

def test_structured():
    result, tokens = call_ollama_structured(
        model="qwen2.5:7b",
        prompt="Classify this task: Read a CSV file and print the first 5 rows",
        system="You are a task classifier.",
        response_schema=ClassificationResult,
        temperature=0.1
    )
    assert isinstance(result, ClassificationResult), f"Wrong type: {type(result)}"
    assert tokens > 0, "Token count should be non-zero"
    print(f"OK: Structured generation returned {type(result).__name__}")
    print(f"    Tokens: {tokens}")

def test_code():
    code, tokens = call_ollama_code(
        model="qwen2.5-coder:7b",
        prompt="Write a Python function that returns the sum of a list of numbers",
        system="You are a Python expert. Return only code, no explanation.",
        temperature=0.1
    )
    assert "def " in code, f"Expected function definition, got: {code[:100]}"
    assert tokens > 0, "Token count should be non-zero"
    print(f"OK: Code generation returned {len(code)} chars, {tokens} tokens")

def test_no_retries():
    """Run structured call 5 times, confirm no retry log messages."""
    import logging
    for i in range(5):
        result, _ = call_ollama_structured(
            model="qwen2.5:7b",
            prompt=f"Classify task {i}: Write a script to sort a list",
            system="You are a task classifier.",
            response_schema=ClassificationResult,
            temperature=0.1
        )
        assert isinstance(result, ClassificationResult)
    print("OK: 5 structured calls, zero retries needed")

if __name__ == "__main__":
    print("Running llm.py migration tests...\n")
    test_availability()
    test_structured()
    test_code()
    test_no_retries()
    print("\nAll tests passed. Migration successful.")
```

---

## Success Criteria

All must pass before session is complete:

- [ ] `pip install outlines` and `llama-cpp-python` succeed
- [ ] `check_ollama_available()` returns True
- [ ] `test_structured()` passes — isinstance check confirmed
- [ ] `test_code()` passes — "def " present in output
- [ ] `test_no_retries()` passes — 5 calls, zero retry logs
- [ ] Grep confirms no callers broken: `grep -r "call_ollama" ai_intern/`

---

## Known Limitations (document in code comments, do not fix)

1. Token counts for constrained calls are estimated (~1 per 3 chars), not exact.
   Affects plan.token_usage accuracy. Acceptable until Outlines exposes raw counts.
2. First call after startup has 5-10s model load latency. Expected, lazy loading.
3. max_retries parameter is non-functional. Noted in docstring.

---

## Rollback If Blocked

If Outlines or llama-cpp-python fail to install or integrate, do NOT use the
OpenAI compatibility shim. Use the llama-server.exe HTTP API directly instead.

The llama-server.exe running at http://127.0.0.1:8080 exposes its own REST API
that is close to (but not identical to) the Ollama API. The rollback path is:

1. Save broken llm.py as llm_constrained_attempt.py
2. Restore llm_ollama_backup.py as llm.py
3. In llm.py, change OLLAMA_BASE_URL to point at http://127.0.0.1:8080
4. Change the generate endpoint from /api/generate to /completion
5. Adjust payload format — llama-server uses "prompt" not "prompt"+"system",
   so pre-format using _format_prompt() before sending
6. Response field is "content" not "response"

llama-server payload format:
```python
payload = {
    "prompt": formatted_prompt,   # pre-formatted with chat template
    "temperature": temperature,
    "n_predict": 1024,
    "stop": ["<|im_end|>", "<|endoftext|>"]
}
# response["content"] contains the generated text
```

This loses constrained decoding but keeps the system running on the GPU
with the new models. Implement as a temporary bridge only.
Start llama-server.exe manually before running the system:
```
E:\llama.cpp\llama-server.exe -m "E:\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf" -ngl 99 -c 4096 --port 8080
```