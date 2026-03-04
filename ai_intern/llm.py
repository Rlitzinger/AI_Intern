from typing import Type
from pydantic import BaseModel
from llama_cpp import Llama
from outlines.models.llamacpp import LlamaCpp as OutlinesLlamaCpp
from outlines.generator import Generator as OutlinesGenerator
import json
from .config import settings
from .logging_config import get_logger

logger = get_logger("llm")


class OllamaConnectionError(Exception):
    """Kept for compatibility. Raised when model cannot load."""
    pass


class OllamaTimeoutError(Exception):
    """Kept for compatibility."""
    pass


class LlamaLoadError(Exception):
    """Raised when llama-cpp-python model fails to load."""
    pass


# ---------------------------------------------------------------------------
# Lazy model singletons
# ---------------------------------------------------------------------------

_instruct_model: Llama | None = None
_coder_model: Llama | None = None


def _get_instruct_model() -> Llama:
    global _instruct_model
    if _instruct_model is None:
        logger.info(f"Loading instruct model: {settings.LLAMA_MODEL_PATH}")
        _instruct_model = Llama(
            model_path=settings.LLAMA_MODEL_PATH,
            n_gpu_layers=settings.LLAMA_N_GPU_LAYERS,
            n_ctx=settings.LLAMA_N_CTX,
            n_threads=settings.LLAMA_N_THREADS,
            verbose=settings.LLAMA_VERBOSE,
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
            verbose=settings.LLAMA_VERBOSE,
        )
        logger.info("Coder model loaded successfully")
    return _coder_model


def _get_model(model_name: str) -> Llama:
    """Route model name string to the correct loaded model."""
    if "coder" in model_name.lower():
        return _get_coder_model()
    return _get_instruct_model()


# ---------------------------------------------------------------------------
# Outlines wrapper singletons + generator cache
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_prompt(system: str, user_prompt: str) -> str:
    """Format using Qwen2.5 chat template (im_start/im_end tokens)."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public utility functions (signatures unchanged from Ollama version)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Availability check (compatibility shim)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Structured generation (constrained decoding via Outlines)
# ---------------------------------------------------------------------------

def call_ollama_structured(
    model: str,
    prompt: str,
    system: str,
    response_schema: Type[BaseModel],
    temperature: float = 0.7,
    max_retries: int = 1,  # noqa: kept for signature compatibility, no longer used.
                           # Outlines makes retries unnecessary -- invalid output
                           # is impossible at the token sampling level.
) -> tuple[BaseModel, int]:
    """
    Generate structured output constrained to response_schema.

    Uses Outlines to enforce schema at token sampling level.
    Output is guaranteed valid JSON -- no JSON parsing failures, no retry needed.

    KNOWN LIMITATION: Token counts are estimated (~1 per 3 chars), not exact.
    Affects plan.token_usage accuracy. Acceptable until Outlines exposes raw counts.

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


# ---------------------------------------------------------------------------
# Code generation (plain text, unconstrained)
# ---------------------------------------------------------------------------

def call_ollama_code(
    model: str,
    prompt: str,
    system: str,
    temperature: float = 0.1,
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
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    code = response["choices"][0]["text"].strip()
    tokens_used = response["usage"]["total_tokens"]

    logger.debug(
        f"Code gen | Model: {model} | Temp: {temperature} "
        f"| Tokens: {tokens_used} | Length: {len(code)} chars"
    )
    logger.debug(f"Code preview: {code[:200]}...")
    return code, tokens_used


# ---------------------------------------------------------------------------
# Generic generation (compatibility shim for any direct callers)
# ---------------------------------------------------------------------------

def call_ollama(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    format: str = "json",  # noqa: kept for signature compatibility
) -> dict:
    """
    Compatibility shim. Prefer call_ollama_structured for typed output.
    Returns dict mimicking old Ollama response structure.
    """
    _ = format  # compatibility shim only
    llm = _get_model(model)
    formatted_prompt = _format_prompt(system, prompt)

    response = llm(
        formatted_prompt,
        temperature=temperature,
        max_tokens=1024,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    text = response["choices"][0]["text"].strip()

    # Mimic Ollama response structure for compatibility
    return {
        "response": text,
        "prompt_eval_count": response["usage"].get("prompt_tokens", 0),
        "eval_count": response["usage"].get("completion_tokens", 0),
    }
