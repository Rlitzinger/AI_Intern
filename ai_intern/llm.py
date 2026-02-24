import requests
import json
import time
from typing import Any, Type
from pydantic import BaseModel, ValidationError
from .config import settings
from .logging_config import get_logger

logger = get_logger("llm")


class OllamaConnectionError(Exception):
    """Raised when Ollama is not reachable."""
    pass


class OllamaTimeoutError(Exception):
    """Raised when an Ollama request times out."""
    pass


def check_ollama_available() -> bool:
    """Ping Ollama to check if it's running. Returns True/False."""
    try:
        response = requests.get(
            f"{settings.OLLAMA_BASE_URL}/api/tags",
            timeout=5
        )
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def _post_with_retry(url: str, payload: dict, timeout: int = None) -> requests.Response:
    """POST to Ollama with timeout, connection error handling, and retry on transient errors."""
    if timeout is None:
        timeout = settings.LLM_CALL_TIMEOUT

    max_attempts = 2  # 1 retry for transient HTTP errors
    for attempt in range(max_attempts):
        try:
            response = requests.post(url, json=payload, timeout=timeout)

            # Retry on transient HTTP errors (502, 503)
            if response.status_code in (502, 503) and attempt < max_attempts - 1:
                logger.warning(f"Transient HTTP {response.status_code}, retrying in 2s...")
                time.sleep(2)
                continue

            response.raise_for_status()
            return response

        except requests.ConnectionError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}. Is it running? "
                f"Start with 'ollama serve' or launch the Ollama desktop app."
            )
        except requests.Timeout:
            raise OllamaTimeoutError(
                f"Ollama request timed out after {timeout}s. "
                f"The model may be loading or the request may be too large."
            )

    # Should not reach here, but just in case
    response.raise_for_status()
    return response


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 3.5 characters for English text."""
    return len(text) // 3


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = max_tokens * 3  # conservative estimate
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def call_ollama(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    format: str = "json"
) -> dict:
    """
    Basic wrapper around Ollama API.
    Returns the raw response from Ollama INCLUDING token counts.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "format": format,
        "stream": False
    }

    response = _post_with_retry(url, payload)
    result = response.json()

    # Extract token usage
    prompt_tokens = result.get("prompt_eval_count", 0)
    response_tokens = result.get("eval_count", 0)
    total_tokens = prompt_tokens + response_tokens

    logger.debug(f"LLM CALL | Model: {model} | Temp: {temperature} | Prompt: {len(prompt)} chars")
    logger.debug(f"Tokens - Prompt: {prompt_tokens}, Response: {response_tokens}, Total: {total_tokens}")
    logger.debug(f"Response: {result['response'][:200]}...")

    return result

def call_ollama_code(
    model: str,
    prompt: str,
    system: str,
    temperature: float = 0.1
) -> tuple[str, int]:
    """
    Calls Ollama for code generation.
    Returns (code_string, tokens_used).

    Unlike call_ollama_structured, this expects plain text responses,
    not JSON. Better for code generation.
    """
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "stream": False
    }

    response = _post_with_retry(url, payload)
    result = response.json()

    # Extract token usage
    prompt_tokens = result.get("prompt_eval_count", 0)
    response_tokens = result.get("eval_count", 0)
    total_tokens = prompt_tokens + response_tokens

    code = result["response"]

    logger.debug(f"LLM CODE GEN | Model: {model} | Temp: {temperature}")
    logger.debug(f"Tokens - Prompt: {prompt_tokens}, Response: {response_tokens}, Total: {total_tokens}")
    logger.debug(f"Code length: {len(code)} chars | First 200: {code[:200]}...")

    return code, total_tokens

def call_ollama_structured(
    model: str,
    prompt: str,
    system: str,
    response_schema: Type[BaseModel],
    temperature: float = 0.7,
    max_retries: int = 1
) -> tuple[BaseModel, int]:
    """
    Calls Ollama and validates response against Pydantic schema.
    Returns (validated_object, total_tokens_used).
    """

    total_tokens = 0

    for attempt in range(max_retries + 1):
        logger.debug(f"Structured call attempt {attempt + 1}/{max_retries + 1}")

        # Call Ollama
        result = call_ollama(
            model=model,
            prompt=prompt,
            system=system,
            temperature=temperature,
            format="json"
        )

        # Accumulate token usage (in case of retries)
        attempt_tokens = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
        total_tokens += attempt_tokens

        # Parse the JSON response
        try:
            response_text = result["response"]
            response_json = json.loads(response_text)

            logger.debug(f"Parsed JSON successfully")

            # Validate against Pydantic schema
            validated_object = response_schema(**response_json)

            logger.debug(f"Validated as {response_schema.__name__} | Total tokens: {total_tokens}")
            return validated_object, total_tokens

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            if attempt == max_retries:
                raise ValueError(f"Ollama did not return valid JSON after {max_retries + 1} attempts: {e}")

        except ValidationError as e:
            logger.warning(f"Schema validation failed: {e}")
            logger.debug(f"Received data: {response_json}")
            if attempt == max_retries:
                raise ValueError(f"Response doesn't match {response_schema.__name__} schema after {max_retries + 1} attempts")

        # If we're retrying, add a note to the prompt
        if attempt < max_retries:
            prompt = f"{prompt}\n\nPREVIOUS ATTEMPT FAILED. Please ensure you return ONLY valid JSON matching the exact structure specified."
