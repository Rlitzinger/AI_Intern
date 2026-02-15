import requests
import json
from typing import Any
from pydantic import BaseModel, ValidationError
from typing import Type

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
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "format": format,
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    
    # Extract token usage
    prompt_tokens = result.get("prompt_eval_count", 0)
    response_tokens = result.get("eval_count", 0)
    total_tokens = prompt_tokens + response_tokens
    
    # Log for debugging
    print(f"\n=== LLM CALL ===")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Tokens - Prompt: {prompt_tokens}, Response: {response_tokens}, Total: {total_tokens}")
    print(f"Response: {result['response'][:200]}...")
    
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

    Made this bc storing python code in a json format is wonky
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "temperature": temperature,
        "stream": False
        # Note: NO "format": "json" - we want raw text
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    
    # Extract token usage
    prompt_tokens = result.get("prompt_eval_count", 0)
    response_tokens = result.get("eval_count", 0)
    total_tokens = prompt_tokens + response_tokens
    
    code = result["response"]
    
    # Log for debugging
    print(f"\n=== LLM CODE GENERATION ===")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Tokens - Prompt: {prompt_tokens}, Response: {response_tokens}, Total: {total_tokens}")
    print(f"Code length: {len(code)} characters")
    print(f"First 200 chars: {code[:200]}...")
    
    return code, total_tokens

def call_ollama_structured(
    model: str,
    prompt: str,
    system: str,
    response_schema: Type[BaseModel],
    temperature: float = 0.7,
    max_retries: int = 1
) -> tuple[BaseModel, int]:  # Now returns (object, token_count)
    """
    Calls Ollama and validates response against Pydantic schema.
    Returns (validated_object, total_tokens_used).
    """
    
    total_tokens = 0
    
    for attempt in range(max_retries + 1):
        print(f"\n🔄 Attempt {attempt + 1}/{max_retries + 1}")
        
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
            
            print(f"📦 Parsed JSON successfully")
            
            # Validate against Pydantic schema
            validated_object = response_schema(**response_json)
            
            print(f"✅ Validated as {response_schema.__name__}")
            print(f"🎫 Total tokens used: {total_tokens}")
            return validated_object, total_tokens
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Raw response: {response_text[:500]}")
            if attempt == max_retries:
                raise ValueError(f"Ollama did not return valid JSON after {max_retries + 1} attempts: {e}")
                
        except ValidationError as e:
            print(f"❌ Schema validation failed:")
            print(f"   {e}")
            print(f"Received data: {response_json}")
            if attempt == max_retries:
                raise ValueError(f"Response doesn't match {response_schema.__name__} schema after {max_retries + 1} attempts")
        
        # If we're retrying, add a note to the prompt
        if attempt < max_retries:
            prompt = f"{prompt}\n\nPREVIOUS ATTEMPT FAILED. Please ensure you return ONLY valid JSON matching the exact structure specified."