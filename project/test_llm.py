# test_llm.py
from llm import call_ollama

result = call_ollama(
    model="qwen2.5:7b-instruct",
    prompt="Return a JSON object with fields: name (string) and age (number). Name should be 'Alice' and age should be 30.",
    system="You are a helpful assistant that always returns valid JSON.",
    temperature=0.1,
    format="json"
)

print(result["response"])