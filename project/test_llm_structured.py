from llm import call_ollama_structured
from schemas import PlanSchema, TaskSchema
from pydantic import BaseModel

# Test 1: Simple schema validation
class SimpleResponse(BaseModel):
    name: str
    age: int
    active: bool

print("=== TEST 1: Simple Schema ===")
result = call_ollama_structured(
    model="qwen2.5:7b-instruct",
    prompt="""Return a JSON object with these fields:
- name: "Alice" (string)
- age: 30 (number)
- active: true (boolean)

Return ONLY the JSON, no other text.""",
    system="You are a helpful assistant that returns valid JSON.",
    response_schema=SimpleResponse,
    temperature=0.1
)

print(f"\nResult type: {type(result)}")
print(f"Result: {result}")
print(f"Name: {result.name} (type: {type(result.name)})")
print(f"Age: {result.age} (type: {type(result.age)})")

# Test 2: Nested schema (like your PlanSchema)
class TaskTest(BaseModel):
    plan_id: str
    task_order: int
    goal: str

class PlanTest(BaseModel):
    request_id: str
    tasks: list[TaskTest]

print("\n\n=== TEST 2: Nested Schema ===")
result2 = call_ollama_structured(
    model="qwen2.5:7b-instruct",
    prompt="""Create a plan with 3 tasks for building a todo app.

Return JSON matching this structure:
{
    "request_id": "test-123",
    "tasks": [
        {"plan_id": "placeholder", "task_order": 0, "goal": "task description"},
        {"plan_id": "placeholder", "task_order": 1, "goal": "task description"},
        {"plan_id": "placeholder", "task_order": 2, "goal": "task description"}
    ]
}

Return ONLY the JSON object.""",
    system="You are a planning assistant.",
    response_schema=PlanTest,
    temperature=0.3
)

print(f"\nResult type: {type(result2)}")
print(f"Request ID: {result2.request_id}")
print(f"Number of tasks: {len(result2.tasks)}")
for task in result2.tasks:
    print(f"  [{task.task_order}] {task.goal}")