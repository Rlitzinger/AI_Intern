from schemas import TaskSchema
from agents import CodingAgent
from uuid import uuid4
from datetime import datetime

# Create a simple coding task
task = TaskSchema(
    plan_id=str(uuid4()),
    task_order=0,
    goal="Write a function that makes an HTTP GET request to a URL and returns the status code"
)

print("=" * 60)
print("🧪 TESTING CODING AGENT")
print("=" * 60)
print(f"\nTask: {task.goal}")
print(f"Task ID: {task.task_id}")
print(f"Initial status: {task.status}")

# Execute the coding agent
updated_task, tokens = CodingAgent.execute_task(task)

print(f"\n📊 Results:")
print(f"Status: {updated_task.status}")
print(f"Tokens used: {tokens}")
print(f"Code length: {len(updated_task.result)} characters")

print(f"\n{'=' * 60}")
print("GENERATED CODE:")
print(f"{'=' * 60}")
print(updated_task.result)
print(f"{'=' * 60}")