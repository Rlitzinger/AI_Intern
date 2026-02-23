import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from schemas import TaskSchema
from agents.planning.classifier import TaskClassifier, TaskVerdict

print("Testing TaskClassifier\n" + "=" * 80)

# Test 1: Simple + Clear → EXECUTE
print("\nTest 1: Simple + Clear")
task1 = TaskSchema(
    plan_id="test",
    task_order=0,
    goal="Write a function to calculate the fibonacci sequence up to n terms"
)
verdict1, reasoning1, tokens1 = TaskClassifier.classify(task1)
print(f"Verdict: {verdict1.value}")
print(f"Reasoning: {reasoning1}")
assert verdict1 == TaskVerdict.EXECUTE, f"Expected EXECUTE, got {verdict1}"
print("✅ Test 1 passed")

# Test 2: Complex + Clear → DECOMPOSE
print("\nTest 2: Complex + Clear")
task2 = TaskSchema(
    plan_id="test",
    task_order=0,
    goal="Build a REST API with user authentication, PostgreSQL database, and file upload endpoints"
)
verdict2, reasoning2, tokens2 = TaskClassifier.classify(task2)
print(f"Verdict: {verdict2.value}")
print(f"Reasoning: {reasoning2}")
assert verdict2 == TaskVerdict.DECOMPOSE, f"Expected DECOMPOSE, got {verdict2}"
print("✅ Test 2 passed")

# Test 3: Simple + Ambiguous → CLARIFY
print("\nTest 3: Simple + Ambiguous")
task3 = TaskSchema(
    plan_id="test",
    task_order=0,
    goal="Write a calculator using the research results"
)
verdict3, reasoning3, tokens3 = TaskClassifier.classify(task3)
print(f"Verdict: {verdict3.value}")
print(f"Reasoning: {reasoning3}")
assert verdict3 == TaskVerdict.CLARIFY, f"Expected CLARIFY, got {verdict3}"
print("✅ Test 3 passed")

# Test 4: Complex + Ambiguous → CLARIFY_THEN_DECOMPOSE
print("\nTest 4: Complex + Ambiguous")
task4 = TaskSchema(
    plan_id="test",
    task_order=0,
    goal="Build a system to handle the data and make it better"
)
verdict4, reasoning4, tokens4 = TaskClassifier.classify(task4)
print(f"Verdict: {verdict4.value}")
print(f"Reasoning: {reasoning4}")
assert verdict4 == TaskVerdict.CLARIFY_THEN_DECOMPOSE, f"Expected CLARIFY_THEN_DECOMPOSE, got {verdict4}"
print("✅ Test 4 passed")

print(f"\n{'=' * 80}")
print(f"All tests passed!")
print(f"Total tokens used: {tokens1 + tokens2 + tokens3 + tokens4}")
