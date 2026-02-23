import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from schemas import RequestSchema
from agents.planning.hierarchical import HierarchicalPlanner

print("Testing HierarchicalPlanner\n" + "=" * 80)

# Test 1: Simple request (should not decompose much)
print("\n🧪 Test 1: Simple request")
request1 = RequestSchema(content="Write a function to calculate fibonacci sequence")
plan1 = HierarchicalPlanner.create_plan(request1)

print(f"\nResult: {len(plan1.tasks)} task(s)")
for i, task in enumerate(plan1.tasks):
    print(f"  Task {i}: {task.goal[:80]}...")

assert len(plan1.tasks) == 1, "Simple request should produce 1 task"
print("✅ Test 1 passed")

# Test 2: Multi-agent request (should decompose)
print("\n🧪 Test 2: Multi-agent workflow")
request2 = RequestSchema(
    content="Research chicken thigh macros and write a Python calculator, then save it as calculator.py"
)
plan2 = HierarchicalPlanner.create_plan(request2)

print(f"\nResult: {len(plan2.tasks)} task(s)")
for i, task in enumerate(plan2.tasks):
    print(f"  Task {i}: {task.goal[:80]}...")

assert len(plan2.tasks) >= 3, "Multi-agent request should produce 3+ tasks"
print("✅ Test 2 passed")

# Test 3: Ambiguous request (should clarify)
print("\n🧪 Test 3: Ambiguous request")
request3 = RequestSchema(content="Make a calculator using the research")
plan3 = HierarchicalPlanner.create_plan(request3)

print(f"\nResult: {len(plan3.tasks)} task(s)")
for i, task in enumerate(plan3.tasks):
    print(f"  Task {i}: {task.goal[:80]}...")
    # Check if clarification added details
    if "calculator" in task.goal.lower():
        assert len(task.goal) > len("Make a calculator using the research"), \
            "Clarified task should be more specific"

print("✅ Test 3 passed")

print(f"\n{'=' * 80}")
print("All tests passed!")
print(f"Total tokens: Plan1={plan1.token_usage}, Plan2={plan2.token_usage}, Plan3={plan3.token_usage}")
