import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ai_intern.schemas import RequestSchema
from ai_intern.orchestration.orchestrator import Orchestrator

if __name__ == "__main__":

    # Test 1: Multi-agent with file I/O (original failing case)
    print("\n" + "=" * 80)
    print("TEST 1: Read + Calculate + Save workflow")
    print("=" * 80)
    user_input = "Read Workouts.csv and calculate average calories, save summary"
    request = RequestSchema(content=user_input)
    orchestrator = Orchestrator(max_retries=2)
    plan = orchestrator.execute_request(request)

    print(f"\nPlan Status: {plan.status}")
    print(f"Total Tasks: {len(plan.tasks)}")

    for i, task in enumerate(plan.tasks):
        agent = task.declared_agent or task.suggested_agent or "inferred"
        print(f"  Task {i} [{agent}]: {task.goal[:60]}...")

    # Verify declared agents are set
    tasks_with_declared = sum(1 for t in plan.tasks if t.declared_agent)
    print(f"\nTasks with declared agent: {tasks_with_declared}/{len(plan.tasks)}")

    if len(plan.tasks) >= 3 and all(t.declared_agent for t in plan.tasks):
        print("PASS: SubtaskSpec working: structured decomposition with agent routing")
    else:
        print("FAIL: SubtaskSpec not working - check decomposer output")

    # Test 2: Simple task (should not decompose)
    print("\n" + "=" * 80)
    print("TEST 2: Simple task (should stay as 1 task)")
    print("=" * 80)
    user_input = "Write a function to calculate fibonacci sequence"
    request = RequestSchema(content=user_input)
    plan = orchestrator.execute_request(request)

    print(f"Tasks: {len(plan.tasks)} (expected: 1)")
    if len(plan.tasks) == 1:
        print("PASS: Simple tasks not over-decomposed")
    else:
        print("FAIL: Over-decomposition still occurring on simple tasks")

    # Test 3: Research + code workflow
    print("\n" + "=" * 80)
    print("TEST 3: Research + Code workflow")
    print("=" * 80)
    user_input = "Research average protein in chicken thighs and write a macro calculator"
    request = RequestSchema(content=user_input)
    plan = orchestrator.execute_request(request)

    print(f"Tasks: {len(plan.tasks)}")
    for i, task in enumerate(plan.tasks):
        agent = task.declared_agent or task.suggested_agent or "inferred"
        print(f"  Task {i} [{agent}]: {task.goal[:60]}...")

    agent_types = [t.declared_agent for t in plan.tasks]
    if "research" in agent_types and "code" in agent_types:
        print("PASS: Research + Code correctly separated")
    else:
        print(f"FAIL: Expected research + code, got: {agent_types}")
