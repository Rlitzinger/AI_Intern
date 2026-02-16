from schemas import RequestSchema, PlanSchema, TaskSchema
from agents import PlanningAgent, CodingAgent, ValidationAgent, Orchestrator
import sqlite3
import json

# main.py - should work exactly like before
if __name__ == "__main__":
    # Try a research task
    user_input = "research the top 10 resturants in greer, sc"
    request = RequestSchema(content=user_input)
    
    orchestrator = Orchestrator(max_retries=2)
    plan = orchestrator.execute_request(request)
    
    print(f"\n{'=' * 80}")
    print(f"RESULT")
    print(f"{'=' * 80}")
    print(f"Plan Status: {plan.status}")
    print(f"Total Tokens: {plan.token_usage}")
    
    for i, task in enumerate(plan.tasks):
        print(f"\nTask {i}:")
        print(f"  Goal: {task.goal[:80]}...")
        print(f"  Status: {task.status}")
        if task.error_message:
            print(f"  Error: {task.error_message}")