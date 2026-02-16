from schemas import RequestSchema, PlanSchema, TaskSchema
from agents import PlanningAgent, CodingAgent, ValidationAgent, Orchestrator
import sqlite3
import json

if __name__ == "__main__":
    # User input
    user_input = "Write a Python function that can determine if a number is prime"
    request = RequestSchema(content=user_input)
    
    print(f"📨 User Request: {request.content}")
    print(f"   Request ID: {request.request_id}")
    
    # Execute with orchestrator (handles everything)
    orchestrator = Orchestrator(max_retries=2)  # 3 total attempts
    plan = orchestrator.execute_request(request)
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")
    print(f"Plan Status: {plan.status}")
    print(f"Total Tokens: {plan.token_usage}")
    print(f"Tasks: {len(plan.tasks)}")
    for i, task in enumerate(plan.tasks):
        print(f"  [{i}] {task.status} - {task.goal[:60]}...")
        if task.retry_count > 0:
            print(f"      Retries: {task.retry_count}")
    
    print(f"\n💡 Run 'python inspect_results.py' to see generated code")