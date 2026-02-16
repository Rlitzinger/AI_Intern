# NON CRITICAL FILE
# This is simply to help the user manually read stuff from teh databse
# so that they can debug files more effetively, rather than
# parsing through the JSON themselves

import sqlite3
import json

def inspect_latest_plan():
    """Load the most recent plan from the database and display task results."""
    conn = sqlite3.connect('ai_intern_db.sqlite')
    cur = conn.cursor()
    
    # Get the most recent plan by created_at timestamp
    cur.execute("""
        SELECT plan_id, request_id, created_at, status, token_usage, tasks 
        FROM plans 
        ORDER BY created_at DESC 
        LIMIT 1
    """)
    
    row = cur.fetchone()
    conn.close()
    
    if row is None:
        print("❌ No plans found in database")
        return
    
    plan_id, request_id, created_at, status, token_usage, tasks_json = row
    tasks = json.loads(tasks_json)
    
    print("=" * 80)
    print("📋 LATEST PLAN INSPECTION")
    print("=" * 80)
    print(f"\nPlan ID: {plan_id}")
    print(f"Request ID: {request_id}")
    print(f"Created: {created_at}")
    print(f"Status: {status}")
    print(f"Token Usage: {token_usage}")
    print(f"Number of Tasks: {len(tasks)}")
    
    print("\n" + "=" * 80)
    print("TASK RESULTS")
    print("=" * 80)
    
    for i, task in enumerate(tasks):
        print(f"\n{'─' * 80}")
        print(f"TASK {i}: {task['goal']}")
        print(f"{'─' * 80}")
        print(f"Task ID: {task['task_id']}")
        print(f"Status: {task['status']}")
        print(f"Created: {task['created_at']}")
        
        if task.get('completed_at'):
            print(f"Completed: {task['completed_at']}")
        
        if task.get('error_message'):
            print(f"❌ Error: {task['error_message']}")
        
        if task.get('result'):
            print(f"\n{'▼' * 40} CODE {'▼' * 40}")
            print(task['result'])
            print(f"{'▲' * 40} END {'▲' * 40}")
            print(f"\Stats:")
            print(f"  - Length: {len(task['result'])} characters")
            print(f"  - Lines: {len(task['result'].splitlines())} lines")
        else:
            print("\n⚠️  No code generated for this task")

        if task.get('test_code'):
            print(f"\n{'▼' * 40} TESTS {'▼' * 40}")
            print(task['test_code'])
            print(f"{'▲' * 40} END {'▲' * 40}")
            print(f"\nTest Stats:")
            print(f"  - Length: {len(task['test_code'])} characters")
            print(f"  - Lines: {len(task['test_code'].splitlines())} lines")
        else:
            print("\n⚠️  No tests generated for this task")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    inspect_latest_plan()