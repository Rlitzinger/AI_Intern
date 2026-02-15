from schemas import RequestSchema, PlanSchema, TaskSchema
from agents import PlanningAgent, CodingAgent
import sqlite3
import json

def save_plan_to_sqlite(plan: PlanSchema):
    """Save plan to database."""
    conn = sqlite3.connect('ai_intern_db.sqlite')
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            plan_id TEXT PRIMARY KEY,
            request_id TEXT,
            schema_version TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            tasks TEXT,
            status TEXT NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            token_usage INTEGER NOT NULL DEFAULT 0
        )
    """)
    
    plan_dict = plan.model_dump(mode='json')
    
    conn.execute("""
        INSERT OR REPLACE INTO plans(plan_id, request_id, schema_version, created_at, completed_at, tasks, status, retry_count, token_usage)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        plan_dict["plan_id"],
        plan_dict["request_id"],
        plan_dict["schema_version"],
        plan_dict["created_at"],
        plan_dict["completed_at"],
        json.dumps(plan_dict["tasks"]),
        plan_dict["status"],
        plan_dict["retry_count"],
        plan_dict["token_usage"]
    ))
    
    conn.commit()
    conn.close()
    print(f"💾 Saved plan {plan.plan_id} to SQLite")

def load_plan_from_sqlite(plan_id: str) -> PlanSchema:
    """Load plan from database."""
    conn = sqlite3.connect('ai_intern_db.sqlite')
    cur = conn.cursor()
    
    cur.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,))
    row = cur.fetchone()
    conn.close()
    
    if row is None:
        raise ValueError(f"Plan {plan_id} not found")
    
    columns = ["plan_id", "request_id", "schema_version", "created_at", 
               "completed_at", "tasks", "status", "retry_count", "token_usage"]
    plan_dict = dict(zip(columns, row))
    plan_dict["tasks"] = json.loads(plan_dict["tasks"])
    
    plan = PlanSchema(**plan_dict)
    print(f"📂 Loaded plan: {plan.plan_id}")
    return plan

# FULL PIPELINE TEST
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 FULL PIPELINE: REQUEST → PLANNING → EXECUTION")
    print("=" * 70)
    
    # === STEP 1: USER INPUT ===
    user_input = "Create a Python function that reads a CSV file and counts the number of rows"
    request = RequestSchema(content=user_input)
    print(f"\n📨 Step 1: User Request")
    print(f"   Request ID: {request.request_id}")
    print(f"   Content: {request.content}")
    
    # === STEP 2: PLANNING ===
    print(f"\n📋 Step 2: Planning Agent")
    plan = PlanningAgent.create_plan(request)
    
    print(f"\n   Generated {len(plan.tasks)} tasks:")
    for i, task in enumerate(plan.tasks):
        print(f"   [{i}] {task.goal}")
    
    # Save plan after planning
    save_plan_to_sqlite(plan)
    
    # === STEP 3: EXECUTION ===
    print(f"\n💻 Step 3: Executing Tasks")
    
    total_execution_tokens = 0
    
    for i, task in enumerate(plan.tasks):
        print(f"\n   --- Task {i} ---")
        print(f"   Goal: {task.goal[:80]}...")
        
        # Execute with CodingAgent
        updated_task, tokens = CodingAgent.execute_task(task)
        total_execution_tokens += tokens
        
        # Update the task in the plan
        plan.tasks[i] = updated_task
        
        print(f"   Status: {updated_task.status}")
        print(f"   Code length: {len(updated_task.result)} chars")
        print(f"   Tokens: {tokens}")
    
    # Update plan status and token usage
    plan.status = "executing"  # Will be "complete" after validation
    plan.token_usage += total_execution_tokens
    
    # Save plan after execution
    save_plan_to_sqlite(plan)
    
    # === STEP 4: VERIFICATION ===
    print(f"\n✅ Step 4: Verification")
    loaded_plan = load_plan_from_sqlite(plan.plan_id)
    
    print(f"   Plan ID: {loaded_plan.plan_id}")
    print(f"   Status: {loaded_plan.status}")
    print(f"   Total tokens: {loaded_plan.token_usage}")
    print(f"   Tasks completed: {sum(1 for t in loaded_plan.tasks if t.status == 'complete')}/{len(loaded_plan.tasks)}")
    
    # Show generated code for each task
    print(f"\n📝 Generated Code Summary:")
    for i, task in enumerate(loaded_plan.tasks):
        print(f"\n   Task {i}: {task.goal[:60]}...")
        print(f"   Status: {task.status}")
        if task.result:
            lines = task.result.split('\n')
            print(f"   Code preview (first 5 lines):")
            for line in lines[:5]:
                print(f"      {line}")
            if len(lines) > 5:
                print(f"      ... ({len(lines) - 5} more lines)")
    
    print(f"\n" + "=" * 70)
    print("🎉 FULL PIPELINE SUCCESSFUL!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  • Request processed: {request.request_id}")
    print(f"  • Plan created: {plan.plan_id}")
    print(f"  • Tasks executed: {len(plan.tasks)}")
    print(f"  • Total tokens used: {plan.token_usage}")
    print(f"  • All data persisted to SQLite")