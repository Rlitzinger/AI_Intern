from schemas import RequestSchema, PlanSchema, TaskSchema
from agents import PlanningAgent, CodingAgent, ValidationAgent
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
    print("=" * 80)
    print("🚀 FULL PIPELINE: REQUEST → PLANNING → EXECUTION → VALIDATION")
    print("=" * 80)
    
    # === STEP 1: USER INPUT ===
    user_input = "Write a Python function that calculates the Fibonacci sequence up to n terms"
    request = RequestSchema(content=user_input)
    print(f"\n📨 Step 1: User Request")
    print(f"   Request ID: {request.request_id}")
    print(f"   Content: {request.content}")
    
    # === STEP 2: PLANNING ===
    print(f"\n📋 Step 2: Planning Agent")
    plan = PlanningAgent.create_plan(request)
    
    print(f"\n   Generated {len(plan.tasks)} task(s):")
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
        
        print(f"   Status after execution: {updated_task.status}")
        print(f"   Code length: {len(updated_task.result)} chars")
        print(f"   Tokens: {tokens}")
    
    plan.token_usage += total_execution_tokens
    plan.status = "executing"
    
    # Save after execution
    save_plan_to_sqlite(plan)
    
    # === STEP 4: VALIDATION ===
    print(f"\n🔍 Step 4: Validating Tasks")
    
    for i, task in enumerate(plan.tasks):
        print(f"\n   --- Task {i} ---")
        
        # Validate with ValidationAgent
        validated_task = ValidationAgent.validate_task(task)
        
        # Update the task in the plan
        plan.tasks[i] = validated_task
        
        print(f"   Final status: {validated_task.status}")
        if validated_task.error_message:
            print(f"   Error: {validated_task.error_message}")
    
    # Update plan status based on task results
    all_validated = all(t.status == "validated" for t in plan.tasks)
    any_failed = any(t.status == "failed" for t in plan.tasks)
    
    if all_validated:
        plan.status = "complete"
    elif any_failed:
        plan.status = "failed"
    else:
        plan.status = "validating"
    
    # Save after validation
    save_plan_to_sqlite(plan)
    
    # === STEP 5: SUMMARY ===
    print(f"\n📊 Step 5: Summary")
    loaded_plan = load_plan_from_sqlite(plan.plan_id)
    
    validated_count = sum(1 for t in loaded_plan.tasks if t.status == "validated")
    failed_count = sum(1 for t in loaded_plan.tasks if t.status == "failed")
    
    print(f"   Plan ID: {loaded_plan.plan_id}")
    print(f"   Plan Status: {loaded_plan.status}")
    print(f"   Total tokens: {loaded_plan.token_usage}")
    print(f"   Tasks validated: {validated_count}/{len(loaded_plan.tasks)}")
    print(f"   Tasks failed: {failed_count}/{len(loaded_plan.tasks)}")
    
    print(f"\n" + "=" * 80)
    if plan.status == "complete":
        print("🎉 FULL PIPELINE SUCCESSFUL!")
    else:
        print("⚠️  PIPELINE COMPLETED WITH ISSUES")
    print("=" * 80)
    
    print(f"\nNext step: Run 'python inspect_results.py' to see generated code")