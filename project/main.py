from schemas import RequestSchema, PlanSchema, TaskSchema
from agents import PlanningAgent
import sqlite3
import json

# Save to SQLite using Pydantic serialization
def save_plan_to_sqlite(plan: PlanSchema):
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
    
    # Use Pydantic's model_dump() for serialization
    plan_dict = plan.model_dump(mode='json')
    
    conn.execute("""
        INSERT INTO plans(plan_id, request_id, schema_version, created_at, completed_at, tasks, status, retry_count, token_usage)
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

# Read from SQLite and reconstruct Pydantic object
def load_plan_from_sqlite(plan_id: str) -> PlanSchema:
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
    
    # Reconstruct Pydantic object with validation
    plan = PlanSchema(**plan_dict)
    print(f"📂 Loaded plan: {plan.plan_id}")
    return plan

# TEST THE FULL PIPELINE
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 TESTING FULL PLANNING PIPELINE")
    print("=" * 60)
    
    # 1. Create request from user input
    user_input = "Build a Python script that scrapes weather data from a website and stores it in a SQLite database"
    request = RequestSchema(content=user_input)
    print(f"\n📨 Created request: {request.request_id}")
    print(f"   Content: {request.content}")
    
    # 2. Generate plan using LLM Planning Agent
    plan = PlanningAgent.create_plan(request)
    
    print(f"\n📋 Generated Plan Details:")
    print(f"   Plan ID: {plan.plan_id}")
    print(f"   Request ID: {plan.request_id}")
    print(f"   Status: {plan.status}")
    print(f"   Tasks: {len(plan.tasks)}")
    
    for i, task in enumerate(plan.tasks):
        print(f"\n   Task {i}:")
        print(f"      ID: {task.task_id}")
        print(f"      Plan ID: {task.plan_id}")
        print(f"      Order: {task.task_order}")
        print(f"      Goal: {task.goal}")
        print(f"      Status: {task.status}")
    
    # 3. Save to SQLite
    print(f"\n💾 Saving to database...")
    save_plan_to_sqlite(plan)
    
    # 4. Load from SQLite
    print(f"\n📂 Loading from database...")
    loaded_plan = load_plan_from_sqlite(plan.plan_id)
    
    # 5. Verify integrity
    print(f"\n✅ Verification:")
    assert loaded_plan.plan_id == plan.plan_id, "Plan ID mismatch"
    print(f"   ✓ Plan ID matches")
    
    assert loaded_plan.request_id == request.request_id, "Request ID mismatch"
    print(f"   ✓ Request ID matches")
    
    assert len(loaded_plan.tasks) == len(plan.tasks), "Task count mismatch"
    print(f"   ✓ Task count matches ({len(plan.tasks)} tasks)")
    
    for i, (original, loaded) in enumerate(zip(plan.tasks, loaded_plan.tasks)):
        assert original.goal == loaded.goal, f"Task {i} goal mismatch"
        assert original.task_order == loaded.task_order, f"Task {i} order mismatch"
        assert original.plan_id == loaded.plan_id, f"Task {i} plan_id mismatch"
    print(f"   ✓ All task details match")
    
    assert loaded_plan.status == "planning", "Status should be 'planning'"
    print(f"   ✓ Status correct: {loaded_plan.status}")
    
    print(f"\n" + "=" * 60)
    print("🎉 FULL PIPELINE TEST SUCCESSFUL!")
    print("=" * 60)