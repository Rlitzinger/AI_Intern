from schemas import RequestSchema, PlanSchema, TaskSchema
import sqlite3
import json

# Create a request using Pydantic
def create_request(user_prompt: str) -> RequestSchema:
    request = RequestSchema(content=user_prompt)
    print(f"Created request: {request.model_dump_json()}")
    return request

# Create a plan using Pydantic
def create_plan(request: RequestSchema) -> PlanSchema:
    plan = PlanSchema(request_id=request.request_id)
    print(f"Created plan: {plan.model_dump_json()}")
    return plan

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
            retry_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    
    # Use Pydantic's model_dump() for serialization
    plan_dict = plan.model_dump(mode='json')
    
    conn.execute("""
        INSERT INTO plans(plan_id, request_id, schema_version, created_at, completed_at, tasks, status, retry_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        plan_dict["plan_id"],
        plan_dict["request_id"],
        plan_dict["schema_version"],
        plan_dict["created_at"],
        plan_dict["completed_at"],
        json.dumps(plan_dict["tasks"]),
        plan_dict["status"],
        plan_dict["retry_count"]
    ))
    
    conn.commit()
    conn.close()
    print(f"Saved plan {plan.plan_id} to SQLite")

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
               "completed_at", "tasks", "status", "retry_count"]
    plan_dict = dict(zip(columns, row))
    plan_dict["tasks"] = json.loads(plan_dict["tasks"])
    
    # Reconstruct Pydantic object with validation
    plan = PlanSchema(**plan_dict)
    print(f"Loaded plan: {plan.model_dump_json()}")
    return plan

# TEST THE ROUND TRIP
if __name__ == "__main__":
    # 1. Create request
    request = create_request("hello respond with a clever riddle")
    
    # 2. Create plan
    plan = create_plan(request)
    
    # 3. Add a task to the plan
    task = TaskSchema(
        plan_id=plan.plan_id,
        task_order=0,
        goal="Generate a clever riddle"
    )
    plan.tasks.append(task)
    
    # 4. Save to SQLite
    save_plan_to_sqlite(plan)
    
    # 5. Load from SQLite
    loaded_plan = load_plan_from_sqlite(plan.plan_id)
    
    # 6. Verify it matches
    assert loaded_plan.plan_id == plan.plan_id
    assert loaded_plan.tasks[0].goal == "Generate a clever riddle"
    assert loaded_plan.status == "planning"
    
    print("✅ Round-trip successful!")