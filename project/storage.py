import sqlite3
import json
from schemas import PlanSchema

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
    
    return PlanSchema(**plan_dict)