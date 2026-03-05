import sqlite3
import json
from .schemas import PlanSchema
from .config import settings
from .logging_config import get_logger

logger = get_logger("storage")


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with WAL mode enabled."""
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_critique_history_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plan_critique_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            plan_id TEXT NOT NULL,
            request_content TEXT,
            cycle INTEGER NOT NULL DEFAULT 0,
            round TEXT NOT NULL,
            agent TEXT NOT NULL,
            finding_type TEXT,
            task_index INTEGER,
            severity TEXT,
            content TEXT NOT NULL,
            confidence_score REAL,
            verdict TEXT,
            survived_revision INTEGER DEFAULT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


def save_critique_event(
    session_id: str,
    plan_id: str,
    request_content: str,
    cycle: int,
    round_name: str,
    agent: str,
    content: str,
    finding_type: str = None,
    task_index: int = None,
    severity: str = None,
    confidence_score: float = None,
    verdict: str = None,
):
    with get_db_connection() as conn:
        ensure_critique_history_table(conn)
        conn.execute("""
            INSERT INTO plan_critique_history
                (session_id, plan_id, request_content, cycle, round, agent,
                 finding_type, task_index, severity, content, confidence_score, verdict)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, plan_id, request_content, cycle, round_name, agent,
            finding_type, task_index, severity, content, confidence_score, verdict
        ))


def mark_findings_survived(session_id: str, cycle: int, survived: bool):
    """Mark whether findings from this cycle persisted into the next."""
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE plan_critique_history
            SET survived_revision = ?
            WHERE session_id = ? AND cycle = ?
        """, (1 if survived else 0, session_id, cycle))


def save_plan_to_sqlite(plan: PlanSchema):
    """Save plan to database using context manager for safety."""
    try:
        with get_db_connection() as conn:
            ensure_critique_history_table(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    request_id TEXT,
                    schema_version TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    tasks TEXT,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    token_usage INTEGER NOT NULL DEFAULT 0,
                    final_answer TEXT
                )
            """)

            plan_dict = plan.model_dump(mode='json')

            conn.execute("""
                INSERT OR REPLACE INTO plans(
                    plan_id, request_id, schema_version, created_at, completed_at,
                    tasks, status, retry_count, token_usage, final_answer
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan_dict["plan_id"],
                plan_dict["request_id"],
                plan_dict["schema_version"],
                plan_dict["created_at"],
                plan_dict["completed_at"],
                json.dumps(plan_dict["tasks"]),
                plan_dict["status"],
                plan_dict["retry_count"],
                plan_dict["token_usage"],
                plan_dict.get("final_answer"),
            ))
    except sqlite3.Error as e:
        logger.error(f"Failed to save plan: {e}")
        raise


def load_plan_from_sqlite(plan_id: str) -> PlanSchema:
    """Load plan from database using context manager."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,))
            row = cur.fetchone()
    except sqlite3.Error as e:
        logger.error(f"Failed to load plan: {e}")
        raise

    if row is None:
        raise ValueError(f"Plan {plan_id} not found")

    columns = [desc[0] for desc in cur.description]
    plan_dict = dict(zip(columns, row))
    plan_dict["tasks"] = json.loads(plan_dict["tasks"])

    return PlanSchema(**plan_dict)


def load_latest_plan() -> PlanSchema:
    """Load the most recent plan from database."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM plans ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
    except sqlite3.Error as e:
        logger.error(f"Failed to load latest plan: {e}")
        raise

    if row is None:
        raise ValueError("No plans found in database")

    columns = [desc[0] for desc in cur.description]
    plan_dict = dict(zip(columns, row))
    plan_dict["tasks"] = json.loads(plan_dict["tasks"])

    return PlanSchema(**plan_dict)
