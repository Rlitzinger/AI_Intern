"""Enhanced plan inspector with tree visualization and JSON output."""

import json
import sys
import os
import argparse

# Force UTF-8 on Windows so tree/box characters don't crash
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .storage import get_db_connection, load_latest_plan, ensure_constraint_audit_table
from .config import settings


def inspect_latest_plan(show_tree=True, json_output=False):
    """Load the most recent plan and display detailed execution info."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT plan_id, request_id, created_at, status, token_usage, tasks, final_answer
                FROM plans ORDER BY created_at DESC LIMIT 1
            """)
            row = cur.fetchone()
    except Exception as e:
        print(f"Database error: {e}")
        return

    if row is None:
        print("No plans found in database")
        return

    plan_id, request_id, created_at, status, token_usage, tasks_json, final_answer = row
    tasks = json.loads(tasks_json)

    if json_output:
        print(json.dumps({
            "plan_id": plan_id,
            "status": status,
            "token_usage": token_usage,
            "task_count": len(tasks),
            "final_answer": final_answer,
            "tasks": [
                {
                    "order": i,
                    "goal": t["goal"],
                    "status": t["status"],
                    "retry_count": t.get("retry_count", 0),
                    "result_length": len(t.get("result") or ""),
                    "error": t.get("error_message"),
                    "depends_on": t.get("depends_on", []),
                }
                for i, t in enumerate(tasks)
            ],
        }, indent=2))
        return

    print("=" * 70)
    print("PLAN INSPECTION")
    print("=" * 70)
    print(f"Plan ID:     {plan_id}")
    print(f"Status:      {status}")
    print(f"Created:     {created_at}")
    print(f"Tokens:      {token_usage}")
    print(f"Tasks:       {len(tasks)}")

    # Task tree
    if show_tree and tasks:
        print(f"\n{'=' * 70}")
        print("TASK TREE")
        print("=" * 70)
        _visualize_task_tree(tasks)

    # Task details
    print(f"\n{'=' * 70}")
    print("TASK DETAILS")
    print("=" * 70)

    for i, task in enumerate(tasks):
        icon = _status_icon(task["status"])
        deps = task.get("depends_on", [])
        dep_str = f" [depends: {deps}]" if deps else ""

        print(f"\n--- Task {i} [{icon} {task['status']}]{dep_str} ---")
        print(f"  Goal: {task['goal'][:80]}")

        if task.get("retry_count", 0) > 0:
            print(f"  Retries: {task['retry_count']}")

        if task.get("error_message"):
            print(f"  Error: {task['error_message'][:150]}")

        if task.get("result"):
            result_type = _detect_result_type(task["result"])
            print(f"  Result: {result_type} ({len(task['result'])} chars)")
            # Show preview
            preview = task["result"][:200].replace("\n", " ")
            print(f"  Preview: {preview}...")

        if task.get("task_output"):
            to = task["task_output"]
            if to.get("data_summary"):
                print(f"  Summary: {to['data_summary'][:150]}")

    # Final answer
    if final_answer:
        print(f"\n{'=' * 70}")
        print("FINAL ANSWER")
        print("=" * 70)
        print(final_answer[:500])

    # Summary stats
    print(f"\n{'=' * 70}")
    print("STATISTICS")
    print("=" * 70)

    status_counts = {}
    for task in tasks:
        s = task["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    for s, count in sorted(status_counts.items()):
        print(f"  {_status_icon(s)} {s}: {count}")

    total_retries = sum(t.get("retry_count", 0) for t in tasks)
    if total_retries > 0:
        print(f"  Total retries: {total_retries}")

    print()


def _visualize_task_tree(tasks):
    """Visualize task tree with dependency lines."""
    for i, task in enumerate(tasks):
        icon = _status_icon(task["status"])
        deps = task.get("depends_on", [])
        is_last = i == len(tasks) - 1
        prefix = "  " if deps else ""
        connector = "└─" if is_last else "├─"
        dep_arrow = f" (← Task {deps[0]})" if deps else ""

        print(f"  {connector} [{icon}] Task {i}: {task['goal'][:55]}{dep_arrow}")


def _status_icon(status):
    return {
        "pending": "?", "executing": "~", "validating": "~",
        "complete": "+", "validated": "+", "failed": "X", "skipped": "-",
    }.get(status, "?")


def _detect_result_type(result):
    r = result.strip()
    if r.startswith(("def ", "import ", "from ", "class ")):
        return "Python code"
    elif r.startswith(("{", "[")):
        return "JSON/structured"
    elif "according to" in r.lower() or "source" in r.lower():
        return "Research"
    elif r.lower().startswith("wrote output"):
        return "File operation"
    elif r.lower().startswith("available files"):
        return "File listing"
    else:
        return "Text"


def show_constraint_audit(session_id: str = None):
    """Show constraint verification history, optionally filtered by session."""
    with get_db_connection() as conn:
        ensure_constraint_audit_table(conn)
        if session_id:
            cur = conn.execute("""
                SELECT session_id, plan_id, replan_cycle, constraint_text, satisfied, checked_at
                FROM constraint_audit
                WHERE session_id = ?
                ORDER BY replan_cycle, id
            """, (session_id,))
        else:
            cur = conn.execute("""
                SELECT session_id, plan_id, replan_cycle, constraint_text, satisfied, checked_at
                FROM constraint_audit
                ORDER BY checked_at DESC, replan_cycle, id
                LIMIT 50
            """)
        rows = cur.fetchall()

    if not rows:
        print("No constraint audit records found.")
        return

    current_session = None
    for row in rows:
        session, plan_id, cycle, text, satisfied, checked_at = row
        if session != current_session:
            print(f"\n{'='*60}")
            print(f"Session: {session[:8]}...  Plan: {plan_id[:8]}...")
            current_session = session
        status = "+" if satisfied else "X"
        print(f"  Cycle {cycle} [{status}] {text[:100]}")


def main():
    parser = argparse.ArgumentParser(description="Inspect AI Intern plan results")
    parser.add_argument("--no-tree", action="store_true", help="Skip tree visualization")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--constraints", action="store_true",
                        help="Show constraint audit history (replan cycles + violations)")
    parser.add_argument("--session", type=str, default=None,
                        help="Filter constraint audit by session ID prefix")
    args = parser.parse_args()

    if args.constraints:
        show_constraint_audit(session_id=args.session)
    else:
        inspect_latest_plan(show_tree=not args.no_tree, json_output=args.json)


if __name__ == "__main__":
    main()
