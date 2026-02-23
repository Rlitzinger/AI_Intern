# Enhanced inspect_results.py - Shows hierarchical planning details

import sqlite3
import json
import sys

def inspect_latest_plan(show_tree=True):
    """Load the most recent plan and display detailed execution info."""
    conn = sqlite3.connect('ai_intern_db.sqlite')
    cur = conn.cursor()

    # Get the most recent plan
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

    # Task tree visualization (if hierarchical planning was used)
    if show_tree and len(tasks) > 1:
        print("\n" + "=" * 80)
        print("🌳 TASK DECOMPOSITION TREE")
        print("=" * 80)
        _visualize_task_tree(tasks)

    # Task execution details
    print("\n" + "=" * 80)
    print("📊 TASK EXECUTION DETAILS")
    print("=" * 80)

    for i, task in enumerate(tasks):
        print(f"\n{'─' * 80}")
        print(f"TASK {i}: {task['goal'][:70]}...")
        print(f"{'─' * 80}")

        # Basic info
        print(f"Task ID: {task['task_id']}")
        print(f"Status: {_status_emoji(task['status'])} {task['status']}")
        print(f"Created: {task['created_at']}")

        if task.get('completed_at'):
            print(f"Completed: {task['completed_at']}")

        if task.get('retry_count', 0) > 0:
            print(f"⚠️  Retries: {task['retry_count']}")

        # Error info
        if task.get('error_message'):
            print(f"\n❌ ERROR:")
            print(f"   {task['error_message'][:200]}")
            if len(task['error_message']) > 200:
                print(f"   ... ({len(task['error_message'])} chars total)")

        # Result preview
        if task.get('result'):
            result_preview = task['result'][:300]
            print(f"\n📄 RESULT ({len(task['result'])} chars):")
            print(f"   {result_preview}...")

            # Detect result type
            result_type = _detect_result_type(task['result'])
            print(f"   Type: {result_type}")
        else:
            print("\n⚠️  No result generated")

        # Test info
        if task.get('test_code'):
            print(f"\n🧪 TESTS ({len(task['test_code'])} chars):")
            # Show first few test names
            test_lines = [line.strip() for line in task['test_code'].split('\n')
                         if line.strip().startswith('# Test:')]
            for test_line in test_lines[:3]:
                print(f"   {test_line}")
            if len(test_lines) > 3:
                print(f"   ... and {len(test_lines) - 3} more tests")

    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)

    status_counts = {}
    for task in tasks:
        status = task['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in sorted(status_counts.items()):
        emoji = _status_emoji(status)
        print(f"{emoji} {status}: {count} tasks")

    total_retries = sum(task.get('retry_count', 0) for task in tasks)
    if total_retries > 0:
        print(f"\n⚠️  Total retries across all tasks: {total_retries}")

    avg_result_length = sum(len(task.get('result', '')) for task in tasks) / len(tasks)
    print(f"\n📊 Average result length: {int(avg_result_length)} characters")

    print("\n" + "=" * 80)

def _visualize_task_tree(tasks):
    """Attempt to visualize task hierarchy based on goals."""
    for i, task in enumerate(tasks):
        indent = "  " if i > 0 else ""
        goal_preview = task['goal'][:60]
        status_emoji = _status_emoji(task['status'])
        print(f"{indent}├─ Task {i}: {status_emoji} {goal_preview}...")

def _status_emoji(status):
    """Return emoji for task status."""
    emoji_map = {
        'pending': '⏳',
        'executing': '🔄',
        'validating': '🔍',
        'complete': '✅',
        'validated': '✅',
        'failed': '❌'
    }
    return emoji_map.get(status, '❓')

def _detect_result_type(result):
    """Detect what type of result this is."""
    result_lower = result.lower().strip()

    if result_lower.startswith(('def ', 'import ', 'from ', 'class ')):
        return 'Python code'
    elif result_lower.startswith('{') or result_lower.startswith('['):
        return 'JSON/structured data'
    elif 'according to' in result_lower or 'source' in result_lower:
        return 'Research synthesis'
    elif result_lower.startswith('wrote output to:'):
        return 'File operation'
    else:
        return 'Text/prose'

if __name__ == "__main__":
    # Support command line args
    show_tree = '--no-tree' not in sys.argv

    inspect_latest_plan(show_tree=show_tree)
