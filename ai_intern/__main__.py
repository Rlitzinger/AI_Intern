import sys
import os
import argparse

# Force UTF-8 output on Windows
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .schemas import RequestSchema
from .orchestration.orchestrator import Orchestrator
from .execution.file import FileAgent
from .llm import check_ollama_available, OllamaConnectionError
from .logging_config import setup_logging
from .config import settings
from .storage import load_latest_plan


def health_check():
    """Check Ollama connectivity and print status."""
    if check_ollama_available():
        print(f"Ollama is running at {settings.OLLAMA_BASE_URL}")
        return True
    else:
        print(f"ERROR: Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}")
        print("Please start Ollama with 'ollama serve' or launch the desktop app.")
        return False


def list_files():
    """List available user data files."""
    files = FileAgent.list_available_files()
    if files:
        print("Available files in user_data/:")
        for f in files:
            print(f"  - {f}")
    else:
        print("No files found in user_data/")


def inspect_latest():
    """Show the latest plan from the database."""
    try:
        plan = load_latest_plan()
    except (ValueError, Exception) as e:
        print(f"No plans found: {e}")
        return

    print(f"\nPlan: {plan.plan_id}")
    print(f"Status: {plan.status}")
    print(f"Tasks: {len(plan.tasks)}")
    print(f"Tokens: {plan.token_usage}")
    print(f"Created: {plan.created_at}")
    print()

    for i, task in enumerate(plan.tasks):
        status_icon = {
            "validated": "+", "complete": "+", "failed": "X",
            "skipped": "-", "pending": "?", "executing": "~",
        }.get(task.status, "?")
        print(f"  [{status_icon}] Task {i}: {task.goal[:70]}")
        if task.status == "failed" and task.error_message:
            print(f"      Error: {task.error_message[:100]}")
        elif task.status == "skipped":
            print(f"      {task.error_message or 'Skipped'}")

    if plan.final_answer:
        print(f"\nFinal Answer:\n{plan.final_answer[:500]}")


def execute_query(query: str):
    """Execute a user query through the orchestrator."""
    if not check_ollama_available():
        print(f"ERROR: Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}")
        print("Please start Ollama with 'ollama serve' or launch the desktop app.")
        sys.exit(1)

    request = RequestSchema(content=query)

    try:
        orchestrator = Orchestrator()
        plan = orchestrator.execute_request(request)
    except OllamaConnectionError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Status: {plan.status} | Tasks: {len(plan.tasks)} | Tokens: {plan.token_usage}")
    print(f"{'=' * 60}")

    for i, task in enumerate(plan.tasks):
        icon = {"validated": "+", "complete": "+", "failed": "X", "skipped": "-"}.get(task.status, "?")
        print(f"  [{icon}] Task {i}: {task.goal[:60]}")

    if plan.final_answer:
        print(f"\n--- Final Answer ---\n{plan.final_answer}")
    print()


def interactive_mode():
    """Interactive REPL mode."""
    print("AI Intern - Interactive Mode")
    print("Type 'quit' or 'exit' to leave. Type 'files' to list data files.\n")

    if not health_check():
        sys.exit(1)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "files":
            list_files()
            continue
        if query.lower() == "inspect":
            inspect_latest()
            continue

        execute_query(query)


def main():
    parser = argparse.ArgumentParser(
        prog="ai_intern",
        description="AI Intern - Multi-agent task execution system",
    )
    parser.add_argument("query", nargs="?", help="Task to execute")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--inspect", action="store_true", help="Show latest plan")
    parser.add_argument("--list-files", action="store_true", help="List available user data files")
    parser.add_argument("--health", action="store_true", help="Check Ollama connection")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug-level logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Warning-level logging only")

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        setup_logging("DEBUG")
    elif args.quiet:
        setup_logging("WARNING")
    else:
        setup_logging(settings.LOG_LEVEL)

    # Ensure directories exist
    settings.USER_DATA_DIR.mkdir(exist_ok=True)
    settings.OUTPUT_DIR.mkdir(exist_ok=True)

    # Handle subcommands
    if args.health:
        sys.exit(0 if health_check() else 1)

    if args.list_files:
        list_files()
        return

    if args.inspect:
        inspect_latest()
        return

    if args.interactive:
        interactive_mode()
        return

    if args.query:
        execute_query(args.query)
        return

    # No args - show help
    parser.print_help()


if __name__ == "__main__":
    main()
