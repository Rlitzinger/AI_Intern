from schemas import RequestSchema
from agents import Orchestrator
import os

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TEST: Read + Calculate + Save workflow")
    print("=" * 80)

    user_input = "Read Workouts.csv and calculate average calories, save summary"
    request = RequestSchema(content=user_input)

    orchestrator = Orchestrator(max_retries=2)
    plan = orchestrator.execute_request(request)

    print(f"\n{'=' * 80}")
    print(f"RESULT")
    print(f"{'=' * 80}")
    print(f"Plan Status: {plan.status}")
    print(f"Total Tasks: {len(plan.tasks)}")

    # Should have 3 tasks
    if len(plan.tasks) >= 3:
        print("✅ Correctly decomposed into multiple tasks")
        for i, task in enumerate(plan.tasks):
            print(f"  Task {i}: {task.goal[:60]}...")
    else:
        print(f"❌ Only created {len(plan.tasks)} task(s) - should be 3")

    # Check if summary file was created
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    summary_files = [f for f in os.listdir(outputs_dir) if 'summary' in f.lower() or 'workout' in f.lower()]

    if summary_files and plan.status == "complete":
        latest = sorted(summary_files)[-1]
        print(f"\n✅ Summary file created: outputs/{latest}")

        with open(os.path.join(outputs_dir, latest), 'r') as f:
            content = f.read()
            if 'average' in content.lower() or 'calories' in content.lower():
                print(f"✅ File contains analysis results ({len(content)} chars)")
            else:
                print("⚠️  File created but may not contain analysis")
    else:
        print("\n❌ No summary file created")
