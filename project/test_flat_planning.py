from schemas import RequestSchema
from agents import Orchestrator
import config

# Temporarily switch to flat planning
config.USE_HIERARCHICAL_PLANNING = False

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TEST 2: Flat Planning (fallback mode)")
    print("=" * 80)

    user_input = "Write a function to calculate fibonacci sequence"
    request = RequestSchema(content=user_input)

    orchestrator = Orchestrator(max_retries=2)
    plan = orchestrator.execute_request(request)

    print(f"\n{'=' * 80}")
    print(f"RESULT")
    print(f"{'=' * 80}")
    print(f"Plan Status: {plan.status}")
    print(f"Planning mode: {'Hierarchical' if config.USE_HIERARCHICAL_PLANNING else 'Flat'}")

    if plan.status == "complete":
        print("✅ Flat planning still works as fallback")
    else:
        print("❌ Flat planning failed")
