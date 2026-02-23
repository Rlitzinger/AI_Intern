# test_file_agent.py
from schemas import RequestSchema
from agents import Orchestrator

if __name__ == "__main__":
    print("=" * 80)
    print("TESTING FILE AGENT")
    print("=" * 80)
    
    # Test 1: List files
    print("\nTest 1: List available files")
    request = RequestSchema(content="List all files in user data")
    orchestrator = Orchestrator(max_retries=2)
    plan = orchestrator.execute_request(request)
    
    print(f"\nResult:\n{plan.tasks[0].result}")
    
    # Test 2: Read Workouts.csv
    print("\n" + "=" * 80)
    print("Test 2: Read workout data")
    request = RequestSchema(content="Read file Workouts.csv")
    plan = orchestrator.execute_request(request)
    
    print(f"\nResult (first 500 chars):\n{plan.tasks[0].result[:500]}")
    
    # Test 3: Read FoodLog.csv
    print("\n" + "=" * 80)
    print("Test 3: Read food log")
    request = RequestSchema(content="Read file FoodLog.csv")
    plan = orchestrator.execute_request(request)
    
    print(f"\nResult (first 500 chars):\n{plan.tasks[0].result[:500]}")