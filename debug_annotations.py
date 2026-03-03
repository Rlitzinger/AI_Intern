from ai_intern.schemas import RequestSchema
from ai_intern.planning.hierarchical import HierarchicalPlanner

cases = [
    "Read Workouts.csv and calculate average calories",
    "Read Workouts.csv, calculate average calories, save summary to outputs/summary.txt",
]
for req_text in cases:
    req = RequestSchema(content=req_text)
    plan = HierarchicalPlanner.create_plan(req)
    print(f"\nRequest: {req_text[:65]}")
    print(f"Tasks: {len(plan.tasks)}")
    for t in plan.tasks:
        g = t.goal[:55]
        print(f"  goal={repr(g)} agent={repr(t.suggested_agent)}")
