"""
plan.py — Print the generated plan for a given prompt.

Usage:
    python plan.py "Read Workouts.csv and calculate average calories"
    python plan.py "Build a note-taking app with save, load, and list"
"""
import sys
import logging
logging.getLogger("ai_intern").setLevel(logging.WARNING)  # suppress INFO logs

from ai_intern.schemas import RequestSchema
from ai_intern.planning.hierarchical import HierarchicalPlanner

prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Prompt: ").strip()

print(f"\nPrompt: {prompt}")
print("-" * 60)

req = RequestSchema(content=prompt)
plan = HierarchicalPlanner.create_plan(req)

print(f"Tasks: {len(plan.tasks)}   Tokens: {plan.token_usage}")
if plan.app_spec:
    summary = plan.app_spec.get("summary", "")
    components = [c["name"] for c in plan.app_spec.get("components", [])]
    print(f"AppSpec: {summary}")
    print(f"Components: {', '.join(components)}")
print()

for i, t in enumerate(plan.tasks):
    agent = t.suggested_agent or "?"
    print(f"  [{i}] [{agent:8}] {t.goal}")
    if t.output_contract:
        print(f"          -> {t.output_contract['output_file']}")
