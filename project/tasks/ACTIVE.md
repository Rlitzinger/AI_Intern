# ACTIVE TASK: Fix classifier to detect multi-action tasks with implicit steps

## What to do
Classifier is missing tasks that combine read/write operations with processing.
Tasks like "Read X, calculate Y, save Z" are being marked as SIMPLE when they're actually COMPLEX (3 actions).

Update classifier to explicitly check for file I/O combined with processing/analysis.

## The Problem

Current:
```
"Read Workouts.csv and calculate average calories, save summary"
→ Classifier: SIMPLE + CLEAR → EXECUTE
→ TaskRouter: Can't route to multiple agents → defaults to CodingAgent
→ Result: Function definition, but no actual file reading or saving
```

Should be:
```
"Read Workouts.csv and calculate average calories, save summary"  
→ Classifier: COMPLEX (read + calculate + save = 3 actions)
→ Decompose into:
   - Task 0: Read file Workouts.csv
   - Task 1: Calculate average calories from workout data
   - Task 2: Save the summary to outputs/workout_summary.txt
```

## Success criteria
- [x] Classifier detects "read X and Y and save Z" as COMPLEX
- [x] Tasks with file I/O + processing are decomposed
- [ ] Test workflow produces actual file with calculated results (run main.py to verify)
- [ ] Non-file tasks still work (pure code generation, pure research)

## Implementation

### Update classifier complexity detection
File: `agents/planning/classifier.py`

In `_build_prompt` method, update the COMPLEXITY section.

Find this part:
```python
CRITICAL: Check for compound actions using "and" or "then":
- "Research X and write Y" = COMPLEX (2 actions: research + code)
- "Write X and save as Y" = COMPLEX (2 actions: code + save)
- "Read X and analyze Y" = COMPLEX (2 actions: read + analyze)
- "Research X, write Y, then save Z" = COMPLEX (3 actions)

Even if each action is simple individually, multiple actions = COMPLEX.
```

Add these additional patterns BEFORE the above section:
```python
CRITICAL: Check for FILE I/O combined with PROCESSING:

File operations (read/save/write) + processing/analysis = ALWAYS COMPLEX:
- "Read X and calculate/analyze Y" = COMPLEX (read + process)
- "Calculate X and save to Y" = COMPLEX (process + save)
- "Read X, process Y, save Z" = COMPLEX (3 actions)
- "Load data from X and compute Y" = COMPLEX (load + compute)

Keywords that indicate file I/O:
- read, load, open, parse (input)
- save, write, output, store, export (output)

Keywords that indicate processing:
- calculate, compute, analyze, summarize, aggregate, average, total

If you see BOTH file I/O AND processing → COMPLEX

Examples:
- "Read Workouts.csv and calculate average calories" = COMPLEX ✓
- "Calculate fibonacci sequence" = SIMPLE ✓ (no file I/O)
- "Read config.json" = SIMPLE ✓ (no processing)
- "Parse CSV and compute statistics, save report" = COMPLEX ✓ (all 3)
```

Also update the SIMPLE definition to be more explicit:
```python
SIMPLE means:
- ONE function, script, or component
- ONE well-defined action (not multiple actions connected by "and")
- NO combination of file I/O + processing + saving
- Less than 100 lines of code
- Single step (no "then", "and then", "followed by")
- No dependencies on other tasks

Examples of SIMPLE:
- "Write a function to calculate fibonacci sequence" (ONE action - pure code)
- "Research protein content in chicken breast" (ONE action - pure research)
- "Read file Workouts.csv" (ONE action - pure file read)
- "Calculate average of [10, 20, 30]" (ONE action - pure computation)

NOT SIMPLE (these are COMPLEX):
- "Read X and calculate Y" (TWO actions: file + processing)
- "Calculate X and save to Y" (TWO actions: processing + file)
- "Research X and write code Y" (TWO actions: research + code)
```

## Validation test

Update main.py:
```python
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
    import os
    summary_files = [f for f in os.listdir('outputs') if 'summary' in f.lower() or 'workout' in f.lower()]
    
    if summary_files and plan.status == "complete":
        latest = sorted(summary_files)[-1]
        print(f"\n✅ Summary file created: outputs/{latest}")
        
        with open(f'outputs/{latest}', 'r') as f:
            content = f.read()
            if 'average' in content.lower() or 'calories' in content.lower():
                print(f"✅ File contains analysis results ({len(content)} chars)")
            else:
                print("⚠️  File created but may not contain analysis")
    else:
        print("\n❌ No summary file created")
```

Expected output:
```
📋 Stage 1: Planning
   Using: HierarchicalPlanner (with classifier)

🔍 Classifying: Read Workouts.csv and calculate average calories, save summary...
📊 Verdict: decompose (or clarify_then_decompose)
💭 Reasoning: Task combines file I/O (read, save) with processing (calculate average)

🔀 Decomposing into subtasks...
  ├─ Subtask 0: Read file Workouts.csv
  ├─ Subtask 1: Calculate average calories from the workout data
  ├─ Subtask 2: Save the analysis summary to outputs/workout_summary.txt

✅ Hierarchical plan created: 3 executable tasks

[Execution...]

✅ Correctly decomposed into multiple tasks
  Task 0: Read file Workouts.csv...
  Task 1: Calculate average calories from the workout data...
  Task 2: Save the analysis summary to outputs/workout_summary.txt...

✅ Summary file created: outputs/2026-02-23_02-45-12_workout_summary.txt
✅ File contains analysis results (245 chars)
```

## Constraints
- DO NOT change verdict mapping logic (_map_to_verdict)
- DO NOT change temperature or model
- ONLY modify the complexity detection in _build_prompt
- Keep all existing complexity checks (they're good)

## Notes
The core issue: Classifier thinks "domain coherence" = "single action"

Just because steps are related (all about workout CSV) doesn't mean they're one action.

File I/O + processing is ALWAYS multi-step, even if conceptually related.