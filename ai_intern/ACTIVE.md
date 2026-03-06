# ACTIVE.md — Wire Spec-Driven Dependencies Into the Task Graph

## Objective
Convert spec-driven component dependencies from prose hints embedded in goal strings
into machine-readable `task.depends_on` edges, and ensure the orchestrator's
`_auto_detect_dependencies` doesn't overwrite them. Four changes, ordered by
dependency (each builds on the previous).

**Time budget:** ~45 minutes of Claude Code execution.

---

## Context for Claude Code

### What this system does
This is a local LLM multi-agent pipeline. Requests flow through:
`Orchestrator → HierarchicalPlanner → TaskRouter → Agent (Coding/Research/File) → Validator`

For app-scale requests (multi-file apps), the planner runs a `SpecGenerator` that
produces an `AppSpec` with `ComponentSpec` objects. Each `ComponentSpec` has a
`depends_on: list[str]` field listing the *names* of other components it imports.

The method `_decompose_task_from_spec` in `hierarchical.py` converts each
`ComponentSpec` into a `TaskSchema`. **The bug:** it encodes `comp.depends_on` as
prose in the goal string (`"Imports from: ColorExtractor (already implemented)"`) but
never populates `task.depends_on: list[int]`.

### What already works (do NOT reimplement)
- `TaskSchema.depends_on: list[int]` field exists (`schemas.py:87`)
- Execution skip logic in `orchestrator.py:124-134` already checks `task.depends_on`
  and skips tasks whose dependencies failed
- `SubtaskSpec.depends_on: list[int]` exists (`schemas.py:50`) and the standard
  decomposition prompt already asks the LLM to populate it
- `ProjectWorkspace` already resolves imports from component names
- The `output_contract` dict on spec-driven tasks already carries `component_name`

### Key file locations (absolute paths)
All source files are in the project root. Key files for this change:

| File | Role |
|---|---|
| `hierarchical.py` | `_decompose_task_from_spec` (Change 1), `_decompose_task` standard path (Change 3) |
| `orchestrator.py` | `_auto_detect_dependencies` (Change 2), execution loop context building (Change 4) |
| `schemas.py` | `TaskSchema`, `SubtaskSpec`, `ComponentSpec` — read-only for this change |
| `spec_generator.py` | `AppSpec`, `ComponentSpec` — read-only for this change |
| `workspace.py` | `ProjectWorkspace` — read-only for this change |

---

## Implementation Order

### Change 1: Map `comp.depends_on` to `task.depends_on` in `_decompose_task_from_spec`

**File:** `hierarchical.py`, method `_decompose_task_from_spec` (starts ~line 561)

**Current behavior (the bug):**
```python
for comp in components:
    goal = (
        f"Write `{comp.output_file}` Python module.\n"
        f"Module name / class: {comp.name}\n"
        f"Public interface (exact signatures): {comp.public_interface}\n"
        f"Single responsibility: {comp.responsibility}"
    )
    if comp.depends_on:
        goal += f"\nImports from: {', '.join(comp.depends_on)} (already implemented)"
    subtask = TaskSchema(
        plan_id=task.plan_id,
        task_order=len(subtasks),
        goal=goal,
        suggested_agent="code",
    )
    # ... output_contract dict is set, but depends_on is never set
    subtasks.append(subtask)
```

`comp.depends_on` contains *component names* (e.g., `["NoteStorage"]`).
`task.depends_on` expects *task indices* (e.g., `[0]`).

**Required change:**
After building all subtasks, do a second pass that resolves component names to task
indices using the `output_contract["component_name"]` that was just set on each subtask.

**Exact implementation:**

After the existing `for comp in components:` loop (after all subtasks are appended),
and *before* the `logger.info(f"Spec-driven decomposition produced...")` line, add:

```python
        # --- Resolve component-name dependencies to task indices ---
        # Build a lookup: component_name -> task_order (index within this subtask list)
        name_to_index = {}
        for i, st in enumerate(subtasks):
            contract = st.output_contract
            if isinstance(contract, dict) and "component_name" in contract:
                name_to_index[contract["component_name"]] = i

        # Now wire up depends_on using the lookup
        for st in subtasks:
            contract = st.output_contract
            if not isinstance(contract, dict):
                continue
            comp_name = contract.get("component_name", "")
            # Find the original ComponentSpec to get its depends_on names
            matching_comps = [c for c in components if c.name == comp_name]
            if not matching_comps:
                continue
            comp_deps = matching_comps[0].depends_on  # list[str] of component names
            resolved = []
            for dep_name in comp_deps:
                if dep_name in name_to_index:
                    resolved.append(name_to_index[dep_name])
                else:
                    logger.warning(
                        f"Component '{comp_name}' depends on '{dep_name}' "
                        f"but no task found for it — dependency dropped"
                    )
            if resolved:
                st.depends_on = sorted(resolved)
                logger.info(
                    f"Task {st.task_order} ({comp_name}) depends on tasks {resolved}"
                )
```

**Why sorted:** Deterministic ordering for reproducible plans and easier debugging.

**Why warn on missing deps:** If SpecGenerator produces a component that references
a dependency that was truncated (by the cap-at-5 logic), that's a real planning
error that should be visible in logs, not silently ignored.

**Verification after this change:**
```bash
cd /path/to/project && python -c "
from planning.hierarchical import HierarchicalPlanner
from planning.spec_generator import AppSpec, ComponentSpec
from schemas import TaskSchema

# Simulate a 3-component spec where B depends on A, C depends on A and B
spec = AppSpec(
    summary='Test app',
    features=['test'],
    components=[
        ComponentSpec(name='CompA', responsibility='base', output_file='outputs/comp_a.py',
                      depends_on=[], public_interface='do_a()'),
        ComponentSpec(name='CompB', responsibility='uses A', output_file='outputs/comp_b.py',
                      depends_on=['CompA'], public_interface='do_b()'),
        ComponentSpec(name='CompC', responsibility='uses A and B', output_file='outputs/comp_c.py',
                      depends_on=['CompA', 'CompB'], public_interface='do_c()'),
    ],
    done_criteria=['test passes'],
)
root_task = TaskSchema(plan_id='test', task_order=0, goal='test app')
subtasks, tokens = HierarchicalPlanner._decompose_task_from_spec(root_task, spec)
for st in subtasks:
    name = st.output_contract['component_name'] if isinstance(st.output_contract, dict) else '?'
    print(f'Task {st.task_order} ({name}): depends_on={st.depends_on}')
# Expected:
# Task 0 (CompA): depends_on=[]
# Task 1 (CompB): depends_on=[0]
# Task 2 (CompC): depends_on=[0, 1]
"
```

---

### Change 2: Make `_auto_detect_dependencies` respect existing `depends_on`

**File:** `orchestrator.py`, method `_auto_detect_dependencies` (starts ~line 547)

**Current behavior (the overwrite bug):**
```python
@staticmethod
def _auto_detect_dependencies(plan: PlanSchema):
    for task in plan.tasks:
        if task.task_order == 0:
            continue
        goal_lower = task.goal.lower()
        if any(kw in goal_lower for kw in dependency_keywords):
            task.depends_on = [task.task_order - 1]           # ← OVERWRITES
        elif task.task_order > 0:
            task.depends_on = [task.task_order - 1]           # ← OVERWRITES
```

This unconditionally assigns `depends_on = [task_order - 1]` to every task, which
means a task that was correctly wired as `depends_on=[0, 2]` by Change 1 would get
overwritten to `depends_on=[previous_task]`.

**Required change:** Skip tasks that already have `depends_on` populated.

**Exact implementation:**

At the top of the for-loop body, after `if task.task_order == 0: continue`, add:

```python
            # Respect dependencies already set by the planner (e.g., spec-driven)
            if task.depends_on:
                logger.debug(
                    f"Task {task.task_order} already has depends_on={task.depends_on}, skipping auto-detect"
                )
                continue
```

**This is the single highest-leverage line in this entire ACTIVE.** Without it,
Change 1 is dead code.

**Verification after this change:**
```bash
cd /path/to/project && python -c "
from schemas import PlanSchema, TaskSchema

plan = PlanSchema(request_id='test', tasks=[
    TaskSchema(plan_id='t', task_order=0, goal='Write base module'),
    TaskSchema(plan_id='t', task_order=1, goal='Write handler using the base', depends_on=[0]),
    TaskSchema(plan_id='t', task_order=2, goal='Write CLI that uses base and handler', depends_on=[0, 1]),
])

from orchestration.orchestrator import Orchestrator
Orchestrator._auto_detect_dependencies(plan)

for t in plan.tasks:
    print(f'Task {t.task_order}: depends_on={t.depends_on}')
# Expected:
# Task 0: depends_on=[]
# Task 1: depends_on=[0]        ← preserved, NOT overwritten to [0]
# Task 2: depends_on=[0, 1]     ← preserved, NOT overwritten to [1]
"
```

---

### Change 3: Wire `SubtaskSpec.depends_on` into tasks in the standard decomposition path

**File:** `hierarchical.py`, method `_decompose_task` (the standard/non-spec path, starts ~line 408)

**Current behavior:**
The LLM returns `SubtaskSpec` objects with `depends_on: list[int]` populated (the
prompt explicitly asks for it with an example). But the conversion to `TaskSchema`
at lines 533-541 drops `depends_on`:

```python
        subtasks = []
        for i, spec in enumerate(filtered_specs):
            task_obj = TaskSchema(
                plan_id="placeholder",
                task_order=i,
                goal=spec.goal,
                declared_agent=spec.agent_type,
                # ← spec.depends_on is NEVER mapped to task_obj.depends_on
            )
            subtasks.append(task_obj)
```

**Required change:** Add `depends_on=spec.depends_on` to the TaskSchema constructor.

**Exact implementation:**

Change the TaskSchema construction to:

```python
            task_obj = TaskSchema(
                plan_id="placeholder",
                task_order=i,
                goal=spec.goal,
                declared_agent=spec.agent_type,
                depends_on=spec.depends_on,
            )
```

**Risk assessment:** This is the change with real behavioral risk. It trusts the
7B model to populate `SubtaskSpec.depends_on` correctly. If the model returns
garbage indices (e.g., `depends_on=[5]` when there are only 3 subtasks), the
execution loop will skip the task (because `plan.tasks[5]` doesn't exist or the
bounds check at `orchestrator.py:127` — `if dep < len(plan.tasks)` — filters it).

**Add a bounds-check safety net** immediately after the TaskSchema construction:

```python
            # Clamp depends_on to valid sibling indices
            task_obj.depends_on = [d for d in task_obj.depends_on if 0 <= d < len(filtered_specs) and d != i]
```

This ensures:
- No out-of-bounds indices
- No self-dependencies
- No negative indices

**What about removing the sequential fallback in `_auto_detect_dependencies`?**
Do NOT remove the `elif task.task_order > 0: task.depends_on = [task.task_order - 1]`
fallback yet. Change 2 already makes it safe (it only fires for tasks where
`depends_on` is empty). The fallback is correct behavior for tasks that came through
the standard path when the LLM returned `depends_on=[]` — sequential execution is
the safe default when no explicit dependency info exists. Removing it would risk
out-of-order execution for non-spec plans where the 7B model didn't populate
dependencies. Leave it for a future change after you have evidence the LLM reliably
populates `depends_on`.

**Verification after this change:**
```bash
cd /path/to/project && python -c "
from schemas import SubtaskSpec

# Simulate what the LLM returns
specs = [
    SubtaskSpec(agent_type='file', goal='Read Workouts.csv', depends_on=[]),
    SubtaskSpec(agent_type='code', goal='Calculate averages', depends_on=[0]),
    SubtaskSpec(agent_type='file', goal='Save results', depends_on=[1]),
]
# Verify depends_on is preserved
for s in specs:
    print(f'{s.agent_type}: depends_on={s.depends_on}')
# This just validates the schema. The real test is an end-to-end run.
"
```

---

### Change 4: Pass only declared dependencies as context (not all previous tasks)

**File:** `orchestrator.py`, method `_execute_task_with_retry` (starts ~line 177)

**Current behavior (lines 194-219):**
Context is built from ALL previous tasks (`t.task_order < task.task_order`),
regardless of whether the current task actually depends on them. This works but
floods the 7B model's context window with irrelevant information, increasing the
chance of attention degradation.

**Required change:** When `task.depends_on` is populated, filter `previous_tasks`
to only include the declared dependencies. Fall back to "all previous" when
`depends_on` is empty (backward compatibility with non-spec plans).

**Exact implementation:**

Replace the context-building block (lines 194-219) with:

```python
            # Build context from previous tasks (#5 - use data_summary when available)
            # When depends_on is set, only include declared dependencies (focused context).
            # When depends_on is empty, include all previous tasks (backward compat).
            if task.depends_on:
                relevant_tasks = [
                    t for t in plan.tasks
                    if t.task_order in task.depends_on and t.result is not None
                ]
            else:
                relevant_tasks = [
                    t for t in plan.tasks
                    if t.task_order < task.task_order and t.result is not None
                ]

            context = {
                'previous_tasks': [
                    {
                        'order': t.task_order,
                        'goal': t.goal,
                        'result': t.result,
                        'status': t.status,
                        'data_summary': (
                            t.task_output.data_summary
                            if t.task_output and t.task_output.data_summary
                            else None
                        ),
                        'file_path': (
                            t.task_output.file_path
                            if t.task_output and t.task_output.file_path
                            else None
                        ),
                        'key_values': (
                            t.task_output.key_values
                            if t.task_output and t.task_output.key_values
                            else None
                        ),
                    }
                    for t in relevant_tasks
                ]
            }
```

**Why this matters for 7B models:** A 5-component app generates 5 tasks. Task 4
(the CLI entry point) may only depend on tasks 0 and 3. Without this change, the
CodingAgent prompt for task 4 includes the full code output of tasks 0, 1, 2, 3 —
potentially 200+ lines of irrelevant code consuming attention budget. With this
change, it only sees tasks 0 and 3.

**Verification after this change — TWO tests required:**

**Test A: Spec-driven plan (exercises the `if task.depends_on` branch).**
```bash
cd /path/to/project && python -c "
from schemas import RequestSchema
from orchestration.orchestrator import Orchestrator

request = RequestSchema(content='Build a simple note-taking app with add, list, and delete')
orchestrator = Orchestrator(max_retries=1)
plan = orchestrator.execute_request(request)

print(f'\nPlan status: {plan.status}')
print(f'Tasks: {len(plan.tasks)}')
for t in plan.tasks:
    contract = t.output_contract
    name = contract.get('component_name', '?') if isinstance(contract, dict) else '?'
    print(f'  Task {t.task_order} ({name}): depends_on={t.depends_on} status={t.status}')
"
```

**Test B: Non-spec plan (verifies backward compatibility).**
This exercises the full chain: standard decomposition → Change 2's sequential
fallback → Change 4's context scoping. The critical check is that every non-first
task receives context from its predecessor (same behavior as before these changes).
```bash
cd /path/to/project && python -c "
from schemas import RequestSchema, PlanSchema, TaskSchema, TaskOutput

# --- Unit test: simulate a 3-task non-spec plan post-Change-2 ---
# After _auto_detect_dependencies, non-spec tasks with empty depends_on
# get sequential deps. Verify Change 4 produces the same context as before.

plan = PlanSchema(request_id='test', tasks=[
    TaskSchema(plan_id='t', task_order=0, goal='Read Workouts.csv',
               result='date,calories\n2024-01-01,350\n2024-01-02,420',
               status='validated'),
    TaskSchema(plan_id='t', task_order=1, goal='Calculate average calories',
               depends_on=[0],   # set by Change 2 fallback
               result='def calc(): return 385.0',
               status='validated'),
    TaskSchema(plan_id='t', task_order=2, goal='Save summary to file',
               depends_on=[1],   # set by Change 2 fallback
               result=None,      # not yet executed
               status='pending'),
])

# Simulate the context-building logic from Change 4 for task 2
task = plan.tasks[2]
if task.depends_on:
    relevant = [t for t in plan.tasks if t.task_order in task.depends_on and t.result is not None]
else:
    relevant = [t for t in plan.tasks if t.task_order < task.task_order and t.result is not None]

context_orders = [t.task_order for t in relevant]
print(f'Task 2 context includes tasks: {context_orders}')
assert context_orders == [1], f'Expected [1], got {context_orders}'

# Also verify task 1 sees task 0
task1 = plan.tasks[1]
if task1.depends_on:
    relevant1 = [t for t in plan.tasks if t.task_order in task1.depends_on and t.result is not None]
else:
    relevant1 = [t for t in plan.tasks if t.task_order < task1.task_order and t.result is not None]

context_orders1 = [t.task_order for t in relevant1]
print(f'Task 1 context includes tasks: {context_orders1}')
assert context_orders1 == [0], f'Expected [0], got {context_orders1}'

print('Backward compatibility PASSED: non-spec tasks receive sequential context')
"
```

**Test C (optional but recommended): Edge case — task with empty `depends_on` and
no sequential fallback.** This can happen if `_auto_detect_dependencies` is skipped
or if a task is manually constructed. Verifies the `else` branch gives all-previous.
```bash
cd /path/to/project && python -c "
from schemas import PlanSchema, TaskSchema

plan = PlanSchema(request_id='test', tasks=[
    TaskSchema(plan_id='t', task_order=0, goal='Step A', result='result A', status='validated'),
    TaskSchema(plan_id='t', task_order=1, goal='Step B', result='result B', status='validated'),
    TaskSchema(plan_id='t', task_order=2, goal='Step C', depends_on=[],     # explicitly empty
               result=None, status='pending'),
])

task = plan.tasks[2]
if task.depends_on:
    relevant = [t for t in plan.tasks if t.task_order in task.depends_on and t.result is not None]
else:
    relevant = [t for t in plan.tasks if t.task_order < task.task_order and t.result is not None]

context_orders = [t.task_order for t in relevant]
print(f'Task 2 (empty depends_on) context includes tasks: {context_orders}')
assert context_orders == [0, 1], f'Expected [0, 1], got {context_orders}'
print('Empty depends_on fallback PASSED: task sees all previous tasks')
"
```

**Why three tests:** Test A validates the new spec-driven path. Test B validates
that the typical non-spec flow (where Change 2's sequential fallback populates
`depends_on`) still produces equivalent behavior. Test C validates the true fallback
edge case where `depends_on` is genuinely empty — this branch rarely fires in
practice (Change 2 fills it for most tasks), but if it ever does, the task must
still receive full context rather than zero context.

---

## Do Not Modify

These files are read-only for this change set. If you find yourself needing to edit
them, stop and document why in your report — it likely means one of the four changes
above has a design error.

- `schemas.py` — `TaskSchema.depends_on` and `SubtaskSpec.depends_on` already exist
- `spec_generator.py` — `ComponentSpec.depends_on` already produces the right data
- `workspace.py` — Already consumes component names correctly
- `routing.py` — Routing is orthogonal to dependency wiring
- `validator.py` — Validation is orthogonal to dependency wiring
- `classifier.py` — Classification is orthogonal to dependency wiring
- Any file under `red_team/` — Council review is orthogonal
- `config.py`, `storage.py`, `llm.py` — Infrastructure, no changes needed

---

## Implementation Checklist (for Claude Code)

1. Read this file fully before writing any code.
2. For each change (1 through 4), in order:
   a. Read the target method in full — `view` the file, don't rely on memory.
   b. If the change is already present, note it in the report and skip.
   c. Implement the change exactly as specified.
   d. Run the verification command for that change.
   e. If verification fails, fix before moving to the next change.
3. After all changes, run the end-to-end test from Change 4's verification.
4. Report:
   - Which changes were implemented vs. already present
   - Output of each verification command
   - Any unexpected issues encountered

---

## Success Criteria

After all four changes:
- Spec-driven tasks have correct `depends_on` indices (not empty, not prose)
- `_auto_detect_dependencies` does NOT overwrite planner-set dependencies
- Standard-path subtasks carry `SubtaskSpec.depends_on` through to `TaskSchema`
- Context passed to agents is scoped to declared dependencies when available
- **Backward compatibility:** Non-spec plans (e.g., "Read Workouts.csv and calculate
  average calories, save summary") produce identical execution behavior to before
  these changes. Specifically: Change 2's sequential fallback assigns `depends_on=
  [task_order-1]` to every non-first task with empty `depends_on`, and Change 4
  scopes context to those sequential deps — which is functionally the same as "all
  previous tasks" in a linear chain. Tests B and C from Change 4 must both pass.
- No tasks receive zero context due to an empty `depends_on` unless they are
  genuinely the first task in the plan (task_order=0)