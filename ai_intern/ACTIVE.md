# ACTIVE: Planner Upgrade — Spec Generation & Project Workspace

## Goal
Move the planner from "list of isolated tasks" to "structured plan with shared context and output
contracts." This is the foundation needed to generate a complete application (e.g. a note-taking
app) from a single prompt. Work in three sequential phases. Do not start Phase 2 until Phase 1
tests pass. Do not start Phase 3 until Phase 2 tests pass.

---

## Current State (do not change these files unless the phase requires it)

| File | Role |
|------|------|
| `ai_intern/planning/hierarchical.py` | Hierarchical planner with classifier + clarifier |
| `ai_intern/planning/classifier.py` | 2x2 complexity/ambiguity classifier |
| `ai_intern/schemas.py` | Pydantic schemas: TaskOutput, TaskSchema, PlanSchema |
| `ai_intern/orchestration/orchestrator.py` | Pipeline: plan → execute → validate |
| `ai_intern/orchestration/routing.py` | Routes tasks to agents by keyword scoring |
| `ai_intern/config.py` | Settings singleton |

---

## Phase 1 — Spec Generation

### What & Why
The clarifier currently produces a single clarified goal string. For app-scale requests,
that is not enough — you need a structured spec (feature list, components, interfaces,
"done" criteria) before decomposition begins. Without a spec, the decomposer has no basis
for defining what each task should produce.

Trigger condition: request is classified as CLARIFY_THEN_DECOMPOSE **and** the goal
contains app-scale keywords ("app", "application", "system", "platform", "tool", "build a").
All other requests continue through the existing clarify → decompose path unchanged.

### New File: `ai_intern/planning/spec_generator.py`

Create this file. It should contain one class: `SpecGenerator`.

```python
class AppSpec(BaseModel):
    """Structured spec produced from an app-scale request."""
    summary: str                    # One sentence: what the app does
    features: list[str]             # 3-6 user-facing features, plain English
    components: list[ComponentSpec] # Technical building blocks (see below)
    done_criteria: list[str]        # How to know the app is complete (2-4 items)

class ComponentSpec(BaseModel):
    name: str           # e.g. "NoteStorage", "NoteAPI", "CLI"
    responsibility: str # One sentence: what this component does
    output_file: str    # Where this component will be written, e.g. "outputs/storage.py"
    depends_on: list[str]  # Names of other components this one imports/uses
    public_interface: str  # Key functions/classes this exposes, plain English
                           # e.g. "save(note), load(id), delete(id), list_all()"
```

`SpecGenerator.generate(task, context) -> tuple[AppSpec, int]` should:
- Call the LLM with `call_ollama_structured` using `AppSpec` as the response schema
- Use `settings.PLANNING_MODEL` and `settings.PLANNING_TEMP`
- Return `(spec, tokens_used)`

The system prompt for the LLM call:
```
You are a software architect. Given a user request, produce a minimal but complete
application spec. Keep features to what was actually requested — do not add scope.
Components should map to individual Python files. Define clear interfaces between them.
```

The user prompt should include:
- The original request
- Instruction to keep features to exactly what was requested (no scope creep)
- Instruction to limit components to 2-5 (small apps don't need 10 files)
- Instruction to make `output_file` paths relative to `outputs/` directory
- Instruction to list `depends_on` using the `name` field of other components

### Changes to `ai_intern/planning/hierarchical.py`

**Add to imports:**
```python
from .spec_generator import SpecGenerator, AppSpec
```

**Add a new method `_is_app_scale(goal: str) -> bool`:**
```python
APP_SCALE_KEYWORDS = [
    "app", "application", "system", "platform", "tool",
    "build a", "create a", "develop a", "full", "complete",
]
# Return True if any keyword is in goal.lower()
```

**Modify the `CLARIFY_THEN_DECOMPOSE` branch in `_decompose_recursive`:**

Before calling `_clarify_task`, check `_is_app_scale(task.goal)`.

If app-scale:
1. Call `SpecGenerator.generate(task, context)` → `(spec, tokens)`
2. Add tokens to total
3. Log the spec summary and component list at INFO level
4. Store the spec on context: `context['app_spec'] = spec`
5. Convert the spec into a clarified goal string for the existing decompose path:
   ```python
   clarified_goal = (
       f"{spec.summary}. "
       f"Components: {', '.join(c.name for c in spec.components)}. "
       f"Features: {'; '.join(spec.features)}"
   )
   ```
6. Create a `TaskSchema` with this as the goal and continue to `_decompose_task` as normal

If not app-scale: use existing `_clarify_task` path unchanged.

**Modify `_decompose_task` to use the spec when available:**

At the top of `_decompose_task`, check `context.get('app_spec')`.

If spec is present, replace the decomposition prompt with a spec-aware version:
- List each component as a candidate subtask
- Instruct the model to create one subtask per component, in dependency order
- Pass `component.output_file` and `component.public_interface` so the model
  can write concrete subtask goals like:
  "Write `NoteStorage` class in `outputs/storage.py` with methods save(note), load(id), delete(id), list_all()"
- Cap at 3 subtasks still — if there are more than 3 components, group related ones

If no spec: use existing decompose prompt unchanged.

### Changes to `ai_intern/schemas.py`

Add `app_spec` field to `PlanSchema` so the spec persists with the plan:

```python
from typing import Any
# In PlanSchema:
app_spec: Optional[dict] = None  # Serialised AppSpec, if generated
```

### Changes to `ai_intern/orchestration/orchestrator.py`

After `HierarchicalPlanner.create_plan(request)` returns, check if the plan has an
`app_spec`. If so, log a formatted summary of the spec at INFO level so it's visible
in the console output. No other changes needed in the orchestrator for Phase 1.

### Phase 1 Test

Run the existing `test_planning.py` with these inputs and verify the results:

```
python test_planning.py "Build a note-taking app"
python test_planning.py "Write a fibonacci function"
python test_planning.py "Research stock prices and write a trend calculator"
```

Expected outcomes:
- "Build a note-taking app" → triggers spec generator, produces 2-3 component-based tasks,
  each task goal references a specific output file and interface
- "Write a fibonacci function" → unchanged, EXECUTE verdict, 1 task
- "Research stock prices..." → unchanged, uses existing clarify path, 2 tasks

---

## Phase 2 — Project Workspace

### What & Why
Tasks currently pass context as summary strings injected into prompts. This works for
single values ("the CSV has 200 rows") but breaks when Task 1's code needs to import
Task 0's class. The workspace gives every task a shared filesystem location and a
manifest file that maps component names to file paths and interfaces.

### New File: `ai_intern/workspace.py`

```python
class ProjectWorkspace:
    """
    Manages a shared directory for a single plan's generated files.

    Layout:
        outputs/{plan_id}/
            project.json       ← manifest: component name → file path + interface
            storage.py         ← generated by task 0
            api.py             ← generated by task 1 (imports storage.py)
            ...
    """
```

Required methods:

**`ProjectWorkspace.__init__(plan_id: str)`**
- Set `self.root = settings.OUTPUT_DIR / plan_id`
- Create the directory if it doesn't exist
- Load `project.json` if it exists, otherwise initialise empty manifest dict

**`ProjectWorkspace.register_component(name, file_path, interface, depends_on)`**
- Add entry to manifest dict
- Write manifest to `project.json` (pretty-printed JSON)
- `file_path` should be relative to workspace root

**`ProjectWorkspace.get_component(name) -> dict | None`**
- Return the manifest entry for `name`, or None if not registered

**`ProjectWorkspace.get_context_for_task(depends_on: list[str]) -> str`**
- For each name in `depends_on`, look up the manifest
- Return a formatted string like:
  ```
  Available components:
  - NoteStorage (outputs/{plan_id}/storage.py): save(note), load(id), delete(id), list_all()
  ```
- This string gets injected into the coding agent's prompt

**`ProjectWorkspace.resolve_imports(depends_on: list[str]) -> str`**
- Return the Python import lines a task should include, based on manifest entries
- e.g. `from outputs.abc123.storage import NoteStorage`

### Changes to `ai_intern/schemas.py`

Add `workspace_root` to `PlanSchema`:
```python
workspace_root: Optional[str] = None  # Path to this plan's workspace directory
```

Add `output_contract` to `TaskSchema`:
```python
output_contract: Optional[dict] = None
# Set by planner when spec is available. Shape:
# {
#   "component_name": "NoteStorage",
#   "output_file": "storage.py",        # relative to workspace root
#   "public_interface": "save(note), load(id), delete(id), list_all()"
# }
```

### Changes to `ai_intern/planning/hierarchical.py`

When the spec is available (app-scale path), after generating subtasks in `_decompose_task`,
iterate over the subtasks and attach `output_contract` to each `TaskSchema`:

```python
for i, subtask in enumerate(subtasks):
    if i < len(spec.components):
        comp = spec.components[i]
        subtask.output_contract = {
            "component_name": comp.name,
            "output_file": comp.output_file,
            "public_interface": comp.public_interface,
        }
```

### Changes to `ai_intern/orchestration/orchestrator.py`

**In `execute_request`**, after `create_plan` returns:
- If `plan.app_spec` is set, create a `ProjectWorkspace(plan.plan_id)`
- Store `str(workspace.root)` in `plan.workspace_root`
- Pass the workspace instance into `_execute_task_with_retry` as a new optional parameter

**In `_execute_task_with_retry`**, if workspace is present and task has `output_contract`:
- Before execution: call `workspace.get_context_for_task(task_depends_on_names)` and
  add the result to the context dict as `context['workspace_context']`
- After successful validation: call `workspace.register_component(...)` using
  `task.output_contract` and the actual output file path from `task.task_output.file_path`

### Changes to `ai_intern/execution/coding.py`

In `build_prompt`, check for `context.get('workspace_context')`.

If present, add a section to the prompt:
```
AVAILABLE COMPONENTS (already implemented — import these, do not rewrite):
{workspace_context}

{resolve_imports output}

Your code must import from the paths shown above.
```

### Phase 2 Test

```
python test_planning.py "Build a note-taking app"
```

Then run the full pipeline:
```
python -m ai_intern "Build a simple note-taking app with save and load"
```

Verify:
- `outputs/{plan_id}/` directory is created
- `project.json` is written and updated after each task completes
- Task 1's generated code contains import statements for Task 0's output
- `inspect_results.py` shows `output_contract` populated on tasks that have it

---

## Phase 3 — Wire It End to End

### What & Why
Phases 1 and 2 build the planner and workspace infrastructure. Phase 3 connects them
so that a single prompt actually produces a runnable application. This phase is mostly
integration work — verifying the pipeline holds together and patching gaps that only
appear when all pieces run together.

### Changes to `ai_intern/validation/validator.py`

For tasks that have an `output_contract`, adjust test generation in `generate_tests`:
- Tell the LLM the expected public interface (from `output_contract['public_interface']`)
- Generate tests that specifically verify each method in the interface exists and is callable
- Do not test implementation details — test the contract

Example addition to the test generation prompt:
```
This code must implement the following public interface:
{output_contract['public_interface']}

Tests must verify:
1. Each listed method/function exists on the class or module
2. Each method accepts the documented parameters without raising TypeError
3. Basic return type is correct (not None unless documented)
```

### Changes to `ai_intern/orchestration/orchestrator.py`

**In `_synthesize_final_answer`**, if `plan.workspace_root` is set:
- List all files in the workspace directory
- Format the final answer as:
  ```
  Application generated in: {workspace_root}

  Files:
    - storage.py     → NoteStorage: save(note), load(id), delete(id), list_all()
    - api.py         → NoteAPI: depends on NoteStorage
    - cli.py         → CLI entry point

  To run: python {workspace_root}/cli.py
  ```

### New Test: `test_app_generation.py`

Create a focused integration test (separate from `test_planning.py`):

```python
"""
End-to-end test: generate a note-taking app from a single prompt.
Verifies the full pipeline produces runnable code, not just a plan.
"""
```

The test should:
1. Run the full orchestrator with `"Build a simple note-taking app with save, load, and list"`
2. Assert `plan.status == "complete"`
3. Assert `plan.workspace_root` exists as a directory
4. Assert at least 2 `.py` files exist in the workspace
5. For each `.py` file, assert it passes Python syntax check (`py_compile`)
6. Assert `project.json` exists and has entries for each component
7. Print a summary of what was generated

### Phase 3 Acceptance Criteria

The implementation is done when `test_app_generation.py` passes with:
- At least 2 Python files generated in the workspace
- Each file passes syntax validation
- Task 1's file contains an import from Task 0's file
- `project.json` reflects the final component registry
- The final answer in the plan output tells the user where the files are and how to run the app

---

## What NOT to Change

- `ai_intern/planning/classifier.py` — the 2x2 classifier is working correctly, do not touch
- `ai_intern/planning/planner.py` — flat planner is kept as fallback, do not modify
- `ai_intern/execution/research.py` — research agent is not involved in app generation
- `ai_intern/execution/analysis.py` — analysis agent is not involved in app generation
- `test_planning.py` — do not modify the test harness; add new tests in separate files
- The existing `_is_obviously_simple` logic — it was recently fixed and is working

---

## Notes for Claude Code

- Work through phases sequentially. Phase 1 first, test it, then Phase 2.
- Run `test_planning.py` after every significant change to catch regressions on existing cases.
- The `call_ollama_structured` function in `llm.py` is the right way to call the LLM with
  a Pydantic schema. Use it for `SpecGenerator`.
- Use `settings.PLANNING_MODEL` and `settings.PLANNING_TEMP` for all new LLM calls in
  the planning layer.
- Keep clarifier temperature at 0.0 or 0.1 to reduce non-determinism.
- All new files go in the appropriate subdirectory:
  - Planning logic → `ai_intern/planning/`
  - Workspace logic → `ai_intern/workspace.py` (top-level utility)
  - Tests → project root alongside `test_planning.py`
- Log generously at INFO level so the console output shows what the spec generator decided.
- Do not add new dependencies beyond what is already in the project. All required libraries
  (pydantic, pathlib, json, sqlite3) are already available.