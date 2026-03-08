# ACTIVE.md — Contract-Driven Structured Output Pipeline

## Mission
Upgrade the planning-to-execution pipeline so that OutputContracts become execution instructions, not just critique metadata. Agents receive contracts telling them what to produce, fill structured `TaskOutput.key_values`, and the orchestrator passes typed key-value data downstream instead of raw prose.

## Architecture Principle
The **planner defines the contract** (what keys to produce, what types, who consumes it). The **agent fills the contract** (generic "fill this contract" capability, not hardcoded output shapes). The **orchestrator enforces the contract** (validates output matches before passing downstream, retries on mismatch).

## Absolute Paths
All source files are in: `project/` (relative to repo root)
- `project/schemas.py`
- `project/config.py`
- `project/llm.py`
- `project/storage.py`
- `project/workspace.py`
- `project/logging_config.py`
- `project/preprocessing.py`
- `project/context.py`
- `project/environment.py`
- `project/classifier.py`
- `project/hierarchical.py`
- `project/planner.py`
- `project/critic.py`
- `project/spec_generator.py`
- `project/orchestrator.py`
- `project/routing.py`
- `project/coding.py`
- `project/research.py`
- `project/file.py`
- `project/analysis.py`
- `project/validator.py`
- `project/error_classifier.py`

**Read each file before editing it. Do not create new files in place of editing existing ones.**

---

## Phase 1: Enrich OutputContract Schema (schemas.py)

### Goal
Make OutputContract carry enough information to be an execution instruction, not just a planning annotation.

### Changes to `project/schemas.py`

1. Add `expected_keys` field to `OutputContract`:

```python
class OutputContract(BaseModel):
    output_type: Literal["python_code", "prose", "structured_data", "file_path", "none"]
    output_format: str  # e.g. "dict with keys: protein_g, fat_g, calories as floats"
    required_by_tasks: list[int] = []
    expected_keys: list[str] = []  # NEW — explicit key names this task must produce in TaskOutput.key_values
```

2. No other schema changes needed. `TaskOutput` already has `key_values: Optional[dict]` — we just need agents to fill it.

### Done When
- `OutputContract` has `expected_keys: list[str]` field
- All existing code still works (field has default empty list)

---

## Phase 2: Emit Contracts During Decomposition (hierarchical.py)

### Goal
Move contract generation INTO the decomposition step. Currently contracts are generated in a separate `_generate_contracts()` pass after decomposition. Instead, the decomposer prompt should declare what each subtask produces.

### Changes to `project/hierarchical.py`

1. **Update `SubtaskSpec`** in `project/schemas.py` — add output declaration fields:

```python
class SubtaskSpec(BaseModel):
    agent_type: Literal["code", "research", "file"]
    goal: str
    input_files: list[str] = []
    output_file: Optional[str] = None
    depends_on: list[int] = []
    produces: Optional[str] = None        # NEW — what this task outputs: "structured_data", "prose", "file_path", "python_code"
    expected_keys: list[str] = []          # NEW — key names this task must put in TaskOutput.key_values
```

2. **Update the decomposition prompt** in `_decompose_task()` to ask for `produces` and `expected_keys`:

Add to the decomposition prompt instructions (after the existing JSON example):
```
Each subtask must also declare:
- produces: what type of output ("structured_data", "prose", "file_path", "python_code")
- expected_keys: list of key names the output must contain (empty list if produces is "prose" or "python_code")

Example with output declarations:
{
  "subtasks": [
    {
      "agent_type": "file",
      "goal": "Read Workouts.csv and return all workout records",
      "input_files": ["Workouts.csv"],
      "output_file": null,
      "depends_on": [],
      "produces": "structured_data",
      "expected_keys": ["column_names", "row_count", "rows"]
    },
    {
      "agent_type": "code",
      "goal": "Calculate average calories burned per workout from the data",
      "input_files": [],
      "output_file": null,
      "depends_on": [0],
      "produces": "structured_data",
      "expected_keys": ["average_calories", "total_workouts", "summary"]
    },
    {
      "agent_type": "file",
      "goal": "Save the workout summary to outputs/workout_summary.txt",
      "input_files": [],
      "output_file": "workout_summary.txt",
      "depends_on": [1],
      "produces": "file_path",
      "expected_keys": ["file_path"]
    }
  ]
}
```

3. **Convert SubtaskSpec fields to OutputContract** when building TaskSchema objects. In `_decompose_task()`, after creating each `TaskSchema` from the `SubtaskSpec`, build the contract:

```python
# After creating task_obj from spec
if spec.produces:
    valid_types = {"python_code", "prose", "structured_data", "file_path", "none"}
    output_type = spec.produces if spec.produces in valid_types else "prose"
    task_obj.output_contract = OutputContract(
        output_type=output_type,
        output_format=f"key_values with keys: {', '.join(spec.expected_keys)}" if spec.expected_keys else "free-form",
        required_by_tasks=[],  # Will be inferred from depends_on
        expected_keys=spec.expected_keys or []
    )
```

4. **Remove or simplify `_generate_contracts()`**. It should now only fill in `required_by_tasks` by cross-referencing `depends_on` fields, not make LLM calls to generate contracts from scratch. The contract output_type and expected_keys are already set from the decomposer.

Simplified version:
```python
@classmethod
def _generate_contracts(cls, tasks: list[TaskSchema]) -> tuple[list[TaskSchema], int]:
    """Fill in required_by_tasks cross-references. No LLM call needed."""
    if len(tasks) < 2:
        return tasks, 0

    for task in tasks:
        if isinstance(task.output_contract, OutputContract):
            # Find which downstream tasks depend on this one
            task.output_contract.required_by_tasks = [
                t.task_order for t in tasks
                if task.task_order in t.depends_on
            ]

    return tasks, 0  # Zero tokens — purely deterministic
```

### Done When
- Decomposer prompt asks for `produces` and `expected_keys`
- SubtaskSpec has both new fields
- OutputContract is built from SubtaskSpec at decomposition time
- `_generate_contracts()` is LLM-free (just cross-references)
- Existing tests/flows still work (new fields have defaults)

---

## Phase 3: Thread Contracts Into Agent Execution (routing.py, coding.py, file.py, research.py)

### Goal
Each agent receives its OutputContract and uses it to structure its output. The contract tells the agent what keys to produce.

### Changes to `project/routing.py`

In `route_task()` and `_route_by_type()`, the contract is already on the task object. No routing changes needed — agents access `task.output_contract` directly.

### Changes to `project/file.py` (FileAgent)

1. After reading a file (CSV/JSON/TXT), populate `TaskOutput.key_values` based on the contract:

```python
# After reading file content, check for contract
contract = task.output_contract
if isinstance(contract, OutputContract) and contract.expected_keys:
    key_values = {}
    # Auto-populate known keys from file read
    if "column_names" in contract.expected_keys and column_names:
        key_values["column_names"] = column_names
    if "row_count" in contract.expected_keys and row_count is not None:
        key_values["row_count"] = row_count
    if "rows" in contract.expected_keys and rows:
        key_values["rows"] = rows  # list of dicts from CSV
    if "file_path" in contract.expected_keys and file_path:
        key_values["file_path"] = str(file_path)
    # Set any remaining expected keys to indicate they weren't found
    for k in contract.expected_keys:
        if k not in key_values:
            key_values[k] = None

    task.task_output = TaskOutput(
        output_type="data" if key_values else "text",
        raw_result=raw_content,
        key_values=key_values,
        column_names=column_names if column_names else None,
        row_count=row_count,
        data_summary=data_summary,
    )
```

2. For file WRITE tasks, populate `key_values` with `{"file_path": "outputs/filename.txt"}`.

**Important**: Read `project/file.py` first to understand the current structure. Integrate the contract-filling logic into the existing flow — do NOT rewrite the entire file.

### Changes to `project/coding.py` (CodingAgent)

1. When the task has an OutputContract with `expected_keys`, add an instruction to the code generation prompt telling the model to print JSON output:

```python
# In the prompt building section, if contract has expected_keys:
if isinstance(task.output_contract, OutputContract) and task.output_contract.expected_keys:
    keys_instruction = (
        f"\n\nOUTPUT REQUIREMENT: Your code must end with a print statement that outputs "
        f"a JSON object with these exact keys: {task.output_contract.expected_keys}\n"
        f"Example: print(json.dumps({{{', '.join(repr(k) + ': value' for k in task.output_contract.expected_keys)}}}))\n"
        f"Import json at the top of your code."
    )
    # Append keys_instruction to the prompt
```

2. After code execution (in orchestrator's `_try_execute_code` or in the agent itself), parse stdout as JSON and populate `task.task_output.key_values`:

```python
import json as _json

# After subprocess execution captures stdout:
if result.stdout.strip():
    try:
        parsed = _json.loads(result.stdout.strip())
        if isinstance(parsed, dict):
            task.task_output.key_values = parsed
    except _json.JSONDecodeError:
        # stdout wasn't JSON — store as data_summary (existing behavior)
        task.task_output.data_summary = result.stdout.strip()
```

**Important**: Read `project/coding.py` first. The prompt building may be in a method or inline. Find the right injection point for the keys instruction.

### Changes to `project/research.py` (ResearchAgent)

1. After the research agent gets its prose result, if the contract has `expected_keys`, run ONE additional structured extraction call:

```python
from ..schemas import OutputContract, TaskOutput

# After getting research prose result:
if isinstance(task.output_contract, OutputContract) and task.output_contract.expected_keys:
    # Build a small extraction schema dynamically
    extraction_prompt = (
        f"Extract these values from the research text below.\n"
        f"Keys to extract: {task.output_contract.expected_keys}\n"
        f"Research text: {prose_result[:2000]}\n\n"
        f"Return a JSON object with the requested keys. Use null for any key you cannot find."
    )

    # Use a simple dict-based extraction (not a full Pydantic schema call)
    # to keep token cost low
    try:
        from ..llm import call_ollama
        raw = call_ollama(
            model=task_model,  # or settings.RESEARCH_MODEL
            prompt=extraction_prompt,
            system="Extract the requested key-value pairs from the text. Return only valid JSON.",
            temperature=0.0,
        )
        parsed = _json.loads(raw["response"])
        if isinstance(parsed, dict):
            task.task_output = TaskOutput(
                output_type="data",
                raw_result=prose_result,
                key_values=parsed,
                data_summary=prose_result[:500],
            )
    except Exception as e:
        logger.warning(f"Research extraction failed: {e}, falling back to prose")
        # Fall through to existing prose-only behavior
```

**Important**: Read `project/research.py` first. Understand how it currently returns results and where to add the extraction pass. This is the most token-expensive change — it adds one LLM call per research task that has expected_keys.

### Done When
- FileAgent populates `key_values` from CSV reads based on contract
- CodingAgent instructs generated code to print JSON with expected keys
- ResearchAgent runs extraction pass when contract has expected_keys
- All three agents set `task.task_output.key_values` when contract is present
- Agents still work normally when no contract is present (backward compat)

---

## Phase 4: Orchestrator Context Builder Upgrade (orchestrator.py)

### Goal
The orchestrator passes structured `key_values` to downstream tasks instead of raw result strings. Add contract validation before passing data.

### Changes to `project/orchestrator.py`

1. **Upgrade context builder** (currently lines ~252-291 in `_execute_task_with_retry`). Change the `previous_tasks` context to prefer `key_values`:

```python
context = {
    'previous_tasks': [
        {
            'order': t.task_order,
            'goal': t.goal,
            # NEW: prefer key_values over raw result
            'key_values': (
                t.task_output.key_values
                if t.task_output and t.task_output.key_values
                else None
            ),
            # Keep raw result as fallback
            'result': t.result if not (t.task_output and t.task_output.key_values) else None,
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
        }
        for t in relevant_tasks
    ]
}
```

2. **Add contract validation** after task execution succeeds but before marking as complete. Add a new method:

```python
@staticmethod
def _validate_contract(task: TaskSchema) -> list[str]:
    """
    Check that task output matches its OutputContract.
    Returns list of missing keys (empty = valid).
    """
    contract = task.output_contract
    if not isinstance(contract, OutputContract):
        return []  # No contract to validate (spec-driven or no contract)
    if not contract.expected_keys:
        return []  # Contract doesn't specify keys

    if not task.task_output or not task.task_output.key_values:
        return contract.expected_keys  # All keys missing

    produced_keys = set(task.task_output.key_values.keys())
    expected_keys = set(contract.expected_keys)
    missing = expected_keys - produced_keys

    return list(missing)
```

3. **Call `_validate_contract` after validation succeeds** (around line 346 where `task.status == "validated"` is checked). If keys are missing, inject the missing keys into the error feedback and retry:

```python
if task.status == "validated":
    # Contract validation — check output has required keys
    missing_keys = self._validate_contract(task)
    if missing_keys and attempt < self.max_retries:
        logger.warning(f"Contract violation: missing keys {missing_keys}")
        task.error_history.append({
            'attempt': attempt + 1,
            'error': f"Output missing required keys: {missing_keys}",
            'test_code': None,
            'category': 'contract_violation',
        })
        task.status = "pending"  # Force retry
        attempt += 1
        continue

    # ... existing success logic (execute code, register component, etc.)
```

### Done When
- Context builder passes `key_values` when available, falls back to `result`
- `_validate_contract` method exists and checks expected_keys
- Contract violations trigger retry with specific error feedback
- Existing flows without contracts still work

---

## Phase 5: Kill _auto_detect_dependencies (orchestrator.py, hierarchical.py)

### Goal
Remove the keyword-based dependency detection in the orchestrator. Dependencies should come from the planner (SubtaskSpec.depends_on) exclusively.

### Changes to `project/hierarchical.py`

1. In `_decompose_task()`, when converting SubtaskSpec to TaskSchema, thread `depends_on` properly:

```python
for i, spec in enumerate(filtered_specs):
    task_obj = TaskSchema(
        plan_id="placeholder",
        task_order=i,
        goal=spec.goal,
        declared_agent=spec.agent_type,
        depends_on=spec.depends_on,  # THREAD THIS THROUGH — currently not being set
    )
    subtasks.append(task_obj)
```

Verify this is actually happening. If `depends_on` is already being set, confirm it's not being overwritten later.

### Changes to `project/orchestrator.py`

1. **Remove `_auto_detect_dependencies` method entirely** (lines ~617-646).

2. **Remove the call to it** (around line 162: `self._auto_detect_dependencies(plan)`).

3. For the flat `PlanningAgent` path (non-hierarchical), tasks won't have `depends_on` set. This is fine — the orchestrator already handles empty `depends_on` by including all previous tasks in context (lines 260-264). That fallback is correct for flat plans.

### Done When
- `_auto_detect_dependencies` method is deleted
- The call to it in `execute_request` is deleted
- `SubtaskSpec.depends_on` flows through to `TaskSchema.depends_on` at decomposition time
- Flat plans (PlanningAgent) still work with sequential assumption

---

## Phase 6: Update Agent Prompt Context Format (coding.py, research.py, file.py)

### Goal
When agents receive context with `key_values`, format it as explicit named variables in the prompt rather than dumping raw text.

### Changes to ALL agent files that build prompts from context

Find where each agent reads `context['previous_tasks']` and builds its prompt. Add a helper or inline logic:

```python
def format_context_for_prompt(previous_tasks: list[dict]) -> str:
    """Format previous task outputs for agent consumption."""
    if not previous_tasks:
        return ""

    lines = ["INPUT FROM PREVIOUS TASKS:"]
    for t in previous_tasks:
        lines.append(f"\n--- Task {t['order']} ({t['goal'][:60]}) ---")

        if t.get('key_values'):
            # Structured output — present as named variables
            for k, v in t['key_values'].items():
                if isinstance(v, list) and len(v) > 5:
                    lines.append(f"  {k} = [{len(v)} items, first 3: {v[:3]}]")
                elif isinstance(v, str) and len(v) > 200:
                    lines.append(f"  {k} = {v[:200]}...")
                else:
                    lines.append(f"  {k} = {v}")
        elif t.get('data_summary'):
            lines.append(f"  Summary: {t['data_summary'][:300]}")
        elif t.get('result'):
            lines.append(f"  Result: {t['result'][:300]}")
        elif t.get('file_path'):
            lines.append(f"  File: {t['file_path']}")

    return "\n".join(lines)
```

This helper can live in `project/context.py` (which already exists) or be inlined in each agent. Putting it in `context.py` is cleaner — single source of truth.

**Important**: Each agent currently has its own way of reading context. Read each agent file to find where context is consumed and replace the raw string injection with this formatted version. Do not assume they all work the same way.

### Done When
- A `format_context_for_prompt()` function exists (in context.py or similar)
- All agents that consume context use it
- When key_values are present, the prompt shows named variables
- When key_values are absent, falls back to existing behavior

---

## Verification Checklist

After all phases, run these test cases manually:

1. **Simple single task**: "Write a fibonacci function" — should work exactly as before (no contract, no key_values)
2. **File + compute**: "Read Workouts.csv and calculate average calories" — Task 0 should produce key_values with column_names/rows, Task 1 should receive structured data
3. **Research + save**: "Research protein in chicken breast and save to outputs/protein.txt" — Research task should extract key_values, file task should receive them
4. **App-scale**: "Build a note-taking app" — Spec-driven path should still work (spec contracts are dicts, not OutputContract, so they bypass the new validation)

If any test fails, fix it before moving to the next. Do not proceed past a broken test.

## Scope Boundaries — Do NOT Do These

- Do not add parallel execution
- Do not add new agent types
- Do not modify the Red Team Council
- Do not modify the SpecGenerator
- Do not modify the classifier
- Do not change the flat PlanningAgent prompt (only hierarchical decomposer changes)
- Do not create new files — all changes go in existing files
- Do not refactor imports or move code between files unless strictly necessary
---

## Tier 1 Follow-Up Issues - 2026-03-06 17:27

The following test cases failed during `test_tier1.py`. Address before marking Tier 1 complete.

### Issue 1: [Annotations] task count in [1,3] - read+calc - agents should be set (not None)
- **Expected**: 1-3 tasks
- **Actual**: majority failed: [False, True, False]
- **Runs**: [False, True, False]
- **Fix area**: planning/hierarchical.py - _decompose_task or classification

### Issue 2: [Annotations] all tasks annotated - read+calc - agents should be set (not None)
- **Expected**: suggested_agent != None for all tasks
- **Actual**: majority failed: [False, True, False]
- **Runs**: [False, True, False]
- **Fix area**: planning/hierarchical.py - _infer_agent_from_goal, _decompose_task agent field, EXECUTE branch

### Issue 3: [Annotations] task count in [2,3] - research then code
- **Expected**: 2-3 tasks
- **Actual**: majority failed: [False, True, False]
- **Runs**: [False, True, False]
- **Fix area**: planning/hierarchical.py - _decompose_task or classification

### Issue 4: [Annotations] task count in [1,3] - file + analysis/code + file
- **Expected**: 1-3 tasks
- **Actual**: majority failed: [False, False, True]
- **Runs**: [False, False, True]
- **Fix area**: planning/hierarchical.py - _decompose_task or classification

### Issue 5: [Annotations] all tasks annotated - file + analysis/code + file
- **Expected**: suggested_agent != None for all tasks
- **Actual**: majority failed: [False, False, True]
- **Runs**: [False, False, True]
- **Fix area**: planning/hierarchical.py - _infer_agent_from_goal, _decompose_task agent field, EXECUTE branch

### Issue 6: [Regression] test_app_generation.py exits 0
- **Expected**: exit code 0
- **Actual**: exit code 1
ist) -> None: ... def load_notes() -> list: ... def list_notes() -> list: ...

  [1m[91mSOME ASSERTIONS FAILED — see above[0m
llama_init_from_model: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
llama_init_from_model: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
Validation failed: ImportError: No module named 'note_persistence'
Unrecoverable error (unrecoverable_import), skipping retries
Task 1 failed after 1 attempts
Task 2 skipped (dependency failed)
1 failed, 1 skipped out of 3 tasks

- **Runs**: n/a
- **Fix area**: Review test_app_generation.py output above - check if Tier 1 changes broke an existing assertion (LLM code-gen flakiness is expected; structural assertion failures are not)

---

## Tier 1 Follow-Up Issues - 2026-03-06 17:40

The following test cases failed during `test_tier1.py`. Address before marking Tier 1 complete.

### Issue 1: [Annotations] task count in [2,3] - CLARIFY+app_scale - vague app request must still produce multi-component plan
- **Expected**: 2-3 tasks
- **Actual**: majority failed: [False, False, False]
- **Runs**: [False, False, False]
- **Fix area**: planning/hierarchical.py - _decompose_task or classification

### Issue 2: [EnvContext] plan goals reference real CSV columns/filename
- **Expected**: at least 1 keyword from ['Calories', 'Exercise', 'Duration_min', 'Heart_Rate', 'test_workouts.csv', 'Heart', 'Duration'] in task goals
- **Actual**: run results: [False, False, True]
- **Runs**: [False, False, True]
- **Fix area**: planning/hierarchical.py - env_header prepended to _decompose_task prompt; or classifier.py env injection; check to_prompt_block() output

### Issue 3: [Regression] test_app_generation.py exits 0
- **Expected**: exit code 0
- **Actual**: exit code 1
ist) -> None: ... def load_notes() -> list: ... def list_notes() -> list: ...

  [1m[91mSOME ASSERTIONS FAILED — see above[0m
llama_init_from_model: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
llama_init_from_model: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
Validation failed: ImportError: No module named 'note_persistence'
Unrecoverable error (unrecoverable_import), skipping retries
Task 1 failed after 1 attempts
Task 2 skipped (dependency failed)
1 failed, 1 skipped out of 3 tasks

- **Runs**: n/a
- **Fix area**: Review test_app_generation.py output above - check if Tier 1 changes broke an existing assertion (LLM code-gen flakiness is expected; structural assertion failures are not)
