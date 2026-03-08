# AI Intern — Hierarchical Multi-Agent System
## Complete Context Document (Current State: March 2026)

---

## Quick Context

- **What**: Multi-agent orchestration using local llama-cpp-python (qwen2.5 models) with hierarchical task decomposition, spec-driven app generation, and an adversarial Red Team Council that reviews every plan before execution.
- **Why**: Learn agent architecture patterns by building from scratch — no LangChain, no LangGraph, no async.
- **Goal**: A local LLM system that autonomously accomplishes complex multi-step tasks, generates multi-file applications, and self-corrects using structured error feedback.
- **Stack**: Python, Pydantic, SQLite, llama-cpp-python (local inference)
- **Current State**: Full pipeline working — classifies, decomposes, generates specs for app-scale requests, runs adversarial council review with replanning, executes per-component, validates, writes to a project workspace, synthesizes final answer.

---

## End-to-End Pipeline (High Level)

```
User Request
     |
     v
[Stage 0: Preprocessor]         Normalise whitespace, strip encoding artefacts
     |
     v
[Stage 1: HierarchicalPlanner]  Environment scan -> Classify -> (optional) SpecGen -> Decompose
     |                           Produces flat ordered list of TaskSchema objects
     |                           Each task has: goal, suggested_agent, declared_agent,
     |                           output_contract, depends_on
     v
[Stage 1.5: Red Team Council]   5-round adversarial review of the plan
     |                           Round 1: Executor + Integrator + Minimalist find issues
     |                           Round 2: Cross-examination (agents dispute each other)
     |                           Round 3: Blue Team defends the plan
     |                           Round 4: Synthesis -- confidence scoring
     |                           Round 5: Constraint Manifest -> replan (up to 3 cycles)
     |                           ConstraintVerifier does deterministic post-replan check
     v
[Stage 2: Orchestrator]         For each task (in depends_on order):
     |                           1. Build context from declared dependencies
     |                           2. Inject workspace context if task has output_contract (dict)
     |                           3. Route to agent (output_contract -> declared_agent ->
     |                              suggested_agent -> keyword scoring)
     |                           4. Validate: syntax -> imports -> LLM tests -> subprocess exec
     |                           5. Contract check: OutputContract.expected_keys present?
     |                           6. On success: write file + register in workspace manifest
     |                           7. On failure: ErrorClassifier -> error_history -> retry (max 2)
     |                           8. _try_execute_code(): run compute tasks, capture stdout
     v
[Stage 3: Final Answer]         Workspace summary (app-scale) OR LLM synthesis (multi-task)
     |                           OR direct result (single task)
     v
[Stage 4: Status + SQLite]      plan.status = "complete" | "failed"; all data persisted
```

---

## Three Execution Modes

| Request type | Example | Planning path | Output |
|---|---|---|---|
| Simple / single-step | "Write a fibonacci function" | EXECUTE leaf | Single code result |
| Research + code + save | "Research chicken macros and write a calculator, save as calc.py" | DECOMPOSE -> 3 tasks | Code using real data |
| App-scale | "Build a note-taking app with save, load, list" | DECOMPOSE/CTD -> SpecGenerator -> per-component tasks | Multi-file workspace |

---

## Project Structure

```
ai_intern/
+-- orchestration/
|   +-- orchestrator.py       # Main pipeline controller (Stages 0-4)
|   +-- routing.py            # TaskRouter -- priority-based agent dispatch
+-- planning/
|   +-- classifier.py         # 3-axis LLM classifier (complexity x ambiguity x scale)
|   +-- hierarchical.py       # Recursive decomposition -- the core planner
|   +-- spec_generator.py     # Two-phase AppSpec generation for app-scale requests
|   +-- context.py            # PlanningContext (file list + agent roster for prompts)
|   +-- environment.py        # EnvironmentScanner (user_data/ scan before every plan)
|   +-- planner.py            # Flat planning fallback (USE_HIERARCHICAL_PLANNING=False)
|   +-- red_team/
|       +-- council.py        # RedTeamCouncil -- 5-round orchestrator
|       +-- executor.py       # ExecutorAgent -- checks executability
|       +-- integrator.py     # IntegratorAgent -- checks integration/data flow
|       +-- minimalist.py     # MinimalistAgent -- checks over-engineering
|       +-- cross_exam.py     # CrossExamination -- agents dispute each other's findings
|       +-- blue_team.py      # BlueTeamDefense -- plan defends itself against findings
|       +-- synthesis.py      # Confidence scoring + ConstraintManifest builder
|       +-- constraint_verifier.py  # Deterministic post-replan checker (no LLM)
|       +-- utils.py
+-- execution/
|   +-- coding.py             # CodingAgent -- Python via qwen2.5-coder
|   +-- research.py           # ResearchAgent -- DuckDuckGo search + LLM synthesis
|   +-- file.py               # FileAgent -- reads user_data/, writes outputs/
|   +-- analysis.py           # AnalysisAgent -- data analysis (routes to CodingAgent)
+-- validation/
|   +-- validator.py          # 4-step: syntax -> imports -> LLM tests -> subprocess exec
|   +-- error_classifier.py   # ErrorClassifier -- categorises errors before retry
|   +-- test_sanitizer.py     # AST-based test cleaner (removes bad imports, relaxes assertions)
+-- schemas.py                # ALL Pydantic models -- the contracts everything talks through
+-- workspace.py              # ProjectWorkspace -- per-plan file directory + manifest
+-- llm.py                    # llama-cpp-python wrappers, Outlines structured decoding
+-- storage.py                # SQLite persistence (plans, critique history, constraint audits)
+-- config.py                 # pydantic_settings (models, temps, timeouts, feature flags)
+-- preprocessing.py          # Request normalisation
+-- logging_config.py         # Structured logger setup

user_data/                    # User input files (CSVs, JSON, text) -- READ ONLY
outputs/                      # All generated files -- WRITE ONLY
outputs/{plan_id}/            # App-scale workspace
    +-- project.json          # Component manifest
    +-- storage.py            # Generated component 0
    +-- api.py                # Generated component 1 (imports storage.py)

plan.py                       # CLI visualizer: shows full plan, spec, contracts, DAG
test_tier1.py                 # 54-test stability harness (54/54 pass)
test_tier2.py                 # 26-test quality harness (26/26 offline + LLM suites 3/3)
inspect_results.py            # DB inspection CLI (--constraints flag for audit history)
```

---

## Schemas (The Contracts Everything Talks Through)

All agent communication flows through Pydantic-validated schemas in `schemas.py`. Nothing passes raw dicts between major stages.

**`TaskSchema`** -- one unit of work:
```python
goal: str                         # Current goal (may be rewritten on retry)
original_goal: str                # Preserved across retries -- reset point
suggested_agent: str | None       # Heuristic annotation from planner (_infer_agent_from_goal)
declared_agent: str | None        # LLM-explicit agent type from decomposer
output_contract: OutputContract | dict | None  # What this task must produce
depends_on: list[int]             # task_order indices this task depends on
error_history: list[dict]         # Per-attempt errors + error categories
error_category: str | None        # Latest ErrorCategory value
task_output: TaskOutput | None    # Structured result (file_path, key_values, data_summary)
```

**`OutputContract`** -- for non-spec tasks (LLM-declared):
```python
output_type: str                  # "python_code", "prose", "structured_data", "file_path"
output_format: str                # Human description of expected format
expected_keys: list[str]          # Key names that must appear in task_output.key_values
required_by_tasks: list[int]      # Downstream task indices that consume this
```

**`SubtaskSpec`** -- what the LLM decomposer declares (then converted to TaskSchema):
```python
agent_type: str                   # "code", "research", "file"
goal: str
input_files: list[str]            # For file tasks that read
output_file: str | None           # For file tasks that write
depends_on: list[int]             # Sibling indices
produces: str                     # "structured_data", "prose", "python_code", "file_path"
expected_keys: list[str]
```

**`AppSpec` / `ComponentSpec`** -- for app-scale spec-driven generation:
```python
# AppSpec
summary: str
features: list[str]              # 3-6 user-facing features
components: list[ComponentSpec]  # 3-6 fine-grained components (<=40 lines each)
done_criteria: list[str]         # 2-4 testable completion criteria

# ComponentSpec
name: str                        # "NoteStorage"
responsibility: str              # "Handles saving/loading notes to disk"
output_file: str                 # "outputs/storage.py" -- snake_case enforced
depends_on: list[str]            # Other component names (resolved to task indices later)
public_interface: str            # "save(note), load(id), list_all()"
```

**Red Team schemas**:
```python
# RedTeamFinding: agent, finding_type, description, task_index, severity, original_index
# CouncilVerdict: approved, high_confidence_findings, constraint_manifest, rounds_used
# ConstraintManifest: must_include_tasks, task_constraints, structural_constraints, must_not_combine
```

---

## Stage 1: Planning Phase (Deep Dive)

### Step 1a: EnvironmentScanner

Before any LLM call, `EnvironmentScanner.scan()` reads `user_data/` and builds:
- `available_files: list[FileInfo]` -- name, extension, size, sample rows (CSV/JSON)
- `current_datetime` -- injected into every planning prompt

This produces an `## Environment` block injected into every LLM prompt so the planner knows what files exist and can reference them by name in decomposed tasks.

### Step 1b: TaskClassifier (3-axis)

Every task node passes through a **3-axis** LLM classifier before decomposition.

```
Axes:               Outputs:
  is_complex        EXECUTE               (simple + clear)
  is_ambiguous  ->  CLARIFY               (simple + vague)
  is_app_scale      DECOMPOSE             (complex + clear)
                    CLARIFY_THEN_DECOMPOSE (complex + vague)
```

Returns a **4-tuple**: `(verdict: TaskVerdict, reasoning: str, tokens: int, is_app_scale: bool)`

**Override rules** applied after classify:
- `EXECUTE + is_app_scale=True` -> forced to `DECOMPOSE` (single task can't produce multi-file app)
- `CLARIFY + is_app_scale=True` -> forced to `CLARIFY_THEN_DECOMPOSE` (must generate spec)

**Hybrid safety net**: If LLM misses app-scale, `_keyword_is_app_scale()` fires using 5 conservative keywords: `" app"`, `"application"`, `"platform"`, `"full stack"`, `"full-stack"`.

### Step 1c: `_decompose_recursive` -- The Decision Tree

```
depth >= max_depth                     -> leaf (no LLM call, safety valve)
depth > 0 AND _is_obviously_simple()   -> leaf (skip LLM call -- prevents over-decomposition)

Classify (-> 4-tuple):

  EXECUTE  -> leaf task, annotate suggested_agent via _infer_agent_from_goal()

  CLARIFY  -> _clarify_task() (LLM rewrites goal, same scope, no new deliverables)
           -> leaf

  DECOMPOSE:
    is_app_scale?  -> SpecGenerator.generate() -> store spec in shared_context['app_spec']
    -> _decompose_task() [spec-aware path if spec present, LLM path otherwise]
    -> for each subtask: recurse(depth+1)

  CLARIFY_THEN_DECOMPOSE:
    is_app_scale?  -> SpecGenerator.generate() -> store spec -> synthetic clarified goal
    else           -> _clarify_task()
    -> lock clarified goal in shared_context['root_clarified_goal']
    -> _decompose_task()
    -> for each subtask: recurse(depth+1)
```

**`_is_obviously_simple`** prevents wasting LLM calls on already-atomic subtasks:
- Simple prefixes: `"read file"`, `"research "`, `"calculate "`, `"save the"`, etc.
- Single-agent prefixes: `"write a script"`, `"implement a"`, `"write \`"` (spec-driven goals)
- No `" and "` / `" then "` connectors -> treat as leaf

### Step 1d: Two Decomposition Paths

**Standard path** (no `app_spec` in context):
- LLM call with `DecompositionResult` schema -> returns `list[SubtaskSpec]`
- Each `SubtaskSpec` has explicit `agent_type`, `depends_on`, `produces`, `expected_keys`
- Post-LLM: strip hallucinated tasks ("research how to", "check if", etc.)
- Cap at 4 subtasks; convert `SubtaskSpec` -> `TaskSchema` with `declared_agent` + `OutputContract`

**Spec-aware path** (`app_spec` present):
- No LLM call -- deterministic from `ComponentSpec` list (zero tokens)
- One `TaskSchema` per component, up to 5
- Each task gets a `dict`-type `output_contract`: `{component_name, output_file, public_interface}`
- Component `depends_on` (by name) resolved to task indices via `name_to_index` lookup
- Goal template: `"Write \`{output_file}\` Python module.\nModule name: {name}\nPublic interface: {public_interface}\nSingle responsibility: {responsibility}"`

### Step 1e: SpecGenerator (Two-Phase)

Called when `is_app_scale=True`. Makes **two** LLM calls:

**Phase 1 -- Intent extraction**: Extracts `summary`, `features`, and `done_criteria`.

**Phase 2 -- Component design**: Designs 3-6 `ComponentSpec` objects with name, responsibility, output_file (snake_case enforced), depends_on, public_interface.

**Safety nets**:
- Fewer than 2 components returned -> appends synthetic `AppCLI` component
- No `capability_owner` (main orchestrating component) -> injects one
- Entrypoint check: `_spec_has_entrypoint()` only accepts `main.py` or `app.py` (strict -- `server.py` does NOT count). Missing entrypoint -> `_inject_entrypoint()` adds a `Main` component.

### Step 1f: Context Propagation Through Recursion

`shared_context` (passed by reference through entire recursion):
```python
{
    'parent_goal': None,
    'root_clarified_goal': None,  # Set once after first clarify/spec, never overwritten
    'environment': env_prompt,    # EnvironmentScanner block injected into every prompt
    'environment_context': env_ctx,
    'app_spec': AppSpec | None,   # Set when SpecGenerator runs
}
```

Each recursive call creates a **local** `subtask_context` (parent_goal, root_clarified_goal, sibling_count, sibling_index) for the classifier call at that depth -- this is NOT the shared context.

### Planning Token Budget (Typical App Request)

| Step | LLM call | Approx tokens |
|---|---|---|
| Classify root task | qwen2.5:7b-instruct | ~300 |
| SpecGenerator Phase 1 | qwen2.5:7b-instruct | ~400 |
| SpecGenerator Phase 2 | qwen2.5:7b-instruct | ~600 |
| _decompose_task_from_spec | None | 0 |
| _is_obviously_simple on each subtask | None | 0 |
| Output contract generation | None | 0 |
| **Total planning** | | ~1,300 |

---

## Stage 1.5: Red Team Council (Adversarial Plan Review)

The council runs after planning and before execution. Up to `MAX_REPLAN_CYCLES=3` replan loops.

### 5-Round Process

**Round 1 -- Independent Red Team** (3 agents):
- `ExecutorAgent` -- executability issues: missing imports, undefined variables, wrong agent assignments
- `IntegratorAgent` -- integration issues: broken data flow, missing task dependencies
- `MinimalistAgent` -- over-engineering: unnecessary tasks, hallucinated steps, scope creep (spec-aware: won't flag multi-component plans if spec justifies them)

Each returns `list[RedTeamFinding]` with `finding_type`, `description`, `task_index`, `severity`.

**Round 2 -- Cross-Examination**:
- Each agent reviews ALL findings from the other two
- Returns `CrossExamResponse`: `{finding_index, verdict: "confirm"|"dispute", reasoning}`
- Any finding disputed by >= 1 other agent is removed from the surviving list

**Round 3 -- Blue Team Defense**:
- The plan defends itself against surviving findings
- Returns `{finding_index, can_rebut: bool, rebuttal: str}`

**Round 4 -- Confidence Synthesis** (deterministic):
- `compute_confidence()` scores each surviving finding 0.0-1.0
  - Base: severity (critical=0.9, high=0.7, medium=0.5, low=0.3)
  - +0.2 if confirmed by cross-exam; -0.3 if blue team rebuts
- `BLOCK_THRESHOLD = 0.6` -- findings scoring >= 0.6 become blocking

**Round 5 -- Constraint Manifest** (only if blocking findings exist):
- `build_constraint_manifest()` converts findings to structured constraints:
  - `must_include_tasks: list[str]` -- task goals that MUST appear in the plan
  - `task_constraints: dict[int, str]` -- per-task-index requirements
  - `structural_constraints: list[str]` -- plan-level rules
  - `must_not_combine: list[tuple]` -- things that must be separate tasks

### Replan Loop

If `approved=False`:
1. `_replan_with_manifest()` augments the original request with hard constraints
2. `HierarchicalPlanner.create_plan()` runs again with augmented request
3. `ConstraintVerifier.check()` runs deterministically (keyword-matching -- no LLM, can't be fooled)
4. `save_constraint_audit()` persists results for `inspect_results.py --constraints`
5. Council reviews the replanned plan (new cycle)

After 3 cycles, proceed with best available plan.

---

## Stage 2: Execution -- How It Responds to the Plan

### Task Router Priority (Ordered)

```
1. output_contract is a dict (spec-driven)  -> CodingAgent (always)
2. declared_agent set by decomposer         -> route to that agent
3. suggested_agent set by planner           -> route to that agent
4. Keyword scoring fallback                 -> winner of {file, code, research, analysis}
```

`declared_agent` and `suggested_agent` are set on ALL `TaskSchema` objects after planning (never None). The router rarely falls through to keyword scoring.

### Context Injection (Before Each Task)

```python
context = {
    'previous_tasks': [           # Only declared dependencies, or all prior if depends_on is empty
        {
            'order': t.task_order,
            'goal': t.goal,
            'key_values': ...,    # Preferred over raw result when present
            'result': ...,        # Only when key_values is absent
            'data_summary': ...,
            'file_path': ...,
        }
    ],
    'workspace_context': ...,     # Available components (spec-driven tasks only)
    'workspace_imports': ...,     # sys.path.insert lines (spec-driven tasks only)
    'error_history': [...],       # Per-attempt errors (only on retry)
}
```

Workspace imports use `sys.path.insert(0, r"...")` + `from storage import NoteStorage` -- never dotted module paths (UUID plan IDs aren't valid Python identifiers).

### CodingAgent

Model: `qwen2.5-coder:7b-instruct` at temperature 0.1. Prompt includes (in order):
1. System role + available libraries block (from ErrorClassifier)
2. Environment block (from EnvironmentScanner)
3. Prior task summaries from context
4. Workspace context (available components + import lines) if spec-driven
5. Error history (on retry) -- specific error + test code that failed

On retry: `_build_correction_prompt()` includes the exact error, the failed test, and the required correction.

### ResearchAgent

DuckDuckGo search (top 5 results) -> LLM synthesis -> key-value extraction. Returns `TaskOutput` with `key_values` for downstream tasks.

### FileAgent

**Read**: CSV/JSON/text from `user_data/`. CSV gets structured metadata. Fuzzy filename matching.
**Write**: Output files to `outputs/`. Uses context to grab prior task result when goal keywords suggest it.

### ValidationAgent (4-Step Pipeline)

1. **Syntax** -- `compile(code, "<string>", "exec")`.
2. **Import check** -- `exec()` in restricted namespace. Catches `ImportError`.
3. **Test generation** -- LLM generates `assert`-based tests. `TestSanitizer` cleans via AST before running: removes extra imports, relaxes hardcoded numeric assertions, warns on undefined names. For spec-driven tasks, tests verify the public interface.
4. **Test execution** -- subprocess with 5-second timeout.

Special: Server code (Flask/FastAPI) starts in subprocess, makes HTTP request, kills process tree (psutil). Random functions skip consistency tests.

### ErrorClassifier

Runs after every failed validation. Categories:
- `retryable_syntax` / `retryable_import` / `retryable_logic` -- LLM can fix
- `unrecoverable_missing_package` / `unrecoverable_timeout` -- abort immediately

If `category.startswith("unrecoverable")` -> abort, no retries.

### Contract Validation

After `status == "validated"`: checks that `task_output.key_values` contains all `output_contract.expected_keys`. If missing AND retries remain -> retry with contract violation in `error_history`. Only applies to `OutputContract` model objects (not spec-driven dict contracts).

### `_try_execute_code` (Compute Tasks Only)

After validation, if goal contains compute keywords (`calculate`, `compute`, `analyze`, `sum`, etc.):
- Runs validated code in subprocess, captures stdout
- Tries to parse stdout as JSON -> `task_output.key_values`
- Falls back to `task_output.data_summary`

### Workspace Registration (Spec-Driven Tasks Only)

After validation:
1. `ws_file_path = workspace.root / rel_file` -- writes code to `outputs/{plan_id}/storage.py`
2. `workspace.register_component(name, file_path, interface, depends_on)` -- updates `project.json`
3. Next task that depends on this component gets `workspace_context` + `workspace_imports` injected

---

## Stage 3: Final Answer Synthesis

- **App-scale** (`plan.workspace_root` set): Reads `project.json`, builds structured summary of all files, interfaces, dependency chains, and entry point.
- **Multi-task** (>1 completed task): LLM synthesis using `data_summary` + `key_values` from all completed tasks. Capped at 200 words.
- **Single task**: Returns `last_task.result` directly.

---

## Model Configuration

```python
PLANNING_MODEL   = "qwen2.5:7b-instruct"        # Classification, decomposition, research, validation
PLANNING_TEMP    = 0.3

CODING_MODEL     = "qwen2.5-coder:7b-instruct"  # Code generation only
CODING_TEMP      = 0.1

VALIDATION_TEMP  = 0.2

CODE_EXEC_TIMEOUT       = 15   # seconds for subprocess code execution
SERVER_STARTUP_TIMEOUT  = 8    # seconds to wait for Flask/FastAPI to start
MAX_RETRIES             = 2    # per task (configurable via Orchestrator(max_retries=N))
MAX_REPLAN_CYCLES       = 3    # Red Team Council replan limit (hardcoded in orchestrator)
MAX_DECOMPOSITION_DEPTH = 3    # Recursion depth limit
```

**LLM Backend**: llama-cpp-python (NOT Ollama HTTP -- backup at `llm_ollama_backup.py`).
- `call_ollama_code()` -- unconstrained text generation
- `call_ollama_structured()` -- Outlines constrained decoding (guaranteed valid JSON matching Pydantic schema)
- Both return `(result, tokens)` -- token tracking is non-negotiable

---

## SQLite Persistence

```
Tables:
  plans                  - Full PlanSchema JSON per plan_id
  plan_critique_history  - Every council round event (R1-R5 findings, cross-exam, blue team)
  constraint_audit       - Per-replan-cycle: violations, satisfied_count
```

`inspect_results.py` shows all tasks, status, result, test_code, error_message.
`inspect_results.py --constraints` shows audit history (cycles, violations, satisfied counts).

---

## Environment Requirements

### LLM Backend (llama-cpp-python)
- Models are lazy singletons loaded on first call
- GPU layers configured in `config.py` (RTX 3070)
- Both models must be available at configured paths

### Python Dependencies
- Python 3.10+
- `pydantic`, `pydantic-settings`, `duckduckgo-search`, `psutil`, `llama-cpp-python`, `outlines`
- Synchronous only -- no async/await anywhere

---

## Common Workflows

### Simple task
```
"Write a fibonacci function"
-> Classify: EXECUTE (simple + clear)
-> suggested_agent = "code"
-> CodingAgent generates function
-> ValidationAgent: syntax, imports, tests, exec all pass
-> status = "validated"
-> final_answer = code result
```

### Research -> Code -> Save
```
"Research chicken thigh macros and write a calculator, save as calc.py"
-> Classify: DECOMPOSE (complex + clear)
-> _is_app_scale? No -> standard decompose (LLM path)
-> Task 0: [research] Research chicken thigh macros -> key_values: {protein, fat, calories}
-> Task 1: [code] Write calculate_macros() using research values from context
-> Task 2: [file] Save Task 1 result to outputs/calc.py
-> final_answer: "File output: outputs/calc.py"
```

### App-scale generation
```
"Build a simple note-taking app with save, load, and list"
-> Classify: DECOMPOSE, is_app_scale=True
-> SpecGenerator Phase 1: intent extraction
-> SpecGenerator Phase 2: NoteStorage, NoteManager, CLI (+ Main entrypoint safety net)
-> _decompose_task_from_spec -> 4 tasks, each with dict output_contract
-> Red Team Council: 5-round review -> approved (or replan if missing entrypoint etc.)
-> Task 0: Write NoteStorage (no deps)
-> Task 1: Write NoteManager (depends_on=[0], workspace_imports injected)
-> Task 2: Write CLI (depends_on=[1])
-> Task 3: Write main.py entrypoint (depends_on=[2])
-> Workspace: outputs/{plan_id}/storage.py, manager.py, cli.py, main.py + project.json
-> final_answer: "Application generated in: outputs/{plan_id}/ ..."
```

---

## Design Principles

1. **No async** -- synchronous execution only, readable call stack
2. **Pydantic schemas are law** -- never bypass with raw dicts between major components
3. **Every LLM call returns `(result, tokens)`** -- token tracking is non-negotiable
4. **Context flows forward only** -- tasks see earlier results, never later ones
5. **No frameworks** -- no LangChain/LangGraph; orchestration is explicit Python
6. **Router priority is explicit and ordered** -- output_contract -> declared_agent -> suggested_agent -> keyword scoring
7. **Error classification before retry** -- unrecoverable errors abort immediately
8. **Adversarial review before execution** -- Red Team Council hardens the plan before any code runs
9. **Deterministic post-replan verification** -- ConstraintVerifier catches constraint violations without LLM judgment

---

## Known Limitations

1. **No user interaction for CLARIFY** -- LLM makes assumptions instead of asking
2. **Research returns prose** -- key_value extraction is LLM-based, can miss values
3. **Sequential execution only** -- independent tasks could run in parallel (not implemented)
4. **App components capped at 5** -- larger apps get truncated
5. **`_is_app_scale` keywords are conservative** -- may miss non-standard app requests
6. **Decomposition non-deterministic** -- LLM may combine or split steps differently each run
7. **Red Team Council token cost** -- 6+ LLM calls per review cycle; app-scale with replanning can cost 15,000+ tokens

---

## Debugging Guide

| Problem | Check |
|---|---|
| LLM not loading | Check llama-cpp-python model paths in `config.py` |
| FileAgent triggered for app task | output_contract dict set? Router should short-circuit to CodingAgent |
| Spec not generated | Does goal contain app-scale keyword? Check `_keyword_is_app_scale` + classifier |
| Subtask over-decomposed | `_is_obviously_simple` not matching? Check goal prefix |
| Wrong file in workspace | Check `output_file` in component spec vs `rel_file` in orchestrator |
| Validation fails | `inspect_results.py` -> `test_code` + `error_message` fields |
| Context not passing | Verify previous task status is "validated" or "complete" |
| Import error in workspace | Check `resolve_imports` -- should use `sys.path.insert`, not dotted path |
| Council replan loop | `inspect_results.py --constraints` -- which constraints survived? |
| Entrypoint missing | SpecGenerator safety net should inject Main; check `_spec_has_entrypoint()` |
| ErrorClassifier aborts early | Check `error_category` -- "unrecoverable_*" skips retries by design |

---

## Testing

```bash
# Tier 1: Planning stability (54/54 tests, 3-run majority vote)
python test_tier1.py

# Tier 2: Execution quality (26 offline tests + LLM suites 3/3)
python test_tier2.py
python test_tier2.py --quick    # Skip LLM suites

# Inspect latest run
python inspect_results.py
python inspect_results.py --constraints

# Full plan visualizer (shows spec, contracts, DAG)
python plan.py "Build a note-taking app with save, load, and list"

# Quick manual test
from ai_intern.schemas import RequestSchema
from ai_intern.orchestration.orchestrator import Orchestrator
request = RequestSchema(content="Write a fibonacci function")
plan = Orchestrator(max_retries=2).execute_request(request)
print(plan.status, plan.token_usage)
```

---

## Current Project State (March 2026)

- **Tier 1 (Grounded Planning): COMPLETE** -- `test_tier1.py` 54/54 stable across 3 runs
- **Tier 2 (Execution Quality): COMPLETE** -- `test_tier2.py` 26/26 offline, all LLM suites 3/3
- **Red Team Council: COMPLETE** -- all 7 hardening changes implemented and stable
- **Current branch**: `claude_rewrite`
- **Next candidates (Tier 3)**: parallel execution, user interaction for CLARIFY verdicts, larger app support (>5 components)
