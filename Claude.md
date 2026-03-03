# AI Intern - Hierarchical Multi-Agent System

## Quick Context
- **What**: Multi-agent orchestration using local Ollama (qwen2.5 models) with hierarchical task decomposition and spec-driven app generation
- **Why**: Learn agent architecture patterns by building from scratch, not using frameworks
- **Goal**: Design a helpful local LLM system that can accomplish complex multi-step tasks autonomously
- **Stack**: Python, Pydantic, SQLite, Ollama
- **Current State**: Full pipeline working — classifies, decomposes, generates specs for app-scale requests, executes per-component, validates, writes to a project workspace, synthesizes final answer

---

## System Overview

The system takes a plain-English request and produces either a direct result (code, research, file operations) or a complete multi-file application. Here is the end-to-end flow:

```
User Request
     │
     ▼
[Preprocessor]          Normalise whitespace, strip artefacts
     │
     ▼
[HierarchicalPlanner]   Classify → (optionally) generate AppSpec → decompose into Tasks
     │                  Produces a flat ordered list of TaskSchema objects
     ▼
[Orchestrator]          For each task (in order):
     │                    1. Build context from prior completed tasks
     │                    2. Inject workspace context if task has output_contract
     │                    3. Route to the right execution agent
     │                    4. Validate the result
     │                    5. On success: write file + register in workspace manifest
     │                    6. On failure: add error to history, retry (max 2 retries)
     ▼
[Final Answer Synthesis] Plain-text or workspace summary
     │
     ▼
[SQLite Persistence]    Plan + all tasks saved after each stage
```

**Two execution modes exist depending on the request type:**

| Request type | Example | Planning path | Output |
|---|---|---|---|
| Simple / multi-step | "Write a fibonacci function" | Classifier → EXECUTE | Single task result |
| Research + code | "Research macros and write a calculator" | Classifier → DECOMPOSE → 2-3 tasks | Code using real data |
| App-scale | "Build a note-taking app with save, load, list" | Classifier → DECOMPOSE/CTD → **SpecGenerator** → per-component tasks | Multi-file workspace |

---

## Project Structure

```
ai_intern/
├── orchestration/
│   ├── orchestrator.py      # Main pipeline controller (Stage 0-4)
│   └── routing.py           # TaskRouter — keyword scoring → agent dispatch
├── planning/
│   ├── classifier.py        # 2x2 LLM classifier (complexity × ambiguity)
│   ├── hierarchical.py      # Recursive decomposition — the core planner
│   ├── spec_generator.py    # AppSpec generation for app-scale requests
│   └── planner.py           # Flat planning (fallback, config flag)
├── execution/
│   ├── coding.py            # CodingAgent — generates Python via qwen2.5-coder
│   ├── research.py          # ResearchAgent — DuckDuckGo search + synthesis
│   ├── file.py              # FileAgent — reads user_data/, writes outputs/
│   └── analysis.py          # AnalysisAgent — data analysis (basic)
├── validation/
│   └── validator.py         # 4-step: syntax → imports → LLM tests → subprocess exec
├── schemas.py               # Pydantic models (RequestSchema, PlanSchema, TaskSchema, TaskOutput)
├── workspace.py             # ProjectWorkspace — per-plan file directory + manifest
├── llm.py                   # Ollama wrappers, token tracking
├── storage.py               # SQLite persistence
├── config.py                # Model assignments, temperatures, feature flags
├── preprocessing.py         # Request normalisation
└── logging_config.py        # Structured logger setup

user_data/                   # User input files (CSVs, JSON, text)
outputs/                     # All generated files
outputs/{plan_id}/           # App-scale workspace (one per plan)
    ├── project.json         # Component manifest
    ├── storage.py           # Generated component 0
    └── api.py               # Generated component 1 (imports storage.py)

test_app_generation.py       # End-to-end test (7 assertions)
test_planning.py             # Planning unit tests
inspect_results.py           # Database inspection CLI
```

---

## Section-by-Section Walkthrough

### 1. Orchestrator (`orchestration/orchestrator.py`)

The top-level controller. `execute_request(request)` runs four stages:

**Stage 0 — Preprocessing**: Normalise the request string (whitespace, encoding artefacts).

**Stage 1 — Planning**: Delegates to `HierarchicalPlanner.create_plan()` (or `PlanningAgent` if `USE_HIERARCHICAL_PLANNING=False`). After planning, logs any `AppSpec` that was generated and calls `_auto_detect_dependencies` to wire sequential task deps. Creates a `ProjectWorkspace` if the plan has an `app_spec`.

**Stage 2 — Execution + Validation**: Iterates tasks in order. For each task:
- Builds `context` dict from all prior completed tasks (includes `data_summary` and `file_path` from `TaskOutput` so downstream agents get compact summaries, not raw multi-KB results)
- If the task has an `output_contract` and prior components exist in the workspace, injects `workspace_context` (component list + interfaces) and `workspace_imports` (`sys.path.insert` lines) into context
- Calls `_execute_task_with_retry(task, plan, workspace)`
  - Routes task → agent → gets result
  - Validates result (ValidationAgent)
  - On `"validated"`: writes code to `workspace/{plan_id}/filename.py`, registers in manifest
  - On failure: records error history, resets goal to `original_goal`, retries (max `max_retries`, default 2)

**Stage 3 — Final Answer**: If the plan has a `workspace_root`, reads `project.json` and builds a structured summary listing all generated files and interfaces. Otherwise synthesises from last task result (or LLM synthesis for multi-task plans).

**Stage 4 — Status**: Sets `plan.status = "complete"` if all tasks are `"validated"` or `"complete"`, otherwise `"failed"`. Saves to SQLite.

---

### 2. Planning (deep dive — see next section)

`HierarchicalPlanner` converts the user's request into an ordered flat list of `TaskSchema` objects. This is the most complex part of the system.

---

### 3. TaskRouter (`orchestration/routing.py`)

Decides which execution agent handles a task using a **weighted keyword scoring system**. Scores four categories (`file`, `code`, `research`, `analysis`) and returns the winner.

**Critical short-circuit**: Tasks with `output_contract` set (i.e. spec-driven component tasks) bypass scoring entirely and always route to `CodingAgent`. This prevents `.py` extension keywords in the goal from accidentally scoring `file` higher than `code`.

Routing rules:
- `output_contract` present → **CodingAgent** (always)
- `file` wins → **FileAgent** (read/write files)
- `code` wins → **CodingAgent** (generate Python)
- `research` wins → **ResearchAgent** (DuckDuckGo search)
- `analysis` wins → **AnalysisAgent**
- Unknown → **CodingAgent** (default)

---

### 4. Execution Agents

**CodingAgent** (`execution/coding.py`): Uses `qwen2.5-coder:7b-instruct` at temperature 0.1. Builds a prompt from: prior task summaries, workspace context (available components + `sys.path` import lines), error history (on retry), and the task goal. Strips markdown fences from output. Returns clean importable Python.

**ResearchAgent** (`execution/research.py`): Runs DuckDuckGo searches, takes top 5 results, synthesises into a prose summary with sources. No API key needed.

**FileAgent** (`execution/file.py`): Reads files from `user_data/` (CSV, JSON, text) and writes output files to `outputs/`. CSV files get structured metadata (`TaskOutput.column_names`, `row_count`, `data_summary`). Smart fuzzy filename matching — if the goal says "workouts" it finds `Workouts.csv`. On write, uses context to grab the previous task's result when goal keywords suggest it.

**AnalysisAgent** (`execution/analysis.py`): Handles data analysis tasks. Routes to CodingAgent internally for complex analysis.

---

### 5. ValidationAgent (`validation/validator.py`)

Four-step pipeline run against every code task result:

1. **Syntax** — `compile(code, "<string>", "exec")`. Fast, no execution.
2. **Import check** — `exec()` with a restricted namespace. Catches `ImportError` / `ModuleNotFoundError`.
3. **Test generation** — LLM generates `assert`-based tests. If the task has an `output_contract`, the prompt instructs tests to verify the public interface (each method exists, accepts documented params, returns correct type).
4. **Test execution** — subprocess runs the tests with a 5-second timeout.

Special handling:
- **Server code** (Flask/FastAPI): starts in subprocess, makes HTTP request, kills process tree (uses `psutil`)
- **Random functions**: skips consistency tests, checks type/structure only
- Non-code tasks (research, file ops): skip directly to `"complete"`

---

### 6. ProjectWorkspace (`workspace.py`)

Manages a per-plan output directory at `outputs/{plan_id}/`.

- **`register_component(name, file_path, interface, depends_on)`**: adds an entry to `project.json` manifest and persists it
- **`get_context_for_task(depends_on)`**: returns a formatted "Available components" string that CodingAgent sees in its prompt
- **`resolve_imports(depends_on)`**: returns `sys.path.insert` lines (not dotted module paths — plan IDs are UUIDs, which are not valid Python identifiers):
  ```python
  import sys
  sys.path.insert(0, r"C:\...\outputs\<plan_id>")
  from storage import NoteStorage
  ```

---

### 7. Schemas (`schemas.py`)

All data is Pydantic-validated. Key models:

**`TaskSchema`** — one unit of work:
- `goal: str` — what to do
- `original_goal: str` — preserved for retry resets
- `output_contract: Optional[dict]` — binds a task to a component: `{component_name, output_file, public_interface}`
- `task_output: Optional[TaskOutput]` — structured result (file path, CSV metadata, data summary)
- `error_history: list[dict]` — per-attempt errors for retry feedback
- `depends_on: list[int]` — task indices this task must wait for

**`PlanSchema`** — a complete plan:
- `tasks: list[TaskSchema]`
- `app_spec: Optional[dict]` — serialised `AppSpec` if generated
- `workspace_root: Optional[str]` — path to `outputs/{plan_id}/` directory
- `token_usage: int` — total LLM tokens consumed
- `final_answer: str` — synthesised response for the user

---

## Deep Dive: Planning Stage

The planning stage is the most layered part of the system. Understanding it fully requires understanding four components that work together: the **Classifier**, the **HierarchicalPlanner**, the **SpecGenerator**, and the **`_is_obviously_simple` guard**.

### TaskClassifier (`planning/classifier.py`)

Every task node passes through a 2x2 LLM classifier before being decomposed or executed.

```
              Clear Goal            Ambiguous Goal
Simple    →   EXECUTE               CLARIFY
Complex   →   DECOMPOSE             CLARIFY_THEN_DECOMPOSE
```

The LLM returns `{is_complex: bool, is_ambiguous: bool, reasoning: str}` (Pydantic-validated). `_map_to_verdict` converts these two booleans to one of four `TaskVerdict` enum values.

- **EXECUTE**: one action, concrete requirements → treat as leaf node
- **CLARIFY**: single action but vague ("make it better") → rewrite goal with concrete details, then leaf
- **DECOMPOSE**: multi-step but well-specified → split into 2-3 subtasks, recurse on each
- **CLARIFY_THEN_DECOMPOSE**: multi-step and vague → clarify OR generate spec first, then decompose

The classifier prompt gives the LLM decision rules: count verbs (read + calculate = complex), check for undefined references (ambiguous). It does not ask about app-scale — that check happens in the planner, not the classifier.

---

### HierarchicalPlanner (`planning/hierarchical.py`)

`create_plan(request)` is the entry point. It:
1. Creates a root `TaskSchema` from the request
2. Calls `_decompose_recursive(root_task, depth=0, shared_context={})`
3. Takes the returned flat list of leaf tasks, assigns `plan_id` + sequential `task_order`, returns `PlanSchema`

The `shared_context` dict is passed **by reference** through the entire recursion. This is how the `AppSpec` generated at depth 0 becomes visible when building subtasks at depth 1 — no global state needed.

#### `_decompose_recursive` — the decision tree

```
depth >= max_depth           → return task as leaf (safety valve)
depth > 0 and _is_obviously_simple(goal)  → return task as leaf (skip LLM call)
                                           (prevents over-decomposition of spec subtasks)

Classify task via LLM
  → EXECUTE               → return [task]
  → CLARIFY               → _clarify_task() → return [clarified_task]
  → DECOMPOSE             → _is_app_scale()?
                               yes → SpecGenerator.generate() → store in shared_context
                               no  → fall through
                           → _decompose_task() → for each subtask: recurse(depth+1)
  → CLARIFY_THEN_DECOMPOSE → _is_app_scale()?
                               yes → SpecGenerator.generate() → store in shared_context
                               no  → _clarify_task() → update shared_context
                           → _decompose_task() → for each subtask: recurse(depth+1)
```

**Why `_is_app_scale` intercepts both DECOMPOSE and CLARIFY_THEN_DECOMPOSE**: the classifier determines ambiguity, not scale. A specific prompt like "Build a note-taking app with save, load, and list" is classified DECOMPOSE (clear). A vague one like "Build an app" gets CLARIFY_THEN_DECOMPOSE. Both should go through `SpecGenerator` — the scale check is independent of the ambiguity check.

```python
APP_SCALE_KEYWORDS = [
    "app", "application", "system", "platform", "tool",
    "build a", "create a", "develop a", "full", "complete",
]
```

---

### SpecGenerator (`planning/spec_generator.py`)

Called when the goal is app-scale. Makes a single LLM call with the `AppSpec` Pydantic schema as the response contract.

```python
class ComponentSpec(BaseModel):
    name: str           # e.g. "NoteStorage"
    responsibility: str # "Handles saving and loading notes to disk"
    output_file: str    # "outputs/storage.py"
    depends_on: list[str] = []  # ["NoteStorage"] — other component names
    public_interface: str  # "save(note), load(id), list_all()"

class AppSpec(BaseModel):
    summary: str
    features: list[str]         # 3-6 user-facing features
    components: list[ComponentSpec]
    done_criteria: list[str]    # 2-4 testable completion criteria
```

The spec is stored in `shared_context['app_spec']` and on `plan.app_spec` (serialised dict). The orchestrator picks it up to create the workspace and to inject workspace context into each task's execution.

---

### `_decompose_task` — standard vs spec-aware paths

After the classify step decides to decompose, `_decompose_task` is called:

**Standard path** (no `app_spec` in context): Makes an LLM call with a structured decomposition prompt. The prompt enforces:
- Max 2-3 subtasks
- Respect agent boundaries (ResearchAgent, CodingAgent, FileAgent are listed)
- Anti-hallucination rules (don't create "research how to use pandas" tasks, don't split a single script into 4 micro-tasks)
- A post-LLM filter that strips hallucinated tasks matching keywords like "research how to", "check if", "merge with existing"
- Hard cap: if LLM returns >3 subtasks, truncate to 3

**Spec-aware path** (`app_spec` present): No LLM call. Creates one `TaskSchema` per component directly from the spec:
```python
goal = (
    f"Write `{comp.name}` class/module in `{comp.output_file}` "
    f"with public interface: {comp.public_interface}. "
    f"Responsibility: {comp.responsibility}."
)
subtask.output_contract = {
    "component_name": comp.name,
    "output_file": comp.output_file,
    "public_interface": comp.public_interface,
}
```

The `output_contract` on each task is the binding contract that ties execution, validation, and workspace registration together.

---

### `_is_obviously_simple` — preventing over-decomposition

After the initial decomposition into spec subtasks (depth 0 → depth 1), each subtask is passed back to `_decompose_recursive`. Without a guard, the classifier would call the LLM again on each subtask, potentially splitting "Write `NoteStorage`..." into further sub-subtasks.

`_is_obviously_simple` short-circuits this by pattern-matching the goal string before making any LLM call:

```python
simple_prefixes   = ["read file", "read csv", "save the", "calculate ", ...]
single_agent      = ["write a script", "write a function", "implement a", "write `", ...]
# "write `" catches spec-driven goals: "Write `NoteStorage` class/module in ..."
```

If the goal starts with any of these prefixes, or has no "and/then" connectors at all, it is treated as a leaf node immediately. This eliminates redundant LLM classification calls on subtasks that are already atomic.

---

### `_clarify_task` — scope-preserving rewrite

For CLARIFY verdicts (not app-scale), the LLM rewrites the goal to be specific without expanding scope. The prompt explicitly forbids adding new deliverables. Returns `{clarified_goal, extracted_details}` (Pydantic-validated, with a `coerce_list_to_str` validator because the LLM occasionally returns a list where a string is expected).

---

### Context propagation through recursion

`shared_context` (passed by reference to every recursive call) carries:
- `app_spec` — the `AppSpec` object, set when SpecGenerator runs
- `root_clarified_goal` — the clarified/summarised goal from depth 0, forwarded to all subtask contexts so decomposition stays coherent

Each recursive call creates a **local** `subtask_context` (not shared) for the subtask:
```python
subtask_context = {
    'parent_goal': task.goal,
    'root_clarified_goal': context.get('root_clarified_goal'),
    'sibling_count': len(subtasks),
    'sibling_index': i
}
```

This means the classifier at depth 1 knows what the parent's goal was and what position in the sibling list the subtask occupies — useful for avoiding ambiguous decompositions.

---

### Planning token cost breakdown (typical app request)

| Step | LLM call | Approx tokens |
|---|---|---|
| Classify root task | `qwen2.5:7b-instruct` | ~300 |
| SpecGenerator | `qwen2.5:7b-instruct` | ~800 |
| `_decompose_task_from_spec` | None (zero cost) | 0 |
| `_is_obviously_simple` guard on each subtask | None | 0 |
| **Total planning** | | ~1,100 |

Per-task execution (CodingAgent + ValidationAgent) adds ~1,500-3,000 tokens per component. Full app generation (3 components) typically uses 6,000-10,000 tokens.

---

## Common Workflows

### Simple task
```
"Write a fibonacci function"
→ Classify: EXECUTE (simple + clear)
→ CodingAgent generates function
→ ValidationAgent validates
→ result stored
```

### Research → Code → Save
```
"Research chicken thigh macros and write a calculator, save as calc.py"
→ Classify: DECOMPOSE (complex + clear)
→ _is_app_scale? No → standard decompose
→ Task 0: Research macros  (ResearchAgent)
→ Task 1: Write calculate_macros() using research values  (CodingAgent + context)
→ Task 2: Save to outputs/calc.py  (FileAgent, reads Task 1 result)
```

### App-scale generation
```
"Build a simple note-taking app with save, load, and list"
→ Classify: DECOMPOSE (complex + clear)
→ _is_app_scale? Yes → SpecGenerator
→ AppSpec: NoteStorage, NoteAPI (or NoteManager), CLI
→ _decompose_task_from_spec → 3 tasks, each with output_contract
→ Task 0: Write NoteStorage (no deps) → coding.py
→ Task 1: Write NoteManager (depends on NoteStorage) → coding.py imports storage
→ Task 2: Write CLI (depends on NoteManager)
→ Workspace: outputs/{plan_id}/storage.py, manager.py, cli.py + project.json
```

---

## Model Configuration
```python
# Planning / Classification / Validation / Research
PLANNING_MODEL = "qwen2.5:7b-instruct"
PLANNING_TEMP  = 0.2

# Code Generation
CODING_MODEL = "qwen2.5-coder:7b-instruct"
CODING_TEMP  = 0.1

# Validation temperature
VALIDATION_TEMP = 0.2
```

---

## Environment Requirements

### Ollama Service
**CRITICAL**: Ollama must be running before executing any code or tests.
- Runs locally at `http://localhost:11434`
- Start: Ollama desktop app or `ollama serve`
- Verify: `curl http://localhost:11434/api/tags`
- Pull models if missing:
  ```
  ollama pull qwen2.5:7b-instruct
  ollama pull qwen2.5-coder:7b-instruct
  ```

### Python Environment
- Python 3.10+
- Dependencies: `requests`, `pydantic`, `duckduckgo-search`, `psutil`
- Synchronous only — no async/await

---

## Testing

### End-to-end app generation
```bash
python test_app_generation.py
```
7 assertions: plan status, workspace dir, ≥2 .py files, all syntax-valid, `project.json` populated, later file imports earlier file, final answer references workspace.

### Planning unit tests
```bash
python test_planning.py
```

### Quick manual test
```python
from ai_intern.schemas import RequestSchema
from ai_intern.orchestration.orchestrator import Orchestrator

request = RequestSchema(content="Write a fibonacci function")
orchestrator = Orchestrator(max_retries=2)
plan = orchestrator.execute_request(request)
print(f"Status: {plan.status}")
print(f"Tokens: {plan.token_usage}")
```

### Inspect latest run
```bash
python inspect_results.py
```

---

## Debugging

| Problem | Check |
|---|---|
| Connection errors | `curl http://localhost:11434/api/tags` — is Ollama running? |
| FileAgent triggered for app task | `output_contract` on task? Router should short-circuit to CodingAgent |
| Spec not generated | Does goal contain app-scale keyword? Check `_is_app_scale` keywords |
| Subtask over-decomposed | `_is_obviously_simple` not matching? Check goal prefix |
| Wrong file in workspace | Check `output_file` in component spec vs `rel_file` in orchestrator |
| Validation fails | `inspect_results.py` → look at `test_code` field, read error message |
| Context not passing | Verify previous task status is `"validated"` or `"complete"` |
| Import error in workspace | Check `resolve_imports` output — should use `sys.path.insert`, not dotted path |

---

## Design Principles

1. **No async** — synchronous execution only, keeps the call stack readable
2. **Pydantic schemas are law** — never bypass with raw dicts
3. **Every LLM call returns `(result, tokens)`** — token tracking is non-negotiable
4. **Context flows forward only** — tasks can see results of earlier tasks, never later ones
5. **No frameworks** — no LangChain/LangGraph; orchestration is explicit Python
6. **Small iterations** — test after each change, don't optimise prematurely
7. **Files go to `outputs/`** — never write to `user_data/`

---

## Known Limitations

1. **No user interaction for CLARIFY verdicts** — LLM makes assumptions instead of asking
2. **Research returns prose** — hard to extract structured values for downstream tasks
3. **Sequential execution only** — independent tasks could run in parallel (not implemented)
4. **Spec components capped at 3** — larger apps get grouped (may lose granularity)
5. **`_is_app_scale` is keyword-based** — may false-positive on non-app requests containing "build a"
6. **Decomposition non-deterministic** — LLM may combine or split steps differently each run

---

## Learning Objectives

1. **Agent orchestration** — coordinating multiple specialised LLM agents
2. **Task decomposition** — classifier-guided recursive breakdown
3. **Spec-driven generation** — structured contracts between planning and execution
4. **Context management** — passing data forward through a pipeline without a framework
5. **Retry logic** — error history feedback to LLM on retry
6. **Validation strategies** — syntax → import → LLM-generated tests → subprocess
7. **Prompt engineering** — anti-hallucination rules, structured JSON output, schema coercion

The goal is to understand how multi-agent systems work at a fundamental level, not to build the most feature-rich system.
