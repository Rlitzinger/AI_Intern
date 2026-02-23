# AI Intern - Hierarchical Multi-Agent System

## Quick Context
- **What**: Multi-agent orchestration using local Ollama (qwen2.5 models) with hierarchical task decomposition
- **Why**: Learn agent architecture patterns by building from scratch, not using frameworks
- **Goal**: Design a helpful local LLM system that can accomplish complex multi-step tasks autonomously
- **Stack**: Python, Pydantic, SQLite, Ollama
- **Current State**: Hierarchical planning with 2x2 classifier working - decomposes complex tasks, passes context between agents, validates results

## Environment Requirements

### Ollama Service
**CRITICAL**: Ollama must be running before executing any code or tests.
- Ollama runs locally at `http://localhost:11434`
- Models used: 
  - `qwen2.5:7b-instruct` (planning, validation, research, classification)
  - `qwen2.5-coder:7b-instruct` (code generation)
- Start Ollama: Run the Ollama desktop app or `ollama serve` in terminal
- Verify: `curl http://localhost:11434/api/tags` should return JSON with model list
- If models missing: 
  - `ollama pull qwen2.5:7b-instruct`
  - `ollama pull qwen2.5-coder:7b-instruct`

### Python Environment
- Python 3.10+
- Dependencies: `requests`, `pydantic`, `duckduckgo-search`, `psutil`
- No async/await - synchronous execution only
- Runs offline after models are downloaded

## Project Structure
```
project/
├── agents/
│   ├── orchestration/
│   │   ├── orchestrator.py      # Main pipeline + retry logic (max 2 retries)
│   │   └── routing.py            # TaskRouter - dispatches to execution agents
│   ├── planning/
│   │   ├── planner.py           # Flat planning (fallback mode)
│   │   ├── classifier.py        # 2x2 complexity/ambiguity classification
│   │   └── hierarchical.py      # Recursive decomposition (default)
│   ├── execution/
│   │   ├── coding.py            # Generates Python code with context
│   │   ├── research.py          # DuckDuckGo search + synthesis
│   │   └── file.py              # Reads user_data/, writes outputs/
│   └── validation/
│       └── validator.py         # Syntax → imports → tests → execution
├── schemas.py         # Pydantic models - strict validation required
├── llm.py            # Ollama wrappers, token tracking
├── storage.py        # SQLite persistence (plans + tasks)
├── config.py         # Model assignments, temperatures, feature flags
├── main.py           # Entry point for testing
└── inspect_results.py # Database inspection tool

user_data/            # User files (CSVs, docs, etc.)
outputs/              # Generated files (.py, .txt, etc.)
```

## Core Architecture Principles

### 1. No Async (Keep It Simple)
Synchronous execution only. No async/await unless explicitly needed later.

### 2. Pydantic Schemas Are Law
- Every data structure uses Pydantic models
- Never bypass validation with manual dict manipulation
- Schemas: `RequestSchema`, `PlanSchema`, `TaskSchema`

### 3. Token Tracking
Every LLM call returns `(result, tokens_used)` tuple. Track usage across entire pipeline.

### 4. Small Iterations
Build incrementally. Test after each change. Don't optimize prematurely.

### 5. No Frameworks
No LangChain/LangGraph/AutoGPT. Build orchestration from scratch to learn the patterns.

### 6. Context Passing
Tasks can access results from previous tasks via context dict:
```python
context = {
    'previous_tasks': [
        {'order': 0, 'goal': '...', 'result': '...', 'status': '...'}
    ]
}
```

## Model Configuration
```python
# Planning/Classification/Validation
model = "qwen2.5:7b-instruct"
temperature = 0.2-0.3

# Code Generation  
model = "qwen2.5-coder:7b-instruct"
temperature = 0.1

# Research Synthesis
model = "qwen2.5:7b-instruct"
temperature = 0.3
```

Low temperatures for consistency. Higher temps only for creative tasks.

## Hierarchical Planning System

### 2x2 Task Classification
```
                Clear Goal          Ambiguous Goal
Simple      →   EXECUTE            CLARIFY
Complex     →   DECOMPOSE          CLARIFY_THEN_DECOMPOSE
```

**Classifier detects:**
- **Complexity**: Can this be done in one step? (lines of code, number of actions)
- **Ambiguity**: Is the desired outcome clearly specified? (vague terms, undefined references)

**Key Insight**: Decomposing an ambiguous task creates a tree of ambiguous subtasks. Must CLARIFY before DECOMPOSE.

### Recursive Decomposition
```
HierarchicalPlanner:
  1. Classify root task
  2. If EXECUTE → return as leaf task
  3. If CLARIFY → make specific, return as leaf
  4. If DECOMPOSE → break into 2-4 subtasks, recurse on each
  5. If CLARIFY_THEN_DECOMPOSE → clarify first, then decompose
  6. Max depth = 3 (prevent infinite loops)
  7. Return flat list of executable tasks (DFS order)
```

**Toggle modes:**
```python
# config.py
USE_HIERARCHICAL_PLANNING = True  # Default
# Set to False for flat planning (1-3 tasks, no decomposition)
```

## Agent Capabilities

### Orchestrator
- Executes full pipeline: Planning → Execution → Validation
- Retry logic: max 2 retries per task (3 total attempts)
- Passes error feedback to CodingAgent on retry
- Tracks token usage across all agents
- Saves to SQLite after each stage

### TaskRouter
Detects task type from keywords and routes to appropriate agent:
- **code**: "write", "function", "script", "implement"
- **research**: "research", "find", "investigate", "search"
- **file**: "read file", "save", "write to outputs"
- **analysis**: "analyze", "evaluate", "compare" (not implemented)
- **decision**: "decide", "recommend", "choose" (not implemented)

### CodingAgent
- Uses `qwen2.5-coder:7b-instruct`
- Reads context from previous tasks (research data, file contents)
- Strips markdown fences from generated code
- Temperature = 0.1 for consistency
- Returns importable Python code (no example usage)

### ResearchAgent
- Uses DuckDuckGo search (no API key needed)
- Synthesizes findings from top 5 results
- Cites sources in response
- Returns prose summary (future: structured JSON output)

### FileAgent
- **Reads**: user_data/ directory (CSV, JSON, text files)
- **Writes**: outputs/ directory with timestamps
- Smart extension detection: `.py` files get code from previous task
- Formats CSV/JSON as readable text when reading
- Checks previous task status before using results

### ValidationAgent
- **Step 1**: Syntax check (`compile()`)
- **Step 2**: Import validation (`exec()` to trigger ImportErrors)
- **Step 3**: Test generation (LLM creates assert-based tests)
- **Step 4**: Test execution (subprocess with 5s timeout)

**Special handling:**
- **Server code** (Flask/FastAPI): Starts server, makes HTTP request, kills process tree
- **Random functions**: Skips consistency tests, checks type/structure only
- **Deterministic code**: Tests for consistent output across calls

## Common Workflows

### Research → Code → Save
```
User: "Research chicken thigh macros and write a calculator, save as calc.py"

HierarchicalPlanner:
  Task 0: Research protein, fat, carbs in chicken thighs per 100g
  Task 1: Write function calculate_macros(grams) using research values  
  Task 2: Save the function to outputs/calc.py

Execution:
  Task 0: ResearchAgent → finds macros → stores in task.result
  Task 1: CodingAgent → reads Task 0 context → generates code with real values
  Task 2: FileAgent → detects .py extension → saves Task 1 code

Result: Working Python file with researched macro values
```

### Read → Analyze → Write
```
User: "Read Workouts.csv and calculate average calories, save summary"

Tasks:
  Task 0: Read file Workouts.csv
  Task 1: Calculate average calories from the workout data
  Task 2: Save the analysis to outputs/workout_summary.txt
```

## Key Design Decisions

### Why Hierarchical Planning?
Flat planning (PlanningAgent) creates vague tasks like "write calculator using research results" without specifying which values. Hierarchical planning with classification forces clarification before execution.

### Why Context Passing?
Agents need to collaborate. Research finds data, Code uses that data, File saves the result. Context dict passes results forward through the pipeline.

### Why Small Local Models?
- Runs offline (privacy, no API costs)
- Fast iteration (no network latency)
- Forces good architecture (can't rely on GPT-4 to fix bad prompts)

**Architectural insight**: Well-designed systems around 7B models can outperform raw interactions with larger cloud models.

### Why SQLite?
- Persistent storage across runs
- Easy inspection with `inspect_results.py`
- Simple schema (plans + tasks)
- No server setup needed

## Debugging & Inspection

### View Latest Plan
```bash
python inspect_results.py
```

Shows:
- Task tree visualization
- Execution details per task
- Retry counts
- Error messages
- Result types (code, research, file ops)
- Summary statistics

### Check Ollama
```bash
curl http://localhost:11434/api/tags
```

### View Database Directly
```python
import sqlite3
conn = sqlite3.connect('ai_intern_db.sqlite')
cur = conn.cursor()
cur.execute("SELECT * FROM plans ORDER BY created_at DESC LIMIT 1")
```

## Known Issues & Limitations

### Current Limitations
1. **No user interaction for CLARIFY verdicts**: LLM makes assumptions instead of asking user
2. **Research returns prose, not structured data**: Hard to parse for downstream tasks
3. **Decomposition can be non-deterministic**: Sometimes combines steps that should be separate
4. **No parallel execution**: Tasks run sequentially (fine for learning, could optimize later)
5. **No task dependencies**: Assumes sequential order, no DAG structure

### Validation Edge Cases
- **Very long code**: Timeout after 5 seconds
- **Server code**: Requires port cleanup (uses psutil to kill process tree)
- **Hardcoded test values**: Fixed for context-driven code, but can still occur

## Next Enhancements (Not Yet Implemented)

- **User interaction**: Ask user for clarification instead of LLM guessing
- **Structured research output**: Return JSON instead of prose
- **Parallel execution**: Independent tasks could run concurrently
- **Task DAG**: Track dependencies explicitly
- **Streaming output**: Show progress in real-time
- **Skill library**: Reusable task patterns
- **Memory/checkpointing**: Resume failed plans

## Testing Guidelines

### Before Running Tests
1. **Start Ollama** (desktop app or `ollama serve`)
2. **Verify models**: `ollama list` should show qwen2.5 models
3. **Check SQLite**: `ai_intern_db.sqlite` should exist or will be created

### Test Pattern
```python
from schemas import RequestSchema
from agents import Orchestrator

user_input = "Your task description here"
request = RequestSchema(content=user_input)

orchestrator = Orchestrator(max_retries=2)
plan = orchestrator.execute_request(request)

# Check results
print(f"Status: {plan.status}")
print(f"Tokens: {plan.token_usage}")

# Inspect outputs
import os
files = os.listdir('outputs')
```

### Good Test Cases
- **Simple**: "Write a fibonacci function" (1 task, EXECUTE verdict)
- **Multi-agent**: "Research X and write Y, save as Z" (3 tasks, context passing)
- **Ambiguous**: "Make it better" (CLARIFY verdict, tests clarification)
- **Complex**: "Build a web scraper with storage" (DECOMPOSE verdict, multiple subtasks)

## File Naming Conventions

### Generated Files
Format: `YYYY-MM-DD_HH-MM-SS_filename.ext`
- Timestamp prevents overwrites
- Original filename preserved
- Extension auto-detected for .py files (checks content)

### User Data Files
Place in `user_data/`:
- CSVs: Formatted as text tables when read
- JSON: Pretty-printed
- Text: Raw content

## When Things Break

### Connection Errors
→ Start Ollama, verify with `curl http://localhost:11434/api/tags`

### Validation Failures
→ Check `inspect_results.py` for error messages
→ Look at test_code in failed task
→ Retry with clearer task description

### Wrong File Extensions
→ FileAgent should detect .py from content, but specify explicitly if needed

### Tasks Not Decomposing
→ Check classifier reasoning in console output
→ Try more explicit phrasing: "Research X, then write Y, then save Z"

### Context Not Passing
→ Verify previous task succeeded (status = 'validated' or 'complete')
→ Check TaskRouter detected correct type
→ Add debug prints in agent's execute_task method

## Critical Reminders

- **Ollama must be running** - All code execution depends on it
- **Pydantic validation is non-negotiable** - Don't bypass schemas
- **Small iterations beat big rewrites** - Test incrementally
- **Hierarchical planning is default** - Set config flag to disable
- **Context flows forward only** - Tasks can't access future results
- **Tokens are tracked everywhere** - Report total usage in plan
- **Files go to outputs/** - Never write to user_data/

## Learning Objectives

This project teaches:
1. **Agent orchestration** - How to coordinate multiple AI agents
2. **Task decomposition** - Breaking complex work into executable steps
3. **Context management** - Passing data between agents
4. **Retry logic** - Handling failures gracefully
5. **Validation strategies** - Testing generated code automatically
6. **Prompt engineering** - Crafting effective LLM instructions
7. **System architecture** - Building maintainable AI systems

The goal is NOT to build the fastest or most feature-rich system. The goal is to understand how multi-agent systems work at a fundamental level.