# AI Intern - Multi-Agent Task Execution System

A multi-agent orchestration system using local Ollama models with hierarchical task decomposition. Built from scratch to learn agent architecture patterns.

## Architecture

```
User Request → Preprocessing → Planning → Execution → Validation → Final Answer
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
              TaskClassifier  Decompose    Clarify
                    │
         ┌─────────┼──────────┬──────────┐
         ▼         ▼          ▼          ▼
     FileAgent  CodingAgent  ResearchAgent  AnalysisAgent
         │         │          │              │
         ▼         ▼          ▼              ▼
     Read/Write  Generate   Web Search    Data Analysis
     user_data/  Python     + Synthesis   + Execution
                 Code
```

### Agent Pipeline
1. **Preprocessing**: Expands abbreviations, resolves file references
2. **Planning**: Classifies task (simple/complex × clear/ambiguous), decomposes if needed
3. **Routing**: Scores task goal to pick the right agent (file, code, research, analysis)
4. **Execution**: Agent generates output (reads files, writes code, searches web, runs analysis)
5. **Validation**: Syntax check → import check (AST) → safety scan → test generation → test execution
6. **Result Synthesis**: Aggregates task results into a final answer

## Setup

### Prerequisites
- Python 3.10+
- Ollama running locally

### Install
```bash
# Install Ollama (https://ollama.ai)
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5-coder:7b-instruct

# Install dependencies
pip install -e .
```

### Verify
```bash
python -m ai_intern --health
```

## Usage

```bash
# Single query
python -m ai_intern "Write a function to calculate fibonacci"

# Interactive mode
python -m ai_intern --interactive

# Inspect latest plan
python -m ai_intern --inspect

# List available data files
python -m ai_intern --list-files

# Verbose/quiet
python -m ai_intern -v "Read Workouts.csv"
python -m ai_intern -q "Write hello world"
```

## Configuration

All settings can be overridden via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_INTERN_PLANNING_MODEL` | `qwen2.5:7b-instruct` | Planning/classification model |
| `AI_INTERN_CODING_MODEL` | `qwen2.5-coder:7b-instruct` | Code generation model |
| `AI_INTERN_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `AI_INTERN_MAX_RETRIES` | `2` | Max retry attempts per task |
| `AI_INTERN_CODE_EXEC_TIMEOUT` | `15` | Code execution timeout (seconds) |
| `AI_INTERN_LOG_LEVEL` | `INFO` | Logging level |

See `.env.example` for all options.

## Project Structure

```
ai_intern/
├── __init__.py              # Package exports
├── __main__.py              # CLI entry point
├── config.py                # Pydantic Settings configuration
├── schemas.py               # Data models (Task, Plan, Request, TaskOutput)
├── llm.py                   # Ollama API wrappers
├── storage.py               # SQLite persistence
├── logging_config.py        # Logging setup
├── preprocessing.py         # Request normalization
├── inspect_results.py       # Plan inspection tool
├── planning/
│   ├── classifier.py        # 2x2 complexity/ambiguity classification
│   ├── hierarchical.py      # Recursive task decomposition
│   └── planner.py           # Flat planning (fallback)
├── execution/
│   ├── coding.py            # Python code generation
│   ├── research.py          # Web search + synthesis
│   ├── file.py              # File I/O (user_data ↔ outputs)
│   └── analysis.py          # Data analysis (generate + execute code)
├── orchestration/
│   ├── orchestrator.py      # Main pipeline + retry + dependency tracking
│   └── routing.py           # Task type scoring + agent dispatch
└── validation/
    └── validator.py         # Syntax, imports (AST), safety, tests

tests/                       # Unit tests (76 tests)
user_data/                   # Input files (CSV, JSON, etc.)
outputs/                     # Generated output files
```

## Example Queries

```
"What files do I have?"
"Read Workouts.csv and tell me my average calories per workout"
"Write a Python function that converts Fahrenheit to Celsius"
"Research the benefits of creatine supplementation"
"Build a simple Flask app that returns the current time"
"avg cals from workouts"
```

## Testing

```bash
python -m pytest tests/ -v
```

## Key Design Decisions

- **No frameworks**: Built from scratch (no LangChain/AutoGPT) to learn the patterns
- **Small local models**: Runs on 7B parameter models via Ollama - forces good architecture
- **Synchronous execution**: No async/await complexity
- **Pydantic schemas**: Strict validation on all data structures
- **AST-based security**: Import validation without executing code
- **Token tracking**: Every LLM call returns usage counts
