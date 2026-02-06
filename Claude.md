# Project AI Intern

Local LLM agentic system using Ollama (qwen2.5:7b-instruct, qwen2.5-coder:7b-instruct), FastAPI, JS/HTML/CSS, Python.

This is the minimal vertical slice needed to run a multi agent system. A simple UI is used to provide necessary information on the status and history of the system.

## Project Architecture
- `project/main.py` - entry point, where the orchestration loop is
- `project/agents.py` - where all gent implementations are
- `project/schemas.py` - where the pydantic models for data structures are
- `project/llm.py` - the wrapper for ollama calls
- `project/storage.py` - the sqlite interations
- `project/config.py` - model names, temperatures, and paths

## Preferences
- small and iterative development is required to help user learn the system

## Model Assignments
- Planning, Critique, Validation: qwen2.5:7b-instruct
- Code Generation: qwen2.5-coder:7b-instruct
- All calls go through llm.py wrapper for logging/tracing

## Goal: Complete the Minimum vertical slice
- the structure will have 5 distinct parts:
    - User Input
        - allow user to input prompts into the system
    - Planning
        - takes user prompts as input
        - defines the scope of the project
        - breaks project into a defined plan with sub tasks
    - Execution
        - executes on the tasks in the plan using appropriate agents
    - Validating
        - run validation on executed tasks
        - validate result is as intended
    - Return
        - return a valid result to the user
- the user has an intuitive UI that allows them to trace activity through the system, see run histories, and observe what is currentl happening.
- Defined schemas are the only data type being transferred between steps. They are required, validated, and enforced.

## Do Not
- Add async/await unless explicitly requested
- Create abstract base classes or factory patterns
- Implement features beyond the current vertical slice
- Use LangChain, LangGraph, or similar frameworks
- Add error handling beyond basic try/except until core flow works

## Today's Tasks (non-exhaustive, not comprehensive):
- [X] Create the claude.md file
- [X] Create a daily devlog file
- [X] Create the project structure
- [] Design the data schemas

## Current State
- [ ] Schemas defined
- [ ] LLM wrapper working
- [ ] Planning agent implemented
- [ ] Executor implemented (stub)
- [ ] Coding agent implemented
- [ ] Validation implemented (stub)
- [ ] SQLite storage working
- [ ] End-to-end flow tested