# AI Intern Architecture

## System Overview
```mermaid
flowchart TD
    User[User Request] --> Orch[Orchestrator]
    Orch --> Plan[PlanningAgent]
    Plan --> Tasks[TaskSchema List]
    Tasks --> Loop{For Each Task}
    
    Loop --> Exec[CodingAgent]
    Exec --> Val[ValidationAgent]
    
    Val --> Pass{Tests Pass?}
    Pass -->|Yes| Done[✅ Task Validated]
    Pass -->|No| Retry{Retries Left?}
    
    Retry -->|Yes| Feedback[Add Error Feedback]
    Feedback --> Exec
    Retry -->|No| Fail[❌ Task Failed]
    
    Done --> Next{More Tasks?}
    Fail --> Next
    Next -->|Yes| Loop
    Next -->|No| Save[💾 Save to SQLite]
    
    Save --> Complete[Plan Complete]
```

## Data Flow
```mermaid
graph LR
    Request[RequestSchema] --> Plan[PlanSchema]
    Plan --> Task1[TaskSchema]
    Plan --> Task2[TaskSchema]
    Plan --> TaskN[TaskSchema...]
    
    Task1 --> Code[result: str]
    Task1 --> Tests[test_code: str]
    Task1 --> Status[status: validated/failed]
    
    Plan --> DB[(SQLite)]
    Task1 --> DB
```

## Component Responsibilities
```mermaid
graph TD
    subgraph Orchestration
        Orch[Orchestrator<br/>Pipeline + Retry Logic]
    end
    
    subgraph Agents
        Plan[PlanningAgent<br/>qwen2.5:7b-instruct<br/>Breaks down requests]
        Code[CodingAgent<br/>qwen2.5-coder:7b-instruct<br/>Generates Python code]
        Val[ValidationAgent<br/>qwen2.5:7b-instruct<br/>Generates & runs tests]
    end
    
    subgraph Infrastructure
        LLM[llm.py<br/>Ollama wrapper + tokens]
        Schema[schemas.py<br/>Pydantic validation]
        Store[storage.py<br/>SQLite persistence]
    end
    
    Orch --> Plan
    Orch --> Code
    Orch --> Val
    
    Plan --> LLM
    Code --> LLM
    Val --> LLM
    
    Plan --> Schema
    Code --> Schema
    Val --> Schema
    
    Orch --> Store
```

## Retry Flow Detail
```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant Code as CodingAgent
    participant Val as ValidationAgent
    participant Task as TaskSchema
    
    Orch->>Code: execute_task(task)
    Code->>Code: Generate code
    Code->>Task: Update result
    Code-->>Orch: (task, tokens)
    
    Orch->>Val: validate_task(task)
    Val->>Val: Syntax check ✓
    Val->>Val: Import check ✓
    Val->>Val: Generate tests
    Val->>Val: Run tests
    
    alt Tests Pass
        Val->>Task: status = "validated"
        Val-->>Orch: (task, tokens)
    else Tests Fail
        Val->>Task: status = "failed"
        Val->>Task: error_message = "..."
        Val-->>Orch: (task, tokens)
        
        Orch->>Task: retry_count++
        
        alt Retries Left
            Orch->>Task: goal += error feedback
            Orch->>Code: execute_task(task) [RETRY]
        else Max Retries
            Orch->>Task: status = "failed"
        end
    end
```

## Token Flow
```mermaid
graph LR
    Plan[Planning<br/>~500 tokens] --> PlanTotal[plan.token_usage]
    
    Code1[CodingAgent<br/>Attempt 1<br/>~350 tokens] --> PlanTotal
    Val1[ValidationAgent<br/>Attempt 1<br/>~650 tokens] --> PlanTotal
    
    Code2[CodingAgent<br/>Attempt 2<br/>~400 tokens] --> PlanTotal
    Val2[ValidationAgent<br/>Attempt 2<br/>~700 tokens] --> PlanTotal
    
    PlanTotal --> Total[Total: ~2600 tokens]
```

## File Structure
```
project/
├── main.py              # Entry point (uses Orchestrator)
├── agents.py            # All agent classes
│   ├── PlanningAgent
│   ├── CodingAgent
│   ├── ValidationAgent
│   └── Orchestrator
├── llm.py               # Ollama wrappers
│   ├── call_ollama()
│   ├── call_ollama_code()
│   └── call_ollama_structured()
├── schemas.py           # Pydantic models
│   ├── RequestSchema
│   ├── PlanSchema
│   └── TaskSchema
├── storage.py           # SQLite operations
│   ├── save_plan_to_sqlite()
│   └── load_plan_from_sqlite()
├── inspect_results.py   # Debug tool
└── ai_intern_db.sqlite  # Database
```

## Status State Machine
```mermaid
stateDiagram-v2
    [*] --> pending: Task created
    
    pending --> executing: CodingAgent starts
    executing --> complete: Code generated
    
    complete --> validating: ValidationAgent starts
    
    validating --> validated: Tests pass ✅
    validating --> failed: Tests fail (no retries left) ❌
    validating --> executing: Tests fail (retry with feedback) 🔄
    
    validated --> [*]
    failed --> [*]
```

## Future: Task Router (Not Yet Built)
```mermaid
flowchart TD
    Task[TaskSchema] --> Router{TaskRouter}
    
    Router -->|"write, code, function"| Code[CodingAgent]
    Router -->|"research, find, compare"| Research[ResearchAgent<br/>⚠️ Not Yet Implemented]
    Router -->|"analyze, evaluate"| Analysis[AnalysisAgent<br/>⚠️ Not Yet Implemented]
    Router -->|unknown| Fail[❌ Unsupported Task Type]
```