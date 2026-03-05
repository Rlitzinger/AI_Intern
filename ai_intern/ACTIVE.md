# ACTIVE TASK: Add Critique Pass + Output Contracts to Planning Layer

## Overview
Upgrade the planning system with two additions:
1. **Output Contracts** — each task declares what it produces and who consumes it
2. **CritiqueAgent** — reviews plans before execution, catches failures before they happen

This runs between planning and execution: `Plan → Critique → (revise if needed) → Execute`

## Files to Modify
- `schemas.py` — add 3 new models, update TaskSchema
- `agents/planning/critic.py` — create new file
- `agents/planning/hierarchical.py` — add contract generation
- `agents/planning/__init__.py` — export CritiqueAgent
- `agents/orchestration/orchestrator.py` — add Stage 1.5 critique pass

---

## Step 1: Update schemas.py

Add these three models to `schemas.py`:

```python
from typing import Literal

class OutputContract(BaseModel):
    """
    Specifies what a task produces and who consumes it.
    Generated during planning, used by CritiqueAgent to detect
    context dependency failures before execution.
    """
    output_type: Literal["python_code", "prose", "structured_data", "file_path", "none"]
    output_format: str  # e.g. "dict with keys: protein_g, fat_g as floats"
    required_by_tasks: list[int] = []  # task_order values that consume this output


class CritiqueIssue(BaseModel):
    """A single issue found during plan critique."""
    severity: Literal["blocking", "warning"]
    issue_type: Literal[
        "context_mismatch",   # Task expects format previous task won't produce
        "missing_task",       # Implicit step not in plan
        "wrong_agent",        # TaskRouter will misroute this task
        "over_decomposed",    # Tasks that could/should be merged
        "bad_contract",       # Output contract is vague or incorrect
    ]
    task_order: int           # Which task has the issue (-1 = plan-level)
    description: str
    suggested_fix: str


class CritiqueResult(BaseModel):
    """Full critique of a plan."""
    approved: bool
    issues: list[CritiqueIssue] = []
    revised_goals: dict[int, str] = {}  # task_order → new goal string
    critique_reasoning: str
```

Also add this field to the existing `TaskSchema`:
```python
output_contract: Optional[OutputContract] = None
```

---

## Step 2: Create agents/planning/critic.py

Create this file in full:

```python
from schemas import PlanSchema, TaskSchema, CritiqueResult, CritiqueIssue, OutputContract
from llm import call_ollama_structured


class CritiqueAgent:
    """
    Critiques a generated plan before execution.

    Checks for:
    - Context dependency failures (task expects format previous task won't produce)
    - Missing implicit steps
    - TaskRouter misroutes (wrong agent will be assigned)
    - Over-decomposition (redundant tasks)
    - Vague output contracts

    Only runs on plans with 3+ tasks (simpler plans skip critique).
    Max 1 revision cycle to prevent infinite loops.
    """

    model = "qwen2.5:7b-instruct"
    temperature = 0.2

    system_prompt = """You are a plan critique expert for a multi-agent AI system.

You review task plans before execution and identify failures that will occur.
Be specific, technical, and actionable. Focus on real failure modes, not theoretical ones.

The system has these agents:
- CodingAgent: generates Python code. Keywords: write, function, script, implement, build, create
- ResearchAgent: web search + synthesis. Keywords: research, find, investigate, search, explore
- FileAgent: reads user_data/ and writes outputs/. Keywords: read file, save, write to outputs
- ValidationAgent: runs after every task automatically (not in task list)

TaskRouter uses keyword matching - it is not smart. If a task goal is ambiguous it will default to CodingAgent."""

    # Mirrors routing.py keyword lists exactly
    ROUTING_RULES = {
        "file": ["read file", "load file", "read csv", "list files", "write output",
                 "save to file", "save to outputs", "save the", "save results",
                 "write to outputs", "write the results", "write to file", "output to file"],
        "code": ["write", "code", "function", "script", "implement", "create", "build", "develop"],
        "research": ["research", "find", "investigate", "search", "explore", "identify"],
    }

    @classmethod
    def critique_plan(cls, plan: PlanSchema) -> tuple[CritiqueResult, int]:
        """
        Critique a plan and return issues + optional revisions.

        Returns:
            tuple: (CritiqueResult, tokens_used)
        """
        if len(plan.tasks) < 3:
            print(f"   ⏭️  Skipping critique (only {len(plan.tasks)} task(s))")
            return CritiqueResult(
                approved=True,
                issues=[],
                revised_goals={},
                critique_reasoning="Plan has fewer than 3 tasks - critique skipped."
            ), 0

        print(f"\n🔎 CritiqueAgent reviewing {len(plan.tasks)}-task plan...")

        # Run deterministic checks first (zero token cost)
        deterministic_issues = cls._run_deterministic_checks(plan)

        # LLM critique pass
        prompt = cls._build_prompt(plan, deterministic_issues)

        result, tokens = call_ollama_structured(
            model=cls.model,
            prompt=prompt,
            system=cls.system_prompt,
            response_schema=CritiqueResult,
            temperature=cls.temperature
        )

        # Merge deterministic issues with LLM issues
        result.issues = deterministic_issues + result.issues

        # Any blocking issue = not approved
        blocking = [i for i in result.issues if i.severity == "blocking"]
        if blocking:
            result.approved = False
            print(f"   ❌ Plan rejected: {len(blocking)} blocking issue(s)")
        else:
            result.approved = True
            print(f"   ✅ Plan approved ({len(result.issues)} warning(s))")

        for issue in result.issues:
            emoji = "❌" if issue.severity == "blocking" else "⚠️"
            print(f"   {emoji} Task {issue.task_order}: [{issue.issue_type}] {issue.description[:80]}")

        return result, tokens

    @classmethod
    def _run_deterministic_checks(cls, plan: PlanSchema) -> list[CritiqueIssue]:
        """
        Checks that don't need an LLM.
        Simulates routing and checks contract vs agent type.
        Zero token cost.
        """
        issues = []

        for task in plan.tasks:
            goal_lower = task.goal.lower()
            detected_type = cls._simulate_routing(goal_lower)

            # Check contract vs what that agent actually produces
            if task.output_contract:
                contract_issue = cls._check_contract_vs_agent(task, detected_type)
                if contract_issue:
                    issues.append(contract_issue)

            # Check context dependency format mismatches
            if task.task_order > 0 and task.output_contract:
                prev_tasks = [t for t in plan.tasks if t.task_order < task.task_order]
                for prev in prev_tasks:
                    if prev.output_contract and task.task_order in prev.output_contract.required_by_tasks:
                        mismatch = cls._check_context_mismatch(prev, task)
                        if mismatch:
                            issues.append(mismatch)

        return issues

    @classmethod
    def _simulate_routing(cls, goal_lower: str) -> str:
        """Simulate TaskRouter.classify_task without importing it."""
        for task_type, keywords in cls.ROUTING_RULES.items():
            if any(kw in goal_lower for kw in keywords):
                return task_type
        return "unknown"

    @classmethod
    def _check_contract_vs_agent(cls, task: TaskSchema, agent_type: str) -> CritiqueIssue | None:
        """Check if the output contract makes sense for the agent that will run this task."""
        contract = task.output_contract

        # ResearchAgent always produces prose
        if agent_type == "research" and contract.output_type not in ["prose", "none"]:
            return CritiqueIssue(
                severity="blocking",
                issue_type="bad_contract",
                task_order=task.task_order,
                description=f"Contract declares output_type='{contract.output_type}' but ResearchAgent always returns prose.",
                suggested_fix="Set output_type='prose' or add a downstream task to parse research output into structured data."
            )

        # FileAgent read tasks produce formatted text, not structured_data
        if agent_type == "file" and "read" in task.goal.lower() and contract.output_type == "structured_data":
            return CritiqueIssue(
                severity="warning",
                issue_type="bad_contract",
                task_order=task.task_order,
                description="FileAgent read tasks return formatted text, not structured data.",
                suggested_fix="Set output_type='prose' and add a processing task to extract values."
            )

        return None

    @classmethod
    def _check_context_mismatch(cls, producer: TaskSchema, consumer: TaskSchema) -> CritiqueIssue | None:
        """Check if what producer outputs matches what consumer needs."""
        if not producer.output_contract or not consumer.output_contract:
            return None

        # Consumer does computation but producer outputs prose
        if (producer.output_contract.output_type == "prose" and
                any(kw in consumer.goal.lower() for kw in
                    ["calculate", "compute", "average", "total", "sum"])):
            return CritiqueIssue(
                severity="warning",
                issue_type="context_mismatch",
                task_order=consumer.task_order,
                description=f"Task {consumer.task_order} does computation but Task {producer.task_order} produces prose. CodingAgent will need to parse unstructured text.",
                suggested_fix="Add a parsing task between them, or explicitly instruct CodingAgent to extract values from prose context."
            )

        return None

    @classmethod
    def _build_prompt(cls, plan: PlanSchema, existing_issues: list[CritiqueIssue]) -> str:
        """Build the LLM critique prompt."""
        task_summary = []
        for task in plan.tasks:
            contract_info = ""
            if task.output_contract:
                contract_info = (
                    f"\n     Contract: {task.output_contract.output_type} — "
                    f"{task.output_contract.output_format}"
                    f"\n     Consumed by tasks: {task.output_contract.required_by_tasks}"
                )
            task_summary.append(f"  Task {task.task_order}: {task.goal}{contract_info}")

        tasks_text = "\n".join(task_summary)

        existing_text = ""
        if existing_issues:
            existing_text = "\n\nDETERMINISTIC CHECKS ALREADY FOUND:\n"
            for issue in existing_issues:
                existing_text += f"  - Task {issue.task_order}: [{issue.issue_type}] {issue.description}\n"

        return f"""Review this execution plan for a multi-agent AI system.

PLAN ({len(plan.tasks)} tasks):
{tasks_text}
{existing_text}

Analyze for these failure modes:

1. CONTEXT MISMATCH: Does any task assume a specific data format from a previous task
   that won't actually be produced?
   Example: Task 2 calls calculate(protein_g=X) but Task 1 returns prose —
   the value is not directly extractable.

2. MISSING TASKS: Is there an implicit step the planner skipped?
   Example: "Read CSV → Save summary" with no analysis task between them.

3. WRONG AGENT: Will TaskRouter misroute any task?
   TaskRouter uses keyword matching. "Analyze the data" has no file/code/research
   keywords and defaults to CodingAgent. Is that correct?

4. OVER-DECOMPOSED: Are any tasks redundant or unnecessarily split?
   Example: "Save research results" + "Write results to file" are the same task.

5. BAD CONTRACT: Is any output contract vague, wrong, or inconsistent with what
   the assigned agent will actually produce?

For each issue: specify severity (blocking/warning), task_order, issue_type, description, suggested_fix.
If the plan is correct, set approved=true with empty issues list.
Only flag real problems, not theoretical ones.

Return JSON matching CritiqueResult schema."""

    @classmethod
    def apply_revisions(cls, plan: PlanSchema, critique: CritiqueResult) -> PlanSchema:
        """
        Apply critic's suggested goal rewrites to the plan.
        Only rewrites goals - does not add/remove tasks.
        Structural changes (missing tasks) require replanning.
        """
        if not critique.revised_goals:
            return plan

        print(f"\n   📝 Applying {len(critique.revised_goals)} revision(s)...")

        for task_order, new_goal in critique.revised_goals.items():
            for task in plan.tasks:
                if task.task_order == task_order:
                    print(f"   Task {task_order} revised:")
                    print(f"     Before: {task.goal[:70]}...")
                    print(f"     After:  {new_goal[:70]}...")
                    task.goal = new_goal

        return plan
```

---

## Step 3: Add contract generation to hierarchical.py

### 3a. Add ContractResult model at the top of hierarchical.py (alongside DecompositionResult):

```python
class ContractResult(BaseModel):
    """LLM response for output contract generation."""
    output_type: str  # validated against allowed literals after
    output_format: str
    required_by_tasks: list[int]
```

### 3b. Add _generate_contracts as a classmethod on HierarchicalPlanner:

```python
@classmethod
def _generate_contracts(cls, tasks: list[TaskSchema]) -> tuple[list[TaskSchema], int]:
    """
    Generate output contracts for a list of tasks.
    Called after decomposition, before returning leaf tasks.
    Only runs when there are 2+ tasks (single tasks don't need contracts).

    Returns:
        tuple: (tasks_with_contracts, tokens_used)
    """
    if len(tasks) < 2:
        return tasks, 0

    task_descriptions = "\n".join([
        f"Task {t.task_order}: {t.goal}" for t in tasks
    ])

    total_tokens = 0
    valid_types = ["python_code", "prose", "structured_data", "file_path", "none"]

    for task in tasks:
        prompt = f"""All tasks in this plan:
{task_descriptions}

For Task {task.task_order}: "{task.goal}"

What does this task produce?

output_type options:
- "python_code": task generates Python functions/scripts
- "prose": task generates text (research summaries, analysis, descriptions)
- "structured_data": task generates parseable data (JSON, CSV, key-value pairs)
- "file_path": task writes a file and returns the path string
- "none": task has no meaningful output for downstream tasks

output_format: Describe specifically what the output looks like.
  python_code example: "function calculate_macros(grams: float) -> dict"
  prose example: "paragraph summary with cited sources"
  structured_data example: "dict with keys: protein_g, fat_g, carbs_g as floats"
  file_path example: "path string like outputs/2024-01-01_summary.txt"

required_by_tasks: Which task numbers (by task_order) consume this output?
  Look at the other tasks - which ones depend on or reference this task's result?

Return JSON with output_type, output_format, required_by_tasks."""

        result, tokens = call_ollama_structured(
            model=cls.model,
            prompt=prompt,
            system="You are a software architect. Specify exact data contracts between system components. Be precise about types and formats.",
            response_schema=ContractResult,
            temperature=0.1
        )
        total_tokens += tokens

        output_type = result.output_type if result.output_type in valid_types else "prose"

        task.output_contract = OutputContract(
            output_type=output_type,
            output_format=result.output_format,
            required_by_tasks=result.required_by_tasks
        )

        print(f"      📋 Task {task.task_order} contract: {output_type} — {result.output_format[:60]}...")

    return tasks, total_tokens
```

### 3c. Call _generate_contracts at the end of create_plan, before returning:

Find the end of `create_plan` where leaf_tasks are assembled, and add:

```python
# Generate output contracts for multi-task plans
if len(leaf_tasks) >= 2:
    print(f"\n📋 Generating output contracts...")
    leaf_tasks, contract_tokens = cls._generate_contracts(leaf_tasks)
    total_tokens += contract_tokens

# ... then continue with building the plan as before
```

### 3d. Add OutputContract import to hierarchical.py:

```python
from schemas import TaskSchema, PlanSchema, RequestSchema, OutputContract
```

---

## Step 4: Update agents/planning/__init__.py

Add CritiqueAgent to exports:

```python
from .planner import PlanningAgent
from .classifier import TaskClassifier, TaskVerdict
from .hierarchical import HierarchicalPlanner
from .critic import CritiqueAgent

__all__ = ['PlanningAgent', 'TaskClassifier', 'TaskVerdict', 'HierarchicalPlanner', 'CritiqueAgent']
```

---

## Step 5: Update orchestrator.py

### 5a. Add import at top:

```python
from ..planning.critic import CritiqueAgent
```

### 5b. Replace Stage 1 in execute_request with this expanded version:

```python
# === STAGE 1: PLANNING ===
print(f"\n📋 Stage 1: Planning")

if USE_HIERARCHICAL_PLANNING:
    print(f"   Using: HierarchicalPlanner (with classifier)")
    plan = HierarchicalPlanner.create_plan(request)
else:
    print(f"   Using: PlanningAgent (flat planning)")
    plan = PlanningAgent.create_plan(request)

print(f"   Generated {len(plan.tasks)} task(s)")
save_plan_to_sqlite(plan)
print(f"   💾 Saved plan after planning")

# === STAGE 1.5: CRITIQUE ===
print(f"\n🔎 Stage 1.5: Plan Critique")

critique, critique_tokens = CritiqueAgent.critique_plan(plan)
plan.token_usage += critique_tokens

if not critique.approved:
    print(f"\n   🔄 Blocking issues found, attempting revision...")

    # Apply goal rewrites first (cheap fix)
    if critique.revised_goals:
        plan = CritiqueAgent.apply_revisions(plan, critique)

    # If missing tasks detected, full replan with critique context
    missing_task_issues = [
        i for i in critique.issues
        if i.issue_type == "missing_task" and i.severity == "blocking"
    ]

    if missing_task_issues:
        print(f"   🔁 Missing tasks detected - replanning with critique context...")
        plan = self._replan_with_critique(request, critique)
        plan.token_usage += critique_tokens  # approximate

    save_plan_to_sqlite(plan)
    print(f"   💾 Saved revised plan")

else:
    # Apply any non-blocking goal rewrites
    if critique.revised_goals:
        plan = CritiqueAgent.apply_revisions(plan, critique)
        save_plan_to_sqlite(plan)
```

### 5c. Add _replan_with_critique as a method on Orchestrator:

```python
def _replan_with_critique(self, request: RequestSchema, critique: CritiqueResult) -> PlanSchema:
    """
    Replan incorporating critique feedback.
    Injects blocking issues as context into the planner.
    Only called when missing_task blocking issues exist.
    """
    issues_text = "\n".join([
        f"- {i.description} → Fix: {i.suggested_fix}"
        for i in critique.issues
        if i.severity == "blocking"
    ])

    augmented_content = f"""{request.content}

PLANNING NOTES (from previous attempt - these issues must be fixed):
{issues_text}"""

    augmented_request = RequestSchema(
        request_id=request.request_id,
        content=augmented_content
    )

    if USE_HIERARCHICAL_PLANNING:
        return HierarchicalPlanner.create_plan(augmented_request)
    else:
        return PlanningAgent.create_plan(augmented_request)
```

### 5d. Add CritiqueResult import to orchestrator.py:

```python
from schemas import RequestSchema, PlanSchema, TaskSchema, CritiqueResult
```

---

## Success Criteria

- [ ] Plans with 3+ tasks show "Stage 1.5: Plan Critique" in output
- [ ] Plans with < 3 tasks skip critique cleanly
- [ ] Output contracts appear on each task after planning
- [ ] Blocking issues prevent execution and trigger revision
- [ ] Warnings log but don't block execution
- [ ] Replan path fires when missing_task blocking issues found
- [ ] Token usage includes critique tokens in plan.token_usage
- [ ] Existing simple tests (fibonacci, single task) still pass unchanged

## Test After Implementation

```python
# Test 1: Simple plan (critique skipped)
request = RequestSchema(content="Write a fibonacci function")
plan = orchestrator.execute_request(request)
# Should show: "Skipping critique (only 1 task)"

# Test 2: Multi-task plan (critique runs)
request = RequestSchema(content="Research chicken thigh macros and write a calculator, save as calc.py")
plan = orchestrator.execute_request(request)
# Should show: Stage 1.5 output with contracts and critique result
# Check: plan.tasks[0].output_contract is not None
```

## Constraints
- DO NOT change TaskRouter keyword lists
- DO NOT change existing agent execute_task signatures
- DO NOT change validation logic
- DO NOT add async
- Keep all existing tests passing
- OutputContract field on TaskSchema must be Optional (backward compatible with existing DB rows)

## Notes

**Why two-pass critique (deterministic + LLM)?**
Deterministic checks catch routing mismatches and contract/agent type conflicts instantly
with zero token cost. The LLM pass catches semantic issues (missing steps, context mismatches)
that require understanding the task goals. Running deterministic first means the LLM prompt
already has known issues listed, so it focuses on finding new ones rather than re-finding obvious ones.

**Why only replan on missing_task blocking issues?**
Goal rewrites are cheap - apply in place. Adding/removing tasks requires the planner
to regenerate the whole structure. Keeping these paths separate prevents over-engineering
the revision loop.

**Token cost estimate:**
Contract generation: ~1 LLM call per task on 3+ task plans = 3-4 extra calls
Critique pass: 1 LLM call per plan
Replan (rare): 1 full planning pass
Total overhead on a typical 3-task plan: ~4-5 extra LLM calls, ~60-90 seconds on your hardware.