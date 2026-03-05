from ..schemas import PlanSchema, TaskSchema, CritiqueResult, CritiqueIssue, OutputContract
from ..llm import call_ollama_structured


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
            print(f"   Skipping critique (only {len(plan.tasks)} task(s))")
            return CritiqueResult(
                approved=True,
                issues=[],
                revised_goals={},
                critique_reasoning="Plan has fewer than 3 tasks - critique skipped."
            ), 0

        print(f"\nCritiqueAgent reviewing {len(plan.tasks)}-task plan...")

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
            print(f"   Plan rejected: {len(blocking)} blocking issue(s)")
        else:
            result.approved = True
            print(f"   Plan approved ({len(result.issues)} warning(s))")

        for issue in result.issues:
            marker = "BLOCKING" if issue.severity == "blocking" else "WARNING"
            print(f"   {marker} Task {issue.task_order}: [{issue.issue_type}] {issue.description[:80]}")

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

            # Only check OutputContract instances (not legacy spec dicts)
            if isinstance(task.output_contract, OutputContract):
                contract_issue = cls._check_contract_vs_agent(task, detected_type)
                if contract_issue:
                    issues.append(contract_issue)

                # Check context dependency format mismatches
                if task.task_order > 0:
                    prev_tasks = [t for t in plan.tasks if t.task_order < task.task_order]
                    for prev in prev_tasks:
                        if (isinstance(prev.output_contract, OutputContract) and
                                task.task_order in prev.output_contract.required_by_tasks):
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
        if not isinstance(producer.output_contract, OutputContract) or not isinstance(consumer.output_contract, OutputContract):
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
            if isinstance(task.output_contract, OutputContract):
                contract_info = (
                    f"\n     Contract: {task.output_contract.output_type} -- "
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
   Example: Task 2 calls calculate(protein_g=X) but Task 1 returns prose --
   the value is not directly extractable.

2. MISSING TASKS: Is there an implicit step the planner skipped?
   Example: "Read CSV -> Save summary" with no analysis task between them.

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

        print(f"\n   Applying {len(critique.revised_goals)} revision(s)...")

        for task_order, new_goal in critique.revised_goals.items():
            for task in plan.tasks:
                if task.task_order == task_order:
                    print(f"   Task {task_order} revised:")
                    print(f"     Before: {task.goal[:70]}...")
                    print(f"     After:  {new_goal[:70]}...")
                    task.goal = new_goal

        return plan
