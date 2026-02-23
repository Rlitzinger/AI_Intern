from schemas import TaskSchema, PlanSchema, RequestSchema
from .classifier import TaskClassifier, TaskVerdict
from llm import call_ollama_structured
from pydantic import BaseModel


class DecompositionResult(BaseModel):
    """LLM response schema for task decomposition."""
    subtasks: list[str]  # List of subtask goal strings


class ClarificationResult(BaseModel):
    """LLM response schema for task clarification."""
    clarified_goal: str
    extracted_details: str  # What was clarified/made explicit


class HierarchicalPlanner:
    """
    Multi-level task decomposition with ambiguity handling.

    Uses TaskClassifier to determine when to:
    - Execute immediately (simple + clear)
    - Decompose into subtasks (complex + clear)
    - Clarify first (simple + ambiguous)
    - Clarify then decompose (complex + ambiguous)

    Key principle: CLARIFY before DECOMPOSE for ambiguous tasks.
    """

    model = "qwen2.5:7b-instruct"
    temperature = 0.3
    max_depth = 3  # Prevent infinite recursion

    @classmethod
    def create_plan(cls, request: RequestSchema) -> PlanSchema:
        """
        Create a hierarchical plan from a request.

        Returns a flat PlanSchema with all leaf tasks (DFS order).
        """
        print(f"\n🌳 HierarchicalPlanner processing request: {request.request_id}")
        print(f"📝 Content: {request.content[:100]}...")

        # Create root task from request
        root_task = TaskSchema(
            plan_id="placeholder",  # Will be set later
            task_order=0,
            goal=request.content
        )

        # Recursively decompose
        leaf_tasks, total_tokens = cls._decompose_recursive(
            task=root_task,
            depth=0,
            context={'parent_goal': None}
        )

        # Build plan from leaf tasks
        plan = PlanSchema(
            request_id=request.request_id,
            tasks=[],
            token_usage=total_tokens
        )

        # Set correct plan_id and task_order
        for i, task in enumerate(leaf_tasks):
            task.plan_id = plan.plan_id
            task.task_order = i
            plan.tasks.append(task)

        print(f"✅ Hierarchical plan created: {len(plan.tasks)} executable tasks")
        print(f"🎫 Tokens consumed: {total_tokens}")

        return plan

    @classmethod
    def _decompose_recursive(
        cls,
        task: TaskSchema,
        depth: int,
        context: dict
    ) -> tuple[list[TaskSchema], int]:
        """
        Recursively decompose task until all subtasks are executable.

        Returns:
            tuple: (list of executable leaf tasks, total tokens used)
        """
        indent = "  " * depth
        print(f"{indent}🔍 Classifying: {task.goal[:60]}...")

        # Check max depth
        if depth >= cls.max_depth:
            print(f"{indent}⚠️  Max depth reached, treating as executable")
            return [task], 0

        # Classify the task
        verdict, reasoning, tokens = TaskClassifier.classify(task, context)
        total_tokens = tokens

        print(f"{indent}📊 Verdict: {verdict.value}")
        print(f"{indent}💭 Reasoning: {reasoning[:80]}...")

        # Handle based on verdict
        if verdict == TaskVerdict.EXECUTE:
            # Base case: task is executable
            print(f"{indent}✅ Executable task (leaf node)")
            return [task], total_tokens

        elif verdict == TaskVerdict.DECOMPOSE:
            # Decompose into subtasks
            print(f"{indent}🔀 Decomposing into subtasks...")
            subtasks, decomp_tokens = cls._decompose_task(task, context)
            total_tokens += decomp_tokens

            # Recursively decompose each subtask
            all_leaf_tasks = []
            for i, subtask in enumerate(subtasks):
                print(f"{indent}  ├─ Subtask {i}: {subtask.goal[:50]}...")

                # Create context for subtask
                subtask_context = {
                    'parent_goal': task.goal,
                    'sibling_count': len(subtasks),
                    'sibling_index': i
                }

                leaf_tasks, sub_tokens = cls._decompose_recursive(
                    subtask,
                    depth + 1,
                    subtask_context
                )
                all_leaf_tasks.extend(leaf_tasks)
                total_tokens += sub_tokens

            return all_leaf_tasks, total_tokens

        elif verdict == TaskVerdict.CLARIFY:
            # Clarify then treat as executable
            print(f"{indent}❓ Clarifying ambiguous task...")
            clarified_task, clarify_tokens = cls._clarify_task(task, context)
            total_tokens += clarify_tokens

            print(f"{indent}✨ Clarified: {clarified_task.goal[:60]}...")

            # After clarification, it should be executable
            # (Don't recurse - assume clarification makes it executable)
            return [clarified_task], total_tokens

        else:  # CLARIFY_THEN_DECOMPOSE
            # Clarify first, then decompose
            print(f"{indent}❓ Clarifying before decomposition...")
            clarified_task, clarify_tokens = cls._clarify_task(task, context)
            total_tokens += clarify_tokens

            print(f"{indent}✨ Clarified: {clarified_task.goal[:60]}...")
            print(f"{indent}🔀 Now decomposing clarified task...")

            # Decompose the clarified task
            subtasks, decomp_tokens = cls._decompose_task(clarified_task, context)
            total_tokens += decomp_tokens

            # Recursively decompose each subtask
            all_leaf_tasks = []
            for i, subtask in enumerate(subtasks):
                print(f"{indent}  ├─ Subtask {i}: {subtask.goal[:50]}...")

                subtask_context = {
                    'parent_goal': clarified_task.goal,
                    'sibling_count': len(subtasks),
                    'sibling_index': i
                }

                leaf_tasks, sub_tokens = cls._decompose_recursive(
                    subtask,
                    depth + 1,
                    subtask_context
                )
                all_leaf_tasks.extend(leaf_tasks)
                total_tokens += sub_tokens

            return all_leaf_tasks, total_tokens

    @classmethod
    def _decompose_task(cls, task: TaskSchema, context: dict) -> tuple[list[TaskSchema], int]:
        """
        Decompose a complex task into 2-4 subtasks.

        Returns:
            tuple: (list of subtasks, tokens used)
        """
        prompt = f"""Task to decompose: {task.goal}

This task is too complex to execute in one step. Break it into 2-4 smaller subtasks.

CRITICAL DECOMPOSITION RULES:

1. FILE OPERATIONS ARE ALWAYS SEPARATE TASKS:
   - If goal mentions "save", "write to file", "output to", create a dedicated save task
   - Example: "Research X and save as Y" → Task 1: Research X, Task 2: Save to Y

2. RESEARCH + CODE ARE ALWAYS SEPARATE TASKS:
   - If goal mentions research AND code generation, split them
   - Example: "Research X and write code" → Task 1: Research X, Task 2: Write code using results

3. EXECUTION ORDER MATTERS:
   - Research tasks come first (gather data)
   - Code/analysis tasks come second (process data)
   - File save tasks come last (store results)

Each subtask should:
- Be independently executable by a single agent
- Have a clear, single purpose
- Reference previous tasks if needed ("using results from previous task")

GOOD DECOMPOSITION:
"Research chicken macros, write calculator, save as calc.py"
→ [
    "Research protein, fat, carb content in chicken per 100g",
    "Write Python function calculate_macros(grams) using the research data",
    "Save the calculator function to outputs/calc.py"
  ]

BAD DECOMPOSITION:
"Research chicken macros, write calculator, save as calc.py"
→ [
    "Research chicken macros",
    "Write calculator and save it"  <- Combines code + save!
  ]

Return a JSON object with a 'subtasks' array containing 2-4 goal strings.

Example:
{{
  "subtasks": [
    "Research protein and fat content in chicken thighs per 100g",
    "Write a Python function that calculates total macros given grams of chicken thighs",
    "Save the calculator function to outputs/calculator.py"
  ]
}}
"""

        result, tokens = call_ollama_structured(
            model=cls.model,
            prompt=prompt,
            system="You are a task decomposition expert. Break complex tasks into clear, actionable subtasks.",
            response_schema=DecompositionResult,
            temperature=cls.temperature
        )

        # Convert goal strings to TaskSchema objects
        subtasks = []
        for i, goal in enumerate(result.subtasks):
            subtasks.append(TaskSchema(
                plan_id="placeholder",
                task_order=i,
                goal=goal
            ))

        return subtasks, tokens

    @classmethod
    def _clarify_task(cls, task: TaskSchema, context: dict) -> tuple[TaskSchema, int]:
        """
        Clarify an ambiguous task by making implicit details explicit.

        Returns:
            tuple: (clarified task, tokens used)
        """
        # Build context info
        context_info = ""
        if context and context.get('parent_goal'):
            context_info = f"\nParent goal: {context['parent_goal']}"

        prompt = f"""Ambiguous task: {task.goal}{context_info}

This task is ambiguous - it lacks specific details needed for execution.

Your job: Make the task CLEAR and SPECIFIC by:
1. Identifying what's vague or missing
2. Making reasonable assumptions to fill in details
3. Rewriting the goal with explicit requirements

Common ambiguities to fix:
- "use the research" → specify which values/data points
- "make it better" → specify what improvement means
- "from task 0" → specify what specific data to use
- "write a calculator" → specify what it calculates and how

Return JSON with:
- clarified_goal: The rewritten, specific task goal
- extracted_details: What you made explicit

Example:
Input: "Write a calculator using the research results"
Output: {{
  "clarified_goal": "Write a Python function calculate_macros(grams) that uses protein=26g and fat=10g per 100g to calculate total macros",
  "extracted_details": "Specified function name, parameter, and concrete macro values to use"
}}
"""

        result, tokens = call_ollama_structured(
            model=cls.model,
            prompt=prompt,
            system="You are a task clarification expert. Make vague requirements explicit and specific.",
            response_schema=ClarificationResult,
            temperature=cls.temperature
        )

        # Create new task with clarified goal
        clarified_task = TaskSchema(
            plan_id=task.plan_id,
            task_order=task.task_order,
            goal=result.clarified_goal
        )

        print(f"      Details added: {result.extracted_details[:60]}...")

        return clarified_task, tokens
