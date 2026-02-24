from ..schemas import TaskSchema, PlanSchema, RequestSchema
from .classifier import TaskClassifier, TaskVerdict
from ..llm import call_ollama_structured
from ..config import settings
from ..logging_config import get_logger
from pydantic import BaseModel

logger = get_logger("hierarchical")


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

    model = settings.PLANNING_MODEL
    temperature = settings.PLANNING_TEMP
    max_depth = settings.MAX_DECOMPOSITION_DEPTH

    @classmethod
    def create_plan(cls, request: RequestSchema) -> PlanSchema:
        """
        Create a hierarchical plan from a request.

        Returns a flat PlanSchema with all leaf tasks (DFS order).
        """
        logger.info(f"HierarchicalPlanner processing request: {request.request_id}")
        logger.info(f"Content: {request.content[:100]}...")

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

        logger.info(f"Hierarchical plan created: {len(plan.tasks)} executable tasks")
        logger.debug(f"Tokens consumed: {total_tokens}")

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
        logger.info(f"{indent}Classifying: {task.goal[:60]}...")

        # Check max depth
        if depth >= cls.max_depth:
            logger.warning(f"{indent}Max depth reached, treating as executable")
            return [task], 0

        # Skip classification for subtasks that are clearly single-action
        # This prevents the LLM from over-decomposing simple leaf tasks
        if depth > 0 and cls._is_obviously_simple(task.goal):
            logger.info(f"{indent}Obviously simple subtask, treating as executable")
            return [task], 0

        # Classify the task
        verdict, reasoning, tokens = TaskClassifier.classify(task, context)
        total_tokens = tokens

        logger.info(f"{indent}Verdict: {verdict.value}")
        logger.debug(f"{indent}Reasoning: {reasoning[:80]}...")

        # Handle based on verdict
        if verdict == TaskVerdict.EXECUTE:
            # Base case: task is executable
            logger.info(f"{indent}Executable task (leaf node)")
            return [task], total_tokens

        elif verdict == TaskVerdict.DECOMPOSE:
            # Decompose into subtasks
            logger.info(f"{indent}Decomposing into subtasks...")
            subtasks, decomp_tokens = cls._decompose_task(task, context)
            total_tokens += decomp_tokens

            # Recursively decompose each subtask
            all_leaf_tasks = []
            for i, subtask in enumerate(subtasks):
                logger.info(f"{indent}  Subtask {i}: {subtask.goal[:50]}...")

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
            logger.info(f"{indent}Clarifying ambiguous task...")
            clarified_task, clarify_tokens = cls._clarify_task(task, context)
            total_tokens += clarify_tokens

            logger.info(f"{indent}Clarified: {clarified_task.goal[:60]}...")

            # After clarification, it should be executable
            # (Don't recurse - assume clarification makes it executable)
            return [clarified_task], total_tokens

        else:  # CLARIFY_THEN_DECOMPOSE
            # Clarify first, then decompose
            logger.info(f"{indent}Clarifying before decomposition...")
            clarified_task, clarify_tokens = cls._clarify_task(task, context)
            total_tokens += clarify_tokens

            logger.info(f"{indent}Clarified: {clarified_task.goal[:60]}...")
            logger.info(f"{indent}Now decomposing clarified task...")

            # Decompose the clarified task
            subtasks, decomp_tokens = cls._decompose_task(clarified_task, context)
            total_tokens += decomp_tokens

            # Recursively decompose each subtask
            all_leaf_tasks = []
            for i, subtask in enumerate(subtasks):
                logger.info(f"{indent}  Subtask {i}: {subtask.goal[:50]}...")

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
        Decompose a complex task into 2-3 subtasks.

        Returns:
            tuple: (list of subtasks, tokens used)
        """
        prompt = f"""Task to decompose: {task.goal}

This task is too complex to execute in one step. Break it into EXACTLY 2-3 subtasks. Never more than 3.

If the task seems to need more than 3 subtasks, you're over-decomposing.
Combine related steps into single subtasks.

ANTI-HALLUCINATION RULES (CRITICAL):

1. DO NOT create research tasks about basic programming operations
   ❌ BAD: "Research how to read CSV files"
   ✅ GOOD: "Read file Workouts.csv"

2. DO NOT create checking/validation/merging tasks unless explicitly requested
   ❌ BAD: "Check if existing data available"
   ❌ BAD: "Merge new data with existing DataFrame"
   ✅ GOOD: Just do the requested operation directly

3. DO NOT add complexity that wasn't requested
   ❌ BAD: Breaking "read CSV" into "research pandas → load file → validate → merge"
   ✅ GOOD: One subtask = "Read file X.csv"

4. ASSUME standard Python libraries are available (pandas, csv, json, etc.)
   - No need to research how to use them
   - No need to check if they're installed

5. File operations are SIMPLE:
   - "Read X.csv" → ONE task, not 3-4 tasks
   - "Save to Y.txt" → ONE task
   - Don't decompose file I/O further

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
"Read Workouts.csv and calculate average calories, save summary"
→ [
    "Read file Workouts.csv",
    "Calculate average calories from the workout data",
    "Save the analysis summary to outputs/workout_summary.txt"
  ]

BAD DECOMPOSITION (hallucinated complexity):
"Read Workouts.csv and calculate average calories, save summary"
→ [
    "Research how to read CSV files",  <- Unnecessary
    "Load Workouts.csv using pandas",  <- Just say "Read file"
    "Check for existing data",         <- Not requested
    "Merge with existing data",        <- Hallucinated
    "Calculate calories",
    "Save summary"
  ]

Return a JSON object with a 'subtasks' array containing 2-3 goal strings.

Example:
{{
  "subtasks": [
    "Read file Workouts.csv",
    "Calculate average calories from the workout data",
    "Save the analysis summary to outputs/workout_summary.txt"
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

        # Enforce max 3 subtasks
        if len(result.subtasks) > 3:
            logger.warning(f"Decomposer returned {len(result.subtasks)} subtasks, truncating to 3")
            result.subtasks = result.subtasks[:3]

        # Sanity check: Filter out hallucinated research/checking tasks
        filtered_subtasks = []
        for goal in result.subtasks:
            goal_lower = goal.lower()

            skip_keywords = [
                "research how to",
                "research pandas",
                "research python",
                "check if",
                "merge with existing",
                "validate data",
                "verify that"
            ]

            if any(keyword in goal_lower for keyword in skip_keywords):
                logger.warning(f"Filtering hallucinated task: {goal[:50]}...")
                continue

            filtered_subtasks.append(goal)

        if len(filtered_subtasks) < len(result.subtasks):
            logger.warning(f"Filtered {len(result.subtasks) - len(filtered_subtasks)} hallucinated task(s)")

        # Convert to TaskSchema
        subtasks = []
        for i, goal in enumerate(filtered_subtasks):
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

        logger.debug(f"Details added: {result.extracted_details[:60]}...")

        return clarified_task, tokens

    @staticmethod
    def _is_obviously_simple(goal: str) -> bool:
        """Check if a goal is obviously a single action (no need to classify via LLM)."""
        goal_lower = goal.lower().strip()

        # Single-action patterns that should never be decomposed further
        simple_prefixes = [
            "read file", "read csv", "load file", "load csv",
            "save the", "save to", "write to", "write the",
            "calculate ", "compute ", "find the",
            "research ", "search for", "look up",
        ]
        if any(goal_lower.startswith(p) for p in simple_prefixes):
            return True

        # If no "and"/"then" connectors, likely single action
        connectors = [" and ", " then ", ", then ", " followed by "]
        if not any(c in goal_lower for c in connectors):
            return True

        return False
