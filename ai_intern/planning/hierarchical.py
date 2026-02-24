from ..schemas import TaskSchema, PlanSchema, RequestSchema
from .classifier import TaskClassifier, TaskVerdict
from ..llm import call_ollama_structured
from ..config import settings
from ..logging_config import get_logger
from pydantic import BaseModel, field_validator

logger = get_logger("hierarchical")


class DecompositionResult(BaseModel):
    """LLM response schema for task decomposition."""
    subtasks: list[str]  # List of subtask goal strings


class ClarificationResult(BaseModel):
    """LLM response schema for task clarification."""
    clarified_goal: str
    extracted_details: str  # What was clarified/made explicit

    @field_validator("extracted_details", "clarified_goal", mode="before")
    @classmethod
    def coerce_list_to_str(cls, v):
        """LLM sometimes returns a list instead of a string — join it."""
        if isinstance(v, list):
            return "; ".join(str(item) for item in v)
        return v


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
        # root_clarified_goal starts as None; set after clarification so all
        # recursive calls share the same stabilised objective string.
        leaf_tasks, total_tokens = cls._decompose_recursive(
            task=root_task,
            depth=0,
            context={'parent_goal': None, 'root_clarified_goal': None}
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

                # Create context for subtask — carry root_clarified_goal forward
                subtask_context = {
                    'parent_goal': task.goal,
                    'root_clarified_goal': context.get('root_clarified_goal'),
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

            # Lock in the clarified goal so ALL downstream decompositions see it
            if not context.get('root_clarified_goal'):
                context['root_clarified_goal'] = clarified_task.goal

            # Decompose the clarified task
            subtasks, decomp_tokens = cls._decompose_task(clarified_task, context)
            total_tokens += decomp_tokens

            # Recursively decompose each subtask
            all_leaf_tasks = []
            for i, subtask in enumerate(subtasks):
                logger.info(f"{indent}  Subtask {i}: {subtask.goal[:50]}...")

                subtask_context = {
                    'parent_goal': clarified_task.goal,
                    'root_clarified_goal': context['root_clarified_goal'],
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
        # Surface the clarified objective as soft context — informational, not a hard constraint
        root_goal = context.get('root_clarified_goal') or context.get('parent_goal')
        objective_block = ""
        if root_goal and root_goal != task.goal:
            objective_block = f"Overall objective: {root_goal}\n\n"

        prompt = f"""{objective_block}Task to decompose: {task.goal}

This task is too complex to execute in one step. Break it into EXACTLY 2-3 subtasks. Never more than 3.

If the task seems to need more than 3 subtasks, you're over-decomposing.
Combine related steps into single subtasks.

AVAILABLE EXECUTION AGENTS — map each subtask to exactly one:
- ResearchAgent  : web search and synthesis of information
- CodingAgent    : writes a complete Python script/function (handles read + process + output in ONE task)
- AnalysisAgent  : generates and runs analysis code against a file
- FileAgent      : reads or writes files (CSV, txt, JSON)

IMPORTANT: CodingAgent can read a file, process data, and output results all in a single task.
Do NOT split "write a script that reads X, calculates Y, and prints Z" — that is ONE CodingAgent task.
Only create separate tasks when different *agents* are genuinely needed.

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

4. ASSUME standard Python libraries are available (pandas, csv, json, yfinance, etc.)
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

This task is ambiguous. Make it specific enough to execute — but DO NOT expand its scope.

CLARIFICATION RULES:
1. Resolve HOW, not WHAT. Fill in missing method, tool, or format — don't add new deliverables.
2. Keep the scope identical to the original. One request in = one request out.
3. Make one reasonable assumption per ambiguity. Don't pile on extras.

GOOD clarification (fills in method, keeps scope):
  Input:  "Research stock prices and write a trend calculator"
  Output: "Use yfinance to fetch 1 year of S&P 500 daily closing prices,
           then write a Python script that calculates a 20-day moving average trend"

BAD clarification (expands scope beyond what was asked):
  Input:  "Research stock prices and write a trend calculator"
  Output: "Scrape Yahoo Finance, analyze historical trends, generate a report WITH
           visualizations AND calculate percentage change AND export to CSV"
           ← added report, visualizations, percentage change, CSV export — none were asked for

Common ambiguities to fix:
- "use the research" → specify which values/data points
- "make it better" → specify what improvement means
- "write a calculator" → specify what it calculates and the method (e.g. moving average)
- "stock prices" → pick one index/ticker and timeframe as a default

Return JSON with:
- clarified_goal: The rewritten goal — same scope, concrete method
- extracted_details: What single assumption you made explicit

Example:
Input: "Write a calculator using the research results"
Output: {{
  "clarified_goal": "Write a Python function calculate_macros(grams) that uses protein=26g and fat=10g per 100g to calculate total macros",
  "extracted_details": "Specified function name, parameter, and concrete macro values from the research"
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
        """
        Check if a goal is obviously a single action (no need to classify via LLM).

        Key insight: "Write a script that reads X, calculates Y, and outputs Z" is ONE
        CodingAgent task even though it has "and" connectors — the conjunctions describe
        implementation steps inside a single script, not separate agent hand-offs.
        """
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

        # A task starting with "write a/the script/function/program..." is a single
        # CodingAgent call regardless of how many internal steps it describes.
        single_agent_prefixes = [
            "write a script", "write the script",
            "write a function", "write the function",
            "write a python", "write python",
            "write a program", "implement a", "implement the",
            "develop a script", "develop a function",
            "create a script", "create a function",
        ]
        if any(goal_lower.startswith(p) for p in single_agent_prefixes):
            return True

        # If no "and"/"then" connectors, likely single action
        connectors = [" and ", " then ", ", then ", " followed by "]
        if not any(c in goal_lower for c in connectors):
            return True

        return False