from ..schemas import TaskSchema, PlanSchema, RequestSchema
from .classifier import TaskClassifier, TaskVerdict
from .spec_generator import SpecGenerator, AppSpec
from ..llm import call_ollama_structured
from ..config import settings
from ..logging_config import get_logger
from pydantic import BaseModel, field_validator

logger = get_logger("hierarchical")


class SubtaskItem(BaseModel):
    """One subtask with an agent annotation."""
    goal: str
    agent: str = "code"  # "code", "research", "file", "analysis"


class DecompositionResult(BaseModel):
    """LLM response schema for task decomposition.

    The LLM returns a list of dicts: [{"goal": "...", "agent": "..."}].
    A field_validator normalises list[str] (old format) into list[SubtaskItem]
    so we're backward-compatible if the 7B model returns the legacy format.
    """
    subtasks: list[SubtaskItem]

    @field_validator("subtasks", mode="before")
    @classmethod
    def coerce_str_list(cls, v):
        """Accept list[str] (legacy) or list[dict] (new format)."""
        if not isinstance(v, list):
            return v
        coerced = []
        for item in v:
            if isinstance(item, str):
                coerced.append({"goal": item, "agent": "code"})
            elif isinstance(item, dict):
                coerced.append(item)
            else:
                coerced.append(item)
        return coerced


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

        # Scan environment before planning so every LLM call sees real context
        from .environment import EnvironmentScanner
        env_ctx = EnvironmentScanner.scan()
        env_prompt = env_ctx.to_prompt_block()
        logger.info(f"Environment: {len(env_ctx.available_files)} file(s) in user_data/")

        # Create root task from request
        root_task = TaskSchema(
            plan_id="placeholder",  # Will be set later
            task_order=0,
            goal=request.content
        )

        # Recursively decompose
        # root_clarified_goal starts as None; set after clarification so all
        # recursive calls share the same stabilised objective string.
        shared_context: dict = {
            'parent_goal': None,
            'root_clarified_goal': None,
            'environment': env_prompt,
            'environment_context': env_ctx,
        }
        leaf_tasks, total_tokens = cls._decompose_recursive(
            task=root_task,
            depth=0,
            context=shared_context
        )

        # Build plan from leaf tasks
        plan = PlanSchema(
            request_id=request.request_id,
            tasks=[],
            token_usage=total_tokens
        )

        # Attach app_spec to plan if one was generated
        app_spec = shared_context.get('app_spec')
        if app_spec is not None:
            plan.app_spec = app_spec.model_dump()

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
            if not task.suggested_agent:
                task.suggested_agent = cls._infer_agent_from_goal(task.goal)
            return [task], 0

        # Classify the task (returns 4-tuple including is_app_scale)
        verdict, reasoning, tokens, is_app_scale = TaskClassifier.classify(task, context)
        total_tokens = tokens

        # Hybrid safety net: LLM OR conservative keyword match.
        # The 7B model sometimes misses obvious app-scale signals. Keywords here are
        # deliberately conservative — only terms that *unambiguously* imply multi-file output.
        if not is_app_scale:
            is_app_scale = cls._keyword_is_app_scale(task.goal)
            if is_app_scale:
                logger.info(f"{indent}App-scale overridden by keyword safety net")

        logger.info(f"{indent}Verdict: {verdict.value} | app_scale={is_app_scale}")
        logger.debug(f"{indent}Reasoning: {reasoning[:80]}...")

        # Handle based on verdict
        if verdict == TaskVerdict.EXECUTE and is_app_scale:
            # Contradiction: EXECUTE + app_scale means the LLM under-estimated complexity.
            # An app-scale request can never be a single executable leaf — override to DECOMPOSE.
            logger.info(f"{indent}EXECUTE overridden to DECOMPOSE (is_app_scale=True)")
            verdict = TaskVerdict.DECOMPOSE

        if verdict == TaskVerdict.EXECUTE:
            # Base case: task is executable — annotate agent if not already set
            logger.info(f"{indent}Executable task (leaf node)")
            if not task.suggested_agent:
                task.suggested_agent = cls._infer_agent_from_goal(task.goal)
            return [task], total_tokens

        elif verdict == TaskVerdict.DECOMPOSE:
            # App-scale requests always go through spec generator, even if classifier
            # says DECOMPOSE (happens when the prompt is specific enough to be "clear")
            if is_app_scale:
                logger.info(f"{indent}App-scale DECOMPOSE — redirecting to SpecGenerator...")
                spec, spec_tokens = SpecGenerator.generate(task, context)
                total_tokens += spec_tokens
                logger.info(f"{indent}Spec: {spec.summary}")
                logger.info(f"{indent}Components: {', '.join(c.name for c in spec.components)}")
                context['app_spec'] = spec
                if not context.get('root_clarified_goal'):
                    context['root_clarified_goal'] = spec.summary

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
            # Check if this is an app-scale request
            if is_app_scale:
                logger.info(f"{indent}App-scale request detected — running SpecGenerator...")
                spec, spec_tokens = SpecGenerator.generate(task, context)
                total_tokens += spec_tokens

                logger.info(f"{indent}Spec summary: {spec.summary}")
                logger.info(f"{indent}Components: {', '.join(c.name for c in spec.components)}")

                # Store spec on context so _decompose_task and orchestrator can access it
                context['app_spec'] = spec

                # Convert spec to a clarified goal string for the decompose path
                clarified_goal = (
                    f"{spec.summary}. "
                    f"Components: {', '.join(c.name for c in spec.components)}. "
                    f"Features: {'; '.join(spec.features)}"
                )
                clarified_task = TaskSchema(
                    plan_id=task.plan_id,
                    task_order=task.task_order,
                    goal=clarified_goal
                )
            else:
                # Not app-scale: use existing clarify path
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

        If context contains an 'app_spec', uses a spec-aware prompt that maps
        each component to a concrete subtask goal.

        Returns:
            tuple: (list of subtasks, tokens used)
        """
        # --- Spec-aware path ---
        spec: AppSpec | None = context.get('app_spec')
        if spec is not None:
            return cls._decompose_task_from_spec(task, spec)

        # --- Standard path ---
        # Surface environment context and clarified objective
        env_block = context.get('environment', '')
        root_goal = context.get('root_clarified_goal') or context.get('parent_goal')
        objective_block = ""
        if root_goal and root_goal != task.goal:
            objective_block = f"Overall objective: {root_goal}\n\n"

        env_header = f"{env_block}\n\n" if env_block else ""
        prompt = f"""{env_header}{objective_block}Task to decompose: {task.goal}

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

For each subtask, specify which agent should handle it:
- "research": web search and information gathering
- "code": write Python code (functions, scripts, classes)
- "file": read or write files (CSV, JSON, TXT)
- "analysis": data analysis (calculate, aggregate, statistics using pandas/code)

Return a JSON object with a 'subtasks' array of objects, each with "goal" and "agent".

Example:
{{
  "subtasks": [
    {{"goal": "Read file Workouts.csv", "agent": "file"}},
    {{"goal": "Calculate average calories from the workout data", "agent": "analysis"}},
    {{"goal": "Save the analysis summary to outputs/workout_summary.txt", "agent": "file"}}
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
        skip_keywords = [
            "research how to",
            "research pandas",
            "research python",
            "check if",
            "merge with existing",
            "validate data",
            "verify that",
        ]
        filtered_items = []
        for item in result.subtasks:
            goal_lower = item.goal.lower()
            if any(kw in goal_lower for kw in skip_keywords):
                logger.warning(f"Filtering hallucinated task: {item.goal[:50]}...")
                continue
            filtered_items.append(item)

        if len(filtered_items) < len(result.subtasks):
            logger.warning(f"Filtered {len(result.subtasks) - len(filtered_items)} hallucinated task(s)")

        # Convert to TaskSchema with agent annotation
        subtasks = []
        for i, item in enumerate(filtered_items):
            task_obj = TaskSchema(
                plan_id="placeholder",
                task_order=i,
                goal=item.goal,
                suggested_agent=item.agent if item.agent else None,
            )
            subtasks.append(task_obj)

        return subtasks, tokens

    # Conservative keyword list — only terms that unambiguously mean multi-file app output.
    # Deliberately excludes "build a", "create a", "tool", "system" (too many false positives).
    _APP_SCALE_KEYWORDS = [
        " app",          # "note-taking app", "mobile app"   (leading space avoids "apply")
        "application",   # "build an application"
        "platform",      # "build a platform"
        "full stack",    # "full stack web service"
        "full-stack",
    ]

    @classmethod
    def _keyword_is_app_scale(cls, goal: str) -> bool:
        """Conservative keyword fallback for app-scale detection."""
        goal_lower = goal.lower()
        return any(kw in goal_lower for kw in cls._APP_SCALE_KEYWORDS)

    @classmethod
    def _decompose_task_from_spec(
        cls, task: TaskSchema, spec: AppSpec
    ) -> tuple[list[TaskSchema], int]:
        """
        Create subtasks directly from a spec, one per component (up to 3).
        Builds concrete goals that include output_file and public_interface.
        Groups components if there are more than 3.
        This path is deterministic from the spec — no LLM call, no context needed.
        """
        components = spec.components

        # Cap at 3: group later components together if needed
        if len(components) > 3:
            logger.warning(
                f"Spec has {len(components)} components — capping at 3 subtasks"
            )
            components = components[:3]

        subtasks = []
        for comp in components:
            goal = (
                f"Write `{comp.name}` class/module in `{comp.output_file}` "
                f"with public interface: {comp.public_interface}. "
                f"Responsibility: {comp.responsibility}."
            )
            if comp.depends_on:
                goal += f" Depends on: {', '.join(comp.depends_on)}."
            subtask = TaskSchema(
                plan_id=task.plan_id,
                task_order=len(subtasks),
                goal=goal,
                suggested_agent="code",  # spec-driven tasks are always code generation
            )
            subtask.output_contract = {
                "component_name": comp.name,
                "output_file": comp.output_file,
                "public_interface": comp.public_interface,
            }
            subtasks.append(subtask)

        logger.info(f"Spec-driven decomposition produced {len(subtasks)} subtasks")
        return subtasks, 0  # No LLM call needed — spec already provides the plan

    @classmethod
    def _clarify_task(cls, task: TaskSchema, context: dict) -> tuple[TaskSchema, int]:
        """
        Clarify an ambiguous task by making implicit details explicit.

        Returns:
            tuple: (clarified task, tokens used)
        """
        # Build context info
        env_block = context.get('environment', '') if context else ''
        context_info = ""
        if context and context.get('parent_goal'):
            context_info = f"\nParent goal: {context['parent_goal']}"

        env_header = f"{env_block}\n\n" if env_block else ""
        prompt = f"""{env_header}Ambiguous task: {task.goal}{context_info}

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

        # Create new task with clarified goal and a heuristic agent annotation
        clarified_task = TaskSchema(
            plan_id=task.plan_id,
            task_order=task.task_order,
            goal=result.clarified_goal,
            suggested_agent=cls._infer_agent_from_goal(result.clarified_goal),
        )

        logger.debug(f"Details added: {result.extracted_details[:60]}...")

        return clarified_task, tokens

    @staticmethod
    def _infer_agent_from_goal(goal: str) -> str | None:
        """Heuristic agent annotation from goal content. Returns None if unclear."""
        g = goal.lower().strip()

        # File operations — explicit file read/write patterns
        if any(g.startswith(p) for p in ["read file", "read csv", "load file", "load csv",
                                          "save to", "write to", "export "]):
            return "file"
        # Save/write the <something> — typically file output
        if g.startswith("save the") or g.startswith("write the"):
            return "file"

        # Research — web search
        if any(g.startswith(p) for p in ["research ", "search for", "look up", "find information",
                                          "investigate "]):
            return "research"

        # Code generation — broad "write a <X>" where X is code artifact
        code_keywords = ["script", "function", "class", "module", "program", "method",
                         "api", "server", "calculator", "parser", "formatter", "generator"]
        if g.startswith("write a") or g.startswith("write `"):
            if any(kw in g for kw in code_keywords):
                return "code"
            return "code"  # "write a" without further context → code
        if any(g.startswith(p) for p in ["implement a", "implement the", "develop a",
                                          "create a script", "create a function", "create a class",
                                          "build a function", "build a script"]):
            return "code"

        # Analysis — computation/statistics over data
        if any(g.startswith(p) for p in ["calculate ", "compute ", "analyze ", "summarize ",
                                          "find the average", "find the max", "find the min"]):
            return "analysis"

        # Full-content fallback: scan anywhere in the goal for strong signals.
        # This catches combined goals like "Read Workouts.csv and calculate average calories"
        # where the prefix check above doesn't match (file name ≠ "file"/"csv").
        if any(kw in g for kw in ["read ", "load ", ".csv", ".json", ".txt"]):
            if any(kw in g for kw in ["calculate", "compute", "analyze", "process",
                                       "write", "generate", "create", "build"]):
                return "code"   # combined read+process → single CodingAgent task
            return "file"       # pure read/load

        if any(kw in g for kw in ["function", "script", "class", "implement", "code"]):
            return "code"

        if any(kw in g for kw in ["calculate", "compute", "analyze", "average", "total", "sum"]):
            return "analysis"

        if any(kw in g for kw in ["research", "search", "look up", "find information"]):
            return "research"

        # Default: any unclassified single-action task falls back to code
        return "code"

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
        # Also matches spec-driven goals like "Write `NoteStorage` class/module in ..."
        single_agent_prefixes = [
            "write a script", "write the script",
            "write a function", "write the function",
            "write a python", "write python",
            "write a program", "implement a", "implement the",
            "develop a script", "develop a function",
            "create a script", "create a function",
            "write `",  # spec-driven: "Write `ComponentName` class/module in ..."
        ]
        if any(goal_lower.startswith(p) for p in single_agent_prefixes):
            return True

        # If no "and"/"then" connectors, likely single action
        connectors = [" and ", " then ", ", then ", " followed by "]
        if not any(c in goal_lower for c in connectors):
            return True

        return False