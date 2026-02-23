from enum import Enum
from schemas import TaskSchema
from llm import call_ollama_structured
from pydantic import BaseModel


class TaskVerdict(Enum):
    """Possible verdicts from task classification."""
    EXECUTE = "execute"              # Simple + Clear → one-shot execution
    DECOMPOSE = "decompose"          # Complex + Clear → break into subtasks
    CLARIFY = "clarify"              # Simple + Ambiguous → ask for clarity
    CLARIFY_THEN_DECOMPOSE = "clarify_then_decompose"  # Complex + Ambiguous


class ClassificationResult(BaseModel):
    """LLM response schema for task classification."""
    is_complex: bool
    is_ambiguous: bool
    reasoning: str


class TaskClassifier:
    """
    2x2 task classification: complexity × ambiguity.

    Determines whether a task should be:
    - Executed immediately (simple + clear)
    - Decomposed into subtasks (complex + clear)
    - Clarified first (simple + ambiguous)
    - Clarified then decomposed (complex + ambiguous)

    Key insight: Decomposing an ambiguous task just creates a tree of ambiguous subtasks.
    We must clarify BEFORE we decompose.
    """

    model = "qwen2.5:7b-instruct"
    temperature = 0.2

    system_prompt = """You are a task analysis expert.
Evaluate tasks for complexity and ambiguity.
Be precise and analytical in your reasoning.
Respond with the exact JSON structure requested."""

    @classmethod
    def classify(cls, task: TaskSchema, context: dict = None) -> tuple[TaskVerdict, str, int]:
        """
        Classify a task into one of 4 verdicts based on complexity and ambiguity.

        Args:
            task: The task to classify
            context: Optional context from parent tasks or previous steps

        Returns:
            tuple: (verdict, reasoning, tokens_used)
        """
        # Build the classification prompt
        prompt = cls._build_prompt(task, context)

        # Call LLM to get classification
        result, tokens = call_ollama_structured(
            model=cls.model,
            prompt=prompt,
            system=cls.system_prompt,
            response_schema=ClassificationResult,
            temperature=cls.temperature
        )

        # Map (is_complex, is_ambiguous) to verdict
        verdict = cls._map_to_verdict(result.is_complex, result.is_ambiguous)

        return verdict, result.reasoning, tokens

    @staticmethod
    def _build_prompt(task: TaskSchema, context: dict = None) -> str:
        """Build the classification prompt."""

        prompt_parts = []

        # Add context if available
        if context and context.get('parent_goal'):
            prompt_parts.append(f"Parent goal: {context['parent_goal']}\n")

        prompt_parts.append(f"Task to classify: {task.goal}\n")

        prompt_parts.append("""
Analyze this task on two dimensions:

1. COMPLEXITY - Can this be accomplished in a single step?

   SIMPLE means:
   - ONE function, script, or component
   - ONE well-defined action (not multiple actions connected by "and")
   - NO combination of file I/O + processing + saving
   - Less than 100 lines of code
   - Single step (no "then", "and then", "followed by")
   - No dependencies on other tasks

   Examples of SIMPLE:
   - "Write a function to calculate fibonacci sequence" (ONE action - pure code)
   - "Research protein content in chicken breast" (ONE action - pure research)
   - "Read file Workouts.csv" (ONE action - pure file read)
   - "Calculate average of [10, 20, 30]" (ONE action - pure computation)

   NOT SIMPLE (these are COMPLEX):
   - "Read X and calculate Y" (TWO actions: file + processing)
   - "Calculate X and save to Y" (TWO actions: processing + file)
   - "Research X and write code Y" (TWO actions: research + code)

   CRITICAL: Check for FILE I/O combined with PROCESSING:

   File operations (read/save/write) + processing/analysis = ALWAYS COMPLEX:
   - "Read X and calculate/analyze Y" = COMPLEX (read + process)
   - "Calculate X and save to Y" = COMPLEX (process + save)
   - "Read X, process Y, save Z" = COMPLEX (3 actions)
   - "Load data from X and compute Y" = COMPLEX (load + compute)

   Keywords that indicate file I/O:
   - read, load, open, parse (input)
   - save, write, output, store, export (output)

   Keywords that indicate processing:
   - calculate, compute, analyze, summarize, aggregate, average, total

   If you see BOTH file I/O AND processing → COMPLEX

   Examples:
   - "Read Workouts.csv and calculate average calories" = COMPLEX (read + calculate)
   - "Calculate fibonacci sequence" = SIMPLE (no file I/O)
   - "Read config.json" = SIMPLE (no processing)
   - "Parse CSV and compute statistics, save report" = COMPLEX (all 3)

   CRITICAL: Check for compound actions using "and" or "then":
   - "Research X and write Y" = COMPLEX (2 actions: research + code)
   - "Write X and save as Y" = COMPLEX (2 actions: code + save)
   - "Read X and analyze Y" = COMPLEX (2 actions: read + analyze)
   - "Research X, write Y, then save Z" = COMPLEX (3 actions)

   Even if each action is simple individually, multiple actions = COMPLEX.

   COMPLEX means:
   - Multiple functions, files, or components
   - More than 100 lines of code
   - Multiple steps or stages
   - Requires integration of parts

   Examples of COMPLEX:
   - "Build a REST API with authentication and database"
   - "Create a web scraper with data storage and analysis"
   - "Write a calculator and save it to a file" (2 separate actions)

2. AMBIGUITY - Is the desired outcome clearly specified?

   CLEAR means:
   - Specific requirements stated
   - Measurable success criteria
   - Concrete values or parameters given
   - One obvious interpretation

   Examples of CLEAR:
   - "Calculate fibonacci up to n terms"
   - "Research protein content in chicken (grams per 100g)"
   - "Use protein=26g, fat=10g to calculate total macros"

   AMBIGUOUS means:
   - Vague requirements
   - Multiple valid interpretations
   - Missing critical details
   - References to undefined data ("use the research", "from above")

   Examples of AMBIGUOUS:
   - "Make it better"
   - "Write a calculator using the research results" (which values? → AMBIGUOUS but still SIMPLE — a calculator is one script)
   - "Implement the functionality" (what functionality?)
   - "Use the data from task 0" (what specific data?)

   IMPORTANT: Ambiguous references to undefined data make a task AMBIGUOUS but do NOT add complexity.
   The complexity is determined by the output artifact, not by the fact that inputs are unclear.
   Example: "Write a function using the research" → SIMPLE (one function) + AMBIGUOUS (unclear inputs).

Analyze the task and respond with:
- is_complex: true if COMPLEX, false if SIMPLE
- is_ambiguous: true if AMBIGUOUS, false if CLEAR
- reasoning: 2-3 sentences explaining your classification

Return JSON matching the ClassificationResult schema.
""")

        return "".join(prompt_parts)

    @staticmethod
    def _map_to_verdict(is_complex: bool, is_ambiguous: bool) -> TaskVerdict:
        """Map complexity/ambiguity booleans to verdict enum."""

        if not is_complex and not is_ambiguous:
            # Simple + Clear → Execute immediately
            return TaskVerdict.EXECUTE

        elif is_complex and not is_ambiguous:
            # Complex + Clear → Decompose into subtasks
            return TaskVerdict.DECOMPOSE

        elif not is_complex and is_ambiguous:
            # Simple + Ambiguous → Clarify first
            return TaskVerdict.CLARIFY

        else:  # is_complex and is_ambiguous
            # Complex + Ambiguous → Clarify THEN decompose
            return TaskVerdict.CLARIFY_THEN_DECOMPOSE
