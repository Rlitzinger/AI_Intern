from enum import Enum
from ..schemas import TaskSchema
from ..llm import call_ollama_structured
from ..config import settings
from ..logging_config import get_logger
from pydantic import BaseModel

logger = get_logger("classifier")


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

    model = settings.PLANNING_MODEL
    temperature = settings.PLANNING_TEMP

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
Classify this task on two dimensions:

1. COMPLEXITY: Does this need more than ONE distinct action?
   SIMPLE = one action (one function, one file read, one search)
   COMPLEX = multiple actions joined by "and"/"then", or file I/O + processing combined

   Decision rule: count the verbs. File I/O (read/save/write/load) + processing (calculate/analyze/summarize) = COMPLEX.

   Examples:
   - "Write a fibonacci function" → SIMPLE (one action)
   - "Read Workouts.csv" → SIMPLE (one action)
   - "Read X and calculate Y" → COMPLEX (read + calculate)
   - "Build a REST API with auth" → COMPLEX (multiple components)

2. AMBIGUITY: Are inputs, outputs, and method all specified?
   CLEAR = specific requirements, concrete values, one obvious interpretation
   AMBIGUOUS = vague terms, undefined references, multiple interpretations

   Examples:
   - "Calculate fibonacci up to n terms" → CLEAR
   - "Make it better" → AMBIGUOUS
   - "Write code using the research" → AMBIGUOUS (which data?)

   Note: Ambiguous inputs don't add complexity. "Write a function using the research" = SIMPLE + AMBIGUOUS.

Return JSON: {"is_complex": bool, "is_ambiguous": bool, "reasoning": "2-3 sentences"}
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
