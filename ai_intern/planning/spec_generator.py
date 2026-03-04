import re
from pydantic import BaseModel
from ..llm import call_ollama_structured
from ..config import settings
from ..logging_config import get_logger
from ..schemas import TaskSchema

logger = get_logger("spec_generator")


class IntentSpec(BaseModel):
    """
    Extracted user intent. Internal scaffolding only — never stored on plan or task.
    Exists only to ground the component design in Call 2.
    """
    primary_action: str
    # The ONE thing the user must be able to do, written as an active verb phrase.
    # Examples:
    #   "turn written text into a generated image"
    #   "track daily workouts and view progress over time"
    #   "store and retrieve personal notes with search"
    # NOT "build a web app" — that describes the container, not the action.

    core_capability: str
    # The specific technical mechanism that makes primary_action possible.
    # Name the specific technology or algorithm required.
    # Examples:
    #   "call an image generation API (e.g. Stable Diffusion, DALL-E, or Replicate)"
    #   "persist workout records to SQLite and compute aggregate stats"
    #   "full-text search over stored note content"
    # If the request doesn't specify, name the most appropriate option and note it's assumed.

    capability_owner: str
    # The PascalCase name of the component that must own core_capability.
    # This component MUST appear in the final AppSpec.
    # Examples: "ImageGenerator", "WorkoutTracker", "NoteSearchEngine"

    constraints: list[str]
    # Explicit constraints from the user's request (platform, language, format, etc.)
    # Leave empty list if none stated. Do NOT invent constraints.
    # Examples: ["web app (HTTP server)", "Python only", "no external database"]


class ComponentSpec(BaseModel):
    name: str           # e.g. "NoteStorage", "NoteAPI", "CLI"
    responsibility: str # One sentence: what this component does
    output_file: str    # Where this component will be written, e.g. "outputs/storage.py"
    depends_on: list[str] = []  # Names of other components this one imports/uses
    public_interface: str  # Key functions/classes this exposes, plain English
                           # e.g. "save(note), load(id), delete(id), list_all()"


class AppSpec(BaseModel):
    """Structured spec produced from an app-scale request."""
    summary: str                    # One sentence: what the app does
    features: list[str]             # 3-6 user-facing features, plain English
    components: list[ComponentSpec] # Technical building blocks
    done_criteria: list[str]        # How to know the app is complete (2-4 items)


class SpecGenerator:
    """
    Generates a structured AppSpec from an app-scale request.
    Uses a two-phase approach: Phase 1 extracts intent (IntentSpec),
    Phase 2 designs components grounded in that intent.
    """

    model = settings.PLANNING_MODEL
    temperature = settings.PLANNING_TEMP

    @classmethod
    def generate(cls, task: TaskSchema, context: dict) -> tuple[AppSpec, int]:
        """
        Two-phase spec generation.
        Phase 1: Extract user intent (IntentSpec) — flat schema, low temperature.
        Phase 2: Design components grounded in intent — IntentSpec injected as ground truth.
        """
        logger.info(f"SpecGenerator: generating spec for: {task.goal[:80]}...")
        total_tokens = 0

        # Phase 1: Intent extraction
        intent, tokens_1 = cls._extract_intent(task.goal)
        total_tokens += tokens_1

        # Log intent so failures are visible immediately
        print(f"      [Intent] Primary action:   {intent.primary_action}")
        print(f"      [Intent] Core capability:  {intent.core_capability}")
        print(f"      [Intent] Capability owner: {intent.capability_owner}  <- must appear in components")

        # Phase 2: Component design grounded in intent
        spec, tokens_2 = cls._design_components(task.goal, intent)
        total_tokens += tokens_2

        # Safety net: verify capability_owner appears in components
        component_names = [c.name for c in spec.components]
        if intent.capability_owner not in component_names:
            print(f"      WARNING: capability_owner '{intent.capability_owner}' missing from {component_names}")
            print(f"      WARNING: Injecting missing component deterministically...")
            logger.warning(f"capability_owner '{intent.capability_owner}' missing — injecting deterministically")
            spec = cls._inject_capability_component(spec, intent)

        logger.info(f"Spec generated: {spec.summary}")
        logger.info(f"Components ({len(spec.components)}): {', '.join(c.name for c in spec.components)}")
        logger.info(f"Features ({len(spec.features)}): {'; '.join(spec.features)}")

        return spec, total_tokens

    @classmethod
    def _extract_intent(cls, goal: str) -> tuple[IntentSpec, int]:
        prompt = f"""User request: {goal}

Extract the user's intent from this request.

primary_action: The ONE thing the user must be able to DO.
  - Write as an active verb phrase ("turn text into an image", "track workouts")
  - Describe the ACTION, not the container ("web app" and "system" are containers, not actions)
  - If the request says "turn X into Y", primary_action is exactly "turn X into Y"

core_capability: The specific technical mechanism that makes primary_action possible.
  - Name the algorithm, API, or data operation required
  - Be concrete: "call image generation API (e.g. Stable Diffusion)" not "handle user input"
  - If multiple options exist, pick the most appropriate and note it is assumed

capability_owner: The PascalCase component name that will own core_capability.
  - This component MUST be created in the final application
  - Name it after what it does, not what layer it is: "ImageGenerator" not "Handler"

constraints: List only explicit requirements from the user's request.
  - Empty list if none stated. Do NOT invent constraints.

Return JSON matching the IntentSpec schema."""

        system = """You are an intent extraction expert.
Your job is to identify what a user actually needs an application to DO.
Focus entirely on the primary user action. Ignore architecture and implementation layers."""

        return call_ollama_structured(
            model=cls.model,
            prompt=prompt,
            system=system,
            response_schema=IntentSpec,
            temperature=0.1  # Intentionally low — extraction is deterministic
        )

    @classmethod
    def _design_components(cls, goal: str, intent: IntentSpec) -> tuple[AppSpec, int]:
        constraints_block = (
            "Constraints from user:\n" + "\n".join(f"  - {c}" for c in intent.constraints)
            if intent.constraints
            else "Constraints: None stated -- use simplest reasonable defaults."
        )

        prompt = f"""User request: {goal}

EXTRACTED INTENT -- treat this as ground truth, not a suggestion:
  Primary action:    {intent.primary_action}
  Core capability:   {intent.core_capability}
  Capability owner:  {intent.capability_owner}
{constraints_block}

Design the application components.

HARD RULES:
1. {intent.capability_owner} MUST be component [0]. It implements: {intent.core_capability}
2. Every other component exists to SUPPORT {intent.capability_owner}
3. 3-5 components total. Each must be implementable in under 50 lines of Python.
4. output_file must follow pattern: "outputs/<snake_case_name>.py"
5. depends_on lists component NAMES (PascalCase), not file paths
6. public_interface lists exact Python signatures only, nothing else

Suggested component order (omit any that are not needed for this specific app):
  1. {intent.capability_owner} -- REQUIRED -- implements {intent.core_capability}
  2. Storage -- persists data to disk (if the app needs persistence)
  3. Handler -- routes HTTP requests to capability + storage components (if web app)
  4. Template -- generates HTML for user interaction (if web app)
  5. Server -- starts HTTP server (if web app)

A CLI app does not need Handler, Template, or Server. Do not add them.

Return JSON matching the AppSpec schema."""

        system = f"""You are a software architect designing minimal Python applications.
The EXTRACTED INTENT block defines what the app must do. Your job is to design the components
that implement it. Component [0] must always be the capability owner named in the intent."""

        return call_ollama_structured(
            model=cls.model,
            prompt=prompt,
            system=system,
            response_schema=AppSpec,
            temperature=cls.temperature
        )

    @classmethod
    def _inject_capability_component(cls, spec: AppSpec, intent: IntentSpec) -> AppSpec:
        """
        Deterministically inject the missing capability component at index 0.
        This is a safety net, not the happy path. It should log a warning every time
        it fires -- consistent firing means the Phase 2 prompt needs adjustment.
        """
        capability_component = ComponentSpec(
            name=intent.capability_owner,
            responsibility=f"Implements the core capability: {intent.core_capability}",
            output_file=f"outputs/{cls._pascal_to_snake(intent.capability_owner)}.py",
            depends_on=[],
            public_interface="execute(input: str) -> str"
        )

        # Insert at front, cap at 5 to avoid over-decomposition
        new_components = [capability_component] + list(spec.components)
        new_components = new_components[:5]

        return AppSpec(
            summary=spec.summary,
            features=spec.features,
            components=new_components,
            done_criteria=spec.done_criteria
        )

    @staticmethod
    def _pascal_to_snake(name: str) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
