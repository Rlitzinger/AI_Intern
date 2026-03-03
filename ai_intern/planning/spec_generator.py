from pydantic import BaseModel
from ..llm import call_ollama_structured
from ..config import settings
from ..logging_config import get_logger
from ..schemas import TaskSchema

logger = get_logger("spec_generator")


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
    Called when a task is classified as CLARIFY_THEN_DECOMPOSE
    and the goal contains app-scale keywords.
    """

    @staticmethod
    def generate(task: TaskSchema, context: dict) -> tuple[AppSpec, int]:
        """
        Generate a structured application spec from the task goal.
        Returns (spec, tokens_used).
        """
        logger.info(f"SpecGenerator: generating spec for: {task.goal[:80]}...")

        system_prompt = (
            "You are a software architect. Given a user request, produce a minimal but complete "
            "application spec. Keep features to what was actually requested — do not add scope. "
            "Components should map to individual Python files. Define clear interfaces between them."
        )

        env_block = context.get('environment', '') if context else ''
        env_header = f"{env_block}\n\n" if env_block else ""

        user_prompt = f"""{env_header}User request: {task.goal}

Generate a structured application spec for this request.

Rules:
- Features: include ONLY what was explicitly requested — no scope creep
- Components: MINIMUM 2, maximum 5. Every app needs at least a data/logic layer AND an entry point.
  Good split: storage.py (data) + cli.py or main.py (interface). Never collapse into 1 file.
- output_file: paths must be relative to outputs/ directory (e.g. "outputs/storage.py")
- depends_on: use the name field of other components (e.g. ["NoteStorage"])
- public_interface: plain English list of key functions/classes
  (e.g. "save(note), load(id), delete(id), list_all()")
- done_criteria: 2-4 concrete, testable statements

Return a JSON object with this exact structure:
{{
  "summary": "One sentence describing what the app does",
  "features": ["feature 1", "feature 2", "feature 3"],
  "components": [
    {{
      "name": "ComponentName",
      "responsibility": "One sentence: what this component does",
      "output_file": "outputs/filename.py",
      "depends_on": [],
      "public_interface": "function1(args), function2(args)"
    }}
  ],
  "done_criteria": ["criteria 1", "criteria 2"]
}}"""

        spec, tokens = call_ollama_structured(
            model=settings.PLANNING_MODEL,
            prompt=user_prompt,
            system=system_prompt,
            response_schema=AppSpec,
            temperature=settings.PLANNING_TEMP
        )

        # Enforce minimum 2 components — add a CLI/main entry point if the model collapsed to 1
        if len(spec.components) < 2:
            first = spec.components[0] if spec.components else None
            first_name = first.name if first else "AppCore"
            cli = ComponentSpec(
                name="AppCLI",
                responsibility="Provides a command-line interface to interact with the application.",
                output_file="outputs/cli.py",
                depends_on=[first_name] if first else [],
                public_interface="main()",
            )
            spec.components.append(cli)
            logger.info(f"Added synthetic AppCLI component (spec had only 1 component)")

        logger.info(f"Spec generated: {spec.summary}")
        logger.info(f"Components ({len(spec.components)}): {', '.join(c.name for c in spec.components)}")
        logger.info(f"Features ({len(spec.features)}): {'; '.join(spec.features)}")

        return spec, tokens
