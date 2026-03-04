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
            "You are a software architect who specialises in decomposing apps into small, "
            "focused Python modules. Each module must be implementable in under 40 lines of code "
            "with no more than 3 simple functions. Never combine unrelated concerns into one module."
        )

        env_block = context.get('environment', '') if context else ''
        env_header = f"{env_block}\n\n" if env_block else ""

        user_prompt = f"""{env_header}User request: {task.goal}

Generate a structured application spec. Split the app into FINE-GRAINED components.

SIZE RULE (critical): Each component must be implementable in ≤40 lines of Python, ≤3 functions.
- If a responsibility requires >40 lines, split it into two smaller components.
- "MainApp" or "AppServer" are forbidden — they are always too big. Split them.

COMPONENT COUNT: 3 to 6 components. Never fewer than 3, never more than 6.

COMPONENT RULES:
- Each component has exactly ONE responsibility. No mixing of concerns.
- output_file: must be relative to outputs/ (e.g. "outputs/storage.py")
- depends_on: list names of other components this module imports
- public_interface: exact Python function signatures (e.g. "save(text: str) -> str")
- done_criteria: 2-4 concrete testable statements

HOW TO SPLIT WEB APPS (follow this pattern exactly):
  1. DataStore     — file I/O only: read/write/list data to a JSON or CSV file
  2. DataModel     — data structure only: a simple dict/dataclass representing one record
  3. RouteHandlers — HTTP request logic: handle_get(path) and handle_post(path, body) -> (int, str)
  4. HTMLTemplates — HTML strings only: get_form() -> str, get_list(items) -> str
  5. Server        — server startup only: run(port: int) — creates HTTPServer and serves forever

HOW TO SPLIT CLI APPS:
  1. DataStore — file I/O: load(), save(record)
  2. Commands  — user commands: cmd_add(args), cmd_list(), cmd_delete(id)
  3. CLI       — entry point: parse_args(argv) -> (command, args), main()

HOW TO SPLIT CALCULATORS/UTILITIES:
  1. Logic     — pure functions: calculate(inputs) -> result
  2. Interface — input/output: prompt_user() -> inputs, display(result)

Return a JSON object:
{{
  "summary": "One sentence describing what the app does",
  "features": ["feature 1", "feature 2"],
  "components": [
    {{
      "name": "ComponentName",
      "responsibility": "One sentence: single responsibility this module handles",
      "output_file": "outputs/filename.py",
      "depends_on": [],
      "public_interface": "function1(arg: type) -> return_type, function2(arg: type) -> return_type"
    }}
  ],
  "done_criteria": ["criteria 1", "criteria 2"]
}}

Example for "Build a note-taking app with save, load, list":
{{
  "summary": "A CLI note-taking app that saves and loads notes as JSON",
  "features": ["save note to disk", "load note by id", "list all notes"],
  "components": [
    {{
      "name": "NoteStore",
      "responsibility": "Reads and writes notes to a JSON file on disk",
      "output_file": "outputs/note_store.py",
      "depends_on": [],
      "public_interface": "save(text: str) -> str, load(id: str) -> dict, list_all() -> list"
    }},
    {{
      "name": "NoteCommands",
      "responsibility": "Implements the add, load, list commands using NoteStore",
      "output_file": "outputs/note_commands.py",
      "depends_on": ["NoteStore"],
      "public_interface": "cmd_add(text: str), cmd_load(id: str), cmd_list()"
    }},
    {{
      "name": "NoteCLI",
      "responsibility": "Parses command-line arguments and calls NoteCommands",
      "output_file": "outputs/note_cli.py",
      "depends_on": ["NoteCommands"],
      "public_interface": "parse_args(argv: list) -> tuple, main()"
    }}
  ],
  "done_criteria": [
    "save() returns a non-empty string id",
    "load(id) returns the saved note dict",
    "list_all() returns a list"
  ]
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
