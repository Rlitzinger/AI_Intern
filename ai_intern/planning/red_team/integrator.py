from ...schemas import PlanSchema, FindingsResult, RedTeamFinding
from ...llm import call_ollama_structured
from ...config import settings
from .utils import format_artifacts_for_prompt


class IntegratorAgent:
    model = settings.PLANNING_MODEL
    temperature = 0.1

    SYSTEM = """You are an adversarial integration reviewer.
Ignore how each task works internally.
Only look at the full artifact set: what files are produced, what each imports,
whether the artifact set forms a runnable application.
If no integration issues exist, return empty findings."""

    @classmethod
    def find_issues(cls, plan: PlanSchema) -> tuple[list[RedTeamFinding], int]:
        artifact_repr = format_artifacts_for_prompt(plan)

        prompt = f"""Artifacts this plan produces:

{artifact_repr}

Analyze the full artifact set ONLY (not individual tasks):
1. Is there an entrypoint? (a file runnable with `python X`)
2. Does each import chain resolve? (if A imports B, does B exist?)
3. Are there import cycles? (A imports B imports A)
4. Dead components? (produced but nothing imports them, no entrypoint)
5. Is the artifact set complete? (can a user actually run this?)

finding_type must be one of:
  missing_entrypoint | import_cycle | broken_dependency | dead_component | incomplete_artifact_set

severity:
  block = app cannot start
  warn  = app starts but missing functionality

DO NOT populate the "agent" field -- leave it null.
Return JSON: {{"findings": [{{"task_index": -1, "finding_type": str,
  "description": str, "evidence": str, "severity": str}}]}}
Return {{"findings": []}} if no issues."""

        result, tokens = call_ollama_structured(
            model=cls.model, prompt=prompt, system=cls.SYSTEM,
            response_schema=FindingsResult, temperature=cls.temperature
        )
        for f in result.findings:
            f.agent = "integrator"
        return result.findings, tokens
