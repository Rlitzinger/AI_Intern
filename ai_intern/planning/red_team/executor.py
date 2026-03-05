from ...schemas import PlanSchema, FindingsResult, RedTeamFinding
from ...llm import call_ollama_structured
from ...config import settings
from .utils import format_plan_for_prompt


class ExecutorAgent:
    model = settings.PLANNING_MODEL
    temperature = 0.1

    SYSTEM = """You are an adversarial execution simulator.
ONE job: find data flow failures. Not validate success.
Trace each task in order. For each task, check whether its inputs
exist based on what prior tasks actually produce.
Be specific. Cite exact task indices.
If no issues exist, return empty findings list."""

    @classmethod
    def find_issues(cls, plan: PlanSchema) -> tuple[list[RedTeamFinding], int]:
        task_repr = format_plan_for_prompt(plan)

        prompt = f"""Plan to simulate (trace in order 0..{len(plan.tasks)-1}):

{task_repr}

For each task, answer:
1. What does this task NEED from context to execute?
2. What did all PRIOR tasks actually produce (check their output contracts)?
3. Is there a mismatch?

Also check:
- Does any task reference a file or class no prior task wrote?
- Does any task produce output no later task consumes (orphaned)?
- Is any task assigned the wrong agent type for what it does?

finding_type must be one of:
  context_mismatch | missing_input | orphaned_output | wrong_agent | missing_file

severity:
  block = execution would fail here
  warn  = execution degrades but continues

DO NOT populate the "agent" field -- leave it null.
Return JSON: {{"findings": [{{"task_index": int, "finding_type": str,
  "description": str, "evidence": str, "severity": str}}]}}
Return {{"findings": []}} if no issues found."""

        result, tokens = call_ollama_structured(
            model=cls.model, prompt=prompt, system=cls.SYSTEM,
            response_schema=FindingsResult, temperature=cls.temperature
        )
        for f in result.findings:
            f.agent = "executor"
        return result.findings, tokens
