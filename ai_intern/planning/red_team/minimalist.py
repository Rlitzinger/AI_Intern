from ...schemas import PlanSchema, FindingsResult, RedTeamFinding
from ...llm import call_ollama_structured
from ...config import settings
from .utils import format_plan_for_prompt


class MinimalistAgent:
    model = settings.PLANNING_MODEL
    temperature = 0.1   # Keep at 0.1 -- handle counterfactual via prompt, not temp

    SYSTEM = """You are an adversarial simplicity reviewer.
ONE job: find over-engineering.
Ask: what is the absolute minimum to fulfill this request?
Compare that minimum to the actual plan.
Flag every task beyond the minimum viable implementation.
If the plan is appropriately scoped, return empty findings."""

    @classmethod
    def find_issues(cls, plan: PlanSchema) -> tuple[list[RedTeamFinding], int]:
        original_request = ""
        spec_context = ""

        if plan.tasks:
            original_request = plan.tasks[0].original_goal or plan.tasks[0].goal

        # If this is a spec-driven plan, inject the spec's justification
        # so Minimalist doesn't flag legitimate multi-component decomposition
        if plan.app_spec:
            spec = plan.app_spec
            spec_context = (
                f"\nThis plan was generated from an app spec. "
                f"The SpecGenerator determined this requires {len(plan.tasks)} components "
                f"because: {spec.get('summary', 'multi-component application')}. "
                f"Components: {', '.join(c['name'] for c in spec.get('components', []))}. "
                f"Only flag over-engineering if the SPEC ITSELF is wrong for the request, "
                f"not if individual tasks seem simple in isolation."
            )

        task_repr = format_plan_for_prompt(plan)

        prompt = f"""Original request: "{original_request}"{spec_context}

Actual plan ({len(plan.tasks)} tasks):
{task_repr}

Step 1: In ONE sentence, state the minimum viable implementation of this request.
How many tasks/files does the minimum require?

Step 2: Compare your minimum to the actual plan.
Is any task unnecessary? Any abstraction that adds no value?
Is the plan decomposed at the wrong granularity (e.g. 4 files where 1 script works)?

finding_type must be one of:
  over_engineered | unnecessary_task | premature_abstraction | wrong_scale

severity:
  block = decomposition makes the request harder to fulfill, not easier
  warn  = extra complexity but plan would still work

Use task_index = -1 for plan-level findings.
DO NOT populate the "agent" field -- leave it null.
Return JSON: {{"findings": [{{"task_index": int, "finding_type": str,
  "description": str, "evidence": str, "severity": str}}]}}
Return {{"findings": []}} if plan matches minimum viable."""

        result, tokens = call_ollama_structured(
            model=cls.model, prompt=prompt, system=cls.SYSTEM,
            response_schema=FindingsResult, temperature=cls.temperature
        )
        for f in result.findings:
            f.agent = "minimalist"
        return result.findings, tokens
