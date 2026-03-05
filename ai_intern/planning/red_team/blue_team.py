from ...schemas import PlanSchema, RequestSchema, RedTeamFinding, BlueTeamResponse, BlueTeamResult
from ...llm import call_ollama_structured
from ...config import settings
from .utils import format_plan_for_prompt


class BlueTeamDefense:
    model = settings.PLANNING_MODEL
    temperature = 0.1

    SYSTEM = """You are the architect who designed this plan.
Your job: find the weakest points in the adversarial findings.
Where are they wrong? Where are they right?
If a finding is correct, admit it -- set can_rebut=false.
Only set can_rebut=true if you have a specific factual reason it is wrong.
Be honest. Correctness over winning."""

    @classmethod
    def run(
        cls,
        plan: PlanSchema,
        request: RequestSchema,
        surviving_findings: list[RedTeamFinding],  # original_index already set
    ) -> tuple[list[BlueTeamResponse], int]:
        if not surviving_findings:
            return [], 0

        task_repr = format_plan_for_prompt(plan)

        # Use original_index in display so responses reference global indices
        findings_repr = "\n".join(
            f"[{f.original_index}] ({f.agent}/{f.finding_type}) Task {f.task_index}: {f.description}\n"
            f"    Evidence: {f.evidence}"
            for f in surviving_findings
        )

        prompt = f"""Original request: "{request.content}"

Your plan:
{task_repr}

Adversarial findings (indices are GLOBAL -- use them as-is in your responses):
{findings_repr}

For each finding, respond with:
- finding_index: the GLOBAL index in brackets above
- can_rebut: true if the finding is wrong, false if correct
- rebuttal: your specific argument (empty string if can_rebut=false)

Return JSON: {{"responses": [list of BlueTeamResponse objects]}}"""

        result, tokens = call_ollama_structured(
            model=cls.model, prompt=prompt, system=cls.SYSTEM,
            response_schema=BlueTeamResult, temperature=cls.temperature
        )
        return result.responses, tokens
