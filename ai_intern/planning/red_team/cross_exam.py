from ...schemas import PlanSchema, RedTeamFinding, CrossExamResponse, CrossExamResult
from ...llm import call_ollama_structured
from ...config import settings
from .utils import format_plan_for_prompt


class CrossExamination:
    model = settings.PLANNING_MODEL
    temperature = 0.1

    SYSTEM = """You are reviewing adversarial findings about a plan.
Your job: filter noise from signal with epistemic rigor.

CONFIRM: Only if you have INDEPENDENT evidence not already stated in the finding.
  - Independent means: you observed it yourself by reading the plan, not by agreeing with the finding's reasoning.
  - If your reasoning would be: "Yes, the finding is correct because [restatement of finding]" -> that is NOT confirm, that is noise.
  - If your reasoning would be: "I also see [specific plan detail not mentioned in the finding]..." -> that IS confirm.

DISPUTE: If you can show the finding misreads the plan specifically.
  - Cite the exact plan line/task/contract that contradicts the finding.
  - "I think it's fine" is not a dispute.

EXTEND: If the finding is correct AND implies a deeper problem not yet surfaced.
  - State the deeper problem explicitly.

Abstain from responding to a finding if you have nothing independent to add.
False confirmations inflate confidence scores and cause unnecessary replanning.
Default to DISPUTE when uncertain -- the Blue Team will catch real issues."""

    @classmethod
    def run(
        cls,
        reviewing_agent: str,
        all_findings: list[RedTeamFinding],   # global list, all agents
        plan: PlanSchema,
    ) -> tuple[list[CrossExamResponse], int]:
        # Filter to findings from other agents -- but preserve original indices
        other_findings = [
            (i, f) for i, f in enumerate(all_findings)
            if f.agent != reviewing_agent
        ]

        if not other_findings:
            return [], 0

        # Build findings repr with GLOBAL indices
        findings_repr = "\n".join(
            f"[{i}] ({f.agent}/{f.finding_type}) Task {f.task_index}: {f.description}\n"
            f"    Evidence: {f.evidence}"
            for i, f in other_findings
        )

        task_repr = format_plan_for_prompt(plan)

        prompt = f"""You are the {reviewing_agent} agent.

The plan:
{task_repr}

Findings from OTHER agents to review (indices are GLOBAL -- use them as-is):
{findings_repr}

For each finding above, respond with:
- finding_index: the GLOBAL index shown in brackets above (e.g. [2] -> finding_index: 2)
- verdict: "confirm" | "dispute" | "extend"
- reasoning: specific reason referencing the plan
- additional_evidence: your own evidence (empty string if disputing)

CONFIRM = you have NEW evidence NOT already stated in the finding (do not restate the finding)
DISPUTE = you can show the finding misreads the plan
EXTEND  = finding is correct AND implies a deeper problem

You must respond to all {len(other_findings)} findings listed above.
Return JSON: {{"responses": [list of CrossExamResponse objects]}}"""

        result, tokens = call_ollama_structured(
            model=cls.model, prompt=prompt, system=cls.SYSTEM,
            response_schema=CrossExamResult, temperature=cls.temperature
        )
        return result.responses, tokens
