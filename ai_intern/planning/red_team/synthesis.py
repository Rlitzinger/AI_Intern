from ...schemas import (
    RedTeamFinding, CrossExamResponse, BlueTeamResponse, ConstraintManifest, PlanSchema
)

BLOCK_THRESHOLD = 0.65   # High-confidence block finding -> triggers revision
WARN_SCORE_CAP  = 0.60   # warn findings CANNOT exceed this (below BLOCK_THRESHOLD)
NOISE_THRESHOLD = 0.25   # Below this: ignore completely


def compute_confidence(
    finding: RedTeamFinding,                  # has original_index set
    all_r1_findings: list[RedTeamFinding],    # full R1 list
    cross_responses: list[CrossExamResponse], # all reference global indices
    blue_responses: list[BlueTeamResponse],   # all reference global indices
) -> float:
    """
    Score a finding 0.0-1.0 based on round survival.
    warn findings are capped at WARN_SCORE_CAP (0.60) -- below BLOCK_THRESHOLD.

    Score breakdown:
      Base (found in R1):                  0.20
      Same type+task_index in other agents: +0.10 each (max +0.20)
      Confirmed in cross-exam (>=1):       +0.25
      Not disputed at all:                 +0.10
      Blue team failed to rebut:           +0.25
    """
    idx = finding.original_index
    score = 0.20

    # Similar findings from other agents (same finding_type + task_index)
    similar = sum(
        1 for f in all_r1_findings
        if f.original_index != idx
        and f.finding_type == finding.finding_type
        and f.task_index == finding.task_index
    )
    score += min(similar * 0.10, 0.20)

    # Cross-examination results (by global original_index)
    relevant_cross = [r for r in cross_responses if r.finding_index == idx]
    confirms = sum(1 for r in relevant_cross if r.verdict == "confirm")
    disputes = sum(1 for r in relevant_cross if r.verdict == "dispute")

    if confirms >= 1:
        score += 0.25
    if disputes == 0:
        score += 0.10

    # Blue team (by global original_index)
    relevant_blue = [r for r in blue_responses if r.finding_index == idx]
    if relevant_blue and not relevant_blue[0].can_rebut:
        score += 0.25

    raw = min(score, 1.0)

    # Severity gate: warn findings cannot trigger re-plan
    if finding.severity == "warn":
        return min(raw, WARN_SCORE_CAP)
    return raw


def build_constraint_manifest(
    high_confidence_findings: list[RedTeamFinding],
    plan: PlanSchema,
) -> ConstraintManifest:
    """Convert high-confidence findings into structured re-planning constraints."""
    task_constraints: dict[int, str] = {}
    structural_constraints: list[str] = []
    must_include_tasks: list[str] = []
    must_not_combine: list[list[str]] = []

    for f in high_confidence_findings:
        constraint = f"[{f.finding_type}] {f.description}"
        if f.task_index >= 0:
            existing = task_constraints.get(f.task_index, "")
            task_constraints[f.task_index] = (existing + "; " + constraint).lstrip("; ")
        else:
            structural_constraints.append(constraint)

        if f.finding_type == "missing_entrypoint":
            must_include_tasks.append(
                "A task that writes a runnable entrypoint (main.py or app.py) "
                "that imports and starts the application"
            )
        if f.finding_type == "wrong_agent" and f.task_index >= 0:
            if f.task_index < len(plan.tasks):
                task = plan.tasks[f.task_index]
                agent = task.declared_agent or task.suggested_agent
                if agent == "code":
                    combo = ["file", "code"]
                    if combo not in must_not_combine:
                        must_not_combine.append(combo)

    return ConstraintManifest(
        task_constraints=task_constraints,
        structural_constraints=structural_constraints,
        must_include_tasks=must_include_tasks,
        must_not_combine=must_not_combine,
    )
