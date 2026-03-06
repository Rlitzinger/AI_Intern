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
    """
    Convert high-confidence findings into structured re-planning constraints.

    Each finding_type maps to a specific, actionable constraint.
    Generic string dumps are used only as fallback for unknown types.
    """
    task_constraints: dict[int, str] = {}
    structural_constraints: list[str] = []
    must_include_tasks: list[str] = []
    must_not_combine: list[list[str]] = []

    for f in high_confidence_findings:
        constraint_str = f"[{f.finding_type}] {f.description}"

        # --- Task-level constraints ---
        if f.task_index >= 0:
            existing = task_constraints.get(f.task_index, "")
            task_constraints[f.task_index] = (existing + "; " + constraint_str).lstrip("; ")

        # --- Structural constraints (per finding_type) ---
        if f.finding_type == "missing_entrypoint":
            structural_constraints.append(constraint_str)
            must_include_tasks.append(
                "A task that writes a runnable entrypoint file (main.py or app.py) "
                "that imports all required components and starts the application. "
                "This file must contain an `if __name__ == '__main__':` block or "
                "call a start_server() / main() function."
            )

        elif f.finding_type == "broken_dependency":
            # Extract the broken dependency name from evidence if possible
            structural_constraints.append(constraint_str)
            must_include_tasks.append(
                f"A task that implements the missing dependency referenced in: {f.evidence[:120]}. "
                "Ensure every import in every component resolves to an actual file in the plan."
            )

        elif f.finding_type == "import_cycle":
            structural_constraints.append(constraint_str)
            must_not_combine.append(["cyclic_import", "dependency"])
            must_include_tasks.append(
                "Restructure component dependencies to eliminate circular imports. "
                "Move shared logic to a new utility module that no other component imports from."
            )

        elif f.finding_type == "dead_component":
            structural_constraints.append(constraint_str)
            must_include_tasks.append(
                f"Remove or integrate the orphaned component described in: {f.description[:120]}. "
                "Every component must either be imported by another component or be the entrypoint."
            )

        elif f.finding_type == "incomplete_artifact_set":
            structural_constraints.append(constraint_str)
            must_include_tasks.append(
                "Ensure the artifact set is complete and runnable. "
                "A user must be able to execute the application with a single `python main.py` "
                "or `python app.py` command after all tasks complete."
            )

        elif f.finding_type == "context_mismatch":
            # Task-level: already added to task_constraints above
            if f.task_index >= 0 and f.task_index < len(plan.tasks):
                task_constraints[f.task_index] = (
                    task_constraints.get(f.task_index, "") +
                    f"; Ensure output format matches what consuming tasks expect"
                ).lstrip("; ")

        elif f.finding_type == "missing_input":
            if f.task_index >= 0:
                structural_constraints.append(constraint_str)
                must_include_tasks.append(
                    f"Add a task that produces the missing input required by task {f.task_index}: "
                    f"{f.evidence[:120]}"
                )

        elif f.finding_type == "orphaned_output":
            # Warn-only in practice -- don't add must_include but note it
            structural_constraints.append(constraint_str)

        elif f.finding_type == "wrong_agent":
            if f.task_index >= 0 and f.task_index < len(plan.tasks):
                task = plan.tasks[f.task_index]
                agent = task.declared_agent or task.suggested_agent
                if agent == "code":
                    combo = ["file", "code"]
                    if combo not in must_not_combine:
                        must_not_combine.append(combo)
                structural_constraints.append(constraint_str)

        else:
            # Unknown finding_type: add as generic structural constraint
            structural_constraints.append(constraint_str)

    return ConstraintManifest(
        task_constraints=task_constraints,
        structural_constraints=structural_constraints,
        must_include_tasks=must_include_tasks,
        must_not_combine=must_not_combine,
    )
