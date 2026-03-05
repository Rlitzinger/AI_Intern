import uuid
from ...schemas import (
    PlanSchema, RequestSchema, RedTeamFinding,
    CouncilVerdict, ConstraintManifest
)
from ...storage import save_critique_event, mark_findings_survived
from ...config import settings
from .executor import ExecutorAgent
from .integrator import IntegratorAgent
from .minimalist import MinimalistAgent
from .cross_exam import CrossExamination
from .blue_team import BlueTeamDefense
from .synthesis import compute_confidence, build_constraint_manifest, BLOCK_THRESHOLD, NOISE_THRESHOLD


class RedTeamCouncil:
    """
    5-round adversarial planning review.
    Drop-in replacement for CritiqueAgent.

    Usage:
        verdict, tokens = RedTeamCouncil.review(plan, request)
    """

    @classmethod
    def review(
        cls,
        plan: PlanSchema,
        request: RequestSchema,
        cycle: int = 0,
        session_id: str = None,
    ) -> tuple[CouncilVerdict, int]:
        # session_id links original plan + re-plan for Historian
        if session_id is None:
            session_id = str(uuid.uuid4())

        total_tokens = 0

        # === ROUND 1: Independent Red Team ===
        print(f"    [R1] Executor...")
        exec_findings, t1 = ExecutorAgent.find_issues(plan)
        print(f"    [R1] Integrator...")
        integ_findings, t2 = IntegratorAgent.find_issues(plan)
        print(f"    [R1] Minimalist...")
        mini_findings, t3 = MinimalistAgent.find_issues(plan)
        total_tokens += t1 + t2 + t3

        # CRITICAL: Assign global original_index immediately, before any filtering
        all_r1 = exec_findings + integ_findings + mini_findings
        for i, f in enumerate(all_r1):
            f.original_index = i

        cls._log_findings(session_id, plan.plan_id, request.content, cycle, all_r1)
        print(f"    [R1] {len(all_r1)} total findings")

        if not all_r1:
            print(f"    [R1] No issues -- approved")
            return CouncilVerdict(
                approved=True, confidence_scores={},
                high_confidence_findings=[], constraint_manifest=None,
                rounds_used=1, total_tokens=total_tokens, session_id=session_id,
                all_findings=[], cross_exam_responses=[], blue_team_responses=[],
            ), total_tokens

        # === ROUND 2: Cross-Examination ===
        # All three agents review the SAME global all_r1 list
        # Each prompt filters to "findings not from your agent" in the prompt text
        print(f"    [R2] Cross-examination...")
        exec_cross, t4  = CrossExamination.run("executor", all_r1, plan)
        integ_cross, t5 = CrossExamination.run("integrator", all_r1, plan)
        mini_cross, t6  = CrossExamination.run("minimalist", all_r1, plan)
        total_tokens += t4 + t5 + t6
        all_cross = exec_cross + integ_cross + mini_cross

        cls._log_cross_exam(session_id, plan.plan_id, request.content, cycle, all_cross)

        # Filter: remove findings disputed by >= 1 other agent
        surviving = cls._filter_disputed(all_r1, all_cross, min_disputes=1)
        print(f"    [R2] {len(surviving)}/{len(all_r1)} findings survived")

        if not surviving:
            return CouncilVerdict(
                approved=True, confidence_scores={},
                high_confidence_findings=[], constraint_manifest=None,
                rounds_used=2, total_tokens=total_tokens, session_id=session_id,
                all_findings=all_r1, cross_exam_responses=all_cross, blue_team_responses=[],
            ), total_tokens

        # === ROUND 3: Blue Team Defense ===
        print(f"    [R3] Blue Team...")
        blue_responses, t7 = BlueTeamDefense.run(plan, request, surviving)
        total_tokens += t7

        cls._log_blue_team(session_id, plan.plan_id, request.content, cycle, blue_responses, surviving)

        # === ROUND 4: Synthesis (deterministic) ===
        # Uses original_index throughout -- no re-indexing after filtering
        confidence_scores: dict[int, float] = {}
        for f in surviving:
            score = compute_confidence(f, all_r1, all_cross, blue_responses)
            confidence_scores[f.original_index] = score

        high_confidence = [
            f for f in surviving
            if confidence_scores.get(f.original_index, 0) >= BLOCK_THRESHOLD
        ]

        print(f"    [R4] {len(high_confidence)} high-confidence findings (threshold={BLOCK_THRESHOLD})")
        cls._log_synthesis(session_id, plan.plan_id, request.content, cycle, surviving, confidence_scores)

        if not high_confidence:
            return CouncilVerdict(
                approved=True, confidence_scores=confidence_scores,
                high_confidence_findings=[], constraint_manifest=None,
                rounds_used=4, total_tokens=total_tokens, session_id=session_id,
                all_findings=all_r1, cross_exam_responses=all_cross, blue_team_responses=blue_responses,
            ), total_tokens

        # === ROUND 5: Constraint Manifest ===
        manifest = build_constraint_manifest(high_confidence, plan)

        return CouncilVerdict(
            approved=False, confidence_scores=confidence_scores,
            high_confidence_findings=high_confidence, constraint_manifest=manifest,
            rounds_used=5, total_tokens=total_tokens, session_id=session_id,
            all_findings=all_r1, cross_exam_responses=all_cross, blue_team_responses=blue_responses,
        ), total_tokens

    @staticmethod
    def _filter_disputed(
        findings: list[RedTeamFinding],
        responses,
        min_disputes: int = 1,
    ) -> list[RedTeamFinding]:
        """
        Remove findings disputed by >= min_disputes other agents.
        Uses original_index for lookup.
        Default min_disputes=1: any single dispute removes the finding.
        """
        dispute_counts: dict[int, int] = {}
        for r in responses:
            if r.verdict == "dispute":
                dispute_counts[r.finding_index] = dispute_counts.get(r.finding_index, 0) + 1
        return [f for f in findings if dispute_counts.get(f.original_index, 0) < min_disputes]

    # === Logging helpers ===

    @staticmethod
    def _log_findings(session_id, plan_id, req_content, cycle, findings):
        for f in findings:
            save_critique_event(
                session_id=session_id, plan_id=plan_id, request_content=req_content,
                cycle=cycle, round_name=f.agent, agent=f.agent,
                content=f.description, finding_type=f.finding_type,
                task_index=f.task_index, severity=f.severity,
            )

    @staticmethod
    def _log_cross_exam(session_id, plan_id, req_content, cycle, responses):
        for r in responses:
            save_critique_event(
                session_id=session_id, plan_id=plan_id, request_content=req_content,
                cycle=cycle, round_name="cross_exam", agent="cross_exam",
                content=r.reasoning, task_index=r.finding_index, verdict=r.verdict,
            )

    @staticmethod
    def _log_blue_team(session_id, plan_id, req_content, cycle, responses, findings):
        for r in responses:
            save_critique_event(
                session_id=session_id, plan_id=plan_id, request_content=req_content,
                cycle=cycle, round_name="blue_team", agent="blue_team",
                content=r.rebuttal or "No rebuttal -- finding confirmed",
                task_index=r.finding_index,
                verdict="rebut" if r.can_rebut else "concede",
            )

    @staticmethod
    def _log_synthesis(session_id, plan_id, req_content, cycle, findings, scores):
        for f in findings:
            save_critique_event(
                session_id=session_id, plan_id=plan_id, request_content=req_content,
                cycle=cycle, round_name="synthesis", agent="synthesis",
                content=f.description, finding_type=f.finding_type,
                task_index=f.task_index, severity=f.severity,
                confidence_score=scores.get(f.original_index),
            )
