from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Any, Literal, NamedTuple, Optional
from uuid import uuid4


class OutputContract(BaseModel):
    """
    Specifies what a task produces and who consumes it.
    Generated during planning, used by CritiqueAgent to detect
    context dependency failures before execution.
    """
    output_type: Literal["python_code", "prose", "structured_data", "file_path", "none"]
    output_format: str  # e.g. "dict with keys: protein_g, fat_g as floats"
    required_by_tasks: list[int] = []  # task_order values that consume this output


class CritiqueIssue(BaseModel):
    """A single issue found during plan critique."""
    severity: Literal["blocking", "warning"]
    issue_type: Literal[
        "context_mismatch",   # Task expects format previous task won't produce
        "missing_task",       # Implicit step not in plan
        "wrong_agent",        # TaskRouter will misroute this task
        "over_decomposed",    # Tasks that could/should be merged
        "bad_contract",       # Output contract is vague or incorrect
    ]
    task_order: int           # Which task has the issue (-1 = plan-level)
    description: str
    suggested_fix: str


class CritiqueResult(BaseModel):
    """Full critique of a plan."""
    approved: bool
    issues: list[CritiqueIssue] = []
    revised_goals: dict[int, str] = {}  # task_order -> new goal string
    critique_reasoning: str


class SubtaskSpec(BaseModel):
    """
    Structured subtask produced by the decomposer.
    Replaces free-form goal strings with explicit agent routing intent.
    """
    agent_type: Literal["code", "research", "file"]
    goal: str                           # Free-form natural language - rich description for the agent
    input_files: list[str] = []         # Files from user_data/ this task needs (must exist)
    output_file: Optional[str] = None   # File this task writes to outputs/ (if any)
    depends_on: list[int] = []          # Indices of subtasks this depends on (0-indexed within siblings)


class TaskOutput(BaseModel):
    """Structured output from a task, beyond just raw text."""
    output_type: Literal["text", "code", "data", "file_path", "error"] = "text"
    raw_result: str = ""
    data_summary: Optional[str] = None  # Brief summary for context passing
    column_names: Optional[list[str]] = None  # For CSV/data tasks
    row_count: Optional[int] = None
    file_path: Optional[str] = None
    key_values: Optional[dict] = None  # Extracted key-value pairs


class RequestSchema(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    schema_version: str = "0.0.1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskSchema(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    plan_id: str
    task_order: int
    goal: str
    original_goal: str = ""  # Immutable copy of the original goal (#9)
    status: Literal["pending", "executing", "validating", "complete", "validated", "failed", "skipped"] = "pending"
    retry_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    schema_version: str = "0.0.1"
    result: Optional[str] = None  # Generated code / raw text
    task_output: Optional[TaskOutput] = None  # Structured output (#5)
    test_code: Optional[str] = None  # Generated tests
    error_message: Optional[str] = None
    error_history: list[dict] = []  # [{attempt, error, test_code}] (#9)
    depends_on: list[int] = []  # Task order values this depends on (#19)
    output_contract: Optional[Any] = None  # dict (spec-driven: component_name, output_file, public_interface) or OutputContract (planning contracts)
    suggested_agent: Optional[str] = None   # "code", "research", "file", "analysis" — set by planner
    declared_agent: Optional[Literal["code", "research", "file"]] = None  # Set from SubtaskSpec
    error_category: Optional[str] = None   # ErrorCategory value from error_classifier


class PlanSchema(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    tasks: list[TaskSchema] = []
    status: Literal["planning", "critiquing", "executing", "validating", "complete", "failed"] = "planning"
    retry_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    schema_version: str = "0.0.1"
    token_usage: int = 0
    final_answer: Optional[str] = None  # (#6) Synthesized final answer
    app_spec: Optional[dict] = None     # Serialised AppSpec, if generated
    workspace_root: Optional[str] = None  # Path to this plan's workspace directory


class AgentResult(NamedTuple):
    """Standard return type for agent execute_task methods."""
    task: TaskSchema
    tokens_used: int


# ---------------------------------------------------------------------------
# Red Team Council schemas
# ---------------------------------------------------------------------------

class RedTeamFinding(BaseModel):
    """A single finding from one red team agent."""
    # NOTE: agent is NOT in the LLM schema -- set server-side after the call.
    # It is Optional here so Pydantic doesn't reject LLM output that omits it.
    agent: Optional[str] = None
    task_index: int = -1      # -1 = plan-level finding, not task-specific
    finding_type: str         # see per-agent prompt for valid values
    description: str
    evidence: str
    severity: Literal["block", "warn"]
    # Original index in all_r1_findings -- set by council, not LLM
    original_index: Optional[int] = None


class FindingsResult(BaseModel):
    """Wrapper schema for LLM findings response."""
    findings: list[RedTeamFinding]


class CrossExamResponse(BaseModel):
    """One agent's response to a finding (references global finding index)."""
    finding_index: int        # Index into all_r1_findings (global, not local)
    verdict: Literal["confirm", "dispute", "extend"]
    reasoning: str
    additional_evidence: str = ""


class CrossExamResult(BaseModel):
    responses: list[CrossExamResponse]


class BlueTeamResponse(BaseModel):
    """Planner's rebuttal attempt. finding_index refs all_r1_findings."""
    finding_index: int        # Index into all_r1_findings (global)
    can_rebut: bool
    rebuttal: str = ""


class BlueTeamResult(BaseModel):
    responses: list[BlueTeamResponse]


class ConstraintManifest(BaseModel):
    """Structured re-planning constraints. NOT freeform text."""
    task_constraints: dict[int, str]     # task_index -> constraint string
    structural_constraints: list[str]
    must_include_tasks: list[str]
    must_not_combine: list[list[str]]    # list[list[str]] not list[tuple] -- JSON compat


class CouncilVerdict(BaseModel):
    """Final output of the Red Team Council."""
    approved: bool
    confidence_scores: dict[int, float]        # finding original_index -> score
    high_confidence_findings: list[RedTeamFinding]
    constraint_manifest: Optional[ConstraintManifest] = None
    rounds_used: int
    total_tokens: int
    session_id: str                            # links original + re-plan cycles
    all_findings: list[RedTeamFinding]         # all R1 findings with original_index set
    cross_exam_responses: list[CrossExamResponse]
    blue_team_responses: list[BlueTeamResponse]
