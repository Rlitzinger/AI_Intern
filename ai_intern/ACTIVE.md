# ACTIVE TASK: Replace CritiqueAgent with Adversarial Red Team Council

## Overview

Replace the single `CritiqueAgent` with a multi-round adversarial council.
The current critique reads the plan and reports issues — literary criticism.
The new system simulates execution, debates findings, forces the planner to defend
its own work, then produces a confidence-scored constraint manifest for revision.

Also: add `plan_critique_history` SQLite table for the Historian agent (tomorrow).

---

## Architecture: 5-Round Adversarial Council

```
Plan
 |
 |-- Round 1: Independent Red Team (3 sequential LLM calls, logically parallel)
 |     |-- ExecutorAgent    -- traces data flow task-by-task
 |     |-- IntegratorAgent  -- reasons about artifact graph
 |     +-- MinimalistAgent  -- counterfactual simplicity check
 |
 |-- Round 2: Cross-Examination (3 sequential LLM calls)
 |     Each agent reviews ALL findings NOT from itself (global indices preserved)
 |     Responds: CONFIRM / DISPUTE / EXTEND per finding
 |
 |-- Round 3: Blue Team Defense (1 LLM call)
 |     Planner defends plan against surviving findings
 |     Prompted as skeptical self-critic, not advocate
 |
 |-- Round 4: Synthesis (deterministic, zero LLM cost)
 |     Confidence score per finding. Severity-gated:
 |       warn findings capped at 0.60 (below BLOCK_THRESHOLD)
 |       block findings can reach 1.0
 |     Only block-severity findings above 0.65 trigger revision
 |
 +-- Round 5: Constraint Manifest + Re-plan (conditional, max 2 cycles)
       ConstraintManifest injected as hard rules into re-planner
       session_id links original and re-planned runs for Historian
```

---

## Files to Create

```
ai_intern/planning/red_team/
|-- __init__.py          (exports RedTeamCouncil)
|-- utils.py             (shared _format_plan, _format_artifacts)
|-- executor.py          (ExecutorAgent)
|-- integrator.py        (IntegratorAgent)
|-- minimalist.py        (MinimalistAgent)
|-- cross_exam.py        (CrossExamination)
|-- blue_team.py         (BlueTeamDefense)
|-- synthesis.py         (compute_confidence, build_constraint_manifest)
+-- council.py           (RedTeamCouncil -- top-level orchestrator)
```

## Files to Modify

- `ai_intern/schemas.py` -- add new schemas (RedTeamFinding, etc.)
- `ai_intern/storage.py` -- add critique history table + helpers
- `ai_intern/orchestration/orchestrator.py` -- swap CritiqueAgent for RedTeamCouncil
- `ai_intern/planning/__init__.py` -- export RedTeamCouncil
- `plan.py` -- update Stage 1.5 display

---

## Pre-flight Check (do this first)

Before writing any new code, verify that `TaskSchema` in `ai_intern/schemas.py` has
these fields. If any are missing, add them with appropriate defaults before proceeding:

```python
declared_agent: Optional[str] = None      # agent explicitly declared by decomposer
suggested_agent: Optional[str] = None     # agent inferred by heuristic
original_goal: Optional[str] = None       # goal before retry feedback appended
output_contract: Optional[Any] = None     # OutputContract or dict (spec-driven)
error_history: list[dict] = Field(default_factory=list)
```

These are referenced by `_format_plan()` in utils.py and by existing orchestrator code.
If they already exist (likely — they're used in the uploaded orchestrator.py), skip this.

---

## Detailed Implementation

---

### 1. New Pydantic Schemas  (add to `ai_intern/schemas.py`)

```python
from typing import Literal, Optional

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
```

---

### 2. SQLite Changes  (`ai_intern/storage.py`)

Add to existing `save_plan_to_sqlite`, call `ensure_critique_history_table(conn)`
before the plan INSERT so the table always exists.

```python
def ensure_critique_history_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plan_critique_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,    -- links original plan + re-plan cycles
            plan_id TEXT NOT NULL,
            request_content TEXT,
            cycle INTEGER NOT NULL DEFAULT 0,
            round TEXT NOT NULL,         -- executor|integrator|minimalist|cross_exam|blue_team|synthesis
            agent TEXT NOT NULL,
            finding_type TEXT,
            task_index INTEGER,
            severity TEXT,
            content TEXT NOT NULL,
            confidence_score REAL,
            verdict TEXT,                -- confirm|dispute|extend|rebut|concede|null
            survived_revision INTEGER DEFAULT NULL,  -- 1=yes 0=no null=unknown
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

def save_critique_event(
    session_id: str,
    plan_id: str,
    request_content: str,
    cycle: int,
    round_name: str,
    agent: str,
    content: str,
    finding_type: str = None,
    task_index: int = None,
    severity: str = None,
    confidence_score: float = None,
    verdict: str = None,
):
    with get_db_connection() as conn:
        ensure_critique_history_table(conn)
        conn.execute("""
            INSERT INTO plan_critique_history
                (session_id, plan_id, request_content, cycle, round, agent,
                 finding_type, task_index, severity, content, confidence_score, verdict)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, plan_id, request_content, cycle, round_name, agent,
            finding_type, task_index, severity, content, confidence_score, verdict
        ))

def mark_findings_survived(session_id: str, cycle: int, survived: bool):
    """Mark whether findings from this cycle persisted into the next."""
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE plan_critique_history
            SET survived_revision = ?
            WHERE session_id = ? AND cycle = ?
        """, (1 if survived else 0, session_id, cycle))
```

Also call `ensure_critique_history_table(conn)` inside `save_plan_to_sqlite`
before the existing INSERT so the table is created on first run automatically.

---

### 3. Shared Utilities  (`red_team/utils.py`)

Both executor and integrator need these. Keep them here, import from everywhere.

```python
from ...schemas import PlanSchema

def format_plan_for_prompt(plan: PlanSchema) -> str:
    """Format all tasks with agent, goal, and contract for LLM prompts."""
    lines = []
    for t in plan.tasks:
        agent = t.declared_agent or t.suggested_agent or "unknown"
        lines.append(f"Task {t.task_order} [{agent}]: {t.goal[:120]}")
        if isinstance(t.output_contract, dict):
            lines.append(f"  output_file : {t.output_contract.get('output_file', 'none')}")
            lines.append(f"  interface   : {t.output_contract.get('public_interface', 'none')}")
            deps = t.output_contract.get("depends_on", [])
            if deps:
                lines.append(f"  depends_on  : {deps}")
        elif t.output_contract is not None:
            oc = t.output_contract
            lines.append(f"  output_type : {oc.output_type}")
            lines.append(f"  output_format: {oc.output_format}")
            lines.append(f"  consumed_by : {oc.required_by_tasks}")
    return "\n".join(lines)

def format_artifacts_for_prompt(plan: PlanSchema) -> str:
    """Format only spec-driven artifacts (component files + interfaces)."""
    lines = []
    for t in plan.tasks:
        if isinstance(t.output_contract, dict):
            name = t.output_contract.get("component_name", "?")
            file = t.output_contract.get("output_file", "?")
            iface = t.output_contract.get("public_interface", "?")
            deps = t.output_contract.get("depends_on", [])
            line = f"{file}  ({name}): {iface}"
            if deps:
                line += f"  -- imports: {deps}"
            lines.append(line)
    if not lines:
        return "(No spec-driven artifacts -- non-app-scale plan)"
    return "\n".join(lines)
```

---

### 4. ExecutorAgent  (`red_team/executor.py`)

```python
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
```

---

### 5. IntegratorAgent  (`red_team/integrator.py`)

```python
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
```

---

### 6. MinimalistAgent  (`red_team/minimalist.py`)

```python
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
        if plan.tasks:
            original_request = plan.tasks[0].original_goal or plan.tasks[0].goal

        task_repr = format_plan_for_prompt(plan)

        prompt = f"""Original request: "{original_request}"

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
```

---

### 7. Cross-Examination  (`red_team/cross_exam.py`)

CRITICAL: All three agents review the SAME global finding list with SAME indices.
Each is filtered to "findings not from your own agent" in the prompt, not in code.
This ensures finding_index in all responses refers to the same global list.

```python
from ...schemas import PlanSchema, RedTeamFinding, CrossExamResult
from ...llm import call_ollama_structured
from ...config import settings
from .utils import format_plan_for_prompt

class CrossExamination:
    model = settings.PLANNING_MODEL
    temperature = 0.1

    SYSTEM = """You are reviewing adversarial findings about a plan.
Filter noise from signal.
CONFIRM only with INDEPENDENT evidence -- not just agreement.
DISPUTE only if you can show the finding misreads the plan specifically.
EXTEND only if the finding reveals a deeper problem not yet surfaced.
Do not be agreeable. False confirmations are worse than missed issues.
Respond to every finding NOT from your own agent."""

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

CONFIRM = you have independent evidence for the same issue
DISPUTE = you can show the finding misreads the plan
EXTEND  = finding is correct AND implies a deeper problem

You must respond to all {len(other_findings)} findings listed above.
Return JSON: {{"responses": [list of CrossExamResponse objects]}}"""

        result, tokens = call_ollama_structured(
            model=cls.model, prompt=prompt, system=cls.SYSTEM,
            response_schema=CrossExamResult, temperature=cls.temperature
        )
        return result.responses, tokens
```

---

### 8. Blue Team Defense  (`red_team/blue_team.py`)

finding_index here also references all_r1_findings global indices.
Only surviving findings are passed in the prompt, but their indices are preserved.

```python
from ...schemas import PlanSchema, RequestSchema, RedTeamFinding, BlueTeamResult
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
```

---

### 9. Synthesis  (`red_team/synthesis.py`)

Deterministic. No LLM. Severity-gated scoring.

```python
from ...schemas import RedTeamFinding, CrossExamResponse, BlueTeamResponse, ConstraintManifest, PlanSchema

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
```

---

### 10. Council Orchestrator  (`red_team/council.py`)

Global index discipline: `original_index` is set on every finding immediately after R1.
All subsequent rounds use `original_index` for reference -- never re-index.
`session_id` is generated once and passed through both review cycles for Historian linking.

```python
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
        # (more aggressive than original spec -- reduces noise at cost of some signal)
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
        responses: list[CrossExamResponse],
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
```

---

### 11. `red_team/__init__.py`

```python
from .council import RedTeamCouncil

__all__ = ["RedTeamCouncil"]
```

---

### 12. Orchestrator Changes  (`orchestrator.py`)

```python
# Add imports at top (remove CritiqueAgent import):
from ..planning.red_team.council import RedTeamCouncil
from ..storage import mark_findings_survived
# Remove: from ..planning.critic import CritiqueAgent

# Replace Stage 1.5 block entirely:

# === STAGE 1.5: RED TEAM COUNCIL ===
print(f"\nStage 1.5: Red Team Council")

# session_id persists across re-plan cycles for Historian linking
import uuid as _uuid
council_session_id = str(_uuid.uuid4())

council_verdict, council_tokens = RedTeamCouncil.review(
    plan, request, cycle=0, session_id=council_session_id
)
plan.token_usage += council_tokens

if not council_verdict.approved and council_verdict.constraint_manifest:
    n = len(council_verdict.high_confidence_findings)
    print(f"\n   {n} blocking issue(s) -- replanning with constraints...")

    plan = self._replan_with_manifest(request, council_verdict.constraint_manifest)
    plan.token_usage += council_tokens  # approximate

    council_verdict_2, council_tokens_2 = RedTeamCouncil.review(
        plan, request, cycle=1, session_id=council_session_id
    )
    plan.token_usage += council_tokens_2

    # Mark whether first-cycle findings survived into second cycle
    mark_findings_survived(council_session_id, cycle=0, survived=not council_verdict_2.approved)

    if not council_verdict_2.approved:
        print(f"   Issues remain after revision -- proceeding with warnings")

save_plan_to_sqlite(plan)
print(f"   Saved plan after council review")

# Remove _replan_with_critique method -- replaced below
```

Add `_replan_with_manifest` method to `Orchestrator` class:

```python
def _replan_with_manifest(
    self, request: RequestSchema, manifest: ConstraintManifest
) -> PlanSchema:
    """Re-plan with structured constraints. Not freeform feedback."""
    lines = []
    for task_idx, c in manifest.task_constraints.items():
        lines.append(f"- Task {task_idx} must satisfy: {c}")
    for c in manifest.structural_constraints:
        lines.append(f"- STRUCTURAL: {c}")
    for goal in manifest.must_include_tasks:
        lines.append(f"- MUST INCLUDE A TASK FOR: {goal}")
    for pair in manifest.must_not_combine:
        if len(pair) == 2:
            lines.append(f"- NEVER combine [{pair[0]}] and [{pair[1]}] in one task")

    augmented_content = (
        f"{request.content}\n\n"
        "HARD PLANNING CONSTRAINTS (non-negotiable, from adversarial review):\n"
        + "\n".join(lines)
    )
    augmented_request = RequestSchema(
        request_id=request.request_id,
        content=augmented_content
    )
    if settings.USE_HIERARCHICAL_PLANNING:
        return HierarchicalPlanner.create_plan(augmented_request)
    else:
        return PlanningAgent.create_plan(augmented_request)
```

---

### 13. plan.py Display Updates

Replace the Stage 1.5 section:

```python
from ai_intern.planning.red_team.council import RedTeamCouncil

_section("Stage 1.5 -- Red Team Council  (Adversarial Review)")

if len(plan.tasks) < 2:
    print(f"  Skipped: {len(plan.tasks)} task -- council only runs on 2+ task plans")
else:
    print(f"  Reviewing {len(plan.tasks)}-task plan...")
    print()

    council_verdict, council_tokens = RedTeamCouncil.review(plan, req)
    plan.token_usage += council_tokens

    status = "APPROVED" if council_verdict.approved else "REJECTED"
    print()
    print(f"  Verdict       : {status}")
    print(f"  Rounds used   : {council_verdict.rounds_used} / 5")
    print(f"  Tokens        : {council_tokens:,}")
    print(f"  R1 findings   : {len(council_verdict.all_findings)}")

    if council_verdict.confidence_scores:
        print()
        print(f"  Confidence scores (surviving findings):")
        for orig_idx, score in sorted(council_verdict.confidence_scores.items()):
            f = next((x for x in council_verdict.all_findings if x.original_index == orig_idx), None)
            label = f"{f.agent}/{f.finding_type}" if f else "unknown"
            filled = int(score * 10)
            bar = "X" * filled + "." * (10 - filled)
            blocked = "  [BLOCK]" if score >= 0.65 else ""
            print(f"    [{orig_idx}] {bar} {score:.2f}  {label}{blocked}")

    if council_verdict.high_confidence_findings:
        print()
        print(f"  High-confidence issues:")
        for f in council_verdict.high_confidence_findings:
            score = council_verdict.confidence_scores.get(f.original_index, 0)
            print(f"    [{f.original_index}] Task {f.task_index:2d}  {score:.2f}  [{f.finding_type}]")
            print(f"           {f.description[:80]}...")

    if council_verdict.constraint_manifest:
        m = council_verdict.constraint_manifest
        total_constraints = (
            len(m.task_constraints) + len(m.structural_constraints) +
            len(m.must_include_tasks) + len(m.must_not_combine)
        )
        print()
        print(f"  Constraint manifest ({total_constraints} constraints):")
        for tidx, c in m.task_constraints.items():
            print(f"    Task {tidx}: {c[:80]}...")
        for c in m.structural_constraints:
            print(f"    Structural: {c[:80]}...")
        for g in m.must_include_tasks:
            print(f"    Must include: {g[:80]}...")
```

---

## Success Criteria

- [ ] `RedTeamCouncil.review()` runs end-to-end without crashing on the color web app example
- [ ] `original_index` is set on all findings immediately after R1, never changes
- [ ] All cross-exam and blue-team responses reference global `original_index` correctly
- [ ] `plan_critique_history` table created on first run
- [ ] Every round's output is logged to the table with correct `session_id`
- [ ] `survived_revision` updated after second cycle via `mark_findings_survived`
- [ ] `warn` findings cannot exceed `WARN_SCORE_CAP` (0.60) -- cannot trigger re-plan
- [ ] Only `block`-severity findings above 0.65 produce a `ConstraintManifest`
- [ ] `plan.py` displays confidence bars and constraint manifest correctly
- [ ] Council short-circuits cleanly at R1 (no issues) or R2 (all disputed)
- [ ] Old `CritiqueAgent` import removed from orchestrator; `critic.py` preserved
- [ ] `_replan_with_critique` removed from orchestrator; `_replan_with_manifest` added
- [ ] `red_team/__init__.py` exports `RedTeamCouncil`

---

## Constraints

- DO NOT delete `critic.py`
- DO NOT modify `HierarchicalPlanner`, `TaskClassifier`, or `SpecGenerator`
- DO NOT add async/await
- DO NOT change existing `PlanSchema` or `TaskSchema` field names
- All `call_ollama_structured` signatures identical to existing usage
- `ConstraintManifest.must_not_combine` is `list[list[str]]` (not tuple -- JSON compat)
- `RedTeamFinding.agent` is `Optional[str] = None` in schema (set server-side, not by LLM)
- `original_index` on `RedTeamFinding` is `Optional[int] = None` (set server-side)
- Cross-exam passes full `all_r1` list to all three agents -- filtering is in the prompt

---

## Key Design Decisions (for reference)

**Why `original_index` instead of list position?**
Filtering steps (_filter_disputed) remove findings from the list. After filtering,
positional indices shift. `original_index` is immutable from R1 onward, so all
cross-exam responses, blue-team responses, and confidence scores stay consistent.

**Why dispute threshold = 1 (not 2)?**
With 2 reviewing agents per finding, the max dispute count is 2. Using >=2 as
threshold means a finding needs both reviewers to dispute it -- very hard to reach,
so most findings survive regardless of quality. threshold=1 is more aggressive
but produces cleaner signal. Tune back to 2 after seeing first run results if too many
real findings are being filtered.

**Why warn findings capped below BLOCK_THRESHOLD?**
Without the cap, accumulated warn scores can exceed 0.65 and trigger a full replan.
A finding the system itself labeled as non-blocking should never be the sole cause
of a replan. The cap enforces this at the scoring level, not the filtering level.

**Why session_id instead of plan_id for Historian?**
A replan generates a new `plan_id`. Without `session_id`, the Historian cannot link
"original plan that failed council" with "revised plan that passed" -- they look like
unrelated plans. `session_id` is stable across cycles and is the joining key for
cross-cycle survival analysis.

---

## Historian Query (tomorrow's task -- for reference)

```sql
-- Find recurring failure patterns (high survival rate = planner keeps making this mistake)
SELECT
    finding_type,
    task_index,
    COUNT(*) as occurrences,
    AVG(survived_revision) as survival_rate,
    AVG(confidence_score) as avg_confidence
FROM plan_critique_history
WHERE survived_revision IS NOT NULL
  AND round IN ('executor', 'integrator', 'minimalist')
GROUP BY finding_type, task_index
HAVING occurrences >= 2
ORDER BY survival_rate DESC, occurrences DESC;
```