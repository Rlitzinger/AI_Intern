# ACTIVE: Tier 2 — Execution Quality Improvements

## Tier 1 Status: COMPLETE

Tier 1 (Grounded Planning) is fully implemented and verified:
- `test_tier1.py` passes 54/54 tests across 3 independent runs (stable)
- `test_planning.py` passes
- `test_app_generation.py` passes

### What Tier 1 Delivered
| Change | File | Status |
|--------|------|--------|
| EnvironmentScanner — pre-planning file/date/time scan | `planning/environment.py` (NEW) | Done |
| LLM-based `is_app_scale` (3-axis classifier) | `planning/classifier.py` | Done |
| Hybrid keyword safety net (`_keyword_is_app_scale`) | `planning/hierarchical.py` | Done |
| Agent annotations on all planned tasks (`suggested_agent`) | `schemas.py`, `hierarchical.py` | Done |
| Router respects `suggested_agent` (planner over keyword scorer) | `orchestration/routing.py` | Done |
| Environment context injected into all planning LLM calls | `hierarchical.py`, `spec_generator.py` | Done |
| Spec generator 2-component minimum + synthetic AppCLI fallback | `planning/spec_generator.py` | Done |

---

## Tier 2 Candidates — Execution Quality

The planning pipeline is now grounded and deliberate. The next bottleneck is execution quality:
the generated code sometimes fails validation and burns all 3 retries. Key improvement areas:

### Candidate A: Smarter Retry Prompting
**Problem**: On retry, the CodingAgent sees the raw error message but gets the same prompt structure.
**Fix**: Build a structured "correction prompt" that includes:
- The failing code (truncated)
- The exact error message
- A specific instruction: "Fix only the error above — do not rewrite everything"
- The test that failed (if available from ValidationAgent)

### Candidate B: Validation Error Classification
**Problem**: The validator retries on all failures equally, but some errors are unrecoverable
(missing external API, hardware requirement). Retrying wastes tokens.
**Fix**: Classify errors before retry:
- `syntax_error` → always retry (LLM can fix)
- `import_error` → check if stdlib vs third-party; retry only if stdlib
- `test_failure` → retry with test-focused correction prompt
- `timeout` → don't retry (inherent code logic issue)

### Candidate C: Context Carryover Quality
**Problem**: When a downstream task needs upstream results (e.g., CodingAgent needs research
data from ResearchAgent), the context is passed as a prose summary. The LLM sometimes ignores it.
**Fix**: Structure the context carryover: instead of raw text, pass `{"type": "research_result",
"key_values": {"protein_per_100g": 26, "fat_per_100g": 10}}` when the upstream task is research.

### Candidate D: Spec-Driven Task Goals
**Problem**: Component task goals like "Write `NoteStorage` class/module in `outputs/storage.py`
with public interface: save_note(note), load_note(id), list_notes(). Responsibility: ..."
are long but sometimes cause the LLM to miss the output file or interface.
**Fix**: Add structured goal template in `_decompose_task_from_spec` with clear section headers:
```
TASK: Code generation (component)
FILE: outputs/storage.py
CLASS: NoteStorage
INTERFACE: save_note(note) -> None, load_note(id) -> dict, list_notes() -> list
RESPONSIBILITY: Saves notes to a JSON file on disk
DEPENDS ON: (none)
```

---

## Notes for Claude Code

- Run `python test_tier1.py` after any changes to verify Tier 1 remains intact
- `test_app_generation.py` is intentionally allowed 2 attempts in `test_tier1.py` because
  LLM code generation is non-deterministic — a single validation failure is not a regression
- The `suggested_agent` field on `TaskSchema` is now set for all tasks — check it before adding
  agent routing logic anywhere in the codebase
- The `_keyword_is_app_scale` safety net uses conservative keywords only
  (`" app"`, `"application"`, `"platform"`, `"full stack"`, `"full-stack"`) — do not expand it
