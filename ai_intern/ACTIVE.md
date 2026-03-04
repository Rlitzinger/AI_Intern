# ACTIVE TASK: SpecGenerator — Intent-First Redesign

## Problem Statement

The SpecGenerator currently produces architecturally valid but functionally hollow specs.
Given "build a web app that allows users to turn their written art ideas into a picture",
it returns a standard CRUD stack (Store, Model, Handler, Template, Server) with no image
generation component. The user's primary objective — turning text into a picture — never
appears in any component.

Root cause: The generator does one LLM call that simultaneously extracts intent AND designs
architecture. At 7B scale, the model pattern-matches to "web app → standard stack" before
it fully processes what the app must DO. Intent gets lost inside structural generation.

The fix is NOT a prompt addition. It is a structural split: two focused LLM calls where
the second is explicitly grounded by the output of the first.

---

## Mental Model

### Current (broken)
```
User request
  → [LLM Call: "generate AppSpec"] ← does everything at once
  → AppSpec with components         ← architecture, no intent guarantee
```

### Target
```
User request
  → [LLM Call 1: "extract intent"]  ← focused, flat schema
  → IntentSpec                       ← primary action, core capability, constraints
      |
      ↓ (injected into prompt)
  → [LLM Call 2: "design components grounded in intent"]
  → AppSpec                          ← architecture derived FROM intent
```

IntentSpec is never exposed to the orchestrator or agents. It is internal scaffolding
that exists only to ground Call 2. Think of it as a chain-of-thought that is structured
rather than free-form.

---

## Why Two Calls Instead of a Better Prompt

One-call approaches fail here for a specific reason: the model must satisfy two competing
objectives simultaneously — understand what the user wants AND produce a valid nested JSON
structure. At 7B scale these compete for the same attention budget. Structural generation
wins because it is the dominant pattern in training data.

Splitting the calls eliminates the competition:
- Call 1 is unconstrained on structure (flat schema, short output) — the model can focus
  entirely on understanding intent
- Call 2 receives the intent as a completed fact in the prompt — it cannot ignore it
  because it is the first thing the model reads

This is strictly more reliable than any prompt engineering approach because it makes
intent-ignorance architecturally impossible, not just instructionally discouraged.

Token cost: ~400-600 extra tokens per app-scale request. Acceptable given that this path
only fires for app-scale requests, and the quality delta is significant.

---

## New Schema: IntentSpec

### File: `ai_intern/planning/spec_generator.py`

Add `IntentSpec` BEFORE `AppSpec`. This is Call 1's response schema.

```python
class IntentSpec(BaseModel):
    """
    Extracted user intent. Internal scaffolding only — never stored on plan or task.
    Exists only to ground the component design in Call 2.
    """
    primary_action: str
    # The ONE thing the user must be able to do, written as an active verb phrase.
    # Examples:
    #   "turn written text into a generated image"
    #   "track daily workouts and view progress over time"
    #   "store and retrieve personal notes with search"
    # NOT "build a web app" — that describes the container, not the action.

    core_capability: str
    # The specific technical mechanism that makes primary_action possible.
    # Name the specific technology or algorithm required.
    # Examples:
    #   "call an image generation API (e.g. Stable Diffusion, DALL-E, or Replicate)"
    #   "persist workout records to SQLite and compute aggregate stats"
    #   "full-text search over stored note content"
    # If the request doesn't specify, name the most appropriate option and note it's assumed.

    capability_owner: str
    # The PascalCase name of the component that must own core_capability.
    # This component MUST appear in the final AppSpec.
    # Examples: "ImageGenerator", "WorkoutTracker", "NoteSearchEngine"

    constraints: list[str]
    # Explicit constraints from the user's request (platform, language, format, etc.)
    # Leave empty list if none stated. Do NOT invent constraints.
    # Examples: ["web app (HTTP server)", "Python only", "no external database"]
```

Keep this schema flat. No nested objects. Four fields, all strings or list[str].
The model must fill this reliably in one shot — complexity here defeats the purpose.

---

## Updated SpecGenerator: Two-Phase `generate()`

### File: `ai_intern/planning/spec_generator.py`

Replace the current `generate()` method with the following structure.
Do not change `AppSpec`, `ComponentSpec`, or any method not listed here.

### `generate()` — orchestrates both phases

```python
@classmethod
def generate(cls, task: TaskSchema, context: dict) -> tuple[AppSpec, int]:
    """
    Two-phase spec generation.
    Phase 1: Extract user intent (IntentSpec) — flat schema, low temperature.
    Phase 2: Design components grounded in intent — IntentSpec injected as ground truth.
    """
    total_tokens = 0

    # Phase 1: Intent extraction
    intent, tokens_1 = cls._extract_intent(task.goal)
    total_tokens += tokens_1

    # Log intent so failures are visible immediately
    print(f"      🎯 Primary action:   {intent.primary_action}")
    print(f"      ⚙️  Core capability:  {intent.core_capability}")
    print(f"      🏗️  Capability owner: {intent.capability_owner}  <- must appear in components")

    # Phase 2: Component design grounded in intent
    spec, tokens_2 = cls._design_components(task.goal, intent)
    total_tokens += tokens_2

    # Safety net: verify capability_owner appears in components
    component_names = [c.name for c in spec.components]
    if intent.capability_owner not in component_names:
        print(f"      ⚠️  WARNING: capability_owner '{intent.capability_owner}' missing from {component_names}")
        print(f"      ⚠️  Injecting missing component deterministically...")
        spec = cls._inject_capability_component(spec, intent)

    return spec, total_tokens
```

### `_extract_intent()` — Phase 1 LLM call

```python
@classmethod
def _extract_intent(cls, goal: str) -> tuple[IntentSpec, int]:
    prompt = f"""User request: {goal}

Extract the user's intent from this request.

primary_action: The ONE thing the user must be able to DO.
  - Write as an active verb phrase ("turn text into an image", "track workouts")
  - Describe the ACTION, not the container ("web app" and "system" are containers, not actions)
  - If the request says "turn X into Y", primary_action is exactly "turn X into Y"

core_capability: The specific technical mechanism that makes primary_action possible.
  - Name the algorithm, API, or data operation required
  - Be concrete: "call image generation API (e.g. Stable Diffusion)" not "handle user input"
  - If multiple options exist, pick the most appropriate and note it is assumed

capability_owner: The PascalCase component name that will own core_capability.
  - This component MUST be created in the final application
  - Name it after what it does, not what layer it is: "ImageGenerator" not "Handler"

constraints: List only explicit requirements from the user's request.
  - Empty list if none stated. Do NOT invent constraints.

Return JSON matching the IntentSpec schema."""

    system = """You are an intent extraction expert.
Your job is to identify what a user actually needs an application to DO.
Focus entirely on the primary user action. Ignore architecture and implementation layers."""

    return call_ollama_structured(
        model=cls.model,
        prompt=prompt,
        system=system,
        response_schema=IntentSpec,
        temperature=0.1  # Intentionally low — extraction is deterministic
    )
```

### `_design_components()` — Phase 2 LLM call

```python
@classmethod
def _design_components(cls, goal: str, intent: IntentSpec) -> tuple[AppSpec, int]:

    constraints_block = (
        "Constraints from user:\n" + "\n".join(f"  - {c}" for c in intent.constraints)
        if intent.constraints
        else "Constraints: None stated — use simplest reasonable defaults."
    )

    prompt = f"""User request: {goal}

EXTRACTED INTENT — treat this as ground truth, not a suggestion:
  Primary action:    {intent.primary_action}
  Core capability:   {intent.core_capability}
  Capability owner:  {intent.capability_owner}
{constraints_block}

Design the application components.

HARD RULES:
1. {intent.capability_owner} MUST be component [0]. It implements: {intent.core_capability}
2. Every other component exists to SUPPORT {intent.capability_owner}
3. 3-5 components total. Each must be implementable in under 50 lines of Python.
4. output_file must follow pattern: "outputs/<snake_case_name>.py"
5. depends_on lists component NAMES (PascalCase), not file paths
6. public_interface lists exact Python signatures only, nothing else

Suggested component order (omit any that are not needed for this specific app):
  1. {intent.capability_owner} — REQUIRED — implements {intent.core_capability}
  2. Storage — persists data to disk (if the app needs persistence)
  3. Handler — routes HTTP requests to capability + storage components (if web app)
  4. Template — generates HTML for user interaction (if web app)
  5. Server — starts HTTP server (if web app)

A CLI app does not need Handler, Template, or Server. Do not add them.

Return JSON matching the AppSpec schema."""

    system = """You are a software architect designing minimal Python applications.
The EXTRACTED INTENT block defines what the app must do. Your job is to design the components
that implement it. Component [0] must always be the capability owner named in the intent."""

    return call_ollama_structured(
        model=cls.model,
        prompt=prompt,
        system=system,
        response_schema=AppSpec,
        temperature=cls.temperature
    )
```

### `_inject_capability_component()` — deterministic safety net

Only called when the LLM ignored the capability_owner constraint. No LLM call.

```python
@classmethod
def _inject_capability_component(cls, spec: AppSpec, intent: IntentSpec) -> AppSpec:
    """
    Deterministically inject the missing capability component at index 0.
    This is a safety net, not the happy path. It should log a warning every time
    it fires — consistent firing means the Phase 2 prompt needs adjustment.
    """
    capability_component = ComponentSpec(
        name=intent.capability_owner,
        responsibility=f"Implements the core capability: {intent.core_capability}",
        output_file=f"outputs/{cls._pascal_to_snake(intent.capability_owner)}.py",
        depends_on=[],
        public_interface="execute(input: str) -> str"
    )

    # Insert at front, cap at 5 to avoid over-decomposition
    new_components = [capability_component] + list(spec.components)
    new_components = new_components[:5]

    return AppSpec(
        summary=spec.summary,
        features=spec.features,
        components=new_components,
        done_criteria=spec.done_criteria
    )

@staticmethod
def _pascal_to_snake(name: str) -> str:
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
```

---

## Logging

The diagnostic tool (`plan.py`) already shows spec components. Add intent logging
immediately after `SpecGenerator.generate()` returns, before the component list prints.

Add this block in `plan.py` (or wherever the spec is displayed after generation):

```python
# After SpecGenerator.generate() is called inside the planning walk-through
if spec:
    print(f"\n      Intent (Phase 1 output):")
    print(f"        Primary action:   {intent_logged_from_generate}")
    print(f"        Core capability:  ...")
    print(f"        Capability owner: ...")
```

The `generate()` method already prints this via the inline print statements.
No additional changes to `plan.py` are needed unless you want to surface IntentSpec
fields in the Step 3 full-plan output.

---

## Validation Tests

```bash
python plan.py "build a web app that allows users to turn their written art ideas into a picture"
python plan.py "build a CLI tool to track daily workouts and show weekly averages"
python plan.py "build a note-taking app with save, load, and search"
python plan.py "write a fibonacci function"
```

### Pass criteria

For the art ideas request:
- [ ] Intent log shows `primary_action` as "turn written text into a generated image" (or equivalent)
- [ ] Intent log shows `capability_owner` as "ImageGenerator" (or equivalent — not "Handler")
- [ ] Component [0] in AppSpec is ImageGenerator with interface `generate(prompt: str) -> ...`
- [ ] No other component has responsibility that duplicates image generation
- [ ] Safety net injection NOT triggered (warning should not appear)

For the fibonacci request:
- [ ] SpecGenerator is NOT called (not app-scale — verify via token count staying low)

For all app-scale requests:
- [ ] Token count is 3000-6000 (two calls, not runaway)
- [ ] Component count is 3-5 (not over-decomposed)
- [ ] Every component has a non-empty `public_interface`
- [ ] Safety net injection logged as WARNING if it fires

## Implementation Status

- [x] `IntentSpec` schema added (flat, 4 fields)
- [x] `SpecGenerator` converted to classmethod-based with `model`/`temperature` class attrs
- [x] `generate()` replaced with two-phase orchestrator
- [x] `_extract_intent()` added (Phase 1, temperature=0.1)
- [x] `_design_components()` added (Phase 2, grounded in intent)
- [x] `_inject_capability_component()` added (deterministic safety net with WARNING logs)
- [x] `_pascal_to_snake()` added
- [x] Import check passes — ready for live validation tests

---

## What NOT to Change

- `AppSpec` schema — downstream decomposition code depends on this shape exactly
- `ComponentSpec` schema — same reason
- `HierarchicalPlanner._decompose_task_from_spec()` — consumes AppSpec unchanged
- `TaskClassifier` — app-scale detection is working correctly
- `plan.py` diagnostic output structure — only add intent logging, do not restructure
- Any file not listed in the Files to Change section

---

## Files to Change

| File | Change |
|------|--------|
| `ai_intern/planning/spec_generator.py` | Add `IntentSpec` schema; replace `generate()` with two-phase version; add `_extract_intent()`, `_design_components()`, `_inject_capability_component()`, `_pascal_to_snake()` |

That is all. One file. The fix is entirely self-contained within the spec generator.

---

## Notes for Claude Code

The safety net (`_inject_capability_component`) must log a warning every time it fires.
If it fires consistently, that signals the Phase 2 prompt needs adjustment — do not
silently mask the failure.

`temperature=0.1` on `_extract_intent` is intentional and must not be changed. Intent
extraction is deterministic — "turn art ideas into a picture" has one correct primary
action. Higher temperatures introduce noise into the one input that grounds everything else.

`IntentSpec` must stay flat. If you find yourself wanting to add nested objects to it,
you are solving a different problem. It is scaffolding, not a data model.

Do not merge `_extract_intent` and `_design_components` back into one call. The separation
is the fix, not an implementation detail.