# ACTIVE TASK: SpecGenerator — Two Targeted Fixes

## What This Fixes

Two specific issues identified from live output analysis:

1. **PascalCase output filenames** — `ComponentSpec.output_file` returns `outputs/ImageCropper.py`
   instead of `outputs/image_cropper.py`. The `_pascal_to_snake()` method exists but only runs
   inside `_inject_capability_component()` (the safety net). The happy path has no enforcement.
   This will cause import failures on Linux/Mac (case-sensitive filesystems).

2. **Tautological `core_capability`** — `_extract_intent()` returned "implement image cropping
   functionality" for a request about cropping images. This restates the action rather than naming
   the mechanism. The `_design_components()` prompt then received this as ground truth and produced
   a generic `execute(input: str) -> str` interface instead of `crop(image_path: str, box: tuple) -> str`.

Both fixes are in one file: `ai_intern/planning/spec_generator.py`

---

## Fix 1: Enforce snake_case on `ComponentSpec.output_file`

### Problem

`ComponentSpec.output_file` has no validator. The Phase 2 prompt says to use snake_case
but the model ignores it. There is no enforcement layer.

`_pascal_to_snake()` already exists as a static method on `SpecGenerator` — it is just
never called on the happy path.

### Solution

Add a `field_validator` to `ComponentSpec` that normalises the path at Pydantic parse time.
Extract the conversion logic into a module-level function `_to_snake_case()` so both
`ComponentSpec` and `SpecGenerator._pascal_to_snake()` share one implementation.

### Exact Changes

**Step 1** — Add a module-level helper directly after the imports, before any class definitions:

```python
def _to_snake_case(name: str) -> str:
    """Convert PascalCase or mixed-case string to snake_case.

    Handles simple cases (ImageCropper -> image_cropper) and
    consecutive-uppercase acronyms (HTTPServer -> http_server).
    """
    # Insert underscore before an uppercase letter that follows a lowercase letter or digit
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    # Insert underscore before an uppercase letter that is followed by a lowercase letter
    # when it is itself preceded by an uppercase letter  (handles "HTTPServer" -> "HTTP_Server")
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s)
    return s.lower()
```

Verify: `_to_snake_case("HTTPServer") == "http_server"` and
`_to_snake_case("ImageCropper") == "image_cropper"` before committing.

**Step 2** — Add a `field_validator` to `ComponentSpec`.
Also add `field_validator` to the import from pydantic if not already present.

```python
from pydantic import BaseModel, field_validator

class ComponentSpec(BaseModel):
    name: str           # PascalCase — do NOT normalise this field
    responsibility: str
    output_file: str
    depends_on: list[str] = []
    public_interface: str

    @field_validator("output_file", mode="before")
    @classmethod
    def normalise_output_file(cls, v: str) -> str:
        """
        Enforce outputs/<snake_case_name>.py regardless of what the LLM returns.

        Handles:
          "outputs/ImageCropper.py"   -> "outputs/image_cropper.py"
          "ImageCropper.py"           -> "outputs/image_cropper.py"
          "outputs/image_cropper.py"  -> "outputs/image_cropper.py"  (no-op)
          "outputs/HTTPServer.py"     -> "outputs/http_server.py"
        """
        if not isinstance(v, str):
            return v

        # Normalise path separators, extract filename only
        filename = v.replace("\\", "/").split("/")[-1]

        # Strip extension
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename

        # Convert stem to snake_case and rebuild canonical path
        return f"outputs/{_to_snake_case(stem)}.py"
```

**Step 3** — Update `SpecGenerator._pascal_to_snake()` to delegate to the shared function:

```python
@staticmethod
def _pascal_to_snake(name: str) -> str:
    return _to_snake_case(name)
```

### What Changes in Practice

The fix happens at Pydantic parse time — before `_decompose_task_from_spec` builds the
task goal string. No changes needed anywhere downstream.

Before:
```
output_file: "outputs/ImageCropper.py"
goal: "Write `outputs/ImageCropper.py` Python module."
```

After:
```
output_file: "outputs/image_cropper.py"
goal: "Write `outputs/image_cropper.py` Python module."
```

---

## Fix 2: Force Concrete `core_capability` in `_extract_intent`

### Problem

The model satisfies the schema with tautological values:
- "implement image cropping functionality"  — restates the action
- "handle user authentication"             — describes the layer, not the mechanism
- "store user data"                        — meaningless

These pass schema validation but give `_design_components` nothing actionable. The
downstream effect is generic interfaces like `execute(input: str) -> str`.

The current prompt gives one good example (image generation API) but a 7B model
generalises poorly from one example. It matches the surface form of the schema field
without matching the substance.

### Solution

**Change A** — Rewrite the `core_capability` instruction block with matched BAD/GOOD pairs
that show the exact failure mode. Add a curated list of standard library defaults so the
model has concrete options when the user hasn't specified one.

Replace the existing `core_capability` block in `_extract_intent`'s prompt:

```
# BEFORE
core_capability: The specific technical mechanism that makes primary_action possible.
  - Name the algorithm, API, or data operation required
  - Be concrete: "call image generation API (e.g. Stable Diffusion)" not "handle user input"
  - If multiple options exist, pick the most appropriate and note it is assumed

# AFTER
core_capability: The specific Python library, function, or API that implements primary_action.
  - MUST name a concrete library or API call — never describe behavior in general terms
  - BAD:  "implement image cropping functionality"    <- restates action, names nothing
  - GOOD: "use Pillow's Image.crop(box) where box=(left, upper, right, lower)"
  - BAD:  "handle HTTP requests for the web app"     <- describes layer, names nothing
  - GOOD: "use Python's http.server.BaseHTTPRequestHandler to route GET/POST requests"
  - BAD:  "store user data on disk"                  <- describes behavior, names nothing
  - GOOD: "persist records as JSON using Python's built-in json module"
  - BAD:  "generate images based on user input"      <- describes behavior, names nothing
  - GOOD: "call the Replicate API with the user's prompt to generate an image (assumed: Replicate)"
  - If the user did not specify a library, choose the most appropriate from this list and note it:
      Images:      Pillow (PIL)
      Database:    sqlite3
      Simple data: json module
      Tabular:     csv module
      HTTP client: requests
      HTTP server: http.server.BaseHTTPRequestHandler
      Image gen:   Replicate API or Stable Diffusion (local)
```

**Change B** — Lower temperature from 0.1 to 0.0 on this call. Intent extraction is
deterministic — there is one correct primary action for any given request.

```python
# BEFORE
temperature=0.1  # Intentionally low — extraction is deterministic

# AFTER
temperature=0.0  # Greedy decoding — extraction is fully deterministic
```

If `call_ollama_structured` raises on `temperature=0.0`, use `0.05` and add a comment.

### Expected Output Delta

Before:
```
[Intent] Core capability:  implement image cropping functionality
→ Component [0] interface: execute(input: str) -> str
```

After:
```
[Intent] Core capability:  use Pillow's Image.crop(box) where box=(left, upper, right, lower)
→ Component [0] interface: crop(image_path: str, box: tuple) -> str
```

The CodingAgent receives the concrete interface in its task goal and can write correct
Pillow code without having to infer what "implement cropping" means.

---

## Validation

```bash
python plan.py "build a web app that allows users to crop images"
python plan.py "build a web app that allows users to turn their written art ideas into a picture"
python plan.py "build a note-taking app with save, load, and search"
```

### Pass Criteria

**Fix 1 — snake_case filenames:**
- [ ] All `output_file` values are `outputs/snake_case_name.py` — no uppercase after `outputs/`
- [ ] `ImageCropper` → `outputs/image_cropper.py`
- [ ] `ArtIdeaServer` → `outputs/art_idea_server.py`
- [ ] If a component named `HTTPServer` appears → `outputs/http_server.py` (acronym case)
- [ ] Goal strings in task list reflect corrected filenames

**Fix 2 — concrete core_capability:**
- [ ] Cropping request: `[Intent] Core capability` mentions Pillow and `Image.crop`
- [ ] Art ideas request: capability mentions a named image generation API
- [ ] Note-taking request: capability mentions `json`, `sqlite3`, or equivalent by name
- [ ] Component [0] interface is domain-specific, not `execute(input: str) -> str`
- [ ] No `core_capability` value ends with the word "functionality"

---

## Files to Change

| File | Change |
|------|--------|
| `ai_intern/planning/spec_generator.py` | Add `_to_snake_case()` module-level function; add `field_validator` to `ComponentSpec`; update `_extract_intent` prompt `core_capability` block; change temperature to `0.0`; update `_pascal_to_snake()` to delegate to `_to_snake_case()` |

One file. No schema changes. No changes to `hierarchical.py`, `schemas.py`, or any other file.

---

## Notes for Claude Code

The two regex substitutions in `_to_snake_case` must run in sequence — the second only
makes sense after the first has run. Do not combine them into one pattern.

Do not normalise `ComponentSpec.name` — it stays PascalCase. `name` is used in
`depends_on` references and in goal strings where PascalCase is intentional and correct.
Only `output_file` gets normalised.

The `normalise_output_file` validator uses `mode="before"` so it runs on the raw LLM
string before any other Pydantic coercion. This is correct — do not change the mode.

If the safety net (`_inject_capability_component`) fires after these changes, its
`output_file` construction already calls `_pascal_to_snake()` which now delegates to
`_to_snake_case()` — so the safety net path is also fixed automatically.