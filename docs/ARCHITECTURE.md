# NoesisEngine Architecture v1.0

> Status: design proposal (no code yet).
> This document is the normative reference ("kyou", the warp thread) for
> every implementation task. All Issues in `docs/issues.json` must align
> with this file. If an Issue drifts from the architecture, the
> architecture wins.

## Design Principles (invariants)

1. Single LLM, multiple personas.
   One underlying model receives different system prompts per persona.
   Cross-model inconsistency is eliminated by construction.

2. Two-stage persona selection.
   Stage 1: the model classifies the conflict category.
   Stage 2: the category deterministically selects a fixed persona set.
   The model never chooses its own personas.

3. Non-coercive output.
   Output presents axes, diffs, and intersection points.
   Consensus is never forced. User choice is preserved.

4. Minority value preservation.
   Rejected opinions are never discarded.
   Each is evaluated for philosophical value and the structural reason
   it was rejected.

5. Pluggable LLM backend.
   Abstract client interface. Anthropic, OpenAI, and a local stub.
   A future local LLM is a drop-in replacement, not a rewrite.

6. Total independence from Po_core.
   No imports, no shared types, no optional dependency.

## Pipeline

```
Input (audio or text)
  -> InputLoader         audio_loader | text_loader
  -> SpeakerSeparator    utterances grouped by speaker_id
  -> UtteranceDecomposer doxa -> claims + premises + implicit values
  -> CategoryClassifier  Stage 1: ETHICAL | RESOURCE | TEMPORAL |
                         VALUE | METHODOLOGICAL
  -> PersonaSelector     Stage 2: category -> fixed persona set
  -> AxisAnalyzer        per persona, per speaker: AxisVector
  -> VectorMath          pairwise diffs, intersections, clusters
  -> MinorityEvaluator   rejected utterances -> philosophical value
  -> OutputFormatter     NoesisOutput (machine + human)
```

## Module Layout

```
noesis_engine/
  __init__.py
  config.py
  core/
    __init__.py
    types.py
    exceptions.py
    pipeline.py
  input/
    __init__.py
    text_loader.py
    audio_loader.py
    speaker_separator.py
  decomposition/
    __init__.py
    utterance_decomposer.py
  classification/
    __init__.py
    category_classifier.py
  persona/
    __init__.py
    base.py
    registry.py
    selector.py
    definitions/
      __init__.py
      kantian.py
      aristotelian.py
      cartesian.py
      hegelian.py
      utilitarian.py
      deontological.py
      virtue_ethics.py
      existentialist.py
      pragmatist.py
  analysis/
    __init__.py
    axis_analyzer.py
    vector_math.py
    minority_evaluator.py
  llm/
    __init__.py
    base.py
    anthropic_client.py
    openai_client.py
    local_client.py
  output/
    __init__.py
    schemas.py
    formatter.py
  api/
    __init__.py
    server.py
  cli.py
tests/
  unit/
  integration/
  fixtures/
docs/
  ARCHITECTURE.md
  TASK_GRAPH.md
  PERSONAS.md
  issues.json
pyproject.toml
.github/workflows/ci.yml
.github/workflows/publish.yml
```

## Core Data Types (contract)

```
Utterance
  speaker_id: str
  text: str
  timestamp: float | None

DecomposedUtterance
  utterance: Utterance
  claims: list[str]
  premises: list[str]
  implicit_values: list[str]

ConflictCategory: Enum
  ETHICAL, RESOURCE, TEMPORAL, VALUE, METHODOLOGICAL

AxisVector
  persona_id: str
  speaker_id: str
  axes: dict[str, float]       # axis_name -> weight in [0,1]
  rationale: str
  confidence: float

SpeakerAnalysis
  speaker_id: str
  dominant_grammar: str         # e.g. "Kantian"
  axis_vectors: list[AxisVector]

MinorityEvaluation
  utterance: Utterance
  philosophical_value: float    # [0,1]
  rejection_reason_structural: str
  preserved_insight: str

NoesisOutput
  category: ConflictCategory
  speaker_analyses: list[SpeakerAnalysis]
  pairwise_axis_diffs: dict[tuple[str, str], float]
  intersection_points: list[str]   # offered, never forced
  minority_evaluations: list[MinorityEvaluation]
```

## Persona-Category Allocation Table (immutable)

```
ETHICAL        -> Kantian, Utilitarian, VirtueEthics
RESOURCE       -> Utilitarian, Deontological, VirtueEthics
TEMPORAL       -> Existentialist, Pragmatist, Hegelian
VALUE          -> Kantian, Aristotelian, Hegelian
METHODOLOGICAL -> Cartesian, Pragmatist, Aristotelian
```

Every category binds at least three fixed personas.
The model may not add or remove personas at any stage.

## LLM Usage Contract

All LLM traffic flows through `LLMClient.chat(system_prompt, messages,
schema=None, temperature=0.0) -> StructuredResponse`.

- Every persona reasoning call uses a JSON schema.
- No free-form persona output reaches downstream stages.
- Temperatures are pinned per call-site:
  - classification: 0.0
  - decomposition:  0.0
  - persona analysis: 0.2
- Backend selection is a configuration concern, not a code concern.

## FastAPI Surface

```
POST /analyze/text      body: {utterances: [...]}
POST /analyze/audio     multipart audio (optional extra)
GET  /personas          list available personas
GET  /categories        list categories + fixed persona sets
GET  /healthz
```

## Package / Release

- PyPI name: `noesis-engine`
- Python: `>=3.10`
- License: AGPL-3.0 (aligned with README)
- Optional extras:
  - `[audio]`     -> whisper, pydub
  - `[openai]`    -> openai
  - `[anthropic]` -> anthropic
  - `[local]`     -> llama-cpp-python (stub target)
  - `[dev]`       -> pytest, mypy, ruff, pydantic-settings

## Non-goals (v1)

- Real diarization (pyannote) -- audio loader uses a naive segmenter.
- Streaming / realtime analysis.
- Persona auto-selection by the model.
- Consensus recommendation. The engine offers intersections, never
  prescriptions.
