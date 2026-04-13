# NoesisEngine

> *"They were not disagreeing. They were speaking different languages."*

NoesisEngine analyzes multi-speaker transcripts and extracts the underlying
philosophical axes that drive each participant's reasoning — revealing *why*
people talk past each other rather than simply cataloguing what they said.

**Status:** Alpha — API and schema may change between releases.

**License:** Proprietary. All rights reserved.
Commercial use requires prior written permission. See [LICENSE](LICENSE).

---

## What it does

Given a meeting transcript (or audio file), NoesisEngine:

1. Decomposes each utterance into structured claim units
2. Clusters claims into issue groups
3. Classifies each issue by conflict category (values, facts, process, etc.)
4. Maps each speaker's reasoning to a philosophical persona and axis vector
5. Computes divergence between speakers per axis
6. Surfaces rejected opinions that retain philosophical value
7. Generates bridge points — shared axes that could ground productive dialogue

Output is a fully typed `AnalysisReport` JSON object.

---

## Installation

### Core (transcript analysis only)

```bash
pip install "noesis-engine @ git+https://github.com/hiroshitanaka-creator/no-sisengine.git"
```

Or from a local clone:

```bash
pip install -e .
```

### Server (to run the FastAPI service)

```bash
pip install -e ".[server]"
```

### Audio (speech-to-text + diarization)

Audio support requires additional system dependencies (ffmpeg, torch).
Install the audio extras after the core package:

```bash
pip install -e ".[server,audio]"
```

> **Note:** Audio analysis also requires a Hugging Face token and a
> pyannote diarization model. Set `NOESIS_AUDIO__DIARIZATION_MODEL`
> to the model ID (e.g. `pyannote/speaker-diarization-3.1`) and ensure
> your HF token is accepted for that model. Without this, `/analyze/audio`
> returns a 500 error.

---

## Quick start

### 1. Configure environment

```bash
# LLM provider — choose one:

# Option A: OpenAI (or compatible API)
export NOESIS_OPENAI__API_KEY=sk-...
export NOESIS_ANALYSIS_MODEL=gpt-4.1          # default

# Option B: Local model (Ollama, LM Studio, vLLM, etc.)
# No API key needed — key defaults to "not-required"
export NOESIS_LOCAL_LLM__BASE_URL=http://localhost:11434/v1
export NOESIS_LOCAL_LLM__MODEL=llama3.2
# When no OpenAI key is set, NoesisEngine automatically uses the local client.
```

### 2. Start the server

```bash
uvicorn noesis_engine.api.app:app --reload
```

### 3. Verify

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "version": "0.1.0", "analysis_model": "gpt-4.1"}
```

---

## API reference

### `GET /health`

Returns service status and active model name.

**Response**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "analysis_model": "gpt-4.1"
}
```

---

### `POST /analyze/transcript`

Analyze a pre-transcribed conversation.

**Request**
```json
{
  "utterances": [
    {
      "utterance_id": "u1",
      "speaker_id": "Alice",
      "text": "We should prioritize speed. Ship now, fix later.",
      "start_sec": 0.0,
      "end_sec": 4.2
    },
    {
      "utterance_id": "u2",
      "speaker_id": "Bob",
      "text": "Shipping broken software damages trust. Quality first.",
      "start_sec": 4.5,
      "end_sec": 9.1
    }
  ],
  "meeting_context": "Sprint planning — release scope decision",
  "debug": false
}
```

**Response** — `AnalysisReport` (abbreviated)
```json
{
  "run_metadata": {"run_id": "...", "analysis_model": "gpt-4.1", "status": "completed", ...},
  "input_diagnostics": {"input_kind": "transcript", "utterance_count": 2, "speaker_count": 2, "issue_count": 1, "warnings": []},
  "issue_analyses": [
    {
      "issue_id": "issue_001",
      "label": "Release quality vs. release speed",
      "category": {"primary_category": "values", ...},
      "persona_readings": [...],
      "divergences": [...],
      "bridge_points": [{"bridge_statement": "Both speakers value user trust; they differ on which risk — delay or defects — poses the greater threat to it.", ...}],
      ...
    }
  ],
  "meeting_level_summary": {"top_conflict_axes": ["risk_tolerance", "time_horizon"], ...}
}
```

---

### `POST /analyze/audio`

Transcribe and diarize audio, then run the full analysis pipeline.

Requires `[audio]` extras and `NOESIS_AUDIO__DIARIZATION_MODEL` to be set.

**Request** — provide exactly one of `path` or `content_base64`:
```json
{
  "path": "/absolute/path/to/meeting.wav",
  "meeting_context": "Optional context string",
  "debug": false
}
```

---

## Environment variables

All variables use the `NOESIS_` prefix. Nested settings use `__` as delimiter.

| Variable | Default | Description |
|---|---|---|
| `NOESIS_ANALYSIS_MODEL` | `gpt-4.1` | LLM model name passed to the active client |
| `NOESIS_ANALYSIS_TEMPERATURE` | `0.2` | Sampling temperature (0.0–1.0) |
| `NOESIS_OPENAI__API_KEY` | *(none)* | OpenAI API key — triggers OpenAI client when set |
| `NOESIS_OPENAI__BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `NOESIS_OPENAI__ORGANIZATION` | *(none)* | OpenAI organization ID |
| `NOESIS_OPENAI__TIMEOUT_SEC` | `60.0` | Request timeout in seconds |
| `NOESIS_LOCAL_LLM__BASE_URL` | `http://localhost:11434/v1` | Local LLM base URL |
| `NOESIS_LOCAL_LLM__MODEL` | `noesis-local` | Local model identifier |
| `NOESIS_LOCAL_LLM__TIMEOUT_SEC` | `60.0` | Request timeout in seconds |
| `NOESIS_AUDIO__DIARIZATION_MODEL` | *(none)* | Pyannote model ID — required for `/analyze/audio` |
| `NOESIS_AUDIO__TRANSCRIPTION_MODEL` | `base` | Whisper model size |
| `NOESIS_AUDIO__LANGUAGE` | *(auto)* | ISO language code hint for Whisper |
| `NOESIS_INCLUDE_DEBUG_ARTIFACTS` | `false` | Include raw LLM outputs in `artifacts` field |

---

## Architecture overview

```
Utterance[] / AudioInput
        │
        ▼
TranscriptNormalizer ──► ClaimDecomposer ──► IssueClusterer
                                                    │
                         ┌──────────────────────────┤
                         ▼                          ▼
               CategoryClassifier         PersonaAnalyzer
                         │                          │
                         └───────────┬──────────────┘
                                     ▼
              VectorAggregator ─► DivergenceAnalyzer ─► DecisionMapper
                                                               │
                              RejectedValueEvaluator ◄─────────┤
                              BridgeBuilder          ◄─────────┘
                                     │
                                     ▼
                               AnalysisReport
```

| Concept | Code location |
|---|---|
| Axis system | `noesis_engine/core/axes.py` |
| Philosophical personas | `noesis_engine/core/persona_catalog.py` |
| All output schemas | `noesis_engine/core/schemas.py` |
| HTTP schemas | `noesis_engine/api/schemas/http.py` |
| LLM port (interface) | `noesis_engine/ports/llm.py` |
| OpenAI adapter | `noesis_engine/adapters/llm/openai_client.py` |
| Local LLM adapter | `noesis_engine/adapters/llm/local_client.py` |
| Settings | `noesis_engine/settings.py` |

---

## Development

```bash
# Install all dev + test dependencies
pip install -e ".[server]"
pip install pytest pytest-cov pytest-asyncio ruff mypy build

# Run tests
pytest --cov=noesis_engine --cov-fail-under=80

# Lint
ruff check .

# Type check
mypy .

# Build distribution
python -m build
```

---

## Contributing

This is a proprietary project. External contributions are not accepted at this
time. For bug reports or licensing inquiries, open an issue:
https://github.com/hiroshitanaka-creator/no-sisengine/issues
