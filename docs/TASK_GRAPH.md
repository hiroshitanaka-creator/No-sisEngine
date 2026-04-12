# NoesisEngine Task Dependency Graph

Legend: `A -> B` means B depends on A.

```
T001 scaffold (pyproject, package skeleton, license)
  -> T002 core types & exceptions
       -> T003 LLM base (LLMClient ABC)
            -> T004 anthropic client
            -> T005 openai client
            -> T006 local client stub
       -> T007 persona base (Persona ABC)
            -> T008 kantian
            -> T009 aristotelian
            -> T010 cartesian
            -> T011 hegelian
            -> T012 utilitarian
            -> T013 deontological
            -> T014 virtue_ethics
            -> T015 existentialist
            -> T016 pragmatist
            -> T017 persona registry
       -> T018 text loader
            -> T019 speaker separator (text-based)
                 -> T020 audio loader (whisper, optional extra)
       -> T021 utterance decomposer (needs T003)
       -> T022 category classifier (needs T003)
            -> T023 persona selector (Stage 2; needs T017, T022)
       -> T024 axis analyzer (needs T007, T017, T003)
            -> T025 vector math
            -> T026 minority evaluator (needs T024, T003)
       -> T027 output schemas + formatter
       -> T028 config management
       -> T029 pipeline orchestrator
            (needs T019, T021, T022, T023, T024, T025, T026, T027)
       -> T030 FastAPI server (needs T029)
       -> T031 CLI entry point (needs T029)
  -> T032 CI workflow (GitHub Actions; needs T001)
  -> T033 PyPI publish workflow (needs T001, T032)
  -> T034 integration tests (needs T029)
  -> T035 docs: persona catalog (needs T008..T016)
```

## Critical Path

```
T001 -> T002 -> T003 -> T007 -> T008..T016 -> T017
     -> T023 -> T024 -> T025 -> T026 -> T027 -> T029
```

Everything on the critical path must be sequenced.
Everything off it (LLM backends, audio, CI, docs) can parallelize.

## 1 Task = 1 PR Rule

Every Issue in `docs/issues.json` owns exactly one target file
(the `file` field). A PR may touch tests and `__init__.py` exports
for that target, but never a sibling module's implementation.

If during implementation a task requires editing a file owned by
another task, stop and split the Issue.
