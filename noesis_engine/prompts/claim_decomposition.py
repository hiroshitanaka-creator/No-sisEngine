from __future__ import annotations

import json

from pydantic import Field

from noesis_engine.core.enums import ClaimActType
from noesis_engine.core.schemas import StrictModel, Utterance
from noesis_engine.ports.llm import StructuredPrompt


class ClaimDraft(StrictModel):
    local_claim_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    source_utterance_ids: list[str] = Field(default_factory=list)
    text_span: str = Field(min_length=1)
    act_type: ClaimActType
    target_local_claim_ids: list[str] = Field(default_factory=list)
    issue_hint: str | None = None
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    explicit_values: list[str] = Field(default_factory=list)
    implicit_values: list[str] = Field(default_factory=list)


class ClaimDecompositionEnvelope(StrictModel):
    chunk_id: str = Field(min_length=1)
    claims: list[ClaimDraft] = Field(default_factory=list)


def build_claim_decomposition_prompt(
    utterances: list[Utterance],
    *,
    chunk_id: str,
    meeting_context: str | None = None,
    temperature: float | None = None,
) -> StructuredPrompt[ClaimDecompositionEnvelope]:
    system_instruction = (
        "Decompose a speaker-tagged transcript chunk into claim units. "
        "Return valid JSON only. "
        "Do not include narrative explanation. "
        "A single utterance may yield multiple claims. "
        "Use act_type only from the allowed enum values. "
        "Use local claim ids such as c1, c2, c3 within this chunk. "
        "Use target_local_claim_ids only when a claim explicitly supports, objects to, or decides another claim. "
        "issue_hint must be a short issue label when inferable, otherwise null."
    )
    payload = {
        "chunk_id": chunk_id,
        "meeting_context": meeting_context,
        "allowed_act_types": [member.value for member in ClaimActType],
        "utterances": [utterance.model_dump(mode="json") for utterance in utterances],
    }
    return StructuredPrompt(
        system_instruction=system_instruction,
        user_input=json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        response_model=ClaimDecompositionEnvelope,
        metadata={"stage": "claim_decomposition", "chunk_id": chunk_id},
        temperature=temperature,
        strict_json=True,
    )


__all__ = [
    "ClaimDecompositionEnvelope",
    "ClaimDraft",
    "build_claim_decomposition_prompt",
]
