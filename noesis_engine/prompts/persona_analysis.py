from __future__ import annotations

import json

from pydantic import Field

from noesis_engine.core.axes import CANONICAL_AXIS_KEYS
from noesis_engine.core.schemas import AxisWeights, ClaimUnit, IssueCluster, PersonaSpec, StrictModel
from noesis_engine.ports.llm import StructuredPrompt


class PersonaReadingDraft(StrictModel):
    speaker_id: str = Field(min_length=1)
    axis_weights: AxisWeights
    alignment_score: float = Field(ge=0.0, le=1.0)
    grammar_summary: str = Field(min_length=1)
    hidden_value: str = Field(min_length=1)
    blind_spot: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_claim_ids: list[str] = Field(default_factory=list)


class PersonaAnalysisEnvelope(StrictModel):
    issue_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    readings: list[PersonaReadingDraft] = Field(default_factory=list)


def build_persona_analysis_prompt(
    issue: IssueCluster,
    claims: list[ClaimUnit],
    persona_spec: PersonaSpec,
    *,
    meeting_context: str | None = None,
    temperature: float | None = None,
) -> StructuredPrompt[PersonaAnalysisEnvelope]:
    system_instruction = (
        "Analyze the issue through the supplied philosophical persona. "
        "Return valid JSON only. "
        "Do not invent new axes. "
        "axis_weights must use only the canonical axis schema and must sum to 1.0. "
        "Produce one reading per speaker present in the issue cluster. "
        "hidden_value means a valuable concern the speaker is preserving. "
        "blind_spot means what the speaker underweights from this persona's perspective."
    )
    payload = {
        "issue": issue.model_dump(mode="json"),
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "meeting_context": meeting_context,
        "canonical_axes": list(CANONICAL_AXIS_KEYS),
        "persona_spec": persona_spec.model_dump(mode="json"),
    }
    return StructuredPrompt(
        system_instruction=system_instruction,
        user_input=json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        response_model=PersonaAnalysisEnvelope,
        metadata={
            "stage": "persona_analysis",
            "issue_id": issue.issue_id,
            "persona_id": persona_spec.persona_id.value,
        },
        temperature=temperature,
        strict_json=True,
    )


__all__ = [
    "PersonaAnalysisEnvelope",
    "PersonaReadingDraft",
    "build_persona_analysis_prompt",
]
