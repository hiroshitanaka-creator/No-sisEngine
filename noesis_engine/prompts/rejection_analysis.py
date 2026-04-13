from __future__ import annotations

import json

from noesis_engine.core.axes import CANONICAL_AXIS_KEYS
from noesis_engine.core.schemas import CategoryResult, ClaimUnit, DecisionMap, IssueCluster, RejectedOpinionAnalysis
from noesis_engine.ports.llm import StructuredPrompt


def build_rejection_analysis_prompt(
    issue: IssueCluster,
    claim: ClaimUnit,
    *,
    category: CategoryResult,
    decision_map: DecisionMap,
    candidate_underrepresented_axes: list[str],
    structural_reason_candidates: list[str],
    routed_persona_ids: list[str],
    divergence_summary: dict[str, float | str | list[str]],
    meeting_context: str | None = None,
    temperature: float | None = None,
) -> StructuredPrompt[RejectedOpinionAnalysis]:
    system_instruction = (
        "Evaluate the philosophical value of a rejected or ignored claim. "
        "Return valid JSON only. "
        "Do not judge whether the claim is empirically true. "
        "Do not confuse value with adoption status. "
        "philosophical_value_score measures whether the claim preserves an underrepresented recognition axis, "
        "protects against future blind spots, or preserves optionality. "
        "structural_rejection_reasons must explain why the claim failed in the discussion structure. "
        "salvage_conditions must state conditions under which the claim's value could be reintegrated."
    )
    payload = {
        "issue": issue.model_dump(mode="json"),
        "claim": claim.model_dump(mode="json"),
        "category": category.model_dump(mode="json"),
        "decision_map": decision_map.model_dump(mode="json"),
        "meeting_context": meeting_context,
        "candidate_underrepresented_axes": candidate_underrepresented_axes,
        "structural_reason_candidates": structural_reason_candidates,
        "routed_persona_ids": routed_persona_ids,
        "divergence_summary": divergence_summary,
        "canonical_axes": list(CANONICAL_AXIS_KEYS),
    }
    return StructuredPrompt(
        system_instruction=system_instruction,
        user_input=json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        response_model=RejectedOpinionAnalysis,
        metadata={"stage": "rejection_analysis", "issue_id": issue.issue_id, "claim_id": claim.claim_id},
        temperature=temperature,
        strict_json=True,
    )


__all__ = ["build_rejection_analysis_prompt"]
