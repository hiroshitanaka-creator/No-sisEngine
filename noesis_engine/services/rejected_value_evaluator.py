from __future__ import annotations

from collections import defaultdict

from noesis_engine.core.axes import CANONICAL_AXIS_KEYS
from noesis_engine.core.schemas import (
    CategoryResult,
    ClaimUnit,
    DecisionMap,
    DivergencePair,
    IssueCluster,
    RejectedOpinionAnalysis,
    SpeakerVector,
)
from noesis_engine.ports.llm import LLMPort
from noesis_engine.prompts.rejection_analysis import build_rejection_analysis_prompt
from noesis_engine.services.persona_router import PersonaRouterService
from noesis_engine.utils.structured_output import generate_structured_or_raise


class RejectedValueEvaluatorService:
    def __init__(
        self,
        llm: LLMPort,
        *,
        max_repair_attempts: int = 1,
        axis_gap_threshold: float = 0.10,
    ) -> None:
        self._llm = llm
        self._max_repair_attempts = max_repair_attempts
        self._axis_gap_threshold = axis_gap_threshold
        self._persona_router = PersonaRouterService()

    def evaluate_issue(
        self,
        issue: IssueCluster,
        claims: list[ClaimUnit],
        category: CategoryResult,
        decision_map: DecisionMap,
        speaker_vectors: list[SpeakerVector],
        divergences: list[DivergencePair],
        *,
        meeting_context: str | None = None,
    ) -> list[RejectedOpinionAnalysis]:
        target_claim_ids = decision_map.rejected_claim_ids + decision_map.ignored_claim_ids
        if not target_claim_ids:
            return []

        claim_by_id = {claim.claim_id: claim for claim in claims}
        vector_by_speaker = {vector.speaker_id: vector for vector in speaker_vectors}
        mean_vector = self._mean_vector(speaker_vectors)
        divergences_by_speaker: dict[str, list[DivergencePair]] = defaultdict(list)
        for pair in divergences:
            divergences_by_speaker[pair.speaker_a].append(pair)
            divergences_by_speaker[pair.speaker_b].append(pair)

        routed_persona_ids = [
            persona.persona_id.value
            for persona in self._persona_router.route_from_result(category)
        ]

        analyses: list[RejectedOpinionAnalysis] = []
        for claim_id in target_claim_ids:
            claim = claim_by_id[claim_id]
            claimant_vector = vector_by_speaker.get(claim.speaker_id)
            candidate_axes = self._candidate_underrepresented_axes(
                claimant_vector,
                mean_vector,
            )
            structural_reasons = self._candidate_structural_reasons(
                claim,
                decision_map,
                divergences_by_speaker.get(claim.speaker_id, []),
                claims,
            )
            divergence_summary = self._divergence_summary(
                divergences_by_speaker.get(claim.speaker_id, []),
            )

            result = generate_structured_or_raise(
                self._llm,
                build_rejection_analysis_prompt(
                    issue,
                    claim,
                    category=category,
                    decision_map=decision_map,
                    candidate_underrepresented_axes=candidate_axes,
                    structural_reason_candidates=structural_reasons,
                    routed_persona_ids=routed_persona_ids,
                    divergence_summary=divergence_summary,
                    meeting_context=meeting_context,
                ),
                max_repair_attempts=self._max_repair_attempts,
            )
            output = result.output
            analyses.append(
                RejectedOpinionAnalysis(
                    claim_id=claim_id,
                    philosophical_value_score=output.philosophical_value_score,
                    underrepresented_axes=output.underrepresented_axes or candidate_axes,
                    structural_rejection_reasons=output.structural_rejection_reasons,
                    salvage_conditions=output.salvage_conditions,
                    confidence=output.confidence,
                )
            )

        return analyses

    def _mean_vector(
        self,
        speaker_vectors: list[SpeakerVector],
    ) -> dict[str, float]:
        if not speaker_vectors:
            return {axis: 0.0 for axis in CANONICAL_AXIS_KEYS}
        mean = {axis: 0.0 for axis in CANONICAL_AXIS_KEYS}
        for vector in speaker_vectors:
            for axis in CANONICAL_AXIS_KEYS:
                mean[axis] += vector.axis_weights[axis]
        for axis in CANONICAL_AXIS_KEYS:
            mean[axis] /= len(speaker_vectors)
        return mean

    def _candidate_underrepresented_axes(
        self,
        claimant_vector: SpeakerVector | None,
        mean_vector: dict[str, float],
    ) -> list[str]:
        if claimant_vector is not None:
            axes = [
                axis
                for axis in CANONICAL_AXIS_KEYS
                if claimant_vector.axis_weights[axis] - mean_vector[axis] >= self._axis_gap_threshold
            ]
            if axes:
                return axes[:4]

        ranked = sorted(CANONICAL_AXIS_KEYS, key=lambda axis: (mean_vector[axis], axis))
        return ranked[:4]

    def _candidate_structural_reasons(
        self,
        claim: ClaimUnit,
        decision_map: DecisionMap,
        claimant_divergences: list[DivergencePair],
        claims: list[ClaimUnit],
    ) -> list[str]:
        reasons: list[str] = []
        if claim.claim_id in decision_map.rejected_claim_ids:
            reasons.append("Encountered explicit objection or incompatibility with the adopted path.")
        if claim.claim_id in decision_map.ignored_claim_ids:
            reasons.append("Received insufficient uptake or integration into the final decision path.")
        if claimant_divergences:
            average_divergence = sum(pair.cosine_distance for pair in claimant_divergences) / len(
                claimant_divergences
            )
            if average_divergence >= 0.25:
                reasons.append("Claimant reasoning diverged materially from the meeting center of gravity.")
        targeted_by_any_support = any(
            claim.claim_id in other_claim.target_claim_ids and other_claim.act_type.value == "support"
            for other_claim in claims
        )
        if not targeted_by_any_support:
            reasons.append("Claim lacked reinforcing support or evidentiary linkage in the discussion.")
        return reasons

    def _divergence_summary(
        self,
        claimant_divergences: list[DivergencePair],
    ) -> dict[str, float | str | list[str]]:
        if not claimant_divergences:
            return {
                "mean_cosine_distance": 0.0,
                "dominant_conflict_axes": [],
                "note": "No divergence data available for claimant.",
            }
        mean_distance = sum(pair.cosine_distance for pair in claimant_divergences) / len(
            claimant_divergences
        )
        axis_counts: dict[str, int] = {}
        for pair in claimant_divergences:
            for axis in pair.conflict_axes:
                axis_counts[axis] = axis_counts.get(axis, 0) + 1
        dominant_conflict_axes = [
            axis
            for axis, _count in sorted(axis_counts.items(), key=lambda item: (-item[1], item[0]))
        ][:4]
        return {
            "mean_cosine_distance": mean_distance,
            "dominant_conflict_axes": dominant_conflict_axes,
            "note": "Derived from claimant pairwise divergence against other speakers in the issue.",
        }


__all__ = ["RejectedValueEvaluatorService"]
