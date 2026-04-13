from __future__ import annotations

from collections import defaultdict

from noesis_engine.core.axes import CANONICAL_AXIS_KEYS, validate_axis_vector
from noesis_engine.core.schemas import ClaimUnit, IssueCluster, PersonaReading, SpeakerVector


class VectorAggregatorService:
    def aggregate_issue(
        self,
        issue: IssueCluster,
        claims: list[ClaimUnit],
        persona_readings: list[PersonaReading],
    ) -> list[SpeakerVector]:
        importance_by_claim_id = {claim.claim_id: claim.importance for claim in claims}
        speaker_claim_ids: dict[str, list[str]] = defaultdict(list)
        for claim in claims:
            speaker_claim_ids[claim.speaker_id].append(claim.claim_id)

        vectors: list[SpeakerVector] = []
        for speaker_id in issue.speaker_ids:
            relevant_readings = [
                reading for reading in persona_readings if reading.speaker_id == speaker_id
            ]
            accumulator = {axis: 0.0 for axis in CANONICAL_AXIS_KEYS}
            source_claim_ids: list[str] = []

            for reading in relevant_readings:
                reading_claim_ids = reading.source_claim_ids or speaker_claim_ids.get(speaker_id, [])
                if reading_claim_ids:
                    avg_importance = sum(
                        importance_by_claim_id.get(claim_id, 0.5) for claim_id in reading_claim_ids
                    ) / len(reading_claim_ids)
                else:
                    avg_importance = 0.5

                weight = max(avg_importance * reading.alignment_score, 1e-9)
                for axis in CANONICAL_AXIS_KEYS:
                    accumulator[axis] += reading.axis_weights[axis] * weight

                for claim_id in reading_claim_ids:
                    if claim_id not in source_claim_ids:
                        source_claim_ids.append(claim_id)

            if sum(accumulator.values()) <= 0.0 and relevant_readings:
                for reading in relevant_readings:
                    for axis in CANONICAL_AXIS_KEYS:
                        accumulator[axis] += reading.axis_weights[axis]

            if sum(accumulator.values()) <= 0.0:
                accumulator[CANONICAL_AXIS_KEYS[0]] = 1.0

            total = sum(accumulator.values())
            normalized = {axis: accumulator[axis] / total for axis in CANONICAL_AXIS_KEYS}
            normalized = validate_axis_vector(
                normalized,
                require_all_keys=True,
                require_normalized=True,
                allow_zero=False,
            )

            vectors.append(
                SpeakerVector(
                    issue_id=issue.issue_id,
                    speaker_id=speaker_id,
                    axis_weights=normalized,
                    source_claim_ids=source_claim_ids,
                )
            )

        return vectors


__all__ = ["VectorAggregatorService"]
