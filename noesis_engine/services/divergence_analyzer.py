from __future__ import annotations

from math import sqrt

from noesis_engine.core.axes import CANONICAL_AXIS_KEYS
from noesis_engine.core.schemas import DivergencePair, IssueCluster, SpeakerVector


class DivergenceAnalyzerService:
    def __init__(
        self,
        *,
        shared_axis_delta_threshold: float = 0.08,
        conflict_axis_delta_threshold: float = 0.20,
    ) -> None:
        self._shared_threshold = shared_axis_delta_threshold
        self._conflict_threshold = conflict_axis_delta_threshold

    def analyze_issue(
        self,
        issue: IssueCluster,
        speaker_vectors: list[SpeakerVector],
    ) -> list[DivergencePair]:
        ordered_vectors = sorted(
            speaker_vectors,
            key=lambda vector: issue.speaker_ids.index(vector.speaker_id)
            if vector.speaker_id in issue.speaker_ids
            else len(issue.speaker_ids),
        )
        pairs: list[DivergencePair] = []
        for left_index in range(len(ordered_vectors)):
            for right_index in range(left_index + 1, len(ordered_vectors)):
                left = ordered_vectors[left_index]
                right = ordered_vectors[right_index]
                delta = {
                    axis: left.axis_weights[axis] - right.axis_weights[axis]
                    for axis in CANONICAL_AXIS_KEYS
                }
                shared_axes = [
                    axis
                    for axis in CANONICAL_AXIS_KEYS
                    if abs(delta[axis]) <= self._shared_threshold
                ]
                conflict_axes = [
                    axis
                    for axis in CANONICAL_AXIS_KEYS
                    if abs(delta[axis]) >= self._conflict_threshold
                ]
                pairs.append(
                    DivergencePair(
                        speaker_a=left.speaker_id,
                        speaker_b=right.speaker_id,
                        cosine_distance=self._cosine_distance(left, right),
                        per_axis_delta=delta,
                        shared_axes=shared_axes,
                        conflict_axes=conflict_axes,
                    )
                )
        return pairs

    def build_divergence_matrix(
        self,
        divergence_pairs: list[DivergencePair],
    ) -> dict[str, dict[str, float]]:
        matrix: dict[str, dict[str, float]] = {}
        for pair in divergence_pairs:
            matrix.setdefault(pair.speaker_a, {})[pair.speaker_b] = pair.cosine_distance
            matrix.setdefault(pair.speaker_b, {})[pair.speaker_a] = pair.cosine_distance
        for speaker_id in list(matrix):
            matrix[speaker_id][speaker_id] = 0.0
        return matrix

    def _cosine_distance(
        self,
        left: SpeakerVector,
        right: SpeakerVector,
    ) -> float:
        dot = sum(left.axis_weights[axis] * right.axis_weights[axis] for axis in CANONICAL_AXIS_KEYS)
        left_norm = sqrt(sum(left.axis_weights[axis] ** 2 for axis in CANONICAL_AXIS_KEYS))
        right_norm = sqrt(sum(right.axis_weights[axis] ** 2 for axis in CANONICAL_AXIS_KEYS))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        similarity = dot / (left_norm * right_norm)
        similarity = min(1.0, max(0.0, similarity))
        return 1.0 - similarity


__all__ = ["DivergenceAnalyzerService"]
