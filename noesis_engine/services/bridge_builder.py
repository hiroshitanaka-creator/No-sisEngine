from __future__ import annotations

from noesis_engine.core.axes import CANONICAL_AXIS_KEYS
from noesis_engine.core.schemas import BridgePoint, DivergencePair, IssueCluster, SpeakerVector


class BridgeBuilderService:
    def __init__(
        self,
        *,
        shared_axis_delta_threshold: float = 0.08,
        conflict_axis_delta_threshold: float = 0.20,
    ) -> None:
        self._shared_threshold = shared_axis_delta_threshold
        self._conflict_threshold = conflict_axis_delta_threshold

    def build_for_issue(
        self,
        issue: IssueCluster,
        speaker_vectors: list[SpeakerVector],
        divergences: list[DivergencePair],
    ) -> list[BridgePoint]:
        if not speaker_vectors:
            return []

        if divergences:
            shared_axes = [
                axis
                for axis in CANONICAL_AXIS_KEYS
                if all(abs(pair.per_axis_delta[axis]) <= self._shared_threshold for pair in divergences)
            ]
            preserved_disagreements = [
                axis
                for axis in CANONICAL_AXIS_KEYS
                if any(abs(pair.per_axis_delta[axis]) >= self._conflict_threshold for pair in divergences)
            ]
        else:
            shared_axes = [
                axis for axis in CANONICAL_AXIS_KEYS if speaker_vectors[0].axis_weights[axis] > 0.0
            ]
            preserved_disagreements = []

        bridge_statement = self._build_bridge_statement(shared_axes, preserved_disagreements)
        return [
            BridgePoint(
                issue_id=issue.issue_id,
                shared_axes=shared_axes,
                preserved_disagreements=preserved_disagreements,
                bridge_statement=bridge_statement,
            )
        ]

    def _build_bridge_statement(
        self,
        shared_axes: list[str],
        preserved_disagreements: list[str],
    ) -> str:
        if shared_axes and preserved_disagreements:
            return (
                "Potential bridge: coordinate on "
                + ", ".join(shared_axes[:3])
                + " while preserving disagreement on "
                + ", ".join(preserved_disagreements[:3])
                + "."
            )
        if shared_axes:
            return "Potential bridge: coordinate on " + ", ".join(shared_axes[:3]) + "."
        if preserved_disagreements:
            return (
                "No strong shared axis dominates; preserve disagreement on "
                + ", ".join(preserved_disagreements[:3])
                + " while searching for reversible next steps."
            )
        return "Insufficient signal to propose a bridge point."


__all__ = ["BridgeBuilderService"]
