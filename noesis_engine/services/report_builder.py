from __future__ import annotations

from noesis_engine.core.axes import CANONICAL_AXIS_KEYS
from noesis_engine.core.enums import DecisionState
from noesis_engine.core.schemas import (
    AnalysisReport,
    BridgePoint,
    CategoryResult,
    DecisionMap,
    DivergencePair,
    InputDiagnostics,
    IssueAnalysis,
    IssueCluster,
    MeetingLevelSummary,
    PersonaReading,
    RejectedOpinionAnalysis,
    RunMetadata,
    SpeakerVector,
)


class ReportBuilderService:
    def build(
        self,
        *,
        run_metadata: RunMetadata,
        input_diagnostics: InputDiagnostics,
        issues: list[IssueCluster],
        categories: dict[str, CategoryResult],
        persona_readings: dict[str, list[PersonaReading]],
        speaker_vectors: dict[str, list[SpeakerVector]],
        divergences: dict[str, list[DivergencePair]],
        decision_maps: dict[str, DecisionMap],
        rejected_opinions: dict[str, list[RejectedOpinionAnalysis]],
        bridge_points: dict[str, list[BridgePoint]],
        artifacts: dict[str, object] | None = None,
        include_debug_artifacts: bool = False,
    ) -> AnalysisReport:
        issue_analyses: list[IssueAnalysis] = []
        for issue in issues:
            issue_id = issue.issue_id
            issue_analyses.append(
                IssueAnalysis(
                    issue_id=issue_id,
                    label=issue.label,
                    category=categories[issue_id],
                    persona_readings=persona_readings.get(issue_id, []),
                    speaker_vectors=speaker_vectors.get(issue_id, []),
                    divergences=divergences.get(issue_id, []),
                    decision_map=decision_maps[issue_id],
                    rejected_opinions=rejected_opinions.get(issue_id, []),
                    bridge_points=bridge_points.get(issue_id, []),
                    summary=self._build_issue_summary(
                        categories[issue_id],
                        decision_maps[issue_id],
                        bridge_points.get(issue_id, []),
                    ),
                )
            )

        meeting_level_summary = self._build_meeting_summary(issue_analyses)
        return AnalysisReport(
            run_metadata=run_metadata,
            input_diagnostics=input_diagnostics,
            issue_analyses=issue_analyses,
            meeting_level_summary=meeting_level_summary,
            artifacts=artifacts or {} if include_debug_artifacts else {},
        )

    def _build_issue_summary(
        self,
        category: CategoryResult,
        decision_map: DecisionMap,
        bridge_points: list[BridgePoint],
    ) -> str:
        parts = [f"Primary category: {category.primary_category.value}."]
        parts.append(f"Decision status: {decision_map.status.value}.")
        if bridge_points:
            parts.append(bridge_points[0].bridge_statement)
        return " ".join(parts)

    def _build_meeting_summary(
        self,
        issue_analyses: list[IssueAnalysis],
    ) -> MeetingLevelSummary:
        conflict_axis_counts: dict[str, int] = {}
        shared_axis_counts: dict[str, int] = {}
        unresolved_issue_ids: list[str] = []

        for issue_analysis in issue_analyses:
            if issue_analysis.decision_map.status == DecisionState.UNRESOLVED:
                unresolved_issue_ids.append(issue_analysis.issue_id)

            for divergence in issue_analysis.divergences:
                for axis in divergence.conflict_axes:
                    conflict_axis_counts[axis] = conflict_axis_counts.get(axis, 0) + 1

            for bridge_point in issue_analysis.bridge_points:
                for axis in bridge_point.shared_axes:
                    shared_axis_counts[axis] = shared_axis_counts.get(axis, 0) + 1

        top_conflict_axes = self._rank_axes(conflict_axis_counts)
        shared_axes = self._rank_axes(shared_axis_counts)
        summary = (
            f"Analyzed {len(issue_analyses)} issues. "
            f"Unresolved issues: {len(unresolved_issue_ids)}. "
            f"Top conflict axes: {', '.join(top_conflict_axes[:3]) if top_conflict_axes else 'none'}."
        )
        return MeetingLevelSummary(
            top_conflict_axes=top_conflict_axes,
            shared_axes=shared_axes,
            unresolved_issue_ids=unresolved_issue_ids,
            summary=summary,
        )

    def _rank_axes(self, counts: dict[str, int]) -> list[str]:
        axis_order = {axis: index for index, axis in enumerate(CANONICAL_AXIS_KEYS)}
        return [
            axis
            for axis, _count in sorted(
                counts.items(),
                key=lambda item: (-item[1], axis_order[item[0]]),
            )
        ]


__all__ = ["ReportBuilderService"]
