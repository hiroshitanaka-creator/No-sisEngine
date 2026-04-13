from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from noesis_engine.core.axes import normalize_axis_vector, validate_axis_delta, validate_axis_subset
from noesis_engine.core.enums import (
    AnalysisStatus,
    ClaimActType,
    ConflictCategory,
    DecisionState,
    PersonaId,
)


def _normalize_axis_weights(value: dict[str, float]) -> dict[str, float]:
    return normalize_axis_vector(value)


def _validate_axis_delta_mapping(value: dict[str, float]) -> dict[str, float]:
    return validate_axis_delta(value, require_all_keys=True)


def _unique_non_empty_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("List entries must be strings.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("List entries must not be empty.")
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


AxisWeights = Annotated[dict[str, float], AfterValidator(_normalize_axis_weights)]
AxisDelta = Annotated[dict[str, float], AfterValidator(_validate_axis_delta_mapping)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RunMetadata(StrictModel):
    run_id: str = Field(min_length=1)
    analysis_model: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=1.0)
    status: AnalysisStatus = AnalysisStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "RunMetadata":
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at must be greater than or equal to started_at.")
        return self


class InputDiagnostics(StrictModel):
    input_kind: str = Field(min_length=1)
    utterance_count: int = Field(ge=0)
    speaker_count: int = Field(ge=0)
    issue_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("warnings", mode="after")
    @classmethod
    def validate_warnings(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class AudioInput(StrictModel):
    path: str | None = None
    content: bytes | None = None
    filename: str | None = None
    content_type: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "AudioInput":
        has_path = self.path is not None and self.path.strip() != ""
        has_content = self.content is not None
        if has_path == has_content:
            raise ValueError("Provide exactly one of 'path' or 'content'.")
        return self


class Utterance(StrictModel):
    utterance_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "Utterance":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be greater than or equal to start_sec.")
        return self


class SpeakerSegment(StrictModel):
    segment_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "SpeakerSegment":
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must be greater than or equal to start_sec.")
        return self


class ClaimUnit(StrictModel):
    claim_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    source_utterance_ids: list[str] = Field(default_factory=list)
    text_span: str = Field(min_length=1)
    act_type: ClaimActType
    target_claim_ids: list[str] = Field(default_factory=list)
    issue_hint: str | None = None
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    explicit_values: list[str] = Field(default_factory=list)
    implicit_values: list[str] = Field(default_factory=list)

    @field_validator(
        "source_utterance_ids",
        "target_claim_ids",
        "explicit_values",
        "implicit_values",
        mode="after",
    )
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class IssueCluster(StrictModel):
    issue_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    speaker_ids: list[str] = Field(min_length=1)

    @field_validator("claim_ids", "speaker_ids", mode="after")
    @classmethod
    def validate_identifier_lists(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class CategoryResult(StrictModel):
    issue_id: str = Field(min_length=1)
    primary_category: ConflictCategory
    secondary_categories: list[ConflictCategory] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("signals", mode="after")
    @classmethod
    def validate_signals(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)

    @field_validator("secondary_categories", mode="after")
    @classmethod
    def validate_secondary_categories(
        cls, value: list[ConflictCategory]
    ) -> list[ConflictCategory]:
        ordered: list[ConflictCategory] = []
        seen: set[ConflictCategory] = set()
        for category in value:
            if category not in seen:
                seen.add(category)
                ordered.append(category)
        return ordered

    @model_validator(mode="after")
    def validate_primary_not_secondary(self) -> "CategoryResult":
        if self.primary_category in self.secondary_categories:
            raise ValueError("primary_category must not appear in secondary_categories.")
        return self


class PersonaSpec(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: PersonaId
    display_name: str = Field(min_length=1)
    category_bindings: list[ConflictCategory] = Field(min_length=1)
    frame_summary: str = Field(min_length=1)
    diagnostic_questions: list[str] = Field(min_length=1)
    axis_prior: AxisWeights
    failure_modes: list[str] = Field(min_length=1)

    @field_validator("category_bindings", mode="after")
    @classmethod
    def validate_category_bindings(
        cls, value: list[ConflictCategory]
    ) -> list[ConflictCategory]:
        ordered: list[ConflictCategory] = []
        seen: set[ConflictCategory] = set()
        for category in value:
            if category not in seen:
                seen.add(category)
                ordered.append(category)
        return ordered

    @field_validator("diagnostic_questions", "failure_modes", mode="after")
    @classmethod
    def validate_text_lists(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class PersonaReading(StrictModel):
    issue_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    persona_id: PersonaId
    axis_weights: AxisWeights
    alignment_score: float = Field(ge=0.0, le=1.0)
    grammar_summary: str = Field(min_length=1)
    hidden_value: str = Field(min_length=1)
    blind_spot: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_claim_ids: list[str] = Field(default_factory=list)

    @field_validator("source_claim_ids", mode="after")
    @classmethod
    def validate_source_claim_ids(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class SpeakerVector(StrictModel):
    issue_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    axis_weights: AxisWeights
    source_claim_ids: list[str] = Field(default_factory=list)

    @field_validator("source_claim_ids", mode="after")
    @classmethod
    def validate_source_claim_ids(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class DivergencePair(StrictModel):
    speaker_a: str = Field(min_length=1)
    speaker_b: str = Field(min_length=1)
    cosine_distance: float = Field(ge=0.0, le=1.0)
    per_axis_delta: AxisDelta
    shared_axes: list[str] = Field(default_factory=list)
    conflict_axes: list[str] = Field(default_factory=list)

    @field_validator("shared_axes", "conflict_axes", mode="after")
    @classmethod
    def validate_axis_lists(cls, value: list[str]) -> list[str]:
        return validate_axis_subset(value)

    @model_validator(mode="after")
    def validate_speakers(self) -> "DivergencePair":
        if self.speaker_a == self.speaker_b:
            raise ValueError("speaker_a and speaker_b must be different speakers.")
        return self


class DecisionMap(StrictModel):
    issue_id: str = Field(min_length=1)
    adopted_claim_id: str | None = None
    rejected_claim_ids: list[str] = Field(default_factory=list)
    ignored_claim_ids: list[str] = Field(default_factory=list)
    status: DecisionState

    @field_validator("rejected_claim_ids", "ignored_claim_ids", mode="after")
    @classmethod
    def validate_claim_lists(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)

    @model_validator(mode="after")
    def validate_state(self) -> "DecisionMap":
        if self.adopted_claim_id is not None and not self.adopted_claim_id.strip():
            raise ValueError("adopted_claim_id must not be blank.")
        if self.status == DecisionState.ADOPTED and self.adopted_claim_id is None:
            raise ValueError("ADOPTED status requires adopted_claim_id.")
        if self.status == DecisionState.UNRESOLVED and self.adopted_claim_id is not None:
            raise ValueError("UNRESOLVED status cannot include adopted_claim_id.")
        if self.adopted_claim_id in self.rejected_claim_ids:
            raise ValueError("adopted_claim_id must not appear in rejected_claim_ids.")
        if self.adopted_claim_id in self.ignored_claim_ids:
            raise ValueError("adopted_claim_id must not appear in ignored_claim_ids.")
        return self


class RejectedOpinionAnalysis(StrictModel):
    claim_id: str = Field(min_length=1)
    philosophical_value_score: float = Field(ge=0.0, le=1.0)
    underrepresented_axes: list[str] = Field(default_factory=list)
    structural_rejection_reasons: list[str] = Field(default_factory=list)
    salvage_conditions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("underrepresented_axes", mode="after")
    @classmethod
    def validate_underrepresented_axes(cls, value: list[str]) -> list[str]:
        return validate_axis_subset(value)

    @field_validator("structural_rejection_reasons", "salvage_conditions", mode="after")
    @classmethod
    def validate_text_lists(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class BridgePoint(StrictModel):
    issue_id: str = Field(min_length=1)
    shared_axes: list[str] = Field(default_factory=list)
    preserved_disagreements: list[str] = Field(default_factory=list)
    bridge_statement: str = Field(min_length=1)

    @field_validator("shared_axes", mode="after")
    @classmethod
    def validate_shared_axes(cls, value: list[str]) -> list[str]:
        return validate_axis_subset(value)

    @field_validator("preserved_disagreements", mode="after")
    @classmethod
    def validate_disagreements(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class IssueAnalysis(StrictModel):
    issue_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    category: CategoryResult
    persona_readings: list[PersonaReading] = Field(default_factory=list)
    speaker_vectors: list[SpeakerVector] = Field(default_factory=list)
    divergences: list[DivergencePair] = Field(default_factory=list)
    decision_map: DecisionMap
    rejected_opinions: list[RejectedOpinionAnalysis] = Field(default_factory=list)
    bridge_points: list[BridgePoint] = Field(default_factory=list)
    summary: str | None = None


class MeetingLevelSummary(StrictModel):
    top_conflict_axes: list[str] = Field(default_factory=list)
    shared_axes: list[str] = Field(default_factory=list)
    unresolved_issue_ids: list[str] = Field(default_factory=list)
    summary: str = ""

    @field_validator("top_conflict_axes", "shared_axes", mode="after")
    @classmethod
    def validate_axis_lists(cls, value: list[str]) -> list[str]:
        return validate_axis_subset(value)

    @field_validator("unresolved_issue_ids", mode="after")
    @classmethod
    def validate_issue_ids(cls, value: list[str]) -> list[str]:
        return _unique_non_empty_strings(value)


class AnalysisReport(StrictModel):
    run_metadata: RunMetadata
    input_diagnostics: InputDiagnostics
    issue_analyses: list[IssueAnalysis] = Field(default_factory=list)
    meeting_level_summary: MeetingLevelSummary
    artifacts: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "AnalysisReport",
    "AudioInput",
    "AxisDelta",
    "AxisWeights",
    "BridgePoint",
    "CategoryResult",
    "ClaimUnit",
    "DecisionMap",
    "DivergencePair",
    "InputDiagnostics",
    "IssueAnalysis",
    "IssueCluster",
    "MeetingLevelSummary",
    "PersonaReading",
    "PersonaSpec",
    "RejectedOpinionAnalysis",
    "RunMetadata",
    "SpeakerSegment",
    "SpeakerVector",
    "StrictModel",
    "Utterance",
]
