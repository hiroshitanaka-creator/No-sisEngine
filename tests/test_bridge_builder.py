from noesis_engine.core.axes import CANONICAL_AXIS_KEYS, normalize_axis_vector
from noesis_engine.core.schemas import DivergencePair, IssueCluster, SpeakerVector
from noesis_engine.services.bridge_builder import BridgeBuilderService


def _make_issue() -> IssueCluster:
    return IssueCluster(
        issue_id="issue_001",
        label="generator budget",
        claim_ids=["c1", "c2"],
        speaker_ids=["alice", "bob"],
    )


def _make_vector(speaker_id: str, **weights: float) -> SpeakerVector:
    return SpeakerVector(
        issue_id="issue_001",
        speaker_id=speaker_id,
        axis_weights=normalize_axis_vector(weights),
        source_claim_ids=["c1"],
    )


def _make_divergence(*, shared_delta: float, conflict_delta: float) -> DivergencePair:
    per_axis = {axis: shared_delta for axis in CANONICAL_AXIS_KEYS}
    per_axis["risk_precaution"] = conflict_delta
    return DivergencePair(
        speaker_a="alice",
        speaker_b="bob",
        cosine_distance=0.5,
        per_axis_delta={axis: float(v) for axis, v in per_axis.items()},
        conflict_axes=["risk_precaution"],
        shared_axes=["consequence_utility"],
    )


def test_bridge_builder_returns_empty_for_no_vectors() -> None:
    service = BridgeBuilderService()
    result = service.build_for_issue(_make_issue(), [], [])
    assert result == []


def test_bridge_builder_no_divergences_uses_vector_weights() -> None:
    service = BridgeBuilderService()
    vectors = [_make_vector("alice", principle_constraint=1.0)]
    result = service.build_for_issue(_make_issue(), vectors, [])
    assert len(result) == 1
    bp = result[0]
    assert "principle_constraint" in bp.shared_axes
    assert bp.preserved_disagreements == []
    assert "coordinate on" in bp.bridge_statement


def test_bridge_builder_with_divergences_shared_and_conflict() -> None:
    service = BridgeBuilderService(
        shared_axis_delta_threshold=0.10,
        conflict_axis_delta_threshold=0.25,
    )
    vectors = [
        _make_vector("alice", principle_constraint=0.6, risk_precaution=0.4),
        _make_vector("bob", principle_constraint=0.3, risk_precaution=0.7),
    ]
    divergence = _make_divergence(shared_delta=0.05, conflict_delta=0.30)
    result = service.build_for_issue(_make_issue(), vectors, [divergence])
    assert len(result) == 1
    bp = result[0]
    assert "risk_precaution" in bp.preserved_disagreements
    assert "coordinate on" in bp.bridge_statement
    assert "preserving disagreement on" in bp.bridge_statement


def test_bridge_builder_only_preserved_disagreements() -> None:
    service = BridgeBuilderService(
        shared_axis_delta_threshold=0.01,
        conflict_axis_delta_threshold=0.01,
    )
    vectors = [
        _make_vector("alice", risk_precaution=1.0),
        _make_vector("bob", consequence_utility=1.0),
    ]
    per_axis = {axis: 0.5 for axis in CANONICAL_AXIS_KEYS}
    divergence = DivergencePair(
        speaker_a="alice",
        speaker_b="bob",
        cosine_distance=0.9,
        per_axis_delta=per_axis,
        conflict_axes=list(CANONICAL_AXIS_KEYS),
        shared_axes=[],
    )
    result = service.build_for_issue(_make_issue(), vectors, [divergence])
    assert len(result) == 1
    bp = result[0]
    assert bp.shared_axes == []
    assert len(bp.preserved_disagreements) > 0
    assert "No strong shared axis" in bp.bridge_statement


def test_bridge_builder_no_signal_at_all() -> None:
    # Deltas between shared_threshold (0.05) and conflict_threshold (0.30)
    # → no shared axes, no preserved disagreements → "Insufficient signal"
    service = BridgeBuilderService(
        shared_axis_delta_threshold=0.05,
        conflict_axis_delta_threshold=0.30,
    )
    vectors = [_make_vector("alice", consequence_utility=1.0)]
    per_axis = {axis: 0.15 for axis in CANONICAL_AXIS_KEYS}
    divergence = DivergencePair(
        speaker_a="alice",
        speaker_b="bob",
        cosine_distance=0.3,
        per_axis_delta=per_axis,
        conflict_axes=[],
        shared_axes=[],
    )
    result = service.build_for_issue(_make_issue(), vectors, [divergence])
    assert len(result) == 1
    assert "Insufficient signal" in result[0].bridge_statement
