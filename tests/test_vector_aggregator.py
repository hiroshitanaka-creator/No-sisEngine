from noesis_engine.core.axes import CANONICAL_AXIS_KEYS, normalize_axis_vector
from noesis_engine.core.enums import PersonaId
from noesis_engine.core.schemas import ClaimUnit, IssueCluster, PersonaReading
from noesis_engine.services.vector_aggregator import VectorAggregatorService


def _full_axis(**weights: float) -> dict[str, float]:
    return normalize_axis_vector(weights)


def test_vector_aggregator_normalizes_weighted_sum() -> None:
    issue = IssueCluster(
        issue_id="issue_001",
        label="budget tradeoff",
        claim_ids=["c1"],
        speaker_ids=["alice"],
    )
    claims = [
        ClaimUnit(
            claim_id="c1",
            speaker_id="alice",
            source_utterance_ids=["u1"],
            text_span="Cut the generator budget.",
            act_type="proposal",
            target_claim_ids=[],
            issue_hint="budget tradeoff",
            importance=1.0,
            confidence=1.0,
            explicit_values=[],
            implicit_values=[],
        )
    ]
    readings = [
        PersonaReading(
            issue_id="issue_001",
            speaker_id="alice",
            persona_id=PersonaId.KANT,
            axis_weights=_full_axis(principle_constraint=1.0),
            alignment_score=1.0,
            grammar_summary="Rules first.",
            hidden_value="Constraint preservation.",
            blind_spot="Aggregate efficiency.",
            confidence=1.0,
            source_claim_ids=["c1"],
        ),
        PersonaReading(
            issue_id="issue_001",
            speaker_id="alice",
            persona_id=PersonaId.MILL,
            axis_weights=_full_axis(consequence_utility=1.0),
            alignment_score=0.5,
            grammar_summary="Utility first.",
            hidden_value="Overall welfare.",
            blind_spot="Rule rigidity.",
            confidence=1.0,
            source_claim_ids=["c1"],
        ),
    ]

    service = VectorAggregatorService()
    vectors = service.aggregate_issue(issue, claims, readings)

    assert len(vectors) == 1
    vector = vectors[0]
    assert list(vector.axis_weights.keys()) == list(CANONICAL_AXIS_KEYS)
    assert vector.axis_weights["principle_constraint"] == 2.0 / 3.0
    assert vector.axis_weights["consequence_utility"] == 1.0 / 3.0


def test_vector_aggregator_is_deterministic() -> None:
    issue = IssueCluster(
        issue_id="issue_001",
        label="budget tradeoff",
        claim_ids=["c1"],
        speaker_ids=["alice"],
    )
    claims = [
        ClaimUnit(
            claim_id="c1",
            speaker_id="alice",
            source_utterance_ids=["u1"],
            text_span="Cut the generator budget.",
            act_type="proposal",
            target_claim_ids=[],
            issue_hint="budget tradeoff",
            importance=0.8,
            confidence=1.0,
            explicit_values=[],
            implicit_values=[],
        )
    ]
    readings = [
        PersonaReading(
            issue_id="issue_001",
            speaker_id="alice",
            persona_id=PersonaId.ARISTOTLE,
            axis_weights=_full_axis(virtue_excellence=1.0),
            alignment_score=0.9,
            grammar_summary="Practical wisdom.",
            hidden_value="Craft excellence.",
            blind_spot="Explicit utility accounting.",
            confidence=1.0,
            source_claim_ids=["c1"],
        )
    ]

    service = VectorAggregatorService()
    first = service.aggregate_issue(issue, claims, readings)[0]
    second = service.aggregate_issue(issue, claims, readings)[0]

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
