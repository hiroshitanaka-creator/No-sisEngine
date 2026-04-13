from noesis_engine.core.enums import ClaimActType, DecisionState
from noesis_engine.core.schemas import ClaimUnit, IssueCluster
from noesis_engine.services.decision_mapper import DecisionMapperService


def _claim(
    claim_id: str,
    speaker_id: str,
    act_type: ClaimActType,
    *,
    targets: list[str] | None = None,
) -> ClaimUnit:
    return ClaimUnit(
        claim_id=claim_id,
        speaker_id=speaker_id,
        source_utterance_ids=[f"u_{claim_id}"],
        text_span=claim_id,
        act_type=act_type,
        target_claim_ids=targets or [],
        issue_hint="generator budget",
        importance=1.0,
        confidence=1.0,
        explicit_values=[],
        implicit_values=[],
    )


def test_decision_mapper_marks_adopted_and_rejected_claims() -> None:
    issue = IssueCluster(
        issue_id="issue_001",
        label="generator budget",
        claim_ids=["p1", "p2", "s1", "o1", "d1"],
        speaker_ids=["alice", "bob"],
    )
    claims = [
        _claim("p1", "alice", ClaimActType.PROPOSAL),
        _claim("p2", "bob", ClaimActType.PROPOSAL),
        _claim("s1", "bob", ClaimActType.SUPPORT, targets=["p1"]),
        _claim("o1", "alice", ClaimActType.OBJECTION, targets=["p2"]),
        _claim("d1", "alice", ClaimActType.DECISION, targets=["p1"]),
    ]

    decision_map = DecisionMapperService().map_issue(issue, claims)

    assert decision_map.adopted_claim_id == "p1"
    assert decision_map.rejected_claim_ids == ["p2"]
    assert decision_map.ignored_claim_ids == []
    assert decision_map.status == DecisionState.ADOPTED


def test_decision_mapper_marks_rejected_issue() -> None:
    issue = IssueCluster(
        issue_id="issue_001",
        label="generator budget",
        claim_ids=["p1", "o1"],
        speaker_ids=["alice", "bob"],
    )
    claims = [
        _claim("p1", "alice", ClaimActType.PROPOSAL),
        _claim("o1", "bob", ClaimActType.OBJECTION, targets=["p1"]),
    ]

    decision_map = DecisionMapperService().map_issue(issue, claims)

    assert decision_map.adopted_claim_id is None
    assert decision_map.rejected_claim_ids == ["p1"]
    assert decision_map.status == DecisionState.REJECTED


def test_decision_mapper_marks_ignored_issue() -> None:
    issue = IssueCluster(
        issue_id="issue_001",
        label="generator budget",
        claim_ids=["p1"],
        speaker_ids=["alice"],
    )
    claims = [_claim("p1", "alice", ClaimActType.PROPOSAL)]

    decision_map = DecisionMapperService().map_issue(issue, claims)

    assert decision_map.adopted_claim_id is None
    assert decision_map.ignored_claim_ids == ["p1"]
    assert decision_map.status == DecisionState.IGNORED


def test_decision_mapper_marks_unresolved_issue() -> None:
    issue = IssueCluster(
        issue_id="issue_001",
        label="generator budget",
        claim_ids=["p1", "p2", "o1"],
        speaker_ids=["alice", "bob"],
    )
    claims = [
        _claim("p1", "alice", ClaimActType.PROPOSAL),
        _claim("p2", "bob", ClaimActType.PROPOSAL),
        _claim("o1", "alice", ClaimActType.OBJECTION, targets=["p2"]),
    ]

    decision_map = DecisionMapperService().map_issue(issue, claims)

    assert decision_map.adopted_claim_id is None
    assert decision_map.rejected_claim_ids == ["p2"]
    assert decision_map.ignored_claim_ids == ["p1"]
    assert decision_map.status == DecisionState.UNRESOLVED
