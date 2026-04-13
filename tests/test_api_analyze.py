from fastapi.testclient import TestClient

from noesis_engine.api.app import create_app
from noesis_engine.core.enums import AnalysisStatus, ConflictCategory, DecisionState
from noesis_engine.core.schemas import (
    AnalysisReport,
    CategoryResult,
    DecisionMap,
    InputDiagnostics,
    IssueAnalysis,
    MeetingLevelSummary,
    RunMetadata,
)
from noesis_engine.settings import Settings


class DummyPipeline:
    def analyze_transcript(self, utterances, *, meeting_context=None, debug=False):
        issue = IssueAnalysis(
            issue_id="issue_001",
            label="generator budget",
            category=CategoryResult(
                issue_id="issue_001",
                primary_category=ConflictCategory.RESOURCE_ALLOCATION,
                secondary_categories=[],
                signals=["budget tradeoff"],
                confidence=0.9,
            ),
            persona_readings=[],
            speaker_vectors=[],
            divergences=[],
            decision_map=DecisionMap(
                issue_id="issue_001",
                adopted_claim_id=None,
                rejected_claim_ids=[],
                ignored_claim_ids=["clm_0001"],
                status=DecisionState.IGNORED,
            ),
            rejected_opinions=[],
            bridge_points=[],
            summary="Primary category: resource_allocation. Decision status: ignored.",
        )
        return AnalysisReport(
            run_metadata=RunMetadata(
                run_id="run_001",
                analysis_model="mock-model",
                temperature=0.2,
                status=AnalysisStatus.SUCCESS,
            ),
            input_diagnostics=InputDiagnostics(
                input_kind="transcript",
                utterance_count=len(utterances),
                speaker_count=len({utterance.speaker_id for utterance in utterances}),
                issue_count=1,
                warnings=[],
            ),
            issue_analyses=[issue],
            meeting_level_summary=MeetingLevelSummary(
                top_conflict_axes=[],
                shared_axes=[],
                unresolved_issue_ids=[],
                summary="Analyzed 1 issues.",
            ),
            artifacts={},
        )

    def analyze_audio(self, audio, *, meeting_context=None, debug=False):
        return self.analyze_transcript([], meeting_context=meeting_context, debug=debug)


def test_transcript_endpoint_returns_schema_valid_report() -> None:
    app = create_app(settings=Settings(), pipeline=DummyPipeline())
    client = TestClient(app)

    response = client.post(
        "/analyze/transcript",
        json={
            "utterances": [
                {
                    "utterance_id": "u1",
                    "speaker_id": "alice",
                    "text": "Cut the generator budget.",
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                }
            ],
            "meeting_context": "Quarterly budget review",
            "debug": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_metadata"]["analysis_model"] == "mock-model"
    assert payload["input_diagnostics"]["utterance_count"] == 1
    assert payload["issue_analyses"][0]["category"]["primary_category"] == "resource_allocation"


def test_transcript_endpoint_surfaces_validation_errors() -> None:
    app = create_app(settings=Settings(), pipeline=DummyPipeline())
    client = TestClient(app)

    response = client.post(
        "/analyze/transcript",
        json={
            "utterances": [
                {
                    "utterance_id": "u1",
                    "speaker_id": "",
                    "text": "Cut the generator budget.",
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                }
            ]
        },
    )

    assert response.status_code == 422
