import json

from noesis_engine.core.axes import normalize_axis_vector
from noesis_engine.core.enums import ConflictCategory
from noesis_engine.core.schemas import CategoryResult, RejectedOpinionAnalysis, Utterance
from noesis_engine.ports.llm import LLMPort, LLMRunMetadata, StructuredGenerationResult, StructuredPrompt
from noesis_engine.prompts.claim_decomposition import ClaimDecompositionEnvelope, ClaimDraft
from noesis_engine.prompts.persona_analysis import PersonaAnalysisEnvelope, PersonaReadingDraft
from noesis_engine.services.pipeline import AnalysisPipeline
from noesis_engine.settings import Settings


class MockLLMPort(LLMPort):
    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def generate_structured(self, prompt: StructuredPrompt):
        stage = str(prompt.metadata.get("stage"))
        payload = json.loads(prompt.user_input)
        metadata = LLMRunMetadata(provider="mock", model="mock-model")

        if stage == "claim_decomposition":
            output = ClaimDecompositionEnvelope(
                chunk_id=payload["chunk_id"],
                claims=[
                    ClaimDraft(
                        local_claim_id="c1",
                        speaker_id="alice",
                        source_utterance_ids=["u1"],
                        text_span="Cut the backup generator budget to ship this quarter.",
                        act_type="proposal",
                        target_local_claim_ids=[],
                        issue_hint="generator budget vs delivery schedule",
                        importance=0.9,
                        confidence=0.95,
                        explicit_values=["speed"],
                        implicit_values=["resource efficiency"],
                    ),
                    ClaimDraft(
                        local_claim_id="c2",
                        speaker_id="bob",
                        source_utterance_ids=["u2"],
                        text_span="Cutting the generator budget increases outage risk.",
                        act_type="objection",
                        target_local_claim_ids=["c1"],
                        issue_hint="generator budget vs delivery schedule",
                        importance=0.85,
                        confidence=0.95,
                        explicit_values=["risk"],
                        implicit_values=["resilience"],
                    ),
                    ClaimDraft(
                        local_claim_id="c3",
                        speaker_id="bob",
                        source_utterance_ids=["u2"],
                        text_span="Keep the generator budget and defer one noncritical feature.",
                        act_type="proposal",
                        target_local_claim_ids=[],
                        issue_hint="generator budget vs delivery schedule",
                        importance=0.9,
                        confidence=0.95,
                        explicit_values=["reliability"],
                        implicit_values=["long-term stability"],
                    ),
                    ClaimDraft(
                        local_claim_id="c4",
                        speaker_id="alice",
                        source_utterance_ids=["u3"],
                        text_span="Decide to keep the generator budget and defer one feature.",
                        act_type="decision",
                        target_local_claim_ids=["c3"],
                        issue_hint="generator budget vs delivery schedule",
                        importance=0.95,
                        confidence=0.99,
                        explicit_values=["decision"],
                        implicit_values=[],
                    ),
                ],
            )
            return StructuredGenerationResult(output=output, metadata=metadata, raw_text=None)

        if stage == "category_classification":
            output = CategoryResult(
                issue_id=payload["issue"]["issue_id"],
                primary_category=ConflictCategory.RESOURCE_ALLOCATION,
                secondary_categories=[ConflictCategory.TEMPORAL_HORIZON],
                signals=["budget tradeoff", "risk allocation"],
                confidence=0.93,
            )
            return StructuredGenerationResult(output=output, metadata=metadata, raw_text=None)

        if stage == "persona_analysis":
            issue_id = payload["issue"]["issue_id"]
            persona_id = payload["persona_spec"]["persona_id"]
            claims = payload["claims"]
            alice_claim_ids = [claim["claim_id"] for claim in claims if claim["speaker_id"] == "alice"]
            bob_claim_ids = [claim["claim_id"] for claim in claims if claim["speaker_id"] == "bob"]

            output = PersonaAnalysisEnvelope(
                issue_id=issue_id,
                persona_id=persona_id,
                readings=[
                    PersonaReadingDraft(
                        speaker_id="alice",
                        axis_weights=normalize_axis_vector(
                            {
                                "resource_efficiency": 0.35,
                                "short_term_urgency": 0.30,
                                "consequence_utility": 0.20,
                                "innovation_optionality": 0.15,
                            }
                        ),
                        alignment_score=0.85,
                        grammar_summary="Alice frames the issue as a delivery-speed tradeoff under resource pressure.",
                        hidden_value="Preserves shipping momentum and budget efficiency.",
                        blind_spot="Underweights outage risk and resilience.",
                        confidence=0.88,
                        source_claim_ids=alice_claim_ids,
                    ),
                    PersonaReadingDraft(
                        speaker_id="bob",
                        axis_weights=normalize_axis_vector(
                            {
                                "risk_precaution": 0.35,
                                "long_term_sustainability": 0.30,
                                "relational_fairness": 0.20,
                                "principle_constraint": 0.15,
                            }
                        ),
                        alignment_score=0.9,
                        grammar_summary="Bob frames the issue as a resilience-preserving safeguard against foreseeable harm.",
                        hidden_value="Preserves infrastructure reliability and future protection.",
                        blind_spot="Underweights schedule pressure and short-term delivery incentives.",
                        confidence=0.9,
                        source_claim_ids=bob_claim_ids,
                    ),
                ],
            )
            return StructuredGenerationResult(output=output, metadata=metadata, raw_text=None)

        if stage == "rejection_analysis":
            claim_id = payload["claim"]["claim_id"]
            candidate_axes = payload["candidate_underrepresented_axes"]
            output = RejectedOpinionAnalysis(
                claim_id=claim_id,
                philosophical_value_score=0.64,
                underrepresented_axes=candidate_axes[:2],
                structural_rejection_reasons=[
                    "The claim lost to a stronger risk-preserving alternative.",
                    "The discussion center of gravity shifted toward resilience and precaution.",
                ],
                salvage_conditions=[
                    "Retain the claim as a contingent fallback if schedule slippage exceeds threshold.",
                    "Reintroduce the efficiency concern during later feature-prioritization review.",
                ],
                confidence=0.87,
            )
            return StructuredGenerationResult(output=output, metadata=metadata, raw_text=None)

        raise AssertionError(f"Unexpected stage: {stage}")


def test_transcript_pipeline_runs_end_to_end() -> None:
    utterances = [
        Utterance(
            utterance_id="u1",
            speaker_id="alice",
            text="We should cut the backup generator budget and ship this quarter.",
            start_sec=0.0,
            end_sec=4.0,
        ),
        Utterance(
            utterance_id="u2",
            speaker_id="bob",
            text="I object. Cutting it increases outage risk, so keep the budget and defer one feature.",
            start_sec=4.0,
            end_sec=9.0,
        ),
        Utterance(
            utterance_id="u3",
            speaker_id="alice",
            text="Then decide to keep the generator budget and defer one feature.",
            start_sec=9.0,
            end_sec=12.0,
        ),
    ]
    settings = Settings()
    pipeline = AnalysisPipeline(llm=MockLLMPort(), settings=settings)

    report = pipeline.analyze_transcript(utterances, meeting_context="Quarterly budget review", debug=False)

    assert report.input_diagnostics.utterance_count == 3
    assert len(report.issue_analyses) >= 1

    issue = report.issue_analyses[0]
    assert issue.category.primary_category == ConflictCategory.RESOURCE_ALLOCATION
    assert issue.decision_map.adopted_claim_id == "clm_0003"
    assert issue.decision_map.rejected_claim_ids == ["clm_0001"]
    assert issue.rejected_opinions
    assert issue.bridge_points
