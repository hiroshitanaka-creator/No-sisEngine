from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from noesis_engine.core.enums import AnalysisStatus
from noesis_engine.core.schemas import (
    AnalysisReport,
    AudioInput,
    CategoryResult,
    ClaimUnit,
    DecisionMap,
    DivergencePair,
    InputDiagnostics,
    IssueCluster,
    PersonaReading,
    RunMetadata,
    SpeakerVector,
    Utterance,
)
from noesis_engine.ports.diarization import DiarizationPort
from noesis_engine.ports.llm import LLMPort
from noesis_engine.ports.transcription import TranscriptionPort
from noesis_engine.services.bridge_builder import BridgeBuilderService
from noesis_engine.services.category_classifier import CategoryClassifierService
from noesis_engine.services.claim_decomposer import ClaimDecomposerService
from noesis_engine.services.decision_mapper import DecisionMapperService
from noesis_engine.services.divergence_analyzer import DivergenceAnalyzerService
from noesis_engine.services.issue_clusterer import IssueClustererService
from noesis_engine.services.persona_analyzer import PersonaAnalyzerService
from noesis_engine.services.persona_router import PersonaRouterService
from noesis_engine.services.rejected_value_evaluator import RejectedValueEvaluatorService
from noesis_engine.services.report_builder import ReportBuilderService
from noesis_engine.services.speaker_segmenter import assign_speakers_to_utterances
from noesis_engine.services.transcript_normalizer import chunk_transcript, normalize_transcript
from noesis_engine.services.vector_aggregator import VectorAggregatorService
from noesis_engine.settings import Settings, get_settings


class AnalysisPipeline:
    def __init__(
        self,
        *,
        llm: LLMPort,
        settings: Settings | None = None,
        transcription_port: TranscriptionPort | None = None,
        diarization_port: DiarizationPort | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.transcription_port = transcription_port
        self.diarization_port = diarization_port

        self.claim_decomposer = ClaimDecomposerService(
            llm,
            max_repair_attempts=self.settings.max_repair_attempts,
        )
        self.issue_clusterer = IssueClustererService()
        self.category_classifier = CategoryClassifierService(
            llm,
            max_repair_attempts=self.settings.max_repair_attempts,
        )
        self.persona_router = PersonaRouterService()
        self.persona_analyzer = PersonaAnalyzerService(
            llm,
            max_repair_attempts=self.settings.max_repair_attempts,
        )
        self.vector_aggregator = VectorAggregatorService()
        self.divergence_analyzer = DivergenceAnalyzerService(
            shared_axis_delta_threshold=self.settings.thresholds.shared_axis_delta_threshold,
            conflict_axis_delta_threshold=self.settings.thresholds.conflict_axis_delta_threshold,
        )
        self.decision_mapper = DecisionMapperService()
        self.rejected_value_evaluator = RejectedValueEvaluatorService(
            llm,
            max_repair_attempts=self.settings.max_repair_attempts,
        )
        self.bridge_builder = BridgeBuilderService(
            shared_axis_delta_threshold=self.settings.thresholds.shared_axis_delta_threshold,
            conflict_axis_delta_threshold=self.settings.thresholds.conflict_axis_delta_threshold,
        )
        self.report_builder = ReportBuilderService()

    def analyze_transcript(
        self,
        utterances: list[Utterance],
        *,
        meeting_context: str | None = None,
        debug: bool = False,
    ) -> AnalysisReport:
        normalized = normalize_transcript(utterances)
        speaker_tagged = assign_speakers_to_utterances(normalized)
        return self._analyze_prepared_utterances(
            speaker_tagged,
            input_kind="transcript",
            meeting_context=meeting_context,
            debug=debug,
        )

    def analyze_audio(
        self,
        audio: AudioInput,
        *,
        meeting_context: str | None = None,
        debug: bool = False,
    ) -> AnalysisReport:
        if self.transcription_port is None:
            raise RuntimeError("Audio analysis requires a configured transcription_port.")
        transcription = self.transcription_port.transcribe(audio)
        utterances = normalize_transcript(list(transcription.utterances))

        diarized_segments = []
        if self.diarization_port is not None:
            diarization = self.diarization_port.diarize(audio)
            diarized_segments = list(diarization.segments)

        speaker_tagged = assign_speakers_to_utterances(
            utterances,
            diarization_segments=diarized_segments,
            min_overlap_ratio=self.settings.thresholds.speaker_merge_overlap_ratio,
        )
        return self._analyze_prepared_utterances(
            speaker_tagged,
            input_kind="audio",
            meeting_context=meeting_context,
            debug=debug,
        )

    def _analyze_prepared_utterances(
        self,
        utterances: list[Utterance],
        *,
        input_kind: str,
        meeting_context: str | None,
        debug: bool,
    ) -> AnalysisReport:
        started_at = datetime.now(timezone.utc)
        run_id = uuid4().hex

        chunks = chunk_transcript(
            utterances,
            max_chars=self.settings.chunking.transcript_chunk_size_chars,
            overlap_chars=self.settings.chunking.transcript_chunk_overlap_chars,
        )
        claims = self.claim_decomposer.decompose_chunks(
            chunks,
            meeting_context=meeting_context,
        )
        issues = self.issue_clusterer.cluster_claims(claims)

        claims_by_id = {claim.claim_id: claim for claim in claims}
        issue_claims = {
            issue.issue_id: [claims_by_id[claim_id] for claim_id in issue.claim_ids if claim_id in claims_by_id]
            for issue in issues
        }

        categories: dict[str, CategoryResult] = {}
        persona_readings: dict[str, list[PersonaReading]] = {}
        speaker_vectors: dict[str, list[SpeakerVector]] = {}
        divergences: dict[str, list[DivergencePair]] = {}
        decision_maps: dict[str, DecisionMap] = {}
        rejected_opinions: dict[str, list[object]] = {}
        bridge_points: dict[str, list[object]] = {}

        for issue in issues:
            issue_claim_list = issue_claims[issue.issue_id]
            category = self.category_classifier.classify_issue(
                issue,
                issue_claim_list,
                meeting_context=meeting_context,
            )
            categories[issue.issue_id] = category

            personas = self.persona_router.route_from_result(category)
            readings = self.persona_analyzer.analyze_issue(
                issue,
                issue_claim_list,
                personas,
                meeting_context=meeting_context,
            )
            persona_readings[issue.issue_id] = readings

            vectors = self.vector_aggregator.aggregate_issue(issue, issue_claim_list, readings)
            speaker_vectors[issue.issue_id] = vectors

            issue_divergences = self.divergence_analyzer.analyze_issue(issue, vectors)
            divergences[issue.issue_id] = issue_divergences

            decision_map = self.decision_mapper.map_issue(issue, issue_claim_list)
            decision_maps[issue.issue_id] = decision_map

            rejected = self.rejected_value_evaluator.evaluate_issue(
                issue,
                issue_claim_list,
                category,
                decision_map,
                vectors,
                issue_divergences,
                meeting_context=meeting_context,
            )
            rejected_opinions[issue.issue_id] = rejected

            bridges = self.bridge_builder.build_for_issue(issue, vectors, issue_divergences)
            bridge_points[issue.issue_id] = bridges

        warnings: list[str] = []
        if len({utterance.speaker_id for utterance in utterances}) < 2:
            warnings.append("Only one speaker detected; divergence analysis may be limited.")
        if not issues:
            warnings.append("No issue clusters were extracted from the transcript.")

        input_diagnostics = InputDiagnostics(
            input_kind=input_kind,
            utterance_count=len(utterances),
            speaker_count=len({utterance.speaker_id for utterance in utterances}),
            issue_count=len(issues),
            warnings=warnings,
        )

        run_metadata = RunMetadata(
            run_id=run_id,
            analysis_model=getattr(self.llm, "model_name", self.settings.locked_analysis_model),
            temperature=self.settings.analysis_temperature,
            status=AnalysisStatus.SUCCESS,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

        include_debug_artifacts = debug or self.settings.include_debug_artifacts
        artifacts = self._build_debug_artifacts(
            utterances=utterances,
            chunks=chunks,
            claims=claims,
            issues=issues,
            categories=categories,
        )

        return self.report_builder.build(
            run_metadata=run_metadata,
            input_diagnostics=input_diagnostics,
            issues=issues,
            categories=categories,
            persona_readings=persona_readings,
            speaker_vectors=speaker_vectors,
            divergences=divergences,
            decision_maps=decision_maps,
            rejected_opinions=rejected_opinions,
            bridge_points=bridge_points,
            artifacts=artifacts,
            include_debug_artifacts=include_debug_artifacts,
        )

    def _build_debug_artifacts(
        self,
        *,
        utterances: list[Utterance],
        chunks: list[object],
        claims: list[ClaimUnit],
        issues: list[IssueCluster],
        categories: dict[str, CategoryResult],
    ) -> dict[str, object]:
        return {
            "normalized_utterances": [utterance.model_dump(mode="json") for utterance in utterances],
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "char_count": chunk.char_count,
                    "utterance_ids": [utterance.utterance_id for utterance in chunk.utterances],
                }
                for chunk in chunks
            ],
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "issues": [issue.model_dump(mode="json") for issue in issues],
            "categories": {
                issue_id: category.model_dump(mode="json")
                for issue_id, category in categories.items()
            },
        }


__all__ = ["AnalysisPipeline"]
