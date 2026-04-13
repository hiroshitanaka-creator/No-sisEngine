from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request

from noesis_engine.api.schemas.http import (
    AudioAnalysisRequest,
    AudioAnalysisResponse,
    TranscriptAnalysisRequest,
    TranscriptAnalysisResponse,
)
from noesis_engine.services.pipeline import AnalysisPipeline


router = APIRouter(prefix="/analyze", tags=["analyze"])


def _get_pipeline(request: Request) -> AnalysisPipeline:
    pipeline = cast(AnalysisPipeline | None, getattr(request.app.state, "pipeline", None))
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Analysis pipeline is not configured.")
    return pipeline


@router.post("/transcript", response_model=TranscriptAnalysisResponse)
def analyze_transcript(
    payload: TranscriptAnalysisRequest,
    request: Request,
) -> TranscriptAnalysisResponse:
    pipeline = _get_pipeline(request)
    report = pipeline.analyze_transcript(
        payload.utterances,
        meeting_context=payload.meeting_context,
        debug=payload.debug,
    )
    return TranscriptAnalysisResponse.model_validate(report.model_dump(mode="json"))


@router.post("/audio", response_model=AudioAnalysisResponse)
def analyze_audio(
    payload: AudioAnalysisRequest,
    request: Request,
) -> AudioAnalysisResponse:
    pipeline = _get_pipeline(request)
    report = pipeline.analyze_audio(
        payload.to_audio_input(),
        meeting_context=payload.meeting_context,
        debug=payload.debug,
    )
    return AudioAnalysisResponse.model_validate(report.model_dump(mode="json"))


__all__ = ["router"]
