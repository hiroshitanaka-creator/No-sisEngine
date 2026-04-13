from __future__ import annotations

from fastapi import FastAPI

from noesis_engine import __version__
from noesis_engine.adapters.audio.pyannote_adapter import PyannoteDiarizationAdapter
from noesis_engine.adapters.audio.whisper_adapter import WhisperTranscriptionAdapter
from noesis_engine.adapters.llm.local_client import LocalLLMClient
from noesis_engine.adapters.llm.openai_client import OpenAIClient
from noesis_engine.api.routers.analyze import router as analyze_router
from noesis_engine.api.schemas.http import HealthResponse
from noesis_engine.services.pipeline import AnalysisPipeline
from noesis_engine.settings import Settings, get_settings


def _build_default_pipeline(settings: Settings) -> AnalysisPipeline:
    if settings.openai.api_key:
        llm = OpenAIClient(settings)
    else:
        llm = LocalLLMClient(settings)

    return AnalysisPipeline(
        llm=llm,
        settings=settings,
        transcription_port=WhisperTranscriptionAdapter(settings),
        diarization_port=PyannoteDiarizationAdapter(settings),
    )


def create_app(
    *,
    settings: Settings | None = None,
    pipeline: AnalysisPipeline | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_pipeline = pipeline or _build_default_pipeline(resolved_settings)

    app = FastAPI(title="NoesisEngine", version=__version__)
    app.state.settings = resolved_settings
    app.state.pipeline = resolved_pipeline

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            analysis_model=resolved_pipeline.llm.model_name,
        )

    app.include_router(analyze_router)
    return app


app = create_app()


__all__ = ["app", "create_app"]
