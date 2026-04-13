from __future__ import annotations

import base64

from pydantic import Field, model_validator

from noesis_engine.core.schemas import AnalysisReport, AudioInput, StrictModel, Utterance


class TranscriptAnalysisRequest(StrictModel):
    utterances: list[Utterance] = Field(min_length=1)
    meeting_context: str | None = None
    debug: bool = False


class AudioAnalysisRequest(StrictModel):
    path: str | None = None
    content_base64: str | None = None
    filename: str | None = None
    content_type: str | None = None
    meeting_context: str | None = None
    debug: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> "AudioAnalysisRequest":
        has_path = self.path is not None and self.path.strip() != ""
        has_content = self.content_base64 is not None and self.content_base64.strip() != ""
        if has_path == has_content:
            raise ValueError("Provide exactly one of 'path' or 'content_base64'.")
        return self

    def to_audio_input(self) -> AudioInput:
        content: bytes | None = None
        if self.content_base64 is not None:
            content = base64.b64decode(self.content_base64.encode("ascii"))
        return AudioInput(
            path=self.path,
            content=content,
            filename=self.filename,
            content_type=self.content_type,
        )


class TranscriptAnalysisResponse(AnalysisReport):
    pass


class AudioAnalysisResponse(AnalysisReport):
    pass


class HealthResponse(StrictModel):
    status: str
    version: str
    analysis_model: str


__all__ = [
    "AudioAnalysisRequest",
    "AudioAnalysisResponse",
    "HealthResponse",
    "TranscriptAnalysisRequest",
    "TranscriptAnalysisResponse",
]
