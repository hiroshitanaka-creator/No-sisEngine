from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript_chunk_size_chars: int = Field(default=12000, ge=1000)
    transcript_chunk_overlap_chars: int = Field(default=500, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingSettings":
        if self.transcript_chunk_overlap_chars >= self.transcript_chunk_size_chars:
            raise ValueError(
                "transcript_chunk_overlap_chars must be smaller than transcript_chunk_size_chars."
            )
        return self


class ThresholdSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shared_axis_delta_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    conflict_axis_delta_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    rejected_value_min_score: float = Field(default=0.50, ge=0.0, le=1.0)
    speaker_merge_overlap_ratio: float = Field(default=0.50, ge=0.0, le=1.0)


class OpenAIAdapterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    timeout_sec: float = Field(default=60.0, gt=0.0)


class LocalLLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(default="http://localhost:11434/v1", min_length=1)
    api_key: str = Field(default="not-required", min_length=1)
    model: str = Field(default="noesis-local", min_length=1)
    timeout_sec: float = Field(default=60.0, gt=0.0)


class AudioAdapterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcription_backend: str = Field(default="whisper", min_length=1)
    transcription_model: str = Field(default="base", min_length=1)
    diarization_backend: str = Field(default="pyannote", min_length=1)
    diarization_model: str | None = None
    language: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NOESIS_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    analysis_model: str = Field(default="gpt-4.1", min_length=1)
    analysis_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_repair_attempts: int = Field(default=1, ge=0, le=1)
    include_debug_artifacts: bool = False

    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    openai: OpenAIAdapterSettings = Field(default_factory=OpenAIAdapterSettings)
    local_llm: LocalLLMSettings = Field(default_factory=LocalLLMSettings)
    audio: AudioAdapterSettings = Field(default_factory=AudioAdapterSettings)

    @field_validator("analysis_model", mode="after")
    @classmethod
    def validate_analysis_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("analysis_model must not be blank.")
        return normalized

    @property
    def locked_analysis_model(self) -> str:
        return self.analysis_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = [
    "AudioAdapterSettings",
    "ChunkingSettings",
    "LocalLLMSettings",
    "OpenAIAdapterSettings",
    "Settings",
    "ThresholdSettings",
    "get_settings",
]
