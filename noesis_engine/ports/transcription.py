from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from noesis_engine.core.schemas import AudioInput, Utterance


@dataclass(frozen=True, slots=True)
class TranscriptionRunMetadata:
    provider: str
    model: str | None = None
    language: str | None = None
    duration_sec: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    utterances: tuple[Utterance, ...]
    metadata: TranscriptionRunMetadata


class TranscriptionPort(ABC):
    @abstractmethod
    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        raise NotImplementedError


__all__ = [
    "TranscriptionPort",
    "TranscriptionResult",
    "TranscriptionRunMetadata",
]
