from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from noesis_engine.core.schemas import AudioInput, SpeakerSegment


@dataclass(frozen=True, slots=True)
class DiarizationRunMetadata:
    provider: str
    model: str | None = None
    duration_sec: float | None = None


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    segments: tuple[SpeakerSegment, ...]
    metadata: DiarizationRunMetadata


class DiarizationPort(ABC):
    @abstractmethod
    def diarize(self, audio: AudioInput) -> DiarizationResult:
        raise NotImplementedError


__all__ = [
    "DiarizationPort",
    "DiarizationResult",
    "DiarizationRunMetadata",
]
