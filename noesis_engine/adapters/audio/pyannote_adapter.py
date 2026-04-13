from __future__ import annotations

import os
import tempfile
from importlib import import_module

from noesis_engine.core.schemas import AudioInput, SpeakerSegment
from noesis_engine.ports.diarization import (
    DiarizationPort,
    DiarizationResult,
    DiarizationRunMetadata,
)
from noesis_engine.settings import Settings, get_settings


class PyannoteDiarizationAdapter(DiarizationPort):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model_name = self._settings.audio.diarization_model

    def diarize(self, audio: AudioInput) -> DiarizationResult:
        if not self._model_name:
            raise RuntimeError("audio.diarization_model must be configured for pyannote diarization.")

        pyannote_audio = import_module("pyannote.audio")
        pipeline_class = getattr(pyannote_audio, "Pipeline")
        pipeline = pipeline_class.from_pretrained(self._model_name)

        audio_path, cleanup_path = self._materialize_audio(audio)
        try:
            annotation = pipeline(audio_path)
        finally:
            if cleanup_path is not None and os.path.exists(cleanup_path):
                os.remove(cleanup_path)

        segments: list[SpeakerSegment] = []
        for index, (turn, _track, speaker) in enumerate(annotation.itertracks(yield_label=True), start=1):
            segments.append(
                SpeakerSegment(
                    segment_id=f"seg_{index:04d}",
                    speaker_id=str(speaker),
                    start_sec=float(turn.start),
                    end_sec=float(turn.end),
                    confidence=None,
                )
            )

        duration_sec = max((segment.end_sec for segment in segments), default=0.0)
        return DiarizationResult(
            segments=tuple(segments),
            metadata=DiarizationRunMetadata(
                provider="pyannote",
                model=self._model_name,
                duration_sec=duration_sec,
            ),
        )

    def _materialize_audio(self, audio: AudioInput) -> tuple[str, str | None]:
        if audio.path is not None:
            return audio.path, None

        suffix = ""
        if audio.filename and "." in audio.filename:
            suffix = "." + audio.filename.rsplit(".", 1)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            assert audio.content is not None
            handle.write(audio.content)
            return handle.name, handle.name


__all__ = ["PyannoteDiarizationAdapter"]
