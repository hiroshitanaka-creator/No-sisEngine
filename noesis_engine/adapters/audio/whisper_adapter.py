from __future__ import annotations

import os
import tempfile
from importlib import import_module

from noesis_engine.core.schemas import AudioInput, Utterance
from noesis_engine.ports.transcription import (
    TranscriptionPort,
    TranscriptionResult,
    TranscriptionRunMetadata,
)
from noesis_engine.settings import Settings, get_settings


class WhisperTranscriptionAdapter(TranscriptionPort):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model_name = self._settings.audio.transcription_model
        self._language = self._settings.audio.language

    def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        whisper = import_module("whisper")
        audio_path, cleanup_path = self._materialize_audio(audio)
        try:
            model = whisper.load_model(self._model_name)
            kwargs = {}
            if self._language:
                kwargs["language"] = self._language
            result = model.transcribe(audio_path, **kwargs)
        finally:
            if cleanup_path is not None and os.path.exists(cleanup_path):
                os.remove(cleanup_path)

        segments = result.get("segments") or []
        utterances: list[Utterance] = []
        if segments:
            for index, segment in enumerate(segments, start=1):
                text = str(segment.get("text", "")).strip()
                if not text:
                    continue
                start_sec = float(segment.get("start", 0.0))
                end_sec = float(segment.get("end", start_sec))
                utterances.append(
                    Utterance(
                        utterance_id=f"utt_{index:04d}",
                        speaker_id="UNKNOWN",
                        text=text,
                        start_sec=start_sec,
                        end_sec=end_sec,
                    )
                )
        else:
            text = str(result.get("text", "")).strip()
            if text:
                utterances.append(
                    Utterance(
                        utterance_id="utt_0001",
                        speaker_id="UNKNOWN",
                        text=text,
                        start_sec=0.0,
                        end_sec=0.0,
                    )
                )

        return TranscriptionResult(
            utterances=tuple(utterances),
            metadata=TranscriptionRunMetadata(
                provider="whisper",
                model=self._model_name,
                language=self._language,
                duration_sec=utterances[-1].end_sec if utterances else 0.0,
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


__all__ = ["WhisperTranscriptionAdapter"]
