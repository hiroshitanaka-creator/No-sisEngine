from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from noesis_engine.core.schemas import Utterance


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    chunk_id: str
    utterances: tuple[Utterance, ...]
    start_index: int
    end_index: int
    char_count: int


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_speaker_id(value: str) -> str:
    normalized = _normalize_whitespace(value)
    if not normalized:
        raise ValueError("speaker_id must not be blank.")
    return normalized.replace(" ", "_")


def normalize_text(value: str) -> str:
    normalized = _normalize_whitespace(value)
    if not normalized:
        raise ValueError("Utterance text must not be blank.")
    return normalized


def normalize_transcript(
    utterances: list[Utterance | Mapping[str, Any]],
) -> list[Utterance]:
    normalized: list[Utterance] = []
    for index, item in enumerate(utterances, start=1):
        if isinstance(item, Utterance):
            data = item.model_dump(mode="python")
        else:
            data = dict(item)

        utterance_id = str(data.get("utterance_id") or f"utt_{index:04d}")
        speaker_id = normalize_speaker_id(str(data.get("speaker_id") or "UNKNOWN"))
        text = normalize_text(str(data.get("text") or ""))
        start_sec = float(data.get("start_sec", 0.0))
        end_sec = float(data.get("end_sec", start_sec))

        normalized.append(
            Utterance(
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                text=text,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )

    normalized.sort(key=lambda utterance: (utterance.start_sec, utterance.end_sec, utterance.utterance_id))
    return normalized


def _estimate_utterance_chars(utterance: Utterance) -> int:
    return len(utterance.text) + len(utterance.speaker_id) + 16


def chunk_transcript(
    utterances: list[Utterance],
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[TranscriptChunk]:
    if not utterances:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative.")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars.")

    chunks: list[TranscriptChunk] = []
    start = 0
    chunk_index = 1

    while start < len(utterances):
        end = start
        char_count = 0
        while end < len(utterances):
            candidate = _estimate_utterance_chars(utterances[end])
            if end > start and char_count + candidate > max_chars:
                break
            char_count += candidate
            end += 1

        chunk_utterances = tuple(utterances[start:end])
        chunks.append(
            TranscriptChunk(
                chunk_id=f"chunk_{chunk_index:04d}",
                utterances=chunk_utterances,
                start_index=start,
                end_index=end,
                char_count=char_count,
            )
        )
        chunk_index += 1

        if end >= len(utterances):
            break

        next_start = end
        overlap_total = 0
        while next_start > start + 1 and overlap_total < overlap_chars:
            next_start -= 1
            overlap_total += _estimate_utterance_chars(utterances[next_start])

        if next_start <= start:
            start = end
        else:
            start = next_start

    return chunks


def normalize_and_chunk_transcript(
    utterances: list[Utterance | Mapping[str, Any]],
    *,
    max_chars: int,
    overlap_chars: int,
) -> tuple[list[Utterance], list[TranscriptChunk]]:
    normalized = normalize_transcript(utterances)
    return normalized, chunk_transcript(
        normalized,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )


__all__ = [
    "TranscriptChunk",
    "chunk_transcript",
    "normalize_and_chunk_transcript",
    "normalize_speaker_id",
    "normalize_text",
    "normalize_transcript",
]
