from __future__ import annotations

from noesis_engine.core.schemas import SpeakerSegment, Utterance


UNKNOWN_SPEAKER_IDS = {"UNKNOWN", "UNASSIGNED", "UNK"}


def _overlap_duration(
    utterance: Utterance,
    segment: SpeakerSegment,
) -> float:
    start = max(utterance.start_sec, segment.start_sec)
    end = min(utterance.end_sec, segment.end_sec)
    return max(0.0, end - start)


def _duration(utterance: Utterance) -> float:
    return max(utterance.end_sec - utterance.start_sec, 1e-9)


def has_meaningful_speaker_labels(utterances: list[Utterance]) -> bool:
    if not utterances:
        return False
    return all(
        utterance.speaker_id.strip() != ""
        and utterance.speaker_id.upper() not in UNKNOWN_SPEAKER_IDS
        for utterance in utterances
    )


def merge_diarization_segments(
    utterances: list[Utterance],
    segments: list[SpeakerSegment],
    *,
    min_overlap_ratio: float = 0.5,
) -> list[Utterance]:
    if not segments:
        return list(utterances)

    ordered_segments = sorted(
        segments,
        key=lambda segment: (segment.start_sec, segment.end_sec, segment.segment_id),
    )
    merged: list[Utterance] = []

    for utterance in utterances:
        best_segment: SpeakerSegment | None = None
        best_overlap_ratio = 0.0
        best_overlap_duration = 0.0

        for segment in ordered_segments:
            overlap = _overlap_duration(utterance, segment)
            if overlap <= 0.0:
                continue
            overlap_ratio = overlap / _duration(utterance)
            if overlap_ratio > best_overlap_ratio or (
                overlap_ratio == best_overlap_ratio and overlap > best_overlap_duration
            ):
                best_segment = segment
                best_overlap_ratio = overlap_ratio
                best_overlap_duration = overlap

        speaker_id = utterance.speaker_id
        if best_segment is not None and (
            best_overlap_ratio >= min_overlap_ratio
            or speaker_id.upper() in UNKNOWN_SPEAKER_IDS
        ):
            speaker_id = best_segment.speaker_id

        merged.append(
            Utterance(
                utterance_id=utterance.utterance_id,
                speaker_id=speaker_id,
                text=utterance.text,
                start_sec=utterance.start_sec,
                end_sec=utterance.end_sec,
            )
        )

    return merged


def assign_speakers_to_utterances(
    utterances: list[Utterance],
    diarization_segments: list[SpeakerSegment] | None = None,
    *,
    min_overlap_ratio: float = 0.5,
) -> list[Utterance]:
    if diarization_segments:
        return merge_diarization_segments(
            utterances,
            diarization_segments,
            min_overlap_ratio=min_overlap_ratio,
        )
    return list(utterances)


__all__ = [
    "UNKNOWN_SPEAKER_IDS",
    "assign_speakers_to_utterances",
    "has_meaningful_speaker_labels",
    "merge_diarization_segments",
]
