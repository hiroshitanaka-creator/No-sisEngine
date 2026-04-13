import pytest

from noesis_engine.core.schemas import Utterance
from noesis_engine.services.transcript_normalizer import (
    TranscriptChunk,
    chunk_transcript,
    normalize_and_chunk_transcript,
    normalize_speaker_id,
    normalize_text,
    normalize_transcript,
)


def _utt(uid: str, speaker: str, text: str, start: float = 0.0, end: float = 1.0) -> Utterance:
    return Utterance(
        utterance_id=uid,
        speaker_id=speaker,
        text=text,
        start_sec=start,
        end_sec=end,
    )


def test_normalize_speaker_id_replaces_spaces() -> None:
    assert normalize_speaker_id("John Doe") == "John_Doe"


def test_normalize_speaker_id_rejects_blank() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        normalize_speaker_id("   ")


def test_normalize_text_rejects_blank() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        normalize_text("   ")


def test_normalize_transcript_sorts_by_start_time() -> None:
    utterances = [
        _utt("u2", "bob", "Second utterance.", 2.0, 3.0),
        _utt("u1", "alice", "First utterance.", 0.0, 1.0),
    ]
    result = normalize_transcript(utterances)
    assert result[0].utterance_id == "u1"
    assert result[1].utterance_id == "u2"


def test_normalize_transcript_normalizes_speaker_id_spaces() -> None:
    utterances = [_utt("u1", "John Doe", "Hello.", 0.0, 1.0)]
    result = normalize_transcript(utterances)
    assert result[0].speaker_id == "John_Doe"


def test_normalize_transcript_accepts_mapping() -> None:
    utterances = [
        {
            "utterance_id": "u1",
            "speaker_id": "alice",
            "text": "Hello.",
            "start_sec": 0.0,
            "end_sec": 1.0,
        }
    ]
    result = normalize_transcript(utterances)
    assert len(result) == 1
    assert result[0].speaker_id == "alice"


def test_chunk_transcript_rejects_zero_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars must be positive"):
        chunk_transcript([_utt("u1", "alice", "Hi.")], max_chars=0, overlap_chars=0)


def test_chunk_transcript_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_chars must be non-negative"):
        chunk_transcript([_utt("u1", "alice", "Hi.")], max_chars=100, overlap_chars=-1)


def test_chunk_transcript_rejects_overlap_ge_max() -> None:
    with pytest.raises(ValueError, match="overlap_chars must be smaller"):
        chunk_transcript([_utt("u1", "alice", "Hi.")], max_chars=10, overlap_chars=10)


def test_chunk_transcript_empty_input() -> None:
    result = chunk_transcript([], max_chars=1000, overlap_chars=0)
    assert result == []


def test_chunk_transcript_single_chunk() -> None:
    utterances = [_utt(f"u{i}", "alice", "Hello.", float(i), float(i + 1)) for i in range(3)]
    chunks = chunk_transcript(utterances, max_chars=10000, overlap_chars=0)
    assert len(chunks) == 1
    assert chunks[0].start_index == 0
    assert chunks[0].end_index == 3


def test_chunk_transcript_creates_overlap() -> None:
    # Each utterance ~41 chars; max_chars=90 fits 2; overlap=40 forces 1-utterance overlap
    utterances = [
        _utt(f"u{i}", "alice", "x" * 20, float(i), float(i + 1)) for i in range(6)
    ]
    chunks = chunk_transcript(utterances, max_chars=90, overlap_chars=40)
    assert len(chunks) > 1
    # Second chunk should start before first chunk ends (overlap)
    second_start = chunks[1].start_index
    first_end = chunks[0].end_index
    assert second_start < first_end


def test_normalize_and_chunk_transcript_returns_both() -> None:
    utterances = [_utt("u1", "alice", "Hello.", 0.0, 1.0)]
    normalized, chunks = normalize_and_chunk_transcript(utterances, max_chars=10000, overlap_chars=0)
    assert len(normalized) == 1
    assert isinstance(chunks[0], TranscriptChunk)
