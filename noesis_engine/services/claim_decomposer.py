from __future__ import annotations

from noesis_engine.core.schemas import ClaimUnit, Utterance
from noesis_engine.ports.llm import LLMPort
from noesis_engine.prompts.claim_decomposition import build_claim_decomposition_prompt
from noesis_engine.services.transcript_normalizer import TranscriptChunk
from noesis_engine.utils.structured_output import generate_structured_or_raise


class ClaimDecomposerService:
    def __init__(
        self,
        llm: LLMPort,
        *,
        max_repair_attempts: int = 1,
    ) -> None:
        self._llm = llm
        self._max_repair_attempts = max_repair_attempts

    def decompose_chunks(
        self,
        chunks: list[TranscriptChunk],
        *,
        meeting_context: str | None = None,
    ) -> list[ClaimUnit]:
        claims: list[ClaimUnit] = []
        claim_counter = 1

        for chunk in chunks:
            result = generate_structured_or_raise(
                self._llm,
                build_claim_decomposition_prompt(
                    list(chunk.utterances),
                    chunk_id=chunk.chunk_id,
                    meeting_context=meeting_context,
                ),
                max_repair_attempts=self._max_repair_attempts,
            )
            envelope = result.output
            local_to_global: dict[str, str] = {}
            drafts_in_order = list(envelope.claims)

            for draft in drafts_in_order:
                local_to_global[draft.local_claim_id] = f"clm_{claim_counter:04d}"
                claim_counter += 1

            for draft in drafts_in_order:
                global_claim_id = local_to_global[draft.local_claim_id]
                source_utterance_ids = [
                    utterance_id
                    for utterance_id in draft.source_utterance_ids
                    if any(utterance.utterance_id == utterance_id for utterance in chunk.utterances)
                ]
                if not source_utterance_ids and chunk.utterances:
                    source_utterance_ids = [chunk.utterances[0].utterance_id]

                target_claim_ids = [
                    local_to_global[target_id]
                    for target_id in draft.target_local_claim_ids
                    if target_id in local_to_global
                ]

                claims.append(
                    ClaimUnit(
                        claim_id=global_claim_id,
                        speaker_id=draft.speaker_id,
                        source_utterance_ids=source_utterance_ids,
                        text_span=draft.text_span.strip(),
                        act_type=draft.act_type,
                        target_claim_ids=target_claim_ids,
                        issue_hint=draft.issue_hint.strip() if draft.issue_hint else None,
                        importance=draft.importance,
                        confidence=draft.confidence,
                        explicit_values=draft.explicit_values,
                        implicit_values=draft.implicit_values,
                    )
                )

        return claims

    def decompose_utterances(
        self,
        utterances: list[Utterance],
        *,
        chunk_id: str = "chunk_0001",
        meeting_context: str | None = None,
    ) -> list[ClaimUnit]:
        chunk = TranscriptChunk(
            chunk_id=chunk_id,
            utterances=tuple(utterances),
            start_index=0,
            end_index=len(utterances),
            char_count=sum(len(utterance.text) for utterance in utterances),
        )
        return self.decompose_chunks([chunk], meeting_context=meeting_context)


__all__ = ["ClaimDecomposerService"]
