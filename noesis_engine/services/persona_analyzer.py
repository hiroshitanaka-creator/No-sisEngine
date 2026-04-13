from __future__ import annotations

from collections import defaultdict

from noesis_engine.core.schemas import ClaimUnit, IssueCluster, PersonaReading, PersonaSpec
from noesis_engine.ports.llm import LLMPort
from noesis_engine.prompts.persona_analysis import build_persona_analysis_prompt
from noesis_engine.utils.structured_output import generate_structured_or_raise


class PersonaAnalyzerService:
    def __init__(
        self,
        llm: LLMPort,
        *,
        max_repair_attempts: int = 1,
    ) -> None:
        self._llm = llm
        self._max_repair_attempts = max_repair_attempts

    def analyze_issue(
        self,
        issue: IssueCluster,
        claims: list[ClaimUnit],
        personas: tuple[PersonaSpec, ...],
        *,
        meeting_context: str | None = None,
    ) -> list[PersonaReading]:
        claim_ids_by_speaker: dict[str, list[str]] = defaultdict(list)
        for claim in claims:
            claim_ids_by_speaker[claim.speaker_id].append(claim.claim_id)

        readings: list[PersonaReading] = []
        for persona_spec in personas:
            result = generate_structured_or_raise(
                self._llm,
                build_persona_analysis_prompt(
                    issue,
                    claims,
                    persona_spec,
                    meeting_context=meeting_context,
                ),
                max_repair_attempts=self._max_repair_attempts,
            )
            envelope = result.output
            seen_speakers: set[str] = set()

            for draft in envelope.readings:
                source_claim_ids = draft.source_claim_ids or claim_ids_by_speaker.get(draft.speaker_id, [])
                readings.append(
                    PersonaReading(
                        issue_id=issue.issue_id,
                        speaker_id=draft.speaker_id,
                        persona_id=persona_spec.persona_id,
                        axis_weights=draft.axis_weights,
                        alignment_score=draft.alignment_score,
                        grammar_summary=draft.grammar_summary,
                        hidden_value=draft.hidden_value,
                        blind_spot=draft.blind_spot,
                        confidence=draft.confidence,
                        source_claim_ids=source_claim_ids,
                    )
                )
                seen_speakers.add(draft.speaker_id)

            for speaker_id in issue.speaker_ids:
                if speaker_id in seen_speakers:
                    continue
                readings.append(
                    PersonaReading(
                        issue_id=issue.issue_id,
                        speaker_id=speaker_id,
                        persona_id=persona_spec.persona_id,
                        axis_weights=persona_spec.axis_prior,
                        alignment_score=0.5,
                        grammar_summary="Fallback reading generated from persona prior due to missing model output.",
                        hidden_value="Insufficient speaker evidence to extract a stronger hidden value.",
                        blind_spot="This fallback may reflect persona prior more than transcript-specific evidence.",
                        confidence=0.1,
                        source_claim_ids=claim_ids_by_speaker.get(speaker_id, []),
                    )
                )

        return readings


__all__ = ["PersonaAnalyzerService"]
