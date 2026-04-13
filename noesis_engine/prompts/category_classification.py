from __future__ import annotations

import json

from noesis_engine.core.enums import ConflictCategory
from noesis_engine.core.schemas import CategoryResult, ClaimUnit, IssueCluster
from noesis_engine.ports.llm import StructuredPrompt


def build_category_classification_prompt(
    issue: IssueCluster,
    claims: list[ClaimUnit],
    *,
    meeting_context: str | None = None,
    temperature: float | None = None,
) -> StructuredPrompt[CategoryResult]:
    system_instruction = (
        "Classify the issue cluster into exactly one primary conflict category. "
        "Return valid JSON only. "
        "Do not add commentary. "
        "secondary_categories may be empty, but primary_category must contain exactly one value from the allowed set. "
        "signals must be short textual reasons grounded in the supplied claims."
    )
    payload = {
        "issue": issue.model_dump(mode="json"),
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "meeting_context": meeting_context,
        "allowed_categories": [member.value for member in ConflictCategory],
    }
    return StructuredPrompt(
        system_instruction=system_instruction,
        user_input=json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        response_model=CategoryResult,
        metadata={"stage": "category_classification", "issue_id": issue.issue_id},
        temperature=temperature,
        strict_json=True,
    )


__all__ = ["build_category_classification_prompt"]
