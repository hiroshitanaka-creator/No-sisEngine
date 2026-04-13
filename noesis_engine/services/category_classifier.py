from __future__ import annotations

from noesis_engine.core.schemas import CategoryResult, ClaimUnit, IssueCluster
from noesis_engine.ports.llm import LLMPort
from noesis_engine.prompts.category_classification import build_category_classification_prompt
from noesis_engine.utils.structured_output import generate_structured_or_raise


class CategoryClassifierService:
    def __init__(
        self,
        llm: LLMPort,
        *,
        max_repair_attempts: int = 1,
    ) -> None:
        self._llm = llm
        self._max_repair_attempts = max_repair_attempts

    def classify_issue(
        self,
        issue: IssueCluster,
        claims: list[ClaimUnit],
        *,
        meeting_context: str | None = None,
    ) -> CategoryResult:
        result = generate_structured_or_raise(
            self._llm,
            build_category_classification_prompt(
                issue,
                claims,
                meeting_context=meeting_context,
            ),
            max_repair_attempts=self._max_repair_attempts,
        )
        category = result.output
        if category.issue_id != issue.issue_id:
            category = CategoryResult(
                issue_id=issue.issue_id,
                primary_category=category.primary_category,
                secondary_categories=category.secondary_categories,
                signals=category.signals,
                confidence=category.confidence,
            )
        return category

    def classify_issues(
        self,
        issues: list[IssueCluster],
        claims_by_issue: dict[str, list[ClaimUnit]],
        *,
        meeting_context: str | None = None,
    ) -> list[CategoryResult]:
        return [
            self.classify_issue(
                issue,
                claims_by_issue.get(issue.issue_id, []),
                meeting_context=meeting_context,
            )
            for issue in issues
        ]


__all__ = ["CategoryClassifierService"]
