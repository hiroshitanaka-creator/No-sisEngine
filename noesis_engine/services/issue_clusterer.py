from __future__ import annotations

from collections import defaultdict

from noesis_engine.core.schemas import ClaimUnit, IssueCluster


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _normalize_issue_hint(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().lower().split())
    return normalized or None


class IssueClustererService:
    def cluster_claims(self, claims: list[ClaimUnit]) -> list[IssueCluster]:
        if not claims:
            return []

        claim_ids = [claim.claim_id for claim in claims]
        claim_index = {claim.claim_id: index for index, claim in enumerate(claims)}
        claim_by_id = {claim.claim_id: claim for claim in claims}
        union_find = _UnionFind(claim_ids)

        issue_hint_roots: dict[str, str] = {}

        for claim in claims:
            normalized_hint = _normalize_issue_hint(claim.issue_hint)
            if normalized_hint is not None:
                root_claim_id = issue_hint_roots.setdefault(normalized_hint, claim.claim_id)
                union_find.union(root_claim_id, claim.claim_id)

            for target_claim_id in claim.target_claim_ids:
                if target_claim_id in claim_by_id:
                    union_find.union(claim.claim_id, target_claim_id)

        grouped: dict[str, list[ClaimUnit]] = defaultdict(list)
        for claim in claims:
            grouped[union_find.find(claim.claim_id)].append(claim)

        sorted_groups = sorted(
            grouped.values(),
            key=lambda group: min(claim_index[claim.claim_id] for claim in group),
        )

        clusters: list[IssueCluster] = []
        for cluster_index, group in enumerate(sorted_groups, start=1):
            ordered_claims = sorted(group, key=lambda claim: claim_index[claim.claim_id])
            speaker_ids: list[str] = []
            seen_speakers: set[str] = set()
            for claim in ordered_claims:
                if claim.speaker_id not in seen_speakers:
                    seen_speakers.add(claim.speaker_id)
                    speaker_ids.append(claim.speaker_id)

            label = self._build_issue_label(ordered_claims)
            clusters.append(
                IssueCluster(
                    issue_id=f"issue_{cluster_index:03d}",
                    label=label,
                    claim_ids=[claim.claim_id for claim in ordered_claims],
                    speaker_ids=speaker_ids,
                )
            )

        return clusters

    def _build_issue_label(self, claims: list[ClaimUnit]) -> str:
        hints = [claim.issue_hint for claim in claims if claim.issue_hint]
        if hints:
            counts: dict[str, int] = {}
            for hint in hints:
                counts[hint] = counts.get(hint, 0) + 1
            return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

        for claim in claims:
            if claim.act_type.value == "proposal":
                return claim.text_span[:80]

        return claims[0].text_span[:80]


__all__ = ["IssueClustererService"]
