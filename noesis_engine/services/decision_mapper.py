from __future__ import annotations

from noesis_engine.core.enums import ClaimActType, DecisionState
from noesis_engine.core.schemas import ClaimUnit, DecisionMap, IssueCluster


class DecisionMapperService:
    def map_issue(
        self,
        issue: IssueCluster,
        claims: list[ClaimUnit],
    ) -> DecisionMap:
        proposals = [claim for claim in claims if claim.act_type == ClaimActType.PROPOSAL]
        proposal_ids = [claim.claim_id for claim in proposals]

        supports = [claim for claim in claims if claim.act_type == ClaimActType.SUPPORT]
        objections = [claim for claim in claims if claim.act_type == ClaimActType.OBJECTION]
        decisions = [claim for claim in claims if claim.act_type == ClaimActType.DECISION]

        adopted_claim_id: str | None = None
        decision_targets: list[str] = []
        for claim in decisions:
            for target_claim_id in claim.target_claim_ids:
                if target_claim_id in proposal_ids and target_claim_id not in decision_targets:
                    decision_targets.append(target_claim_id)

        if decision_targets:
            adopted_claim_id = decision_targets[0]
        elif len(proposal_ids) == 1:
            proposal_id = proposal_ids[0]
            has_support = any(proposal_id in claim.target_claim_ids for claim in supports)
            has_objection = any(proposal_id in claim.target_claim_ids for claim in objections)
            if has_support and not has_objection:
                adopted_claim_id = proposal_id

        rejected_claim_ids: list[str] = []
        for target_claim_id in decision_targets[1:]:
            if target_claim_id not in rejected_claim_ids:
                rejected_claim_ids.append(target_claim_id)

        for objection in objections:
            for target_claim_id in objection.target_claim_ids:
                if target_claim_id in proposal_ids and target_claim_id != adopted_claim_id:
                    if target_claim_id not in rejected_claim_ids:
                        rejected_claim_ids.append(target_claim_id)

        ignored_claim_ids = [
            proposal_id
            for proposal_id in proposal_ids
            if proposal_id != adopted_claim_id and proposal_id not in rejected_claim_ids
        ]

        if adopted_claim_id is not None:
            status = DecisionState.ADOPTED
        elif rejected_claim_ids and not ignored_claim_ids:
            status = DecisionState.REJECTED
        elif ignored_claim_ids and not rejected_claim_ids:
            status = DecisionState.IGNORED
        else:
            status = DecisionState.UNRESOLVED

        return DecisionMap(
            issue_id=issue.issue_id,
            adopted_claim_id=adopted_claim_id,
            rejected_claim_ids=rejected_claim_ids,
            ignored_claim_ids=ignored_claim_ids,
            status=status,
        )


__all__ = ["DecisionMapperService"]
