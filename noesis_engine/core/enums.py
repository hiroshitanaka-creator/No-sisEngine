from __future__ import annotations

from enum import StrEnum


class ConflictCategory(StrEnum):
    ETHICAL_CONFLICT = "ethical_conflict"
    RESOURCE_ALLOCATION = "resource_allocation"
    TEMPORAL_HORIZON = "temporal_horizon"
    VALUE_CONFLICT = "value_conflict"
    METHODOLOGICAL_CONFLICT = "methodological_conflict"


class PersonaId(StrEnum):
    KANT = "kant"
    MILL = "mill"
    ARISTOTLE = "aristotle"
    LEVINAS = "levinas"
    RAWLS = "rawls"
    SEN = "sen"
    SARTRE = "sartre"
    DEWEY = "dewey"
    JONAS = "jonas"
    NIETZSCHE = "nietzsche"
    CONFUCIUS = "confucius"
    MACINTYRE = "macintyre"
    DESCARTES = "descartes"
    BACON = "bacon"
    POPPER = "popper"
    KUHN = "kuhn"


class ClaimActType(StrEnum):
    PROPOSAL = "proposal"
    OBJECTION = "objection"
    SUPPORT = "support"
    EVIDENCE = "evidence"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    QUESTION = "question"
    UNCERTAINTY = "uncertainty"


class DecisionState(StrEnum):
    ADOPTED = "adopted"
    REJECTED = "rejected"
    IGNORED = "ignored"
    UNRESOLVED = "unresolved"


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


__all__ = [
    "AnalysisStatus",
    "ClaimActType",
    "ConflictCategory",
    "DecisionState",
    "PersonaId",
]
