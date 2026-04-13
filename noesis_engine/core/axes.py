from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Final


CANONICAL_AXIS_KEYS: Final[tuple[str, ...]] = (
    "principle_constraint",
    "consequence_utility",
    "virtue_excellence",
    "procedural_rigor",
    "short_term_urgency",
    "long_term_sustainability",
    "resource_efficiency",
    "risk_precaution",
    "autonomy_agency",
    "relational_fairness",
    "innovation_optionality",
)
CANONICAL_AXIS_SET: Final[frozenset[str]] = frozenset(CANONICAL_AXIS_KEYS)
AXIS_VALIDATION_TOLERANCE: Final[float] = 1e-6


class AxisValidationError(ValueError):
    pass


def canonical_axis_keys() -> tuple[str, ...]:
    return CANONICAL_AXIS_KEYS


def is_canonical_axis_key(value: str) -> bool:
    return value in CANONICAL_AXIS_SET


def empty_axis_vector() -> dict[str, float]:
    return {axis: 0.0 for axis in CANONICAL_AXIS_KEYS}


def validate_axis_subset(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise AxisValidationError("Axis list entries must be strings.")
        normalized = value.strip()
        if not normalized:
            raise AxisValidationError("Axis list entries must not be empty.")
        if normalized not in CANONICAL_AXIS_SET:
            raise AxisValidationError(f"Unknown canonical axis: {normalized}")
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def validate_axis_vector(
    vector: Mapping[str, float],
    *,
    require_all_keys: bool = True,
    require_normalized: bool = False,
    allow_zero: bool = False,
    tolerance: float = AXIS_VALIDATION_TOLERANCE,
) -> dict[str, float]:
    if not isinstance(vector, Mapping):
        raise AxisValidationError("Axis vector must be a mapping of axis names to floats.")

    unknown_keys = [key for key in vector if key not in CANONICAL_AXIS_SET]
    if unknown_keys:
        raise AxisValidationError(
            f"Unknown axis keys: {', '.join(sorted(unknown_keys))}"
        )

    if require_all_keys:
        missing_keys = [key for key in CANONICAL_AXIS_KEYS if key not in vector]
        if missing_keys:
            raise AxisValidationError(
                f"Missing axis keys: {', '.join(missing_keys)}"
            )

    ordered: dict[str, float] = {}
    total = 0.0

    for key in CANONICAL_AXIS_KEYS:
        raw_value = vector.get(key, 0.0)

        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise AxisValidationError(
                f"Axis '{key}' must be numeric, got {type(raw_value).__name__}."
            )

        value = float(raw_value)
        if not isfinite(value):
            raise AxisValidationError(f"Axis '{key}' must be finite.")
        if value < -tolerance:
            raise AxisValidationError(f"Axis '{key}' must be >= 0.0.")
        if value > 1.0 + tolerance:
            raise AxisValidationError(f"Axis '{key}' must be <= 1.0.")

        if abs(value) <= tolerance:
            value = 0.0

        ordered[key] = value
        total += value

    if total <= tolerance and not allow_zero:
        raise AxisValidationError("Axis vector must contain at least one positive weight.")

    if require_normalized and abs(total - 1.0) > tolerance:
        raise AxisValidationError(
            f"Axis vector must sum to 1.0, got {total:.12f}."
        )

    return ordered


def normalize_axis_vector(
    vector: Mapping[str, float],
    *,
    tolerance: float = AXIS_VALIDATION_TOLERANCE,
) -> dict[str, float]:
    validated = validate_axis_vector(
        vector,
        require_all_keys=False,
        require_normalized=False,
        allow_zero=False,
        tolerance=tolerance,
    )
    total = sum(validated.values())
    if total <= tolerance:
        raise AxisValidationError("Cannot normalize an all-zero axis vector.")

    normalized = {
        key: validated[key] / total
        for key in CANONICAL_AXIS_KEYS
    }
    return validate_axis_vector(
        normalized,
        require_all_keys=True,
        require_normalized=True,
        allow_zero=False,
        tolerance=tolerance,
    )


def validate_axis_delta(
    delta: Mapping[str, float],
    *,
    require_all_keys: bool = True,
    tolerance: float = AXIS_VALIDATION_TOLERANCE,
) -> dict[str, float]:
    if not isinstance(delta, Mapping):
        raise AxisValidationError("Axis delta must be a mapping of axis names to floats.")

    unknown_keys = [key for key in delta if key not in CANONICAL_AXIS_SET]
    if unknown_keys:
        raise AxisValidationError(
            f"Unknown axis keys: {', '.join(sorted(unknown_keys))}"
        )

    if require_all_keys:
        missing_keys = [key for key in CANONICAL_AXIS_KEYS if key not in delta]
        if missing_keys:
            raise AxisValidationError(
                f"Missing axis keys: {', '.join(missing_keys)}"
            )

    ordered: dict[str, float] = {}
    for key in CANONICAL_AXIS_KEYS:
        if key not in delta:
            if require_all_keys:
                raise AxisValidationError(f"Missing axis key: {key}")
            continue

        raw_value = delta[key]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise AxisValidationError(
                f"Axis delta '{key}' must be numeric, got {type(raw_value).__name__}."
            )

        value = float(raw_value)
        if not isfinite(value):
            raise AxisValidationError(f"Axis delta '{key}' must be finite.")
        if abs(value) <= tolerance:
            value = 0.0
        ordered[key] = value

    return ordered


__all__ = [
    "AXIS_VALIDATION_TOLERANCE",
    "AxisValidationError",
    "CANONICAL_AXIS_KEYS",
    "CANONICAL_AXIS_SET",
    "canonical_axis_keys",
    "empty_axis_vector",
    "is_canonical_axis_key",
    "normalize_axis_vector",
    "validate_axis_delta",
    "validate_axis_subset",
    "validate_axis_vector",
]
