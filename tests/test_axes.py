import pytest

from noesis_engine.core.axes import (
    CANONICAL_AXIS_KEYS,
    AxisValidationError,
    canonical_axis_keys,
    empty_axis_vector,
    is_canonical_axis_key,
    normalize_axis_vector,
    validate_axis_delta,
    validate_axis_subset,
    validate_axis_vector,
)


def test_canonical_axis_keys_returns_tuple() -> None:
    keys = canonical_axis_keys()
    assert isinstance(keys, tuple)
    assert len(keys) == 11
    assert keys is CANONICAL_AXIS_KEYS


def test_is_canonical_axis_key_known() -> None:
    assert is_canonical_axis_key("principle_constraint") is True


def test_is_canonical_axis_key_unknown() -> None:
    assert is_canonical_axis_key("not_a_real_axis") is False


def test_empty_axis_vector_all_zeros() -> None:
    vec = empty_axis_vector()
    assert list(vec.keys()) == list(CANONICAL_AXIS_KEYS)
    assert all(v == 0.0 for v in vec.values())


def test_validate_axis_subset_valid() -> None:
    result = validate_axis_subset(["principle_constraint", "risk_precaution"])
    assert result == ["principle_constraint", "risk_precaution"]


def test_validate_axis_subset_deduplicates() -> None:
    result = validate_axis_subset(["principle_constraint", "principle_constraint"])
    assert result == ["principle_constraint"]


def test_validate_axis_subset_rejects_unknown() -> None:
    with pytest.raises(AxisValidationError, match="Unknown canonical axis"):
        validate_axis_subset(["not_a_real_axis"])


def test_validate_axis_subset_rejects_empty_string() -> None:
    with pytest.raises(AxisValidationError, match="must not be empty"):
        validate_axis_subset([""])


def test_validate_axis_vector_rejects_unknown_key() -> None:
    vec = {axis: 0.1 for axis in CANONICAL_AXIS_KEYS}
    vec["not_real"] = 0.1
    with pytest.raises(AxisValidationError, match="Unknown axis keys"):
        validate_axis_vector(vec)


def test_validate_axis_vector_rejects_missing_key() -> None:
    vec = {axis: 0.1 for axis in CANONICAL_AXIS_KEYS}
    del vec["principle_constraint"]
    with pytest.raises(AxisValidationError, match="Missing axis keys"):
        validate_axis_vector(vec, require_all_keys=True)


def test_validate_axis_vector_rejects_negative_value() -> None:
    vec = {axis: 0.0 for axis in CANONICAL_AXIS_KEYS}
    vec["principle_constraint"] = -0.5
    with pytest.raises(AxisValidationError, match=">= 0.0"):
        validate_axis_vector(vec)


def test_validate_axis_vector_rejects_value_above_one() -> None:
    vec = {axis: 0.0 for axis in CANONICAL_AXIS_KEYS}
    vec["principle_constraint"] = 1.5
    with pytest.raises(AxisValidationError, match="<= 1.0"):
        validate_axis_vector(vec)


def test_validate_axis_vector_rejects_all_zero_when_not_allowed() -> None:
    vec = {axis: 0.0 for axis in CANONICAL_AXIS_KEYS}
    with pytest.raises(AxisValidationError, match="at least one positive weight"):
        validate_axis_vector(vec, allow_zero=False)


def test_validate_axis_vector_allows_all_zero_explicitly() -> None:
    vec = {axis: 0.0 for axis in CANONICAL_AXIS_KEYS}
    result = validate_axis_vector(vec, allow_zero=True)
    assert result["principle_constraint"] == 0.0


def test_normalize_axis_vector_sums_to_one() -> None:
    vec = normalize_axis_vector({"principle_constraint": 0.75, "risk_precaution": 0.25})
    assert abs(sum(vec.values()) - 1.0) < 1e-9
    assert abs(vec["principle_constraint"] - 0.75) < 1e-9


def test_validate_axis_delta_allows_negative() -> None:
    delta = {axis: -0.5 for axis in CANONICAL_AXIS_KEYS}
    result = validate_axis_delta(delta)
    assert result["principle_constraint"] == -0.5


def test_validate_axis_delta_rejects_missing_keys() -> None:
    delta = {axis: 0.1 for axis in CANONICAL_AXIS_KEYS}
    del delta["principle_constraint"]
    with pytest.raises(AxisValidationError, match="Missing axis keys"):
        validate_axis_delta(delta)
