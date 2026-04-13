import pytest
from pydantic import BaseModel

from noesis_engine.ports.llm import LLMPort, LLMRunMetadata, StructuredGenerationResult, StructuredPrompt
from noesis_engine.utils.structured_output import (
    StructuredOutputExecutionError,
    StructuredOutputFailure,
    StructuredOutputRunResult,
    build_repair_prompt,
    generate_structured_or_raise,
    run_structured_output,
)


class _SimpleModel(BaseModel):
    value: str


_METADATA = LLMRunMetadata(provider="mock", model="mock-model")


def _prompt(response_model: type = _SimpleModel) -> StructuredPrompt:
    return StructuredPrompt(
        system_instruction="Return JSON.",
        user_input='{"value": "hello"}',
        response_model=response_model,
        metadata={"stage": "test"},
    )


class _SuccessLLM(LLMPort):
    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def generate_structured(self, prompt: StructuredPrompt) -> StructuredGenerationResult:
        return StructuredGenerationResult(
            output=_SimpleModel(value="hello"),
            metadata=_METADATA,
            raw_text=None,
        )


class _FailThenSuccessLLM(LLMPort):
    def __init__(self) -> None:
        self._calls = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def generate_structured(self, prompt: StructuredPrompt) -> StructuredGenerationResult:
        self._calls += 1
        if self._calls == 1:
            exc = ValueError("first call failed")
            exc.raw_text = '{"value": "repaired"}'  # type: ignore[attr-defined]
            raise exc
        return StructuredGenerationResult(
            output=_SimpleModel(value="repaired"),
            metadata=_METADATA,
            raw_text=None,
        )


class _AlwaysFailLLM(LLMPort):
    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def generate_structured(self, prompt: StructuredPrompt) -> StructuredGenerationResult:
        raise RuntimeError("always fails")


def test_structured_output_execution_error_stores_failure() -> None:
    failure = StructuredOutputFailure(message="oops", attempts=1, cause_type="ValueError")
    exc = StructuredOutputExecutionError(failure)
    assert exc.failure is failure
    assert str(exc) == "oops"


def test_run_structured_output_success() -> None:
    result = run_structured_output(_SuccessLLM(), _prompt())
    assert result.ok
    assert result.result is not None
    assert result.result.output.value == "hello"
    assert result.attempts == 1


def test_run_structured_output_no_repair_on_zero_attempts() -> None:
    result = run_structured_output(_AlwaysFailLLM(), _prompt(), max_repair_attempts=0)
    assert not result.ok
    assert result.failure is not None
    assert result.failure.cause_type == "RuntimeError"
    assert result.attempts == 1


def test_run_structured_output_repairs_on_first_failure() -> None:
    llm = _FailThenSuccessLLM()
    result = run_structured_output(llm, _prompt(), max_repair_attempts=1)
    assert result.ok
    assert result.result is not None
    assert result.result.output.value == "repaired"
    assert result.attempts == 2


def test_run_structured_output_returns_failure_after_exhausted_repairs() -> None:
    result = run_structured_output(_AlwaysFailLLM(), _prompt(), max_repair_attempts=1)
    assert not result.ok
    assert result.failure is not None
    assert result.attempts == 2


def test_generate_structured_or_raise_succeeds() -> None:
    result = generate_structured_or_raise(_SuccessLLM(), _prompt())
    assert result.output.value == "hello"


def test_generate_structured_or_raise_raises_on_failure() -> None:
    with pytest.raises(StructuredOutputExecutionError) as exc_info:
        generate_structured_or_raise(_AlwaysFailLLM(), _prompt(), max_repair_attempts=0)
    assert "always fails" in str(exc_info.value)


def test_build_repair_prompt_contains_original_instruction() -> None:
    prompt = _prompt()
    repair = build_repair_prompt(prompt, raw_text='{"bad": true}', error_message="Missing field")
    assert "repairing" in repair.system_instruction
    assert "Missing field" in repair.user_input
    assert repair.metadata.get("repair_attempt") is True
    assert repair.strict_json is True


def test_run_structured_output_coerces_dict_output() -> None:
    class _DictLLM(LLMPort):
        @property
        def provider_name(self) -> str:
            return "mock"

        @property
        def model_name(self) -> str:
            return "mock-model"

        def generate_structured(self, prompt: StructuredPrompt) -> StructuredGenerationResult:
            return StructuredGenerationResult(
                output={"value": "from_dict"},
                metadata=_METADATA,
                raw_text=None,
            )

    result = run_structured_output(_DictLLM(), _prompt())
    assert result.ok
    assert result.result is not None
    assert result.result.output.value == "from_dict"


def test_structured_output_run_result_ok_false_when_failure_set() -> None:
    failure = StructuredOutputFailure(message="err", attempts=1)
    run_result = StructuredOutputRunResult(result=None, failure=failure, attempts=1)
    assert not run_result.ok
