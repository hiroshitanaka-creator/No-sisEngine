from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from pydantic import BaseModel, ValidationError

from noesis_engine.ports.llm import (
    LLMPort,
    SchemaModelT,
    StructuredGenerationResult,
    StructuredPrompt,
)


@dataclass(frozen=True, slots=True)
class StructuredOutputFailure:
    message: str
    attempts: int
    raw_text: str | None = None
    cause_type: str | None = None


@dataclass(frozen=True, slots=True)
class StructuredOutputRunResult(Generic[SchemaModelT]):
    result: StructuredGenerationResult[SchemaModelT] | None
    failure: StructuredOutputFailure | None
    attempts: int

    @property
    def ok(self) -> bool:
        return self.result is not None and self.failure is None


class StructuredOutputExecutionError(RuntimeError):
    def __init__(self, failure: StructuredOutputFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def _extract_raw_text(exc: Exception) -> str | None:
    for attr_name in ("raw_text", "raw_output", "text"):
        value = getattr(exc, attr_name, None)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
    return None


def _coerce_output(
    prompt: StructuredPrompt[SchemaModelT],
    result: StructuredGenerationResult[Any],
) -> StructuredGenerationResult[SchemaModelT]:
    response_model = prompt.response_model
    output = result.output

    if isinstance(output, response_model):
        return StructuredGenerationResult(
            output=output,
            metadata=result.metadata,
            raw_text=result.raw_text,
        )

    if isinstance(output, BaseModel):
        coerced = response_model.model_validate(output.model_dump())
    elif isinstance(output, dict):
        coerced = response_model.model_validate(output)
    elif isinstance(output, str):
        coerced = response_model.model_validate_json(output)
    elif result.raw_text is not None:
        coerced = response_model.model_validate_json(result.raw_text)
    else:
        raise TypeError(
            "Structured generation result could not be coerced into the requested response model."
        )

    return StructuredGenerationResult(
        output=coerced,
        metadata=result.metadata,
        raw_text=result.raw_text,
    )


def build_repair_prompt(
    prompt: StructuredPrompt[SchemaModelT],
    *,
    raw_text: str,
    error_message: str,
) -> StructuredPrompt[SchemaModelT]:
    repair_instruction = (
        "You are repairing a malformed structured response. "
        "Return valid JSON only. "
        "Do not add commentary. "
        "Preserve the intended semantics where possible. "
        "If fields are missing, infer the minimum valid content from the provided data."
    )
    repair_input = (
        "{\n"
        '  "original_system_instruction": '
        + repr(prompt.system_instruction)
        + ",\n"
        '  "validation_error": '
        + repr(error_message)
        + ",\n"
        '  "malformed_output": '
        + repr(raw_text)
        + "\n}"
    )
    metadata = dict(prompt.metadata)
    metadata["repair_attempt"] = True
    return StructuredPrompt(
        system_instruction=repair_instruction,
        user_input=repair_input,
        response_model=prompt.response_model,
        metadata=metadata,
        temperature=prompt.temperature,
        strict_json=True,
    )


def run_structured_output(
    llm: LLMPort,
    prompt: StructuredPrompt[SchemaModelT],
    *,
    max_repair_attempts: int = 1,
) -> StructuredOutputRunResult[SchemaModelT]:
    attempts = 0
    try:
        attempts += 1
        first_result = llm.generate_structured(prompt)
        coerced = _coerce_output(prompt, first_result)
        return StructuredOutputRunResult(result=coerced, failure=None, attempts=attempts)
    except Exception as exc:
        failure = StructuredOutputFailure(
            message=str(exc) or "Structured generation failed.",
            attempts=attempts,
            raw_text=_extract_raw_text(exc),
            cause_type=type(exc).__name__,
        )

    if max_repair_attempts <= 0:
        return StructuredOutputRunResult(result=None, failure=failure, attempts=attempts)

    try:
        attempts += 1
        repair_prompt = (
            build_repair_prompt(prompt, raw_text=failure.raw_text, error_message=failure.message)
            if failure.raw_text is not None
            else prompt
        )
        repaired_result = llm.generate_structured(repair_prompt)
        coerced = _coerce_output(prompt, repaired_result)
        return StructuredOutputRunResult(result=coerced, failure=None, attempts=attempts)
    except Exception as exc:
        final_failure = StructuredOutputFailure(
            message=str(exc) or failure.message,
            attempts=attempts,
            raw_text=_extract_raw_text(exc) or failure.raw_text,
            cause_type=type(exc).__name__,
        )
        return StructuredOutputRunResult(result=None, failure=final_failure, attempts=attempts)


def generate_structured_or_raise(
    llm: LLMPort,
    prompt: StructuredPrompt[SchemaModelT],
    *,
    max_repair_attempts: int = 1,
) -> StructuredGenerationResult[SchemaModelT]:
    outcome = run_structured_output(
        llm,
        prompt,
        max_repair_attempts=max_repair_attempts,
    )
    if not outcome.ok or outcome.result is None:
        assert outcome.failure is not None
        raise StructuredOutputExecutionError(outcome.failure)
    return outcome.result


__all__ = [
    "StructuredOutputExecutionError",
    "StructuredOutputFailure",
    "StructuredOutputRunResult",
    "build_repair_prompt",
    "generate_structured_or_raise",
    "run_structured_output",
]
