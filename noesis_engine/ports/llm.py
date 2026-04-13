from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, TypeVar

from pydantic import BaseModel


SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class StructuredPrompt(Generic[SchemaModelT]):
    system_instruction: str
    user_input: str
    response_model: type[SchemaModelT]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    strict_json: bool = True


@dataclass(frozen=True, slots=True)
class LLMRunMetadata:
    provider: str
    model: str
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class StructuredGenerationResult(Generic[SchemaModelT]):
    output: SchemaModelT
    metadata: LLMRunMetadata
    raw_text: str | None = None


class LLMPort(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        prompt: StructuredPrompt[SchemaModelT],
    ) -> StructuredGenerationResult[SchemaModelT]:
        raise NotImplementedError


__all__ = [
    "LLMPort",
    "LLMRunMetadata",
    "SchemaModelT",
    "StructuredGenerationResult",
    "StructuredPrompt",
]
