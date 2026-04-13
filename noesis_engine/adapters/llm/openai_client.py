from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError

from noesis_engine.ports.llm import (
    LLMPort,
    LLMRunMetadata,
    SchemaModelT,
    StructuredGenerationResult,
    StructuredPrompt,
)
from noesis_engine.settings import Settings, get_settings


class RemoteStructuredGenerationError(RuntimeError):
    def __init__(self, message: str, *, raw_text: str | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text


class OpenAIClient(LLMPort):
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._base_url = (self._settings.openai.base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key = self._settings.openai.api_key
        self._organization = self._settings.openai.organization
        self._timeout_sec = self._settings.openai.timeout_sec
        self._model_name = self._settings.locked_analysis_model
        self._default_temperature = self._settings.analysis_temperature

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_structured(
        self,
        prompt: StructuredPrompt[SchemaModelT],
    ) -> StructuredGenerationResult[SchemaModelT]:
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": prompt.system_instruction},
                {"role": "user", "content": prompt.user_input},
            ],
            "temperature": self._default_temperature if prompt.temperature is None else prompt.temperature,
        }
        if prompt.strict_json:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._organization:
            headers["OpenAI-Organization"] = self._organization

        try:
            with httpx.Client(timeout=self._timeout_sec) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenAI structured generation request failed: {exc}") from exc

        data = response.json()
        raw_text = self._extract_content(data)
        try:
            output = prompt.response_model.model_validate_json(raw_text)
        except ValidationError as exc:
            raise RemoteStructuredGenerationError(
                "OpenAI returned malformed structured output.",
                raw_text=raw_text,
            ) from exc

        usage = data.get("usage", {})
        metadata = LLMRunMetadata(
            provider=self.provider_name,
            model=self._model_name,
            request_id=response.headers.get("x-request-id"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=self._extract_finish_reason(data),
            latency_ms=None,
        )
        return StructuredGenerationResult(
            output=output,
            metadata=metadata,
            raw_text=raw_text,
        )

    def _extract_content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RemoteStructuredGenerationError("OpenAI response did not contain any choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
                    elif "text" in part and isinstance(part["text"], str):
                        text_parts.append(part["text"])
            return "".join(text_parts)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=True)
        raise RemoteStructuredGenerationError("OpenAI response did not contain textual message content.")

    def _extract_finish_reason(self, data: dict[str, Any]) -> str | None:
        choices = data.get("choices") or []
        if not choices:
            return None
        finish_reason = choices[0].get("finish_reason")
        return str(finish_reason) if finish_reason is not None else None


__all__ = ["OpenAIClient", "RemoteStructuredGenerationError"]
