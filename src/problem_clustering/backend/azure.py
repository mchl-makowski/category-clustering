"""Azure Foundry backend (ADR-0001). Not exercised in tests: the dev machine has no API access."""

from __future__ import annotations

import warnings
from typing import Any
from urllib.parse import urlparse

import numpy as np
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..config import BackendConfig
from .base import ChatReply, ContentFilteredError, EmbeddingReply, ToolCall, Usage

# Hosts of Azure OpenAI / Foundry resource endpoints (`azure_endpoint` of AzureOpenAI)
KNOWN_ENDPOINT_HOSTS = (".openai.azure.com", ".cognitiveservices.azure.com", ".services.ai.azure.com")


class FoundrySettings(BaseSettings):
    """Secrets and endpoint from env: FOUNDRY_API_KEY, FOUNDRY_ENDPOINT, FOUNDRY_API_VERSION."""

    model_config = SettingsConfigDict(env_prefix="FOUNDRY_", env_file=".env", extra="ignore")

    api_key: SecretStr
    endpoint: str
    api_version: str = "2024-10-21"

    @field_validator("endpoint")
    @classmethod
    def _resource_endpoint(cls, value: str) -> str:
        """Must be the resource base URL, e.g. https://<resource>.openai.azure.com/ (no /openai/... path)."""
        url = urlparse(value.strip())
        example = "e.g. https://<resource>.openai.azure.com/"
        if url.scheme != "https" or not url.hostname:
            raise ValueError(f"FOUNDRY_ENDPOINT must be an https resource URL, {example}; got {value!r}")
        if url.path.rstrip("/"):
            raise ValueError(f"FOUNDRY_ENDPOINT must be the resource base URL without a path, {example}; "
                             f"got {value!r} (the SDK adds /openai/deployments/... itself)")
        if not url.hostname.endswith(KNOWN_ENDPOINT_HOSTS):
            warnings.warn(f"FOUNDRY_ENDPOINT host {url.hostname!r} is not a known Azure OpenAI host "
                          f"({', '.join('*' + h for h in KNOWN_ENDPOINT_HOSTS)})", UserWarning, stacklevel=2)
        return value.strip()


class AzureFoundryBackend:
    def __init__(self, config: BackendConfig, settings: FoundrySettings | None = None):
        from openai import AzureOpenAI

        settings = settings or FoundrySettings()
        self.config = config
        self.client = AzureOpenAI(
            api_key=settings.api_key.get_secret_value(),
            azure_endpoint=settings.endpoint,
            api_version=settings.api_version,
            max_retries=0,  # RetryingBackend (runtime.max_retries) is the only retry layer
        )

    def embed(self, texts: list[str]) -> EmbeddingReply:
        resp = self.client.embeddings.create(model=self.config.embedding_deployment, input=texts)
        vectors = np.array([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        return EmbeddingReply(vectors=vectors, usage=Usage(prompt_tokens=resp.usage.prompt_tokens))

    def chat(self, messages, tools=None, response_format=None, model=None) -> ChatReply:
        kwargs: dict[str, Any] = {"model": model or self.config.chat_deployment, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if _is_content_filter_error(exc):  # prompt blocked: HTTP 400, code content_filter
                raise ContentFilteredError(str(exc)) from exc
            raise
        choice = resp.choices[0]
        msg = choice.message
        if choice.finish_reason == "content_filter" or getattr(msg, "refusal", None):
            raise ContentFilteredError(f"reply blocked (finish_reason={choice.finish_reason!r})")
        return ChatReply(
            content=msg.content,
            tool_calls=[
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in msg.tool_calls or []
            ],
            usage=Usage(
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            ),
            finish_reason=choice.finish_reason or "stop",
        )


def _is_content_filter_error(exc: Exception) -> bool:
    """openai.BadRequestError whose error code is `content_filter` (matched by name: no import needed)."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    return "BadRequestError" in names and getattr(exc, "code", None) == "content_filter"
