"""Azure Foundry backend (ADR-0001). Not exercised in tests: the dev machine has no API access."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..config import BackendConfig
from .base import ChatReply, EmbeddingReply, ToolCall, Usage


class FoundrySettings(BaseSettings):
    """Secrets and endpoint from env: FOUNDRY_API_KEY, FOUNDRY_ENDPOINT, FOUNDRY_API_VERSION."""

    model_config = SettingsConfigDict(env_prefix="FOUNDRY_", env_file=".env", extra="ignore")

    api_key: SecretStr
    endpoint: str
    api_version: str = "2024-10-21"


class AzureFoundryBackend:
    def __init__(self, config: BackendConfig, settings: FoundrySettings | None = None):
        from openai import AzureOpenAI

        settings = settings or FoundrySettings()
        self.config = config
        self.client = AzureOpenAI(
            api_key=settings.api_key.get_secret_value(),
            azure_endpoint=settings.endpoint,
            api_version=settings.api_version,
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
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
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
