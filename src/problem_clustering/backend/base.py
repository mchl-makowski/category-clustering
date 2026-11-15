"""Backend protocol and reply types. Shapes mirror the Azure OpenAI chat/embeddings replies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ToolCall:
    """One function tool call, as in `choices[0].message.tool_calls[i]`."""

    id: str
    name: str
    arguments: str  # JSON string, exactly as Azure returns it

    def args(self) -> dict[str, Any]:
        return json.loads(self.arguments or "{}")


@dataclass
class ChatReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"

    def json(self) -> Any:
        """Parse `content` produced under a JSON `response_format`."""
        return json.loads(self.content or "null")


@dataclass
class EmbeddingReply:
    vectors: np.ndarray  # shape (len(texts), dim)
    usage: Usage = field(default_factory=Usage)  # completion_tokens always 0


class TransientBackendError(RuntimeError):
    """A call that may succeed if retried (rate limit, timeout, 5xx). See `retry.is_transient`."""


class Backend(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingReply:
        """One embeddings request for `texts` (callers batch; see `embed_texts`)."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> ChatReply:
        """One chat completion. `model` overrides the configured chat deployment (e.g. the Judge)."""


def embed_texts(backend: Backend, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed any number of texts in batches; returns an (n, dim) array."""
    if not texts:
        return np.zeros((0, 0))
    parts = [backend.embed(texts[i : i + batch_size]).vectors for i in range(0, len(texts), batch_size)]
    return np.vstack(parts)


def json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Azure structured-output `response_format`. `name` also keys Fake backend default replies."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
