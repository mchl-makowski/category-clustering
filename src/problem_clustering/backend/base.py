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


class ContentFilteredError(RuntimeError):
    """Azure content filter blocked the prompt (HTTP 400 `content_filter`) or the reply
    (`finish_reason="content_filter"` / refusal). Never retried; every LLM step falls back."""


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


def embed_texts(backend: Backend, texts: list[str], batch_size: int = 64, max_workers: int = 1,
                max_chars: int | None = None, desc: str | None = None) -> np.ndarray:
    """Embed any number of texts in batches (up to `max_workers` batches in parallel); returns an (n, dim) array.

    Each text is cut to `max_chars` first (embedding model token limit; see `backend.embed_max_chars`).
    `desc` reports progress in texts (leave it None for one-off lookups)."""
    from contextlib import nullcontext

    from ..concurrency import parallel_map
    from ..logs import progress

    if not texts:
        return np.zeros((0, 0))
    if max_chars:
        texts = [t[:max_chars] for t in texts]
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    with progress(total=len(texts), desc=desc) if desc else nullcontext() as bar:

        def embed(batch: list[str]):
            vectors = backend.embed(batch).vectors
            if bar is not None:
                bar.update(len(batch))
            return vectors

        return np.vstack(parallel_map(embed, batches, max_workers))


def json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Azure structured-output `response_format`. `name` also keys Fake backend default replies."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
