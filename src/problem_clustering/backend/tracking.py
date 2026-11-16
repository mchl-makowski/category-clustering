"""Counts calls and tokens of any backend. `run_method` wraps the given backend with it."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .base import Backend, ChatReply, EmbeddingReply


@dataclass
class UsageCounters:
    llm_calls: int = 0
    embedding_calls: int = 0
    prompt_tokens: int = 0  # chat prompt tokens
    completion_tokens: int = 0
    embedding_tokens: int = 0
    runtime_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


class TrackedBackend:
    def __init__(self, inner: Backend, counters: UsageCounters | None = None):
        self.inner = inner
        self.counters = counters or UsageCounters()
        self._lock = threading.Lock()  # thread-safe under parallel_map

    def embed(self, texts: list[str]) -> EmbeddingReply:
        reply = self.inner.embed(texts)
        with self._lock:
            self._add_embed(reply)
        return reply

    def _add_embed(self, reply: EmbeddingReply) -> None:
        self.counters.embedding_calls += 1
        self.counters.embedding_tokens += reply.usage.prompt_tokens

    def chat(self, messages, tools=None, response_format=None, model=None) -> ChatReply:
        reply = self.inner.chat(messages, tools=tools, response_format=response_format, model=model)
        with self._lock:
            self._add_chat(reply)
        return reply

    def _add_chat(self, reply: ChatReply) -> None:
        self.counters.llm_calls += 1
        self.counters.prompt_tokens += reply.usage.prompt_tokens
        self.counters.completion_tokens += reply.usage.completion_tokens
