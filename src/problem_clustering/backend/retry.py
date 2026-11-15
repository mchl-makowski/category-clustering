"""Retry with exponential backoff on transient errors, around any backend."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

from ..config import RuntimeConfig
from .base import Backend, ChatReply, EmbeddingReply, TransientBackendError

T = TypeVar("T")

# openai SDK exception classes worth retrying (matched by name: no import needed offline)
_TRANSIENT_NAMES = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TransientBackendError):
        return True
    return any(cls.__name__ in _TRANSIENT_NAMES for cls in type(exc).__mro__)


def backoff_delays(rt: RuntimeConfig) -> list[float]:
    """Wait before each retry: initial, initial*multiplier, ... capped at backoff_max_s."""
    return [min(rt.backoff_initial_s * rt.backoff_multiplier**i, rt.backoff_max_s) for i in range(rt.max_retries)]


class RetryingBackend:
    def __init__(self, inner: Backend, runtime: RuntimeConfig,
                 log: Callable[[str], None] = lambda _: None, sleep: Callable[[float], None] = time.sleep):
        self.inner = inner
        self.delays = backoff_delays(runtime)
        self.log = log
        self.sleep = sleep

    def _call(self, what: str, fn: Callable[[], T]) -> T:
        for attempt in range(len(self.delays) + 1):
            try:
                return fn()
            except Exception as exc:
                if attempt == len(self.delays) or not is_transient(exc):
                    raise
                delay = self.delays[attempt]
                self.log(f"retry {attempt + 1}/{len(self.delays)} {what} after {type(exc).__name__}: "
                         f"{exc} (wait {delay:.1f}s)")
                self.sleep(delay)
        raise AssertionError("unreachable")

    def embed(self, texts: list[str]) -> EmbeddingReply:
        return self._call("embed", lambda: self.inner.embed(texts))

    def chat(self, messages, tools=None, response_format=None, model=None) -> ChatReply:
        return self._call("chat", lambda: self.inner.chat(
            messages, tools=tools, response_format=response_format, model=model))
