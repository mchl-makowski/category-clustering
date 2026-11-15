"""Disk cache over any backend, keyed by content hash of (backend kind, model, input).

- Embeddings are cached per text, so a rerun with different batching still hits.
- Chat replies are cached per full request (messages, tools, response_format, model).
- One JSON file per entry, written atomically, so concurrent callers are safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import BackendConfig
from .base import Backend, ChatReply, EmbeddingReply, ToolCall, Usage


@dataclass
class CacheStats:
    embed_hits: int = 0
    embed_misses: int = 0  # texts sent to the underlying backend
    chat_hits: int = 0
    chat_misses: int = 0

    def summary(self) -> str:
        return (f"cache embed_hits={self.embed_hits} embed_calls={self.embed_misses} "
                f"chat_hits={self.chat_hits} chat_calls={self.chat_misses}")


def content_key(*parts: Any) -> str:
    """sha256 of the canonical JSON of `parts`."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachedBackend:
    def __init__(self, inner: Backend, cache_dir: str | Path, config: BackendConfig):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.config = config
        self.stats = CacheStats()
        self._lock = threading.Lock()

    # --- Backend protocol ------------------------------------------------
    def embed(self, texts: list[str]) -> EmbeddingReply:
        model = self.config.embedding_deployment
        keys = [content_key(self.config.kind, "embed", model, t) for t in texts]
        entries = [self._read("embed", k) for k in keys]
        missing = [i for i, e in enumerate(entries) if e is None]
        if missing:
            reply = self.inner.embed([texts[i] for i in missing])
            shares = _split_tokens(reply.usage.prompt_tokens, len(missing))
            for j, i in enumerate(missing):
                entries[i] = {"vector": reply.vectors[j].tolist(), "tokens": shares[j]}
                self._write("embed", keys[i], entries[i])
        with self._lock:
            self.stats.embed_hits += len(texts) - len(missing)
            self.stats.embed_misses += len(missing)
        vectors = np.array([e["vector"] for e in entries], dtype=float) if entries else np.zeros((0, 0))
        return EmbeddingReply(vectors=vectors, usage=Usage(prompt_tokens=sum(e["tokens"] for e in entries)))

    def chat(self, messages, tools=None, response_format=None, model=None) -> ChatReply:
        key = content_key(self.config.kind, "chat", model or self.config.chat_deployment,
                          messages, tools, response_format)
        entry = self._read("chat", key)
        if entry is not None:
            with self._lock:
                self.stats.chat_hits += 1
            return _reply_from(entry)
        reply = self.inner.chat(messages, tools=tools, response_format=response_format, model=model)
        self._write("chat", key, _reply_to(reply))
        with self._lock:
            self.stats.chat_misses += 1
        return reply

    # --- storage ---------------------------------------------------------
    def _path(self, kind: str, key: str) -> Path:
        return self.dir / kind / key[:2] / f"{key}.json"

    def _read(self, kind: str, key: str) -> dict | None:
        path = self._path(kind, key)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write(self, kind: str, key: str, entry: dict) -> None:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
        os.replace(tmp, path)


def _split_tokens(total: int, n: int) -> list[int]:
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def _reply_to(reply: ChatReply) -> dict:
    return {
        "content": reply.content,
        "tool_calls": [{"id": t.id, "name": t.name, "arguments": t.arguments} for t in reply.tool_calls],
        "usage": {"prompt_tokens": reply.usage.prompt_tokens, "completion_tokens": reply.usage.completion_tokens},
        "finish_reason": reply.finish_reason,
    }


def _reply_from(entry: dict) -> ChatReply:
    return ChatReply(
        content=entry["content"],
        tool_calls=[ToolCall(**t) for t in entry["tool_calls"]],
        usage=Usage(**entry["usage"]),
        finish_reason=entry["finish_reason"],
    )
