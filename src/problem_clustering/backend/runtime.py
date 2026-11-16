"""Composes the runtime wrappers `run_method` puts around the given backend:

    TrackedBackend( CachedBackend( RetryingBackend( inner ) ) )

Usage is tracked outside the cache, so reported LLM calls/tokens are the same on a
cached rerun (comparable across Methods); the log line from `CacheStats` says how
many of them actually reached the backend.
"""

from __future__ import annotations

from ..config import Config
from .base import Backend
from .cache import CachedBackend, CacheStats
from .retry import RetryingBackend


def with_runtime(backend: Backend, config: Config) -> tuple[Backend, CacheStats | None]:
    """Wrap `backend` with retry/backoff and the disk cache per `config.runtime`.

    Returns (wrapped backend, cache stats or None when the cache is off)."""
    rt = config.runtime
    backend = RetryingBackend(backend, rt)
    if rt.cache_dir:
        cached = CachedBackend(backend, rt.cache_dir, config.backend)
        return cached, cached.stats
    return backend, None
