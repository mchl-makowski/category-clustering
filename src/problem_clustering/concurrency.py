"""Bounded parallelism for independent model calls (naming, Judge checks, Method B verification ...).

    results = parallel_map(fn, items, config.runtime.max_concurrency)

Results keep input order, so output is the same as a sequential loop. The first
error (in input order) is raised after in-flight calls finish. Backends used from
`fn` must be thread-safe: the runtime wrappers, TrackedBackend and FakeBackend are.
Retry/backoff and the disk cache apply per call, via the backend `run_method` passes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Iterable[T], max_workers: int) -> list[R]:
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = [pool.submit(fn, x) for x in items]
        return [f.result() for f in futures]
