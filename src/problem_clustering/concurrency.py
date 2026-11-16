"""Bounded parallelism for independent model calls (naming, Judge checks, Method B verification ...).

    results = parallel_map(fn, items, config.runtime.max_concurrency, desc="naming")

Results keep input order, so output is the same as a sequential loop. The first
error (in input order) is raised after in-flight calls finish. Backends used from
`fn` must be thread-safe: the runtime wrappers, TrackedBackend and FakeBackend are.
Retry/backoff and the disk cache apply per call, via the backend `run_method` passes.
With `desc`, processed/total is reported through `logs.progress` (bar or ~10% lines).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

from .logs import progress

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Iterable[T], max_workers: int, desc: str | None = None) -> list[R]:
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in (progress(items, desc=desc) if desc else items)]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        futures = [pool.submit(fn, x) for x in items]
        if not desc:
            return [f.result() for f in futures]
        with progress(total=len(futures), desc=desc) as bar:
            for _ in as_completed(futures):
                bar.update()
            return [f.result() for f in futures]  # raises inside the bar: no "done" line after an error
