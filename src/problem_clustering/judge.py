"""Judge: label-free evaluation of any RunResult (Method A, B or C).

- Mixed check: every multi-Ticket Problem, up to `judge.sample_size` sampled Tickets,
  "do these describe one Problem?".
- Split check: each Problem vs its `judge.neighbours` nearest Problems by centroid,
  pairs deduplicated, "are these the same Problem?".
- `judge.max_problems` limits both checks to the N largest Problems.

The Judge always calls `judge.deployment` (passed as `model=`), never the run's chat
model, and counts its own usage separately from the Method's. X-0000 is never judged.
Checks run in parallel (`runtime.max_concurrency`) through the same retry + disk cache
stack as `run_method`. A check blocked by the content filter gets verdict "error" and is
left out of purity / split rate. Relies only on RunResult fields, so it works for every Method.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backend import Backend, ContentFilteredError, TrackedBackend, UsageCounters, embed_texts, json_schema_format
from .backend.runtime import with_runtime
from .concurrency import parallel_map
from .config import Config
from .logs import capture
from .clustering import centroids, nearest_pairs, sample_members, unit_rows
from .prompts import (
    MIXED_SCHEMA, MIXED_SCHEMA_NAME, SPLIT_SCHEMA, SPLIT_SCHEMA_NAME, mixed_messages, split_messages,
)
from .result import EMPTY_PROBLEM_ID, RunResult
from .text import model_texts

JUDGE_COLUMNS = ["check", "problem_id", "other_problem_id", "verdict", "reason"]
ERROR_VERDICT = "error"  # check got no verdict (e.g. content filter); excluded from purity / split rate
FILTERED_REASON = "content filter: the Judge prompt or reply was blocked"

log = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    checks: pd.DataFrame  # JUDGE_COLUMNS; one row per Mixed check and per Split pair
    usage: UsageCounters = field(default_factory=UsageCounters)  # Judge usage only

    @property
    def mixed_checks(self) -> pd.DataFrame:
        return self.checks[self.checks["check"] == "mixed"]

    @property
    def split_checks(self) -> pd.DataFrame:
        return self.checks[self.checks["check"] == "split"]

    @property
    def purity(self) -> float | None:
        """Share of Mixed-checked Problems judged to be one Problem ("error" checks excluded)."""
        m = self.mixed_checks
        m = m[m["verdict"] != ERROR_VERDICT]
        return float((m["verdict"] == "one_problem").mean()) if len(m) else None

    @property
    def split_rate(self) -> float | None:
        """Share of judged Problem pairs flagged as a Split Problem ("error" checks excluded)."""
        s = self.split_checks
        s = s[s["verdict"] != ERROR_VERDICT]
        return float((s["verdict"] == "split").mean()) if len(s) else None


def judge(run_result: RunResult, config: Config, backend: Backend) -> JudgeResult:
    """Judge a run; its log records are appended to `run_result.log` (the run's `.txt`)."""
    with capture(run_result.log):
        return _judge(run_result, config, backend)


def _judge(run_result: RunResult, config: Config, backend: Backend) -> JudgeResult:
    cfg = config.judge
    start = time.perf_counter()
    tracked = TrackedBackend(with_runtime(backend, config)[0])  # same retry + disk cache as run_method
    texts = model_texts(run_result.tickets)

    members: dict[str, list] = {}
    for row, pid in run_result.assignments["problem_id"].items():
        if pid != EMPTY_PROBLEM_ID:
            members.setdefault(pid, []).append(row)
    order = [pid for pid in run_result.catalogue["problem_id"] if pid in members]
    order += [pid for pid in members if pid not in order]
    order.sort(key=lambda pid: -len(members[pid]))  # stable: catalogue order within equal size
    if cfg.max_problems is not None:
        order = order[: cfg.max_problems]

    log.info(f"judge: {len(order)} Problems, deployment={cfg.deployment}")
    workers = config.runtime.max_concurrency
    mixed_fmt = json_schema_format(MIXED_SCHEMA_NAME, MIXED_SCHEMA)

    def mixed_check(pid: str) -> dict:
        sample = [texts[r] for r in sample_members(members[pid], cfg.sample_size)]
        row = {"check": "mixed", "problem_id": pid, "other_problem_id": None}
        try:
            reply = tracked.chat(mixed_messages(config.granularity, sample), response_format=mixed_fmt,
                                 model=cfg.deployment).json()
        except ContentFilteredError:
            log.warning(f"judge: Mixed check of {pid} blocked by content filter")
            return {**row, "verdict": ERROR_VERDICT, "reason": FILTERED_REASON}
        return {**row, "verdict": "mixed" if reply["mixed"] else "one_problem", "reason": str(reply["reason"])}

    split_fmt = json_schema_format(SPLIT_SCHEMA_NAME, SPLIT_SCHEMA)

    def split_check(pair: tuple[str, str]) -> dict:
        a, b = pair
        sa = [texts[r] for r in sample_members(members[a], cfg.split_sample_size)]
        sb = [texts[r] for r in sample_members(members[b], cfg.split_sample_size)]
        row = {"check": "split", "problem_id": a, "other_problem_id": b}
        try:
            reply = tracked.chat(split_messages(config.granularity, sa, sb), response_format=split_fmt,
                                 model=cfg.deployment).json()
        except ContentFilteredError:
            log.warning(f"judge: Split check of {a} / {b} blocked by content filter")
            return {**row, "verdict": ERROR_VERDICT, "reason": FILTERED_REASON}
        return {**row, "verdict": "split" if reply["same_problem"] else "distinct", "reason": str(reply["reason"])}

    rows = parallel_map(mixed_check, [pid for pid in order if len(members[pid]) >= 2], workers,
                        desc="judge Mixed checks")
    pairs = _split_pairs(run_result, order, members, texts, cfg.neighbours, config, tracked)
    rows += parallel_map(split_check, pairs, workers, desc="judge Split checks")

    tracked.counters.runtime_s = time.perf_counter() - start
    result = JudgeResult(checks=pd.DataFrame(rows, columns=JUDGE_COLUMNS), usage=tracked.counters)
    log.info(f"judge: mixed_checks={len(result.mixed_checks)} split_checks={len(result.split_checks)} "
             f"purity={_fmt(result.purity)} split_rate={_fmt(result.split_rate)} "
             f"llm_calls={tracked.counters.llm_calls}")
    return result


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def judge_safely(run_result: RunResult, config: Config, backend: Backend) -> JudgeResult | None:
    """`judge`, but a Judge failure never loses a finished run: on any error a WARNING is
    logged (and appended to `run_result.log`) and None is returned, so the run can still be written."""
    with capture(run_result.log):
        try:
            return _judge(run_result, config, backend)
        except Exception as exc:  # noqa: BLE001 - any Judge failure; the run itself is done
            log.warning(f"judge failed, run written without Judge: {type(exc).__name__}: {exc}")
            return None


def _split_pairs(run_result, order, members, texts, k, config, backend) -> list[tuple[str, str]]:
    """Deduplicated (a, b) pairs: each judged Problem with its k nearest judged Problems by centroid."""
    if len(order) < 2 or k < 1:
        return []
    vectors = _ticket_vectors(run_result, [r for pid in order for r in members[pid]], texts, config, backend)
    cents = centroids([np.vstack([vectors[r] for r in members[pid]]) for pid in order])
    return [(order[i], order[j]) for i, j in nearest_pairs(cents, k)]


def _ticket_vectors(run_result, rows, texts, config, backend) -> dict:
    """Unit vectors per ticket row: reused from the run when present, otherwise embedded (Judge usage)."""
    vectors: dict = {}
    if run_result.embeddings is not None and run_result.embedding_rows is not None:
        vectors = dict(zip(run_result.embedding_rows, run_result.embeddings))
    missing = [r for r in rows if r not in vectors]
    if missing:
        new = embed_texts(backend, [texts[r] for r in missing], config.backend.embed_batch_size,
                          config.runtime.max_concurrency, config.backend.embed_max_chars, desc="judge embeddings")
        vectors.update(zip(missing, new))
    keys = list(vectors)
    return dict(zip(keys, unit_rows(np.vstack([vectors[r] for r in keys])))) if keys else {}
