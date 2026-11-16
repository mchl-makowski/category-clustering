"""Method B: coarse clustering + LLM verification (split / merge until stable).

1. agglomerative clusters at `method_b.coarse_threshold`;
2. each iteration: the LLM checks (in parallel) a sample of every not-yet-verified
   multi-Ticket cluster for being one Problem; a mixed cluster is re-clustered at a tighter
   threshold (minus `split_step`, tightened further until it actually splits);
3. then nearest-centroid neighbour pairs are LLM-checked for a merge (sequential:
   each merge changes which pairs are still open);
4. stop when an iteration changes nothing, or after `max_iterations`.
Naming is done by run_method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..backend import ContentFilteredError, json_schema_format
from ..clustering import (
    agglomerative, centroids, drafts_from_groups, groups_from_labels, nearest_pairs, sample_members, unit_rows,
)
from ..concurrency import parallel_map
from ..logs import progress
from ..prompts import (
    MERGE_SCHEMA, MERGE_SCHEMA_NAME, VERIFICATION_SCHEMA, VERIFICATION_SCHEMA_NAME, merge_messages,
    verification_messages,
)
from ..result import ProblemDraft
from . import MethodContext

log = logging.getLogger(__name__)


@dataclass
class _Cluster:
    positions: list[int]  # positions into ctx.texts / embeddings
    threshold: float  # distance threshold this cluster was formed at

    @property
    def key(self) -> frozenset[int]:
        return frozenset(self.positions)


def run(ctx: MethodContext) -> list[ProblemDraft]:
    cfg = ctx.config.method_b
    vectors = ctx.embeddings()
    unit = unit_rows(vectors)
    texts = list(ctx.texts)
    clusters = _cluster(vectors, list(range(len(texts))), cfg.coarse_threshold, cfg.linkage)
    log.info(f"method_b: coarse threshold={cfg.coarse_threshold} -> {len(clusters)} candidate Problems")

    verdicts: dict[frozenset[int], bool] = {}  # cluster -> is one Problem (never re-asked)
    verify_fmt = json_schema_format(VERIFICATION_SCHEMA_NAME, VERIFICATION_SCHEMA)

    def ask_one_problem(c: _Cluster) -> bool:
        sample = [texts[p] for p in sample_members(c.positions, cfg.sample_size)]
        try:
            reply = ctx.backend.chat(verification_messages(ctx.config.granularity, sample), response_format=verify_fmt)
        except ContentFilteredError:
            log.warning(f"method_b: verification blocked by content filter, candidate kept as one Problem "
                        f"({len(c.positions)} Tickets)")
            return True
        return bool(reply.json()["one_problem"])

    def verify(current: list[_Cluster], iteration: int) -> int:
        """Ask, in parallel, about every multi-Ticket cluster not verified yet; returns how many were asked."""
        todo = list({c.key: c for c in current if len(c.positions) >= 2 and c.key not in verdicts}.values())
        replies = parallel_map(ask_one_problem, todo, ctx.config.runtime.max_concurrency,
                               desc=f"method_b iteration {iteration} verification")
        for c, verdict in zip(todo, replies):
            verdicts[c.key] = verdict
        return len(todo)

    rejected: set[frozenset[frozenset[int]]] = set()  # merge pairs the LLM said no to
    merge_fmt = json_schema_format(MERGE_SCHEMA_NAME, MERGE_SCHEMA)
    per_side_sample = max(1, cfg.sample_size // 2)  # merge check shows two groups

    def merge_step(current: list[_Cluster], iteration: int) -> tuple[list[_Cluster], int]:
        merged_into: dict[int, _Cluster] = {}
        pairs = _neighbour_pairs(unit, current, cfg.merge_neighbours)
        for i, j in progress(pairs, desc=f"method_b iteration {iteration} merge checks"):
            if i in merged_into or j in merged_into:
                continue
            a, b = current[i], current[j]
            pair = frozenset((a.key, b.key))
            if pair in rejected or verdicts.get(a.key | b.key) is False:
                continue  # never re-ask a rejected pair, never recreate a cluster judged mixed
            msgs = merge_messages(
                ctx.config.granularity,
                [texts[p] for p in sample_members(a.positions, per_side_sample)],
                [texts[p] for p in sample_members(b.positions, per_side_sample)],
            )
            try:
                same = bool(ctx.backend.chat(msgs, response_format=merge_fmt).json()["same_problem"])
            except ContentFilteredError:
                log.warning("method_b: merge check blocked by content filter, not merged")
                same = False
            if not same:
                rejected.add(pair)
                continue
            joined = _Cluster(sorted(a.positions + b.positions), max(a.threshold, b.threshold))
            log.debug(f"method_b: merged candidates of {len(a.positions)} + {len(b.positions)} Tickets")
            verdicts[joined.key] = True
            merged_into[i] = merged_into[j] = joined
        out, seen = [], set()
        for k, c in enumerate(current):
            c = merged_into.get(k, c)
            if id(c) not in seen:
                seen.add(id(c))
                out.append(c)
        return out, len(merged_into) // 2

    for iteration in range(1, cfg.max_iterations + 1):
        next_clusters: list[_Cluster] = []
        verified = verify(clusters, iteration)
        mixed = splits = 0
        for c in clusters:
            if len(c.positions) < 2 or verdicts[c.key]:
                next_clusters.append(c)
                continue
            mixed += 1
            parts = _split(vectors, c, cfg.split_step, cfg.linkage)
            splits += len(parts) > 1
            next_clusters.extend(parts)
        clusters, merges = merge_step(next_clusters, iteration)
        log.info(f"method_b iteration {iteration}: verified={verified} mixed={mixed} splits={splits} "
                f"merges={merges} Problems={len(clusters)}")
        if not splits and not merges:
            log.info(f"method_b: stable after {iteration} iteration(s)")
            break
    else:
        log.info(f"method_b: stopped at max_iterations={cfg.max_iterations}")

    return drafts_from_groups(list(ctx.texts.index), vectors, [c.positions for c in clusters])


def _cluster(vectors: np.ndarray, positions: list[int], threshold: float, linkage: str) -> list[_Cluster]:
    labels = agglomerative(vectors[positions], threshold, linkage)
    return [_Cluster([positions[i] for i in group], threshold) for group in groups_from_labels(labels)]


def _neighbour_pairs(unit: np.ndarray, clusters: list[_Cluster], k: int) -> list[tuple[int, int]]:
    """Deduplicated (i, j) pairs of each cluster with its k nearest clusters by centroid, closest first."""
    if len(clusters) < 2:
        return []
    return nearest_pairs(centroids([unit[c.positions] for c in clusters]), k)


def _split(vectors: np.ndarray, c: _Cluster, step: float, linkage: str) -> list[_Cluster]:
    """Re-cluster at a tighter threshold until it splits; unsplittable (identical vectors) -> unchanged."""
    threshold = c.threshold
    while threshold > 0:
        threshold = max(threshold - step, 0.0)
        parts = _cluster(vectors, c.positions, threshold, linkage)
        if len(parts) > 1:
            return parts
    return [c]
