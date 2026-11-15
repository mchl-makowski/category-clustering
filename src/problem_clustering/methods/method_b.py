"""Method B: coarse clustering + LLM verification (split / merge until stable).

1. agglomerative clusters at `method_b.coarse_threshold`;
2. each iteration: the LLM checks a sample of every not-yet-verified multi-Ticket
   cluster for being one Problem; a mixed cluster is re-clustered at a tighter
   threshold (minus `split_step`, tightened further until it actually splits);
3. then nearest-centroid neighbour pairs are LLM-checked for a merge;
4. stop when an iteration changes nothing, or after `max_iterations`.
Naming is done by run_method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backend import json_schema_format
from ..clustering import agglomerative, centroid_confidence, groups_from_labels
from ..concurrency import parallel_map
from ..naming import sample_members
from ..prompts import (
    MERGE_SCHEMA, MERGE_SCHEMA_NAME, VERIFICATION_SCHEMA, VERIFICATION_SCHEMA_NAME, merge_messages,
    verification_messages,
)
from ..result import ProblemDraft
from . import MethodContext


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
    texts = list(ctx.texts)
    clusters = _cluster(vectors, list(range(len(texts))), cfg.coarse_threshold, cfg.linkage)
    ctx.log(f"method_b: coarse threshold={cfg.coarse_threshold} -> {len(clusters)} clusters")

    verdicts: dict[frozenset[int], bool] = {}  # cluster -> is one Problem (never re-asked)
    verify_fmt = json_schema_format(VERIFICATION_SCHEMA_NAME, VERIFICATION_SCHEMA)

    def ask_one_problem(c: _Cluster) -> bool:
        sample = [texts[p] for p in sample_members(c.positions, cfg.sample_size)]
        reply = ctx.backend.chat(verification_messages(ctx.config.granularity, sample), response_format=verify_fmt)
        return bool(reply.json()["one_problem"])

    def verify(current: list[_Cluster]) -> int:
        """Ask, in parallel, about every multi-Ticket cluster not verified yet; returns how many were asked."""
        todo = list({c.key: c for c in current if len(c.positions) >= 2 and c.key not in verdicts}.values())
        for c, verdict in zip(todo, parallel_map(ask_one_problem, todo, ctx.config.runtime.max_concurrency)):
            verdicts[c.key] = verdict
        return len(todo)

    rejected: set[frozenset[frozenset[int]]] = set()  # merge pairs the LLM said no to
    merge_fmt = json_schema_format(MERGE_SCHEMA_NAME, MERGE_SCHEMA)
    half = max(1, cfg.sample_size // 2)

    def merge_step(current: list[_Cluster]) -> tuple[list[_Cluster], int]:
        merged_into: dict[int, _Cluster] = {}
        for i, j in _neighbour_pairs(vectors, current, cfg.merge_neighbours):
            if i in merged_into or j in merged_into:
                continue
            a, b = current[i], current[j]
            pair = frozenset((a.key, b.key))
            if pair in rejected or verdicts.get(a.key | b.key) is False:
                continue  # never re-ask a rejected pair, never recreate a cluster judged mixed
            msgs = merge_messages(
                ctx.config.granularity,
                [texts[p] for p in sample_members(a.positions, half)],
                [texts[p] for p in sample_members(b.positions, half)],
            )
            if not bool(ctx.backend.chat(msgs, response_format=merge_fmt).json()["same_problem"]):
                rejected.add(pair)
                continue
            joined = _Cluster(sorted(a.positions + b.positions), max(a.threshold, b.threshold))
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
        verified = verify(clusters)
        mixed = splits = 0
        for c in clusters:
            if len(c.positions) < 2 or verdicts[c.key]:
                next_clusters.append(c)
                continue
            mixed += 1
            parts = _split(vectors, c, cfg.split_step, cfg.linkage)
            splits += len(parts) > 1
            next_clusters.extend(parts)
        clusters, merges = merge_step(next_clusters)
        ctx.log(f"method_b iteration {iteration}: verified={verified} mixed={mixed} splits={splits} "
                f"merges={merges} clusters={len(clusters)}")
        if not splits and not merges:
            ctx.log(f"method_b: stable after {iteration} iteration(s)")
            break
    else:
        ctx.log(f"method_b: stopped at max_iterations={cfg.max_iterations}")

    rows = list(ctx.texts.index)
    drafts = []
    for c in clusters:
        conf = centroid_confidence(vectors[c.positions])
        drafts.append(ProblemDraft(members=[rows[p] for p in c.positions], confidences=[float(x) for x in conf]))
    return drafts


def _cluster(vectors: np.ndarray, positions: list[int], threshold: float, linkage: str) -> list[_Cluster]:
    labels = agglomerative(vectors[positions], threshold, linkage)
    return [_Cluster([positions[i] for i in group], threshold) for group in groups_from_labels(labels)]


def _neighbour_pairs(vectors: np.ndarray, clusters: list[_Cluster], k: int) -> list[tuple[int, int]]:
    """Deduplicated (i, j) pairs of each cluster with its k nearest clusters by centroid, closest first."""
    if len(clusters) < 2 or k < 1:
        return []
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cents = np.vstack([unit[c.positions].mean(axis=0) for c in clusters])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    sim = cents @ cents.T
    np.fill_diagonal(sim, -np.inf)
    pairs: dict[tuple[int, int], float] = {}
    for i in range(len(clusters)):
        for j in np.argsort(-sim[i])[:k]:
            pairs[(min(i, int(j)), max(i, int(j)))] = float(sim[i, j])
    return sorted(pairs, key=lambda p: (-pairs[p], p))


def _split(vectors: np.ndarray, c: _Cluster, step: float, linkage: str) -> list[_Cluster]:
    """Re-cluster at a tighter threshold until it splits; unsplittable (identical vectors) -> unchanged."""
    threshold = c.threshold
    while threshold > 0:
        threshold = max(threshold - step, 0.0)
        parts = _cluster(vectors, c.positions, threshold, linkage)
        if len(parts) > 1:
            return parts
    return [c]
