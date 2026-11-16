"""Embedding clustering and sampling helpers shared by the Methods and the Judge."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering

from .result import ProblemDraft


def agglomerative(vectors: np.ndarray, distance_threshold: float, linkage: str = "average") -> np.ndarray:
    """Cluster labels from cosine-distance agglomerative clustering."""
    if len(vectors) < 2:
        return np.zeros(len(vectors), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage=linkage, distance_threshold=distance_threshold
    )
    return model.fit_predict(vectors)


def hdbscan(vectors: np.ndarray, min_cluster_size: int = 2, min_samples: int | None = None) -> np.ndarray:
    """Cluster labels from cosine HDBSCAN; each noise point gets its own label (no "unassigned")."""
    n = len(vectors)
    if n < max(2, min_cluster_size):
        labels = np.full(n, -1)
    else:
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="cosine", copy=True)
        labels = model.fit_predict(vectors)
    noise = np.flatnonzero(labels < 0)
    labels = labels.copy()
    labels[noise] = labels.max(initial=-1) + 1 + np.arange(len(noise))
    return labels


def centroid_confidence(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity of each vector to the group centroid; 1.0 for a one-Ticket group."""
    if len(vectors) == 1:
        return np.ones(1)
    unit = unit_rows(vectors)
    return np.clip(unit @ centroids([unit])[0], 0.0, 1.0)


def groups_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    """Positions per label, in order of first appearance."""
    _, first = np.unique(labels, return_index=True)
    return [np.flatnonzero(labels == labels[i]) for i in sorted(first)]


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Each row scaled to unit length (zero rows stay zero)."""
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def centroids(groups: Sequence[np.ndarray]) -> np.ndarray:
    """Unit-length centroid per group of unit vectors; shape (len(groups), dim)."""
    return unit_rows(np.vstack([np.asarray(g).mean(axis=0) for g in groups]))


def nearest_pairs(unit_centroids: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Deduplicated (i, j), i < j: each centroid with its k nearest others, closest pair first."""
    n = len(unit_centroids)
    if n < 2 or k < 1:
        return []
    sim = unit_centroids @ unit_centroids.T
    np.fill_diagonal(sim, -np.inf)
    pairs: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in np.argsort(-sim[i], kind="stable")[: min(k, n - 1)]:
            pairs[(min(i, int(j)), max(i, int(j)))] = float(sim[i, j])
    return sorted(pairs, key=lambda p: (-pairs[p], p))


def sample_members(members: list, k: int) -> list:
    """Deterministic, evenly spread sample of up to k members."""
    if len(members) <= k:
        return list(members)
    step = len(members) / k
    return [members[int(i * step)] for i in range(k)]


def drafts_from_groups(rows: list, vectors: np.ndarray, groups: Sequence[Sequence[int]]) -> list[ProblemDraft]:
    """One ProblemDraft per group of positions into `rows`/`vectors`; confidence = similarity to the centroid."""
    drafts = []
    for positions in groups:
        positions = list(positions)
        conf = centroid_confidence(vectors[positions])
        drafts.append(ProblemDraft(members=[rows[p] for p in positions], confidences=[float(c) for c in conf]))
    return drafts
