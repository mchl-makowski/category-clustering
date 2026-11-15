"""Embedding clustering helpers reused by Methods A and B."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative(vectors: np.ndarray, distance_threshold: float, linkage: str = "average") -> np.ndarray:
    """Cluster labels from cosine-distance agglomerative clustering."""
    if len(vectors) < 2:
        return np.zeros(len(vectors), dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage=linkage, distance_threshold=distance_threshold
    )
    return model.fit_predict(vectors)


def centroid_confidence(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity of each vector to the group centroid; 1.0 for a one-Ticket group."""
    if len(vectors) == 1:
        return np.ones(1)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    centroid = unit.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    return np.clip(unit @ centroid, 0.0, 1.0)


def groups_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    """Positions per label, in order of first appearance."""
    _, first = np.unique(labels, return_index=True)
    return [np.flatnonzero(labels == labels[i]) for i in sorted(first)]
