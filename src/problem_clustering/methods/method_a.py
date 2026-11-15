"""Method A: embeddings + clustering (baseline). Naming is done by run_method."""

from __future__ import annotations

from ..clustering import agglomerative, centroid_confidence, groups_from_labels
from ..result import ProblemDraft
from . import MethodContext


def run(ctx: MethodContext) -> list[ProblemDraft]:
    cfg = ctx.config.method_a
    vectors = ctx.embeddings()
    labels = agglomerative(vectors, cfg.distance_threshold, cfg.linkage)
    rows = list(ctx.texts.index)
    drafts = []
    for positions in groups_from_labels(labels):
        conf = centroid_confidence(vectors[positions])
        drafts.append(ProblemDraft(members=[rows[p] for p in positions], confidences=[float(c) for c in conf]))
    ctx.log(f"method_a: {cfg.algorithm} threshold={cfg.distance_threshold} -> {len(drafts)} Problems")
    return drafts
