"""`run_method`: the shared pipeline around every Method.

1. model input per Ticket (Tytuł + Opis, PII-masked); empty Tickets -> X-0000, never sent;
2. the Method turns non-empty Tickets into ProblemDrafts;
3. the shared naming step names any unnamed Problem;
4. ids `<M>-0001…` by size (desc), assignments + Problem Catalogue + usage + log.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .backend import Backend, TrackedBackend
from .config import Config
from .methods import MethodContext, get_method
from .naming import name_problems
from .result import (
    ASSIGNMENT_COLUMNS, CATALOGUE_COLUMNS, EMPTY_PROBLEM_DESCRIPTION, EMPTY_PROBLEM_ID, EMPTY_PROBLEM_NAME,
    ProblemDraft, RunResult,
)
from .text import model_texts
from .tickets import ID_COLUMN, TITLE_COLUMN


def run_method(tickets: pd.DataFrame, method: str, config: Config, backend: Backend) -> RunResult:
    method = method.lower()
    method_fn = get_method(method)
    started_at = datetime.now()
    start = time.perf_counter()
    tracked = TrackedBackend(backend)
    log: list[str] = []

    def say(line: str) -> None:
        log.append(f"[{datetime.now():%H:%M:%S}] {line}")

    say(f"run method={method} tickets={len(tickets)}")
    say("config " + json.dumps(config.model_dump(), ensure_ascii=False))

    all_texts = model_texts(tickets)
    texts = all_texts.dropna()
    empty_rows = [i for i in tickets.index if i not in texts.index]
    say(f"non-empty={len(texts)} empty(X-0000)={len(empty_rows)}")

    ctx = MethodContext(tickets=tickets, texts=texts, config=config, backend=tracked, log=say)
    drafts = method_fn(ctx) if len(texts) else []
    _check_partition(drafts, texts.index)
    drafts.sort(key=lambda d: (-len(d.members), min(d.members)))

    name_problems(drafts, texts, config, tracked)

    assignments = pd.DataFrame(index=tickets.index, columns=ASSIGNMENT_COLUMNS, dtype=object)
    catalogue_rows = []
    prefix = method.upper()
    for k, draft in enumerate(drafts, start=1):
        pid = f"{prefix}-{k:04d}"
        levels = draft.confidence_levels or [None] * len(draft.members)
        for row, conf, level in zip(draft.members, draft.confidences, levels):
            assignments.loc[row] = [pid, draft.name, float(conf), level]
        catalogue_rows.append(_catalogue_row(pid, draft.name, draft.description, draft.members, tickets, config))
    if empty_rows:
        for row in empty_rows:
            assignments.loc[row] = [EMPTY_PROBLEM_ID, EMPTY_PROBLEM_NAME, 1.0, None]
        catalogue_rows.append(
            _catalogue_row(EMPTY_PROBLEM_ID, EMPTY_PROBLEM_NAME, EMPTY_PROBLEM_DESCRIPTION, empty_rows, tickets, config)
        )
    assignments["confidence"] = assignments["confidence"].astype(float)

    usage = tracked.counters
    usage.runtime_s = time.perf_counter() - start
    say(f"problems={len(drafts)} (+X-0000)" if empty_rows else f"problems={len(drafts)}")
    say("totals " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in usage.as_dict().items()))

    return RunResult(
        method=method,
        config=config,
        tickets=tickets,
        assignments=assignments,
        catalogue=pd.DataFrame(catalogue_rows, columns=CATALOGUE_COLUMNS),
        usage=usage,
        log=log,
        started_at=started_at,
        embeddings=ctx._embeddings,
        embedding_rows=list(texts.index) if ctx._embeddings is not None else None,
    )


def _check_partition(drafts: list[ProblemDraft], rows: pd.Index) -> None:
    seen = [r for d in drafts for r in d.members]
    if len(seen) != len(set(seen)) or set(seen) != set(rows):
        raise RuntimeError("Method must assign every non-empty Ticket to exactly one Problem")
    for d in drafts:
        if len(d.confidences) != len(d.members):
            raise RuntimeError("ProblemDraft.confidences must align with members")


def _catalogue_row(pid, name, description, members, tickets, config) -> dict:
    examples = []
    for row in members[: config.output.example_tickets]:
        title = tickets.loc[row, TITLE_COLUMN]
        title = "" if pd.isna(title) else str(title).strip()
        examples.append(f"{tickets.loc[row, ID_COLUMN]}: {title}")
    return {
        "problem_id": pid,
        "problem_name": name,
        "description": description,
        "ticket_count": len(members),
        "example_tickets": "\n".join(examples),
    }
