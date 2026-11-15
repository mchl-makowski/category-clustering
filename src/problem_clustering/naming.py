"""Shared Polish naming step: one LLM call per Problem that has no name yet (in parallel)."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .backend import Backend, ContentFilteredError, json_schema_format
from .concurrency import parallel_map
from .config import Config
from .prompts import NAMING_SCHEMA, NAMING_SCHEMA_NAME, naming_messages
from .result import ProblemDraft


def sample_members(members: list[int], k: int) -> list[int]:
    """Deterministic, evenly spread sample of up to k members."""
    if len(members) <= k:
        return list(members)
    step = len(members) / k
    return [members[int(i * step)] for i in range(k)]


FALLBACK_NAME_CHARS = 100
FALLBACK_DESCRIPTION = "Nazwa zastępcza z tytułu pierwszego zgłoszenia (odpowiedź modelu zablokowana przez filtr treści)."


def name_problems(drafts: list[ProblemDraft], texts: pd.Series, config: Config, backend: Backend,
                  log: Callable[[str], None] = lambda _: None) -> None:
    """Fill `name`/`description` in place for drafts that lack them.

    A reply blocked by the content filter falls back to the first Ticket's title (first line of its text)."""
    fmt = json_schema_format(NAMING_SCHEMA_NAME, NAMING_SCHEMA)
    todo = [d for d in drafts if not (d.name and d.description)]

    def ask(draft: ProblemDraft) -> dict | None:
        sample = [texts[i] for i in sample_members(draft.members, config.naming.sample_size)]
        try:
            return backend.chat(naming_messages(config.granularity, sample), response_format=fmt).json()
        except ContentFilteredError:
            return None

    for draft, reply in zip(todo, parallel_map(ask, todo, config.runtime.max_concurrency)):
        if reply is None:
            title = texts[draft.members[0]].splitlines()[0].strip()[:FALLBACK_NAME_CHARS]
            reply = {"name": title, "description": FALLBACK_DESCRIPTION}
            log(f"naming: blocked by content filter, Problem named after its first Ticket: {title!r}")
        draft.name = draft.name or str(reply["name"]).strip()
        draft.description = draft.description or str(reply["description"]).strip()
