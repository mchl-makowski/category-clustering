"""Shared Polish naming step: one LLM call per Problem that has no name yet."""

from __future__ import annotations

import pandas as pd

from .backend import Backend, json_schema_format
from .config import Config
from .prompts import NAMING_SCHEMA, NAMING_SCHEMA_NAME, naming_messages
from .result import ProblemDraft


def sample_members(members: list[int], k: int) -> list[int]:
    """Deterministic, evenly spread sample of up to k members."""
    if len(members) <= k:
        return list(members)
    step = len(members) / k
    return [members[int(i * step)] for i in range(k)]


def name_problems(drafts: list[ProblemDraft], texts: pd.Series, config: Config, backend: Backend) -> None:
    """Fill `name`/`description` in place for drafts that lack them."""
    fmt = json_schema_format(NAMING_SCHEMA_NAME, NAMING_SCHEMA)
    for draft in drafts:
        if draft.name and draft.description:
            continue
        sample = [texts[i] for i in sample_members(draft.members, config.naming.sample_size)]
        reply = backend.chat(naming_messages(config.granularity, sample), response_format=fmt).json()
        draft.name = draft.name or str(reply["name"]).strip()
        draft.description = draft.description or str(reply["description"]).strip()
