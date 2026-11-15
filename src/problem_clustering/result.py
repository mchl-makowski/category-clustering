"""Domain result types shared by all Methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .backend.tracking import UsageCounters
from .config import Config

EMPTY_PROBLEM_ID = "X-0000"
EMPTY_PROBLEM_NAME = "Brak opisu"
EMPTY_PROBLEM_DESCRIPTION = "Zgłoszenia bez tytułu i opisu; pominięte przez wszystkie kroki modelu."

# Confidence Levels an LLM picks (label -> numeric confidence).
CONFIDENCE_LEVELS: dict[str, float] = {
    "pewne": 1.0,
    "wysokie": 0.8,
    "średnie": 0.6,
    "niskie": 0.4,
    "zgadywane": 0.2,
}

ASSIGNMENT_COLUMNS = ["problem_id", "problem_name", "confidence", "confidence_level"]
CATALOGUE_COLUMNS = ["problem_id", "problem_name", "description", "ticket_count", "example_tickets"]


@dataclass
class ProblemDraft:
    """What a Method returns per Problem; ids and missing names are filled by `run_method`.

    `members` are row labels of the tickets DataFrame (non-empty Tickets only).
    `confidences` aligns with `members`; `confidence_levels` too, or None (Methods A/B).
    """

    members: list[int]
    confidences: list[float]
    confidence_levels: list[str] | None = None
    name: str | None = None
    description: str | None = None


@dataclass
class RunResult:
    method: str
    config: Config
    tickets: pd.DataFrame  # original export, unchanged
    assignments: pd.DataFrame  # same index as tickets; ASSIGNMENT_COLUMNS
    catalogue: pd.DataFrame  # CATALOGUE_COLUMNS, X-0000 last when present
    usage: UsageCounters
    log: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    # Embeddings of grouped text for non-empty Tickets, if the Method computed them
    # (rows = `embedding_rows`, ticket row labels). Lets the Judge reuse them.
    embeddings: np.ndarray | None = None
    embedding_rows: list[int] | None = None

    def tickets_with_assignments(self) -> pd.DataFrame:
        return pd.concat([self.tickets, self.assignments], axis=1)
