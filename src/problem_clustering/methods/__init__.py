"""Method registry by naming convention: method "x" lives in `methods/method_x.py`
and exposes `run(ctx: MethodContext) -> list[ProblemDraft]`.

Adding Method B or C = adding `method_b.py` / `method_c.py`; nothing here changes.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..backend import Backend, embed_texts
from ..config import Config
from ..result import ProblemDraft

MethodFn = Callable[["MethodContext"], list[ProblemDraft]]


@dataclass
class MethodContext:
    tickets: pd.DataFrame  # original export (do NOT send non-Tytuł/Opis columns to a model)
    texts: pd.Series  # grouped model input for non-empty Tickets only, indexed by ticket row label
    config: Config
    backend: Backend  # usage-tracked
    log: Callable[[str], None]
    _embeddings: np.ndarray | None = field(default=None, repr=False)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed any texts (batched, parallel, cut to `backend.embed_max_chars`)."""
        cfg = self.config
        return embed_texts(self.backend, texts, cfg.backend.embed_batch_size, cfg.runtime.max_concurrency,
                           cfg.backend.embed_max_chars)

    def embeddings(self) -> np.ndarray:
        """Embeddings of `texts` (row order = `texts.index`), computed once per run."""
        if self._embeddings is None:
            self._embeddings = self.embed(list(self.texts))
        return self._embeddings


def get_method(name: str) -> MethodFn:
    if not re.fullmatch(r"[a-z]", name or ""):
        raise ValueError(f"Unknown method {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.method_{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unknown method {name!r}") from exc
    return module.run
