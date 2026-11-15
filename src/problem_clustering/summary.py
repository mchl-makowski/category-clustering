"""Optional Symptom Summary, part of the shared text-preparation step.

With `symptom_summary.enabled`, each non-empty Ticket's model input (masked
`Tytuł` + `Opis`) is rewritten by the LLM into one normalised Polish line
(symptom + context). That line replaces the raw text in `MethodContext.texts`,
so every Method groups on it. Empty Tickets never get here (X-0000).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .backend import Backend, ContentFilteredError, json_schema_format
from .concurrency import parallel_map
from .config import Config
from .prompts import SUMMARY_SCHEMA, SUMMARY_SCHEMA_NAME, summary_messages


def symptom_summaries(texts: pd.Series, config: Config, backend: Backend,
                      log: Callable[[str], None]) -> pd.Series:
    """One LLM call per Ticket (in parallel); returns a Series aligned with `texts` holding single-line summaries."""
    fmt = json_schema_format(SUMMARY_SCHEMA_NAME, SUMMARY_SCHEMA)

    def summarise(text: str) -> str | None:
        try:
            reply = backend.chat(summary_messages(config.granularity, text), response_format=fmt).json()
        except ContentFilteredError:
            return None
        return " ".join(str(reply["summary"]).split())

    summaries = parallel_map(summarise, list(texts), config.runtime.max_concurrency)
    out = {}
    for (row, text), summary in zip(texts.items(), summaries):
        if summary is None:
            log(f"summary row={row}: blocked by content filter, raw masked text grouped instead")
        out[row] = summary or " ".join(text.split())  # never hand an empty string to grouping
        if config.symptom_summary.log_summaries:
            log(f"summary row={row}: {out[row]}")
    log(f"symptom summaries={len(out)}")
    return pd.Series(out, index=texts.index, dtype=object)
