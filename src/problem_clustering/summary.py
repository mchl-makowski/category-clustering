"""Optional Symptom Summary, part of the shared text-preparation step.

With `symptom_summary.enabled`, each non-empty Ticket's model input (masked
`Tytuł` + `Opis`) is rewritten by the LLM into one normalised Polish line
(symptom + context). That line replaces the raw text in `MethodContext.texts`,
so every Method groups on it. Empty Tickets never get here (X-0000).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .backend import Backend, json_schema_format
from .concurrency import parallel_map
from .config import Config, GranularityLevel
from .prompts import problem_definition

SUMMARY_SCHEMA_NAME = "symptom_summary"
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
TICKET_HEADER = "Zgłoszenie:\n"


def summary_messages(level: GranularityLevel, text: str) -> list[dict]:
    system = (
        f"{problem_definition(level)}\n\n"
        "Jesteś analitykiem Service Desk. Przepisz zgłoszenie na jedną znormalizowaną linię po polsku: "
        "obserwowalny objaw + kontekst (system, komponent, wersja, lokalizacja), bez danych osobowych, "
        "powitań i zbędnych szczegółów. Szczegółowość kontekstu dopasuj do poziomu szczegółowości powyżej. "
        "Odpowiedz w JSON: {\"summary\": ...}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": TICKET_HEADER + text}]


def symptom_summaries(texts: pd.Series, config: Config, backend: Backend,
                      log: Callable[[str], None]) -> pd.Series:
    """One LLM call per Ticket; returns a Series aligned with `texts` holding single-line summaries."""
    fmt = json_schema_format(SUMMARY_SCHEMA_NAME, SUMMARY_SCHEMA)

    def summarise(text: str) -> str:
        reply = backend.chat(summary_messages(config.granularity, text), response_format=fmt).json()
        return " ".join(str(reply["summary"]).split())

    summaries = parallel_map(summarise, list(texts), config.runtime.max_concurrency)
    out = {}
    for (row, text), summary in zip(texts.items(), summaries):
        out[row] = summary or " ".join(text.split())  # never hand an empty string to grouping
        if config.symptom_summary.log_summaries:
            log(f"summary row={row}: {out[row]}")
    log(f"symptom summaries={len(out)}")
    return pd.Series(out, index=texts.index, dtype=object)
