"""Model input preparation shared by all Methods.

Only `Tytuł` + `Opis` ever reach a model (acceptance criterion), with minimal
PII masking. No other cleaning. Output keeps the original text.
"""

import re

import pandas as pd

from .tickets import DESCRIPTION_COLUMN, TITLE_COLUMN

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PESEL = re.compile(r"(?<!\d)\d{11}(?!\d)")
_PHONE = re.compile(
    r"(?<![\w+])(?:\+48[\s-]?)?(?:\d{3}[\s-]?\d{3}[\s-]?\d{3}|\d{2}[\s-]\d{3}[\s-]\d{2}[\s-]\d{2})(?!\w)"
)


def mask_pii(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PESEL.sub("[PESEL]", text)  # before phone: 11 digits would partly match a phone
    return _PHONE.sub("[TELEFON]", text)


def _clean_cell(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def model_texts(tickets: pd.DataFrame) -> pd.Series:
    """Per Ticket: masked `Tytuł` + newline + `Opis`, or None when both are empty (→ X-0000)."""

    def build(row) -> str | None:
        parts = [p for p in (_clean_cell(row[TITLE_COLUMN]), _clean_cell(row[DESCRIPTION_COLUMN])) if p]
        return mask_pii("\n".join(parts)) if parts else None

    return tickets[[TITLE_COLUMN, DESCRIPTION_COLUMN]].apply(build, axis=1).astype(object)
