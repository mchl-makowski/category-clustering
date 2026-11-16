"""Excel export loading: the Ticket data contract."""

from pathlib import Path

import pandas as pd

ID_COLUMN = "Numer I/SR"
TITLE_COLUMN = "Tytuł"
DESCRIPTION_COLUMN = "Opis"
COLUMNS = ["Typ", ID_COLUMN, "Zgłoszenie", TITLE_COLUMN, DESCRIPTION_COLUMN, "Status", "Grupa odpowiedzialna"]


def load_tickets(path: str | Path) -> pd.DataFrame:
    """Read the Service Desk export and check the 7 exact Polish column names.

    Rows keep file order and a 0..n-1 RangeIndex; all columns pass through unchanged.
    """
    df = pd.read_excel(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Export is missing required columns: {missing}; found {list(df.columns)}")
    return df.reset_index(drop=True)
