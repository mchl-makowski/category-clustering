import pandas as pd
import pytest

from problem_clustering import load_tickets

COLUMNS = ["Typ", "Numer I/SR", "Zgłoszenie", "Tytuł", "Opis", "Status", "Grupa odpowiedzialna"]


def test_loads_export_with_exact_polish_columns(fixture_path):
    df = load_tickets(fixture_path)
    assert list(df.columns) == COLUMNS
    assert len(df) == 20
    assert df["Numer I/SR"].iloc[0] == "INC000001"


def test_rejects_export_missing_a_column(tmp_path, fixture_path):
    bad = pd.read_excel(fixture_path).rename(columns={"Tytuł": "Tytul"})
    path = tmp_path / "bad.xlsx"
    bad.to_excel(path, index=False)
    with pytest.raises(ValueError, match="Tytuł"):
        load_tickets(path)
