import re

import pandas as pd
import pytest

from problem_clustering import load_config, load_tickets, run_method, write_outputs
from problem_clustering.backend import FakeBackend


@pytest.fixture
def result(fixture_path):
    cfg = load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.6})
    return run_method(load_tickets(fixture_path), "a", cfg, FakeBackend())


def test_writes_timestamped_workbook_and_log(result, tmp_path):
    paths = write_outputs(result, None, tmp_path)
    assert re.fullmatch(r"\d{6}_\d{4}_a\.xlsx", paths["xlsx"].name)
    assert re.fullmatch(r"\d{6}_\d{4}_a\.txt", paths["txt"].name)
    assert paths["xlsx"].stem == paths["txt"].stem
    assert pd.ExcelFile(paths["xlsx"]).sheet_names == ["Tickets", "Problem Catalogue", "Judge", "Metrics"]


def test_tickets_sheet_keeps_original_columns_unchanged(result, tmp_path, fixture_path):
    xlsx = write_outputs(result, None, tmp_path)["xlsx"]
    out = pd.read_excel(xlsx, sheet_name="Tickets")
    original = pd.read_excel(fixture_path)
    assert list(out.columns) == list(original.columns) + [
        "problem_id", "problem_name", "confidence", "confidence_level"]
    pd.testing.assert_frame_equal(out[original.columns], original)
    assert "jan.kowalski@firma.pl" in out.loc[1, "Opis"]
    assert out["problem_id"].str.fullmatch(r"(A-\d{4}|X-0000)").all()


def test_problem_catalogue_sheet(result, tmp_path):
    xlsx = write_outputs(result, None, tmp_path)["xlsx"]
    cat = pd.read_excel(xlsx, sheet_name="Problem Catalogue")
    assert list(cat.columns) == ["problem_id", "problem_name", "description", "ticket_count", "example_tickets"]
    assert cat["ticket_count"].sum() == 20
    assert cat["example_tickets"].str.contains("INC").all()


def test_log_file_has_config_progress_and_totals(result, tmp_path):
    txt = write_outputs(result, None, tmp_path)["txt"].read_text(encoding="utf-8")
    assert "distance_threshold" in txt
    assert "llm_calls=" in txt and "runtime_s=" in txt
