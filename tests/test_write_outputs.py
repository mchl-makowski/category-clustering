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


def test_judge_sheet_lists_every_check_with_verdict_and_reason(result, tmp_path):
    from problem_clustering import judge

    jr = judge(result, result.config, FakeBackend())
    xlsx = write_outputs(result, jr, tmp_path)["xlsx"]
    sheet = pd.read_excel(xlsx, sheet_name="Judge")
    assert list(sheet.columns) == ["check", "problem_id", "other_problem_id", "verdict", "reason"]
    assert len(sheet) == len(jr.checks)
    assert set(sheet["check"]) == {"mixed", "split"}
    assert sheet["reason"].notna().all()


def _metrics(xlsx):
    m = pd.read_excel(xlsx, sheet_name="Metrics")
    assert list(m.columns) == ["metric", "value"]
    return dict(zip(m["metric"], m["value"]))


def test_metrics_sheet_has_all_metrics_with_judge_usage_separate(result, tmp_path):
    from problem_clustering import judge

    fake = FakeBackend()
    fake.on("judge_mixed", {"mixed": False, "reason": "ok"})
    fake.on("judge_split", {"same_problem": True, "reason": "to samo"})
    jr = judge(result, result.config, fake)
    m = _metrics(write_outputs(result, jr, tmp_path)["xlsx"])
    sizes = result.catalogue.loc[result.catalogue["problem_id"] != "X-0000", "ticket_count"]
    assert m["method"] == "a"
    assert m["problem_count"] == len(sizes)
    assert m["one_ticket_share"] == pytest.approx((sizes == 1).mean())
    assert m["size_max"] == sizes.max()
    assert m["size_median"] == pytest.approx(sizes.median())
    assert m["empty_tickets"] == 2
    assert m["purity"] == 1.0
    assert m["split_rate"] == 1.0
    assert m["llm_calls"] == result.usage.llm_calls
    assert m["runtime_s"] == pytest.approx(result.usage.runtime_s)
    assert m["judge_llm_calls"] == fake.chat_calls
    for key in ["size_min", "size_mean", "size_p75", "prompt_tokens", "completion_tokens", "embedding_calls",
                "embedding_tokens", "judge_mixed_checks", "judge_split_pairs", "judge_prompt_tokens",
                "judge_completion_tokens", "judge_runtime_s"]:
        assert key in m
    assert not any("cost" in str(k) for k in m)


def test_metrics_without_judge_leave_judge_metrics_blank(result, tmp_path):
    xlsx = write_outputs(result, None, tmp_path)["xlsx"]
    m = _metrics(xlsx)
    assert m["problem_count"] > 0
    assert pd.isna(m["purity"]) and pd.isna(m["split_rate"])
    assert pd.read_excel(xlsx, sheet_name="Judge").empty


def test_log_file_has_config_progress_and_totals(result, tmp_path):
    txt = write_outputs(result, None, tmp_path)["txt"].read_text(encoding="utf-8")
    assert "distance_threshold" in txt
    assert "llm_calls=" in txt and "runtime_s=" in txt
