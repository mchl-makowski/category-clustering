import re

import pandas as pd

from problem_clustering import compare, judge, load_config, load_tickets, run_method, write_outputs
from problem_clustering.backend import FakeBackend


def _run(fixture_path, out_dir, threshold, with_judge=True):
    cfg = load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": threshold})
    result = run_method(load_tickets(fixture_path), "a", cfg, FakeBackend())
    jr = judge(result, cfg, FakeBackend()) if with_judge else None
    return write_outputs(result, jr, out_dir)["xlsx"]


def test_compare_two_runs_one_column_per_run(fixture_path, tmp_path):
    first = _run(fixture_path, tmp_path / "runs", 0.6)
    second = _run(fixture_path, tmp_path / "runs", 0.2, with_judge=False)
    out = compare([first, second], tmp_path / "cmp")
    assert re.fullmatch(r"\d{6}_\d{4}_comparison\.xlsx", out.name)
    table = pd.read_excel(out, sheet_name="Comparison", index_col=0)
    assert list(table.columns) == [first.stem, second.stem]
    for metric in ["problem_count", "one_ticket_share", "purity", "split_rate", "llm_calls", "runtime_s",
                   "judge_llm_calls"]:
        assert metric in table.index
    m1 = dict(zip(*pd.read_excel(first, sheet_name="Metrics").T.values))
    assert table.loc["problem_count", first.stem] == m1["problem_count"]
    assert table.loc["problem_count", first.stem] != table.loc["problem_count", second.stem]
    assert pd.isna(table.loc["purity", second.stem])
