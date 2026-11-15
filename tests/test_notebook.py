"""The comparison notebook runs top to bottom on the fixture with the Fake backend."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "compare.ipynb"
PUBLIC_API = {"load_config", "load_tickets", "run_method", "judge", "judge_safely", "write_outputs", "compare"}


def _code_cells():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]


@pytest.fixture
def executed(monkeypatch, tmp_path, capsys):
    """Execute every code cell in order; the parameter cell's OUT_DIR is pointed at tmp_path."""
    monkeypatch.chdir(NOTEBOOK.parent)
    shown: list = []
    ns: dict = {"display": lambda *a, **k: shown.extend(a)}
    params, *rest = _code_cells()
    exec(params, ns)
    ns["OUT_DIR"] = tmp_path
    for src in rest:
        exec(src, ns)
    return ns, shown, capsys.readouterr().out, tmp_path


def test_notebook_imports_only_the_public_api():
    imports = re.findall(r"^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$", "\n".join(_code_cells()), re.M)
    for module, names in imports:
        if module and module.startswith("problem_clustering"):
            assert module in {"problem_clustering", "problem_clustering.backend"}, module
            if module == "problem_clustering":
                assert {n.strip() for n in names.split(",")} <= PUBLIC_API


def test_runs_all_three_methods_judges_and_writes_workbooks(executed):
    ns, _, _, out = executed
    assert set(ns["workbooks"]) == {"a", "b", "c"}
    for method, path in ns["workbooks"].items():
        assert re.fullmatch(rf"\d{{6}}_\d{{4}}_{method}\.xlsx", Path(path).name)
        assert len(pd.read_excel(path, sheet_name="Judge")) > 0  # Judge ran
    assert len(list(out.glob("*_comparison.xlsx"))) == 1


def test_comparison_table_has_one_column_per_method(executed):
    ns, shown, _, _ = executed
    table = ns["comparison"]
    assert len(table.columns) == 3
    assert {"problem_count", "purity", "split_rate", "llm_calls"} <= set(table.index), table.index.tolist()
    assert any(obj is table for obj in shown)


def test_problem_browser_shows_name_description_and_tickets_per_method(executed):
    ns, shown, printed, _ = executed
    text = printed + "\n".join(o.to_string() for o in shown if isinstance(o, pd.DataFrame))
    for path in ns["workbooks"].values():
        catalogue = pd.read_excel(path, sheet_name="Problem Catalogue")
        tickets = pd.read_excel(path, sheet_name="Tickets")
        for row in catalogue.itertuples():
            assert row.problem_name in text and row.description in text
        for title in tickets["Tytuł"].dropna():
            assert title in text


def test_loads_existing_workbooks_instead_of_running(executed, monkeypatch, tmp_path_factory):
    ns, _, _, _ = executed
    loaded_out = tmp_path_factory.mktemp("loaded")
    params, *rest = _code_cells()
    ns2: dict = {"display": lambda *a, **k: None}
    exec(params, ns2)
    ns2["OUT_DIR"] = loaded_out
    ns2["WORKBOOKS"] = dict(ns["workbooks"])

    def boom(*a, **k):
        raise AssertionError("run_method called although every workbook was given")

    ns2["run_method"] = ns2["judge_safely"] = boom  # imports live in the parameter cell
    for src in rest:
        exec(src, ns2)
    assert ns2["workbooks"] == ns["workbooks"]
    assert len(ns2["comparison"].columns) == 3
