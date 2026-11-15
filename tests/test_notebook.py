import json
from pathlib import Path

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "explore.ipynb"


def test_notebook_runs_method_a_on_fixture(monkeypatch):
    monkeypatch.chdir(NOTEBOOK.parent)
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    ns: dict = {"display": lambda *a, **k: None}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), ns)
    assert ns["result"].assignments["problem_id"].notna().all()
    assert len(ns["result"].catalogue) > 1
