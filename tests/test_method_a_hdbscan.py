import re

import pytest

from problem_clustering import load_config, load_tickets, run_method
from problem_clustering.backend import FakeBackend
from problem_clustering.cli import main

EMPTY_ROWS = {17, 18}


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def hdbscan_config(**params):
    overrides = {"backend.kind": "fake", "method_a.algorithm": "hdbscan"}
    overrides.update({f"method_a.{k}": v for k, v in params.items()})
    return load_config(None, overrides)


def test_hdbscan_assigns_every_ticket_a_problem(tickets):
    a = run_method(tickets, "a", hdbscan_config(min_cluster_size=2), FakeBackend()).assignments
    assert len(a) == len(tickets)
    assert a["problem_id"].notna().all()
    for i, pid in a["problem_id"].items():
        assert pid == "X-0000" if i in EMPTY_ROWS else re.fullmatch(r"A-\d{4}", pid)


def test_hdbscan_noise_tickets_become_one_ticket_problems_with_confidence_one(tickets):
    a = run_method(tickets, "a", hdbscan_config(min_cluster_size=3), FakeBackend()).assignments
    a = a[a["problem_id"] != "X-0000"]
    sizes = a["problem_id"].map(a["problem_id"].value_counts())
    assert not ((sizes > 1) & (sizes < 3)).any()  # anything below min_cluster_size is noise
    noise = a[sizes == 1]
    assert len(noise) >= 1
    assert (noise["confidence"] == 1.0).all()


def test_hdbscan_params_load_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("method_a:\n  algorithm: hdbscan\n  min_cluster_size: 4\n  min_samples: 2\n")
    cfg = load_config(path).method_a
    assert (cfg.algorithm, cfg.min_cluster_size, cfg.min_samples) == ("hdbscan", 4, 2)


def test_hdbscan_runs_from_cli_with_param_overrides(tmp_path, fixture_path):
    code = main([
        "run", "--method", "a", "--input", str(fixture_path), "--out-dir", str(tmp_path),
        "--set", "backend.kind=fake", "--set", "method_a.algorithm=hdbscan",
        "--set", "method_a.min_cluster_size=3", "--set", "method_a.min_samples=1", "--no-judge",
    ])
    assert code == 0
    log = next(tmp_path.glob("*.txt")).read_text(encoding="utf-8")
    assert "hdbscan" in log and "min_cluster_size=3" in log and "min_samples=1" in log
