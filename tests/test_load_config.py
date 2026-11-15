import pytest
from pydantic import ValidationError

from problem_clustering import load_config


def test_defaults_without_yaml():
    cfg = load_config()
    assert cfg.granularity == "medium"
    assert cfg.method_a.algorithm == "agglomerative"
    assert 0 < cfg.method_a.distance_threshold < 2


def test_yaml_values_and_dotted_cli_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("granularity: coarse\nmethod_a:\n  distance_threshold: 0.4\n", encoding="utf-8")

    cfg = load_config(path)
    assert cfg.granularity == "coarse"
    assert cfg.method_a.distance_threshold == 0.4

    cfg = load_config(path, ["granularity=fine", "method_a.distance_threshold=0.25"])
    assert cfg.granularity == "fine"
    assert cfg.method_a.distance_threshold == 0.25


def test_overrides_may_be_a_dict():
    cfg = load_config(None, {"backend.kind": "fake"})
    assert cfg.backend.kind == "fake"


def test_invalid_values_and_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        load_config(None, ["granularity=extreme"])
    with pytest.raises(ValidationError):
        load_config(None, ["method_a.no_such_knob=1"])
