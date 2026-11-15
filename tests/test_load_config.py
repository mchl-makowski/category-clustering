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


def test_deployment_names_from_env_win_over_yaml(tmp_path, monkeypatch):
    path = tmp_path / "cfg.yaml"
    path.write_text("backend:\n  embedding_deployment: yaml-emb\n  chat_deployment: yaml-chat\n"
                    "judge:\n  deployment: yaml-judge\n", encoding="utf-8")
    assert load_config(path).backend.chat_deployment == "yaml-chat"

    monkeypatch.setenv("FOUNDRY_EMBEDDING_DEPLOYMENT", "env-emb")
    monkeypatch.setenv("FOUNDRY_CHAT_DEPLOYMENT", "env-chat")
    monkeypatch.setenv("FOUNDRY_JUDGE_DEPLOYMENT", "env-judge")
    cfg = load_config(path)
    assert (cfg.backend.embedding_deployment, cfg.backend.chat_deployment, cfg.judge.deployment) == (
        "env-emb", "env-chat", "env-judge")


def test_unset_deployment_env_keeps_yaml_and_cli_set_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FOUNDRY_CHAT_DEPLOYMENT", "env-chat")
    cfg = load_config(None, ["backend.chat_deployment=cli-chat"])
    assert cfg.backend.chat_deployment == "cli-chat"  # an explicit --set is the most specific
    assert cfg.backend.embedding_deployment == "text-embedding-3-large"
    assert cfg.judge.deployment == "gpt-4o"
