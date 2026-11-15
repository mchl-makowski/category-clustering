import pytest

from problem_clustering import judge, load_config, load_tickets, run_method
from problem_clustering.backend import FakeBackend


@pytest.fixture
def config():
    return load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.6})


@pytest.fixture
def result(fixture_path, config):
    return run_method(load_tickets(fixture_path), "a", config, FakeBackend())


def _sizes(result):
    cat = result.catalogue
    return dict(zip(cat["problem_id"], cat["ticket_count"]))


def test_mixed_check_runs_on_every_multi_ticket_problem_only(result, config):
    jr = judge(result, config, FakeBackend())
    mixed = jr.checks[jr.checks["check"] == "mixed"]
    sizes = _sizes(result)
    expected = {pid for pid, n in sizes.items() if n > 1 and pid != "X-0000"}
    assert expected  # fixture has multi-Ticket Problems
    assert set(mixed["problem_id"]) == expected
    assert len(mixed) == len(expected)
    assert mixed["verdict"].isin(["mixed", "one_problem"]).all()
    assert mixed["reason"].astype(str).str.len().gt(0).all()


def test_split_pairs_are_nearest_neighbours_deduplicated(result, config):
    jr = judge(result, config, FakeBackend())
    split = jr.checks[jr.checks["check"] == "split"]
    n_problems = len([p for p in result.catalogue["problem_id"] if p != "X-0000"])
    assert n_problems > 4
    pairs = [frozenset(p) for p in zip(split["problem_id"], split["other_problem_id"])]
    assert all(len(p) == 2 for p in pairs)
    assert len(pairs) == len(set(pairs))  # deduplicated
    assert "X-0000" not in set(split["problem_id"]) | set(split["other_problem_id"])
    # every Problem appears with its 3 neighbours: at least 3 pairs each, at most n*3 pairs overall
    for pid in result.catalogue["problem_id"]:
        if pid != "X-0000":
            assert sum(pid in p for p in pairs) >= 3
    assert len(pairs) <= n_problems * 3
    assert split["verdict"].isin(["split", "distinct"]).all()


def test_max_problems_limits_judged_problems_to_largest(result, fixture_path):
    cfg = load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.6, "judge.max_problems": 2})
    jr = judge(result, cfg, FakeBackend())
    largest = list(result.catalogue["problem_id"][:2])
    judged = set(jr.checks["problem_id"]) | set(jr.checks["other_problem_id"].dropna())
    assert judged <= set(largest)
    split = jr.checks[jr.checks["check"] == "split"]
    assert len(split) == 1


def test_judge_uses_its_own_deployment_and_granularity(result):
    cfg = load_config(None, {"backend.kind": "fake", "backend.chat_deployment": "run-model",
                             "judge.deployment": "judge-model", "granularity": "fine"})
    fake = FakeBackend()
    judge(result, cfg, fake)
    assert fake.chat_inputs
    assert {c["model"] for c in fake.chat_inputs} == {"judge-model"}
    assert all("SZCZEGÓŁOWY" in c["messages"][0]["content"] for c in fake.chat_inputs)


def test_judge_usage_is_its_own(result, config):
    method_usage = result.usage.as_dict()
    fake = FakeBackend()
    jr = judge(result, config, fake)
    assert jr.usage.llm_calls == fake.chat_calls == len(jr.checks)
    assert jr.usage.prompt_tokens > 0
    assert result.usage.as_dict() == method_usage  # Method usage untouched
    assert fake.embed_calls == 0  # run embeddings reused


def test_judge_embeds_when_run_has_no_embeddings(result, config):
    result.embeddings = None
    result.embedding_rows = None
    fake = FakeBackend()
    jr = judge(result, config, fake)
    assert fake.embed_calls > 0
    assert jr.usage.embedding_calls == fake.embed_calls
    assert len(jr.checks[jr.checks["check"] == "split"]) > 0


def test_judge_sends_only_masked_title_and_description(result, config, fixture_path):
    fake = FakeBackend()
    judge(result, config, fake)
    sent = "\n".join(fake.all_input_texts())
    assert "jan.kowalski@firma.pl" not in sent
    tickets = load_tickets(fixture_path)
    for value in tickets["Numer I/SR"]:
        assert str(value) not in sent


def test_scripted_verdicts_drive_purity_and_split_rate(result, config):
    fake = FakeBackend()
    fake.on("judge_mixed", {"mixed": True, "reason": "dwa objawy"})
    fake.on("judge_split", {"same_problem": False, "reason": "różne"})
    jr = judge(result, config, fake)
    assert jr.purity == 0.0
    assert jr.split_rate == 0.0
    assert set(jr.checks.loc[jr.checks["check"] == "mixed", "reason"]) == {"dwa objawy"}


def test_judge_retries_transient_errors(result):
    cfg = load_config(None, {"backend.kind": "fake", "runtime.backoff_initial_s": 0.0})
    clean = judge(result, cfg, FakeBackend())
    jr = judge(result, cfg, FakeBackend().fail_next(2))
    assert jr.checks.equals(clean.checks)


def test_second_identical_judge_is_served_from_the_disk_cache(result, tmp_path):
    cfg = load_config(None, {"backend.kind": "fake", "runtime.cache_dir": str(tmp_path / "cache")})
    first = judge(result, cfg, FakeBackend())
    fake = FakeBackend()
    second = judge(result, cfg, fake)
    assert fake.chat_calls == 0
    assert second.checks.equals(first.checks)
    assert second.usage.llm_calls == first.usage.llm_calls  # usage counted outside the cache
