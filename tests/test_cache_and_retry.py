import threading
import time

import pytest

from problem_clustering import load_config, load_tickets, run_method
from problem_clustering.backend import FakeBackend, TransientBackendError
from problem_clustering.backend.fake import DEFAULT_HANDLERS
from problem_clustering.concurrency import parallel_map


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def make_config(tmp_path, **overrides):
    base = {
        "backend.kind": "fake",
        "method_a.distance_threshold": 0.6,
        "runtime.cache_dir": str(tmp_path / "cache"),
        "runtime.backoff_initial_s": 0.0,
    }
    base.update(overrides)
    return load_config(None, base)


def test_second_identical_run_makes_no_backend_calls_and_same_outputs(tickets, tmp_path):
    config = make_config(tmp_path)
    first_fake, second_fake = FakeBackend(), FakeBackend()
    first = run_method(tickets, "a", config, first_fake)
    second = run_method(tickets, "a", config, second_fake)

    assert first_fake.embed_calls > 0 and first_fake.chat_calls > 0
    assert second_fake.embed_calls == 0
    assert second_fake.chat_calls == 0
    assert second.assignments.equals(first.assignments)
    assert second.catalogue.equals(first.catalogue)


@pytest.mark.parametrize("override", [
    {"backend.embedding_deployment": "other-embedding"},
    {"backend.chat_deployment": "other-chat"},
])
def test_changing_deployment_misses_the_cache(tickets, tmp_path, override):
    run_method(tickets, "a", make_config(tmp_path), FakeBackend())
    fake = FakeBackend()
    run_method(tickets, "a", make_config(tmp_path, **override), fake)
    if "backend.embedding_deployment" in override:
        assert fake.embed_calls > 0
    else:
        assert fake.chat_calls > 0


def test_changed_ticket_text_misses_only_for_that_ticket(tickets, tmp_path):
    config = make_config(tmp_path)
    run_method(tickets, "a", config, FakeBackend())
    edited = tickets.copy()
    edited.loc[0, "Opis"] = "Zupełnie nowy opis problemu z drukarką sieciową"
    fake = FakeBackend()
    run_method(edited, "a", config, fake)
    assert len(fake.embed_inputs) == 1
    assert "drukarką sieciową" in fake.embed_inputs[0]


def test_log_reports_cache_hits_vs_calls(tickets, tmp_path):
    config = make_config(tmp_path)
    run_method(tickets, "a", config, FakeBackend())
    result = run_method(tickets, "a", config, FakeBackend())
    line = next(entry for entry in result.log if "cache " in entry)
    assert "embed_calls=0" in line and "chat_calls=0" in line
    assert "embed_hits=18" in line  # 20 fixture Tickets, 2 empty


def test_transient_errors_are_retried_then_run_succeeds(tickets, tmp_path):
    fake = FakeBackend().fail_next(2)
    clean = run_method(tickets, "a", make_config(tmp_path / "a"), FakeBackend())
    result = run_method(tickets, "a", make_config(tmp_path / "b", **{"runtime.max_retries": 2}), fake)
    assert result.assignments.equals(clean.assignments)
    assert sum("retry" in entry for entry in result.log) == 2


def test_transient_errors_surface_after_max_retries(tickets, tmp_path):
    fake = FakeBackend().fail_next(3)
    with pytest.raises(TransientBackendError):
        run_method(tickets, "a", make_config(tmp_path, **{"runtime.max_retries": 2}), fake)


def _in_flight_recorder():
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def slow_naming(req):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return DEFAULT_HANDLERS["problem_naming"](req)

    return state, slow_naming


@pytest.mark.parametrize("limit", [1, 3])
def test_naming_calls_run_in_parallel_up_to_the_concurrency_limit(tickets, tmp_path, limit):
    state, slow_naming = _in_flight_recorder()
    fake = FakeBackend().on("problem_naming", slow_naming)
    parallel = run_method(tickets, "a", make_config(tmp_path, **{"runtime.max_concurrency": limit}), fake)
    assert fake.chat_calls > limit
    assert state["max"] == limit
    sequential = run_method(tickets, "a", make_config(tmp_path / "seq", **{"runtime.max_concurrency": 1}),
                            FakeBackend())
    assert parallel.catalogue.equals(sequential.catalogue)


def test_parallel_map_keeps_input_order_and_raises_first_error():
    assert parallel_map(lambda x: x * x, [3, 1, 2], max_workers=3) == [9, 1, 4]
    assert parallel_map(lambda x: x, [], max_workers=3) == []

    def boom(x):
        if x == 2:
            raise ValueError("two")
        return x

    with pytest.raises(ValueError, match="two"):
        parallel_map(boom, [1, 2, 3], max_workers=2)


def test_runtime_settings_load_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "runtime:\n  cache_dir: .cache/x\n  max_concurrency: 4\n  max_retries: 3\n"
        "  backoff_initial_s: 0.5\n  backoff_multiplier: 3\n  backoff_max_s: 10\n",
        encoding="utf-8",
    )
    rt = load_config(path).runtime
    assert (rt.cache_dir, rt.max_concurrency, rt.max_retries) == (".cache/x", 4, 3)
    assert (rt.backoff_initial_s, rt.backoff_multiplier, rt.backoff_max_s) == (0.5, 3, 10)


def test_non_transient_errors_are_not_retried(tickets, tmp_path):
    fake = FakeBackend().fail_next(1, ValueError("bad request"))
    with pytest.raises(ValueError):
        run_method(tickets, "a", make_config(tmp_path, **{"runtime.max_retries": 5}), fake)
    assert fake.embed_calls == 1
