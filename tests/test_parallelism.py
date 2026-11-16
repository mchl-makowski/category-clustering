"""Independent model calls run in parallel, bounded by runtime.max_concurrency; results match a sequential run."""

import threading
import time

import pytest

from problem_clustering import judge, load_config, load_tickets, run_method
from problem_clustering.backend import FakeBackend
from problem_clustering.backend.fake import DEFAULT_HANDLERS


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def make_config(limit, **overrides):
    return load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.6,
                              "runtime.max_concurrency": limit, **overrides})


def slow(schema):
    """Default reply for `schema`, slowed down, recording the peak number of calls in flight."""
    state = {"now": 0, "max": 0, "calls": 0}
    lock = threading.Lock()

    def reply(req):
        with lock:
            state["now"] += 1
            state["calls"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return DEFAULT_HANDLERS[schema](req)

    return state, reply


@pytest.mark.parametrize("limit", [1, 3])
def test_symptom_summaries_run_in_parallel_up_to_the_limit(tickets, limit):
    state, reply = slow("symptom_summary")
    over = {"symptom_summary.enabled": True}
    parallel = run_method(tickets, "a", make_config(limit, **over), FakeBackend().on("symptom_summary", reply))
    assert state["calls"] > limit and state["max"] == limit
    sequential = run_method(tickets, "a", make_config(1, **over), FakeBackend())
    assert parallel.assignments.equals(sequential.assignments)


@pytest.mark.parametrize("limit", [1, 3])
def test_method_b_verification_runs_in_parallel_up_to_the_limit(tickets, limit):
    state, reply = slow("cluster_verification")
    over = {"method_b.coarse_threshold": 0.8}
    parallel = run_method(tickets, "b", make_config(limit, **over), FakeBackend().on("cluster_verification", reply))
    assert state["calls"] > limit and state["max"] == limit
    sequential = run_method(tickets, "b", make_config(1, **over), FakeBackend())
    assert parallel.assignments.equals(sequential.assignments)


@pytest.mark.parametrize("schema", ["judge_mixed", "judge_split"])
@pytest.mark.parametrize("limit", [1, 3])
def test_judge_checks_run_in_parallel_up_to_the_limit(tickets, schema, limit):
    result = run_method(tickets, "a", make_config(1), FakeBackend())
    state, reply = slow(schema)
    parallel = judge(result, make_config(limit), FakeBackend().on(schema, reply))
    assert state["calls"] > limit and state["max"] == limit
    sequential = judge(result, make_config(1), FakeBackend())
    assert parallel.checks.equals(sequential.checks)


class SlowEmbedFake(FakeBackend):
    def __init__(self):
        super().__init__()
        self.in_flight = self.peak = 0
        self._count_lock = threading.Lock()

    def embed(self, texts):
        with self._count_lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(0.02)
        try:
            return super().embed(texts)
        finally:
            with self._count_lock:
                self.in_flight -= 1


def test_embedding_batches_run_in_parallel_and_keep_order(tickets):
    fake = SlowEmbedFake()
    parallel = run_method(tickets, "a", make_config(3, **{"backend.embed_batch_size": 2}), fake)
    assert fake.embed_calls > 3 and fake.peak == 3
    sequential = run_method(tickets, "a", make_config(1), FakeBackend())
    assert parallel.assignments.equals(sequential.assignments)
