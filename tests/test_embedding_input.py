"""Embedding input is cut to backend.embed_max_chars (model token limit); LLM prompts keep the full text."""

import pytest

from problem_clustering import judge, load_config, load_tickets, run_method
from problem_clustering.backend import FakeBackend

CAP = 40


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def cfg(**over):
    return load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.6,
                              "backend.embed_max_chars": CAP, **over})


def test_default_cap_is_below_the_embedding_model_limit():
    assert 0 < load_config().backend.embed_max_chars <= 8191


@pytest.mark.parametrize("method", ["a", "b", "c"])
def test_every_embedding_input_is_cut_to_the_cap(tickets, method):
    fake = FakeBackend()
    run_method(tickets, method, cfg(), fake)
    assert fake.embed_inputs
    assert max(len(t) for t in fake.embed_inputs) == CAP
    assert any(tickets.loc[0, "Opis"].strip() in m["content"]
               for c in fake.chat_inputs for m in c["messages"] if m.get("content"))  # prompts uncut


def test_judge_embedding_input_is_cut_to_the_cap(tickets):
    result = run_method(tickets, "a", cfg(), FakeBackend())
    result.embeddings = result.embedding_rows = None
    fake = FakeBackend()
    judge(result, cfg(), fake)
    assert fake.embed_inputs and max(len(t) for t in fake.embed_inputs) == CAP
