import re

import pandas as pd
import pytest

from problem_clustering import load_config, load_tickets, run_method
from problem_clustering.backend import FakeBackend

EMPTY_ROWS = {17, 18}  # fixture rows with blank Tytuł and Opis
VPN_ROWS = [0, 1, 2, 3, 19]
PRINTER_ROWS = [7, 8, 9]


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


@pytest.fixture
def config():
    return load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.6})


def test_every_ticket_assigned_once_with_method_prefix(tickets, config):
    result = run_method(tickets, "a", config, FakeBackend())
    a = result.assignments
    assert len(a) == len(tickets)
    assert list(a.index) == list(tickets.index)
    for i, pid in a["problem_id"].items():
        if i in EMPTY_ROWS:
            assert pid == "X-0000"
            assert a.loc[i, "problem_name"] == "Brak opisu"
        else:
            assert re.fullmatch(r"A-\d{4}", pid)
    assert a["confidence_level"].isna().all()
    assert a["confidence"].between(0, 1.0 + 1e-9).all()


def test_catalogue_lists_every_problem_with_name_description_count_examples(tickets, config):
    result = run_method(tickets, "a", config, FakeBackend())
    cat = result.catalogue
    counts = result.assignments["problem_id"].value_counts()
    assert set(cat["problem_id"]) == set(counts.index)
    for _, row in cat.iterrows():
        assert row["ticket_count"] == counts[row["problem_id"]]
        assert row["problem_name"]
        assert row["description"]
        assert row["example_tickets"]
    assert cat.set_index("problem_id").loc["X-0000", "problem_name"] == "Brak opisu"


def test_near_duplicates_share_a_problem_and_distinct_problems_differ(tickets, config):
    a = run_method(tickets, "a", config, FakeBackend()).assignments
    vpn = set(a.loc[VPN_ROWS, "problem_id"])
    printer = set(a.loc[PRINTER_ROWS, "problem_id"])
    assert len(vpn) == 1 and len(printer) == 1
    assert vpn != printer


def test_one_ticket_problems_have_confidence_one(tickets, config):
    a = run_method(tickets, "a", config, FakeBackend()).assignments
    sizes = a["problem_id"].map(a["problem_id"].value_counts())
    singles = a[(sizes == 1) & (a["problem_id"] != "X-0000")]
    assert len(singles) >= 1
    assert (singles["confidence"] == 1.0).all()


def test_tighter_distance_threshold_gives_more_problems(tickets):
    loose = load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.9})
    tight = load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.1})
    n_loose = run_method(tickets, "a", loose, FakeBackend()).assignments["problem_id"].nunique()
    n_tight = run_method(tickets, "a", tight, FakeBackend()).assignments["problem_id"].nunique()
    assert n_tight > n_loose


def test_empty_tickets_trigger_no_model_call(tickets, config):
    fake = FakeBackend()
    run_method(tickets, "a", config, fake)
    assert len(fake.embed_inputs) == len(tickets) - len(EMPTY_ROWS)
    assert all(t.strip() for t in fake.embed_inputs)


def test_only_title_and_description_reach_the_backend(tickets, config):
    fake = FakeBackend()
    run_method(tickets, "a", config, fake)
    sent = "\n".join(fake.all_input_texts())
    for col in ["Typ", "Numer I/SR", "Zgłoszenie", "Status", "Grupa odpowiedzialna"]:
        for value in tickets[col].dropna().astype(str).unique():
            assert value not in sent, f"{col} value {value!r} leaked to model input"


def test_pii_is_masked_in_model_input_but_output_keeps_original(tickets, config):
    fake = FakeBackend()
    result = run_method(tickets, "a", config, fake)
    sent = "\n".join(fake.all_input_texts())
    for pii in ["jan.kowalski@firma.pl", "anna.nowak@firma.pl", "601 234 567", "601-234-568",
                "22 123 45 67", "85010112345"]:
        assert pii not in sent
    assert "[EMAIL]" in sent and "[TELEFON]" in sent and "[PESEL]" in sent
    assert "jan.kowalski@firma.pl" in result.tickets.loc[1, "Opis"]
    pd.testing.assert_frame_equal(result.tickets, tickets)


def test_granularity_level_sets_problem_definition_in_naming_prompt(tickets):
    prompts = {}
    for level in ["coarse", "fine"]:
        cfg = load_config(None, {"backend.kind": "fake", "granularity": level,
                                 "method_a.distance_threshold": 0.6})
        fake = FakeBackend()
        run_method(tickets, "a", cfg, fake)
        assert fake.chat_calls > 0
        prompts[level] = fake.chat_inputs[0]["messages"][0]["content"]
    assert prompts["coarse"] != prompts["fine"]


def test_usage_counters_recorded_in_result_and_log(tickets, config):
    fake = FakeBackend()
    result = run_method(tickets, "a", config, fake)
    u = result.usage
    assert u.embedding_calls == fake.embed_calls >= 1
    assert u.llm_calls == fake.chat_calls == len(result.catalogue) - 1  # all but X-0000 named
    assert u.prompt_tokens > 0 and u.completion_tokens > 0
    assert u.runtime_s >= 0
    log = "\n".join(result.log)
    assert "llm_calls" in log and "embedding_calls" in log


def test_unknown_method_is_rejected(tickets, config):
    with pytest.raises(ValueError, match="method"):
        run_method(tickets, "z", config, FakeBackend())
