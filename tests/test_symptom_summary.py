import pandas as pd
import pytest

from problem_clustering import load_config, load_tickets, run_method
from problem_clustering.backend import FakeBackend
from problem_clustering.prompts import GRANULARITY_PARAGRAPHS

EMPTY_ROWS = {17, 18}
VPN_ROWS = [0, 1, 2, 3, 19]
PRINTER_ROWS = [7, 8, 9]
SUMMARY_SCHEMA = "symptom_summary"


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def cfg(**over):
    base = {"backend.kind": "fake", "method_a.distance_threshold": 0.6}
    base.update(over)
    return load_config(None, base)


def summary_calls(fake):
    return [c for c in fake.chat_inputs
            if ((c["response_format"] or {}).get("json_schema") or {}).get("name") == SUMMARY_SCHEMA]


def test_enabled_groups_on_symptom_summary_not_raw_text(tickets):
    fake = FakeBackend()
    # every non-empty Ticket summarised to the same line -> all land in one Problem
    fake.on(SUMMARY_SCHEMA, {"summary": "drukarka nie drukuje w biurze"})
    result = run_method(tickets, "a", cfg(**{"symptom_summary.enabled": True}), fake)
    n_non_empty = len(tickets) - len(EMPTY_ROWS)
    assert fake.embed_inputs == ["drukarka nie drukuje w biurze"] * n_non_empty
    a = result.assignments.drop(index=list(EMPTY_ROWS))
    assert set(a.loc[VPN_ROWS + PRINTER_ROWS, "problem_id"]) == {"A-0001"}
    assert a["problem_id"].nunique() == 1


def test_disabled_by_default_makes_no_summary_calls(tickets):
    fake = FakeBackend()
    run_method(tickets, "a", cfg(), fake)
    assert summary_calls(fake) == []
    assert any("\n" in t for t in fake.embed_inputs)  # raw Tytuł + newline + Opis was grouped


def test_summary_prompt_has_masked_title_and_description_and_granularity_paragraph(tickets):
    fake = FakeBackend()
    run_method(tickets, "a", cfg(**{"symptom_summary.enabled": True, "granularity": "fine"}), fake)
    calls = summary_calls(fake)
    system, user = calls[0]["messages"][0]["content"], calls[0]["messages"][1]["content"]
    assert GRANULARITY_PARAGRAPHS["fine"] in system
    assert tickets.loc[0, "Tytuł"].strip() in user and tickets.loc[0, "Opis"].strip() in user
    sent = "\n".join(m["content"] for c in calls for m in c["messages"])
    assert "jan.kowalski@firma.pl" not in sent and "[EMAIL]" in sent
    for col in ["Typ", "Numer I/SR", "Zgłoszenie", "Status", "Grupa odpowiedzialna"]:
        for value in tickets[col].dropna().astype(str).unique():
            assert value not in sent, f"{col} value {value!r} leaked to summary prompt"


def test_empty_tickets_skip_summary_and_all_model_calls(tickets):
    fake = FakeBackend()
    result = run_method(tickets, "a", cfg(**{"symptom_summary.enabled": True}), fake)
    n_non_empty = len(tickets) - len(EMPTY_ROWS)
    assert len(summary_calls(fake)) == n_non_empty
    assert len(fake.embed_inputs) == n_non_empty
    assert set(result.assignments.loc[list(EMPTY_ROWS), "problem_id"]) == {"X-0000"}


def test_output_keeps_original_text_and_summary_optionally_logged(tickets):
    line = "vpn rozłącza się po aktualizacji klienta"
    quiet = run_method(tickets, "a", cfg(**{"symptom_summary.enabled": True}),
                       FakeBackend().on(SUMMARY_SCHEMA, {"summary": line}))
    pd.testing.assert_frame_equal(quiet.tickets, tickets)
    assert not any(line in entry for entry in quiet.log)

    loud = run_method(tickets, "a", cfg(**{"symptom_summary.enabled": True, "symptom_summary.log_summaries": True}),
                      FakeBackend().on(SUMMARY_SCHEMA, {"summary": line}))
    assert sum(line in entry for entry in loud.log) == len(tickets) - len(EMPTY_ROWS)


def test_summary_calls_counted_in_usage(tickets):
    fake = FakeBackend()
    result = run_method(tickets, "a", cfg(**{"symptom_summary.enabled": True}), fake)
    assert result.usage.llm_calls == fake.chat_calls == len(summary_calls(fake)) + len(result.catalogue) - 1
