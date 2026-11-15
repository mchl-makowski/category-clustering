"""Azure content filter: a filtered reply never aborts a run; every LLM step falls back and logs it."""

import pytest

from problem_clustering import judge, load_config, load_tickets, run_method
from problem_clustering.backend import ContentFilteredError, FakeBackend, content_filtered, json_schema_format

EMPTY_ROWS = [17, 18]


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def cfg(**over):
    return load_config(None, {"backend.kind": "fake", "method_a.distance_threshold": 0.6, **over})


def filtered_log_lines(result):
    return [line for line in result.log if "content filter" in line]


def test_fake_content_filtered_reply_raises_content_filtered_error():
    fake = FakeBackend().on("problem_naming", content_filtered())
    with pytest.raises(ContentFilteredError):
        fake.chat([{"role": "user", "content": "x"}], response_format=json_schema_format("problem_naming", {}))


def test_filtered_symptom_summary_falls_back_to_raw_masked_text(tickets):
    fake = FakeBackend().on("symptom_summary", content_filtered())
    result = run_method(tickets, "a", cfg(**{"symptom_summary.enabled": True}), fake)
    plain = run_method(tickets, "a", cfg(), FakeBackend())
    assert result.assignments.equals(plain.assignments)
    assert "jan.kowalski@firma.pl" not in "\n".join(fake.embed_inputs)
    assert len(filtered_log_lines(result)) == len(tickets) - len(EMPTY_ROWS)


def test_filtered_naming_uses_first_ticket_title(tickets):
    fake = FakeBackend().on("problem_naming", content_filtered())
    result = run_method(tickets, "a", cfg(), fake)
    cat = result.catalogue.set_index("problem_id")
    first_row = result.assignments.index[result.assignments["problem_id"] == "A-0001"][0]
    assert cat.loc["A-0001", "problem_name"] == tickets.loc[first_row, "Tytuł"].strip()
    assert cat["description"].astype(str).str.len().gt(0).all()
    assert filtered_log_lines(result)


def test_filtered_method_b_verification_keeps_the_cluster(tickets):
    fake = FakeBackend().on("cluster_verification", content_filtered())
    result = run_method(tickets, "b", cfg(), fake)
    accepted = run_method(tickets, "b", cfg(), FakeBackend())  # Fake default: every cluster is one Problem
    assert result.assignments.equals(accepted.assignments)
    assert filtered_log_lines(result)


def test_filtered_method_b_merge_check_does_not_merge(tickets):
    over = {"method_b.coarse_threshold": 0.3}
    fake = FakeBackend().on("problem_merge", content_filtered())
    result = run_method(tickets, "b", cfg(**over), fake)
    rejected = run_method(tickets, "b", cfg(**over), FakeBackend())  # Fake default: every merge rejected
    assert result.assignments.equals(rejected.assignments)
    assert filtered_log_lines(result)


def test_filtered_method_c_agent_gives_each_ticket_a_new_guessed_problem(tickets):
    fake = FakeBackend().on(lambda req: bool(req.tools), content_filtered())
    result = run_method(tickets, "c", cfg(), fake)
    non_empty = result.assignments.drop(index=EMPTY_ROWS)
    assert non_empty["problem_id"].nunique() == len(non_empty)
    assert set(non_empty["confidence_level"]) == {"zgadywane"}
    assert filtered_log_lines(result)


def test_filtered_method_c_consolidation_merges_nothing(tickets):
    over = {"method_c.consolidate_threshold": 1.5}  # every Problem is a merge candidate
    fake = FakeBackend().on("method_c_consolidation", content_filtered())
    result = run_method(tickets, "c", cfg(**over), fake)
    plain = run_method(tickets, "c", cfg(**over), FakeBackend())  # Fake default: no merges
    assert result.assignments.equals(plain.assignments)
    assert filtered_log_lines(result)


@pytest.fixture
def run_a(tickets):
    return run_method(tickets, "a", cfg(), FakeBackend())


def test_filtered_judge_checks_are_errors_excluded_from_the_rates(run_a):
    fake = FakeBackend().on("judge_mixed", content_filtered()).on("judge_split", content_filtered())
    jr = judge(run_a, cfg(), fake)
    assert len(jr.checks) and set(jr.checks["verdict"]) == {"error"}
    assert jr.checks["reason"].str.contains("content filter").all()
    assert jr.purity is None and jr.split_rate is None


def test_filtered_judge_check_does_not_count_in_the_denominator(run_a):
    calls = {"n": 0}

    def first_filtered(req):
        calls["n"] += 1
        return content_filtered() if calls["n"] == 1 else {"mixed": False, "reason": "ok"}

    jr = judge(run_a, cfg(**{"runtime.max_concurrency": 1}), FakeBackend().on("judge_mixed", first_filtered))
    mixed = jr.mixed_checks
    assert (mixed["verdict"] == "error").sum() == 1
    assert jr.purity == 1.0
