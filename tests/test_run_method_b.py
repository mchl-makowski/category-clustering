import re

import pytest

from problem_clustering import load_config, load_tickets, run_method
from problem_clustering.backend import FakeBackend

EMPTY_ROWS = {17, 18}
VPN_ROWS = [0, 1, 2, 3, 19]
PRINTER_ROWS = [7, 8, 9]


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def cfg(**overrides):
    return load_config(None, {"backend.kind": "fake", **{f"method_b.{k}": v for k, v in overrides.items()}})


def test_output_format_matches_method_a_with_b_prefix(tickets):
    a = run_method(tickets, "a", load_config(None, {"backend.kind": "fake"}), FakeBackend())
    b = run_method(tickets, "b", cfg(), FakeBackend())
    assert list(b.assignments.columns) == list(a.assignments.columns)
    assert list(b.catalogue.columns) == list(a.catalogue.columns)
    assert list(b.assignments.index) == list(tickets.index)
    for i, pid in b.assignments["problem_id"].items():
        assert pid == "X-0000" if i in EMPTY_ROWS else re.fullmatch(r"B-\d{4}", pid)
    assert b.assignments["confidence_level"].isna().all()
    assert b.assignments["confidence"].between(0, 1.0 + 1e-9).all()
    counts = b.assignments["problem_id"].value_counts()
    assert dict(zip(b.catalogue["problem_id"], b.catalogue["ticket_count"])) == counts.to_dict()
    assert b.catalogue["problem_name"].all() and b.catalogue["description"].all()


TOPICS = {"vpn": ["vpn"], "printer": ["drukar", "drukuje", "wydruk"], "outlook": ["outlook", "kalendarz"],
          "sap": ["sap"], "teams": ["teams", "kamer"]}


def topics_of(text):
    t = text.lower()
    return {name for name, keys in TOPICS.items() if any(k in t for k in keys)}


def bullets(text):
    return [line[2:] for line in text.splitlines() if line.startswith("- ")]


def verify_by_topic(req):
    found = set().union(*(topics_of(t) for t in bullets(req.last_user)))
    return {"one_problem": len(found) <= 1, "reason": "tematy: " + ",".join(sorted(found))}


def test_cluster_flagged_mixed_gets_split(tickets):
    config = cfg(coarse_threshold=1.01, split_step=0.41)  # everything starts in one coarse cluster
    stable = run_method(tickets, "b", config, FakeBackend())
    assert stable.assignments.loc[[*VPN_ROWS, *PRINTER_ROWS], "problem_id"].nunique() == 1

    fake = FakeBackend().on("cluster_verification", verify_by_topic)
    a = run_method(tickets, "b", config, fake).assignments
    vpn, printer = set(a.loc[VPN_ROWS, "problem_id"]), set(a.loc[PRINTER_ROWS, "problem_id"])
    assert len(vpn) == 1 and len(printer) == 1 and vpn != printer


def merge_if_both_vpn(req):
    group_a, _, group_b = req.last_user.partition("Grupa B:")
    same = all(topics_of(t) == {"vpn"} for t in bullets(group_a) + bullets(group_b))
    return {"same_problem": same, "reason": "oba o VPN" if same else "różne"}


def test_confirmed_merge_joins_neighbouring_clusters(tickets):
    config = cfg(coarse_threshold=0.1)  # fine start: VPN near-duplicates in separate clusters
    unmerged = run_method(tickets, "b", config, FakeBackend()).assignments
    assert unmerged.loc[VPN_ROWS, "problem_id"].nunique() > 1

    fake = FakeBackend().on("problem_merge", merge_if_both_vpn)
    a = run_method(tickets, "b", config, fake).assignments
    assert a.loc[VPN_ROWS, "problem_id"].nunique() == 1
    vpn_pid = a.loc[VPN_ROWS[0], "problem_id"]
    assert (a["problem_id"] == vpn_pid).sum() == len(VPN_ROWS)
    assert any(c["response_format"]["json_schema"]["name"] == "problem_merge" for c in fake.chat_inputs)


def iteration_lines(result):
    return [line for line in result.log if re.search(r"method_b iteration \d+:", line)]


def test_stops_when_nothing_changes_and_logs_iteration(tickets):
    result = run_method(tickets, "b", cfg(), FakeBackend())  # defaults: every cluster one Problem, no merges
    lines = iteration_lines(result)
    assert len(lines) == 1
    assert re.search(r"splits=0 merges=0", lines[0])
    assert any("stable" in line for line in result.log)


def test_log_records_splits_and_merges_per_iteration(tickets):
    fake = FakeBackend().on("cluster_verification", verify_by_topic).on("problem_merge", merge_if_both_vpn)
    result = run_method(tickets, "b", cfg(coarse_threshold=1.01, split_step=0.41), fake)
    lines = iteration_lines(result)
    assert re.search(r"splits=[1-9]", lines[0])
    assert all(re.search(r"verified=\d+ mixed=\d+ splits=\d+ merges=\d+", line) for line in lines)


def never_stable():
    always_mixed = {"one_problem": False, "reason": "zawsze mieszany"}
    return FakeBackend().on("cluster_verification", always_mixed).on("problem_merge", {"same_problem": True, "reason": ""})


def test_stops_at_max_iterations_when_never_stable(tickets):
    result = run_method(tickets, "b", cfg(coarse_threshold=1.01, split_step=0.01, max_iterations=2), never_stable())
    assert len(iteration_lines(result)) == 2
    assert any("max_iterations=2" in line for line in result.log)


def test_default_cap_is_five_iterations(tickets):
    assert cfg().method_b.max_iterations == 5
    result = run_method(tickets, "b", cfg(coarse_threshold=1.01, split_step=0.01), never_stable())
    assert 1 <= len(iteration_lines(result)) <= 5


def test_verification_sees_at_most_sample_size_tickets(tickets):
    fake = FakeBackend().on("cluster_verification", verify_by_topic)
    run_method(tickets, "b", cfg(coarse_threshold=1.01, split_step=0.41, sample_size=3), fake)
    checks = [c for c in fake.chat_inputs if c["response_format"]["json_schema"]["name"] == "cluster_verification"]
    assert checks
    assert all(1 <= len(bullets(c["messages"][-1]["content"])) <= 3 for c in checks)


def test_prompts_carry_granularity_and_only_masked_title_and_description(tickets):
    systems = {}
    for level in ["coarse", "fine"]:
        config = load_config(None, {"backend.kind": "fake", "granularity": level, "method_b.coarse_threshold": 1.01,
                                    "method_b.split_step": 0.41})
        fake = FakeBackend().on("cluster_verification", verify_by_topic).on("problem_merge", merge_if_both_vpn)
        run_method(tickets, "b", config, fake)
        by_schema = {c["response_format"]["json_schema"]["name"]: c["messages"][0]["content"] for c in fake.chat_inputs}
        assert {"cluster_verification", "problem_merge"} <= set(by_schema)
        systems[level] = by_schema
        sent = "\n".join(fake.all_input_texts())
        for col in ["Typ", "Numer I/SR", "Zgłoszenie", "Status", "Grupa odpowiedzialna"]:
            for value in tickets[col].dropna().astype(str).unique():
                assert value not in sent, f"{col} value {value!r} leaked to model input"
        for pii in ["jan.kowalski@firma.pl", "601 234 567", "85010112345"]:
            assert pii not in sent
    for step in ["cluster_verification", "problem_merge"]:
        assert systems["coarse"][step] != systems["fine"][step]
