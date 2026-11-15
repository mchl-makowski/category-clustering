import json
import re

import pytest

from problem_clustering import load_config, load_tickets, run_method
from problem_clustering.backend import ChatReply, FakeBackend, ToolCall, tool_reply

EMPTY_ROWS = {17, 18}
VPN_ROWS = [0, 1, 2, 3, 19]
PRINTER_ROWS = [7, 8, 9]
LEVELS = {"pewne": 1.0, "wysokie": 0.8, "średnie": 0.6, "niskie": 0.4, "zgadywane": 0.2}


@pytest.fixture
def tickets(fixture_path):
    return load_tickets(fixture_path)


def cfg(**overrides):
    return load_config(None, {"backend.kind": "fake", **overrides})


def test_every_ticket_gets_a_c_problem_with_confidence_from_its_level(tickets):
    a = run_method(tickets, "c", cfg(), FakeBackend()).assignments
    assert list(a.index) == list(tickets.index)
    for i, row in a.iterrows():
        if i in EMPTY_ROWS:
            assert row["problem_id"] == "X-0000"
        else:
            assert re.fullmatch(r"C-\d{4}", row["problem_id"])
            assert row["confidence_level"] in LEVELS
            assert row["confidence"] == LEVELS[row["confidence_level"]]
    assert a.loc[VPN_ROWS, "problem_id"].nunique() == 1
    assert a.loc[PRINTER_ROWS, "problem_id"].nunique() == 1
    assert a.loc[VPN_ROWS[0], "problem_id"] != a.loc[PRINTER_ROWS[0], "problem_id"]


OUTLOOK = "Outlook nie synchronizuje kalendarza"  # row 4


def ticket_is(fragment):
    """Rule predicate: agent request for the Ticket whose text contains `fragment`."""
    return lambda req: bool(req.tools) and fragment in ticket_message(req.messages)


def ticket_message(messages):
    return next(m["content"] for m in messages if m["role"] == "user")


def tool_results(messages):
    return [json.loads(m["content"]) for m in messages if m.get("role") == "tool"]


def agent_requests(fake, fragment):
    """Recorded agent requests for one Ticket (its first user message holds the Ticket text)."""
    return [c["messages"] for c in fake.chat_inputs if c["tools"] and fragment in ticket_message(c["messages"])]


def raw_tool_reply(name, arguments):
    return ChatReply(tool_calls=[ToolCall(id="call_raw", name=name, arguments=arguments)],
                     finish_reason="tool_calls")


def test_scripted_create_then_assign_land_in_expected_problems(tickets):
    fake = FakeBackend()
    fake.on(ticket_is("VPN rozłącza się po aktualizacji FortiClient"), tool_reply(
        ("create_problem", {"name": "VPN zrywa się po aktualizacji FortiClient 7.2",
                            "description": "Połączenie VPN zrywa się po aktualizacji.", "confidence_level": "pewne"})))

    def second(req):
        found = tool_results(req.messages)
        if not found:
            return tool_reply(("search_similar_tickets", {"query": "VPN FortiClient", "k": 3}))
        if len(found) == 1:
            return tool_reply(("get_problem", {"problem_id": found[0]["results"][0]["problem_id"]}))
        return tool_reply(("assign_to_problem", {"problem_id": found[1]["problem_id"], "confidence_level": "niskie"}))

    fake.on(ticket_is("Zrywa połączenie VPN"), second)
    a = run_method(tickets, "c", cfg(), fake).assignments
    assert a.loc[0, "problem_id"] == a.loc[1, "problem_id"]
    assert a.loc[0, "problem_name"] == "VPN zrywa się po aktualizacji FortiClient 7.2"
    assert (a.loc[0, "confidence_level"], a.loc[0, "confidence"]) == ("pewne", 1.0)
    assert (a.loc[1, "confidence_level"], a.loc[1, "confidence"]) == ("niskie", 0.4)
    search, problem = tool_results(agent_requests(fake, "Zrywa połączenie VPN")[-1])
    assert len(search["results"]) == 1  # only already-classified Tickets are searched
    assert problem["name"] == "VPN zrywa się po aktualizacji FortiClient 7.2"
    assert problem["ticket_count"] == 1


def test_first_ticket_search_sees_empty_catalogue(tickets):
    fake = FakeBackend()
    run_method(tickets, "c", cfg(), fake)
    (first,) = tool_results(agent_requests(fake, "VPN rozłącza się po aktualizacji FortiClient")[-1])
    assert first["results"] == []


@pytest.mark.parametrize("cap", [6, 3])
def test_agent_capped_at_max_tool_calls_then_ticket_gets_new_problem(tickets, cap):
    fake = FakeBackend()
    fake.on(ticket_is(OUTLOOK), tool_reply(("search_similar_tickets", {"query": "Outlook", "k": 2})))
    result = run_method(tickets, "c", cfg(**{"method_c.max_tool_calls": cap}), fake)
    assert len(agent_requests(fake, OUTLOOK)) == cap
    a = result.assignments
    assert (a.loc[4, "confidence_level"], a.loc[4, "confidence"]) == ("zgadywane", 0.2)
    assert a.loc[4, "problem_id"] not in set(a.loc[:3, "problem_id"])  # a new Problem
    assert a.loc[4, "problem_name"]  # named by the shared naming step
    assert result.catalogue.set_index("problem_id").loc[a.loc[4, "problem_id"], "description"]


def test_parallel_tool_calls_count_toward_the_cap(tickets):
    fake = FakeBackend()
    fake.on(ticket_is(OUTLOOK), tool_reply(*[("search_similar_tickets", {"query": "Outlook", "k": 1})] * 4))
    result = run_method(tickets, "c", cfg(), fake)
    requests = agent_requests(fake, OUTLOOK)
    assert len(requests) == 2  # 4 + 2 tool calls = 6
    assert len(tool_results(requests[-1])) == 4
    assert result.assignments.loc[4, "confidence_level"] == "zgadywane"


def test_reply_without_tool_call_is_nudged_and_counts_toward_the_cap(tickets):
    fake = FakeBackend()
    fake.on(ticket_is(OUTLOOK), "To chyba nowy problem.")
    result = run_method(tickets, "c", cfg(), fake)
    requests = agent_requests(fake, OUTLOOK)
    assert len(requests) == 6
    assert requests[-1][-1]["role"] == "user" and "assign_to_problem" in requests[-1][-1]["content"]
    assert result.assignments.loc[4, "confidence_level"] == "zgadywane"


def test_invalid_confidence_level_and_unknown_problem_id_are_reported_back_without_crash(tickets):
    fake = FakeBackend()
    script = [
        tool_reply(("get_problem", {"problem_id": "C-9999"})),
        tool_reply(("assign_to_problem", {"problem_id": "C-9999", "confidence_level": "pewne"})),
        tool_reply(("create_problem", {"name": "X", "description": "Y", "confidence_level": "bardzo"})),
        raw_tool_reply("create_problem", "{nie json"),
        tool_reply(("create_problem", {"name": "Outlook kalendarz", "description": "Brak synchronizacji.",
                                       "confidence_level": "Średnie"})),
    ]
    fake.on(ticket_is(OUTLOOK), lambda req: script[len(tool_results(req.messages))])
    a = run_method(tickets, "c", cfg(), fake).assignments
    errors = tool_results(agent_requests(fake, OUTLOOK)[-1])
    assert len(errors) == 4 and all("error" in e for e in errors)
    assert a.loc[4, "problem_name"] == "Outlook kalendarz"
    assert (a.loc[4, "confidence_level"], a.loc[4, "confidence"]) == ("średnie", 0.6)


def always_create(req):
    title = ticket_message(req.messages).split("\n")[1]
    return tool_reply(("create_problem", {"name": title, "description": title, "confidence_level": "wysokie"}))


def consolidation_points(fake):
    """Number of Tickets the agent had seen when each consolidation request was made."""
    seen, points = set(), []
    for c in fake.chat_inputs:
        if c["tools"]:
            seen.add(ticket_message(c["messages"]))
        elif (c["response_format"] or {}).get("json_schema", {}).get("name") == "method_c_consolidation":
            points.append(len(seen))
    return sorted(set(points))


def merge_vpn(req):
    ids = [pid for pid, name in re.findall(r"^\[(C-\d{4})\] (.*)$", req.last_user, re.M) if "VPN" in name]
    merges = [{"problem_ids": ids, "name": "VPN po aktualizacji FortiClient",
               "description": "VPN rozłącza się po aktualizacji."}] if len(ids) > 1 else []
    return {"merges": merges + [{"problem_ids": ["C-9999", "C-0001"], "name": "x", "description": "x"}]}


def test_consolidation_runs_every_n_tickets_and_at_end_and_reassigns_merged_tickets(tickets):
    fake = FakeBackend()
    fake.on(lambda req: bool(req.tools), always_create)
    fake.on("method_c_consolidation", merge_vpn)
    over = {"method_c.consolidate_every": 5, "method_c.consolidate_threshold": 2.0}
    result = run_method(tickets, "c", cfg(**over), fake)
    assert consolidation_points(fake) == [5, 10, 15, 18]
    a = result.assignments
    assert a.loc[VPN_ROWS, "problem_id"].nunique() == 1
    assert (a.loc[VPN_ROWS, "problem_name"] == "VPN po aktualizacji FortiClient").all()
    assert (a.loc[VPN_ROWS, "confidence_level"] == "wysokie").all()
    assert a.loc[PRINTER_ROWS, "problem_id"].nunique() == 3  # not merged
    vpn = result.catalogue.set_index("problem_id").loc[a.loc[0, "problem_id"]]
    assert vpn["ticket_count"] == len(VPN_ROWS)
    assert any("consolidation" in line for line in result.log)


def test_consolidation_by_default_runs_once_at_the_end_on_a_small_file(tickets):
    fake = FakeBackend()
    fake.on(lambda req: bool(req.tools), always_create)
    run_method(tickets, "c", cfg(**{"method_c.consolidate_threshold": 2.0}), fake)
    assert consolidation_points(fake) == [18]


def test_prompts_carry_granularity_paragraph_and_only_masked_title_and_description(tickets):
    from problem_clustering.prompts import GRANULARITY_PARAGRAPHS

    for level in ["coarse", "fine"]:
        fake = FakeBackend()
        fake.on(lambda req: bool(req.tools), always_create)
        over = {"granularity": level, "method_c.consolidate_threshold": 2.0}
        result = run_method(tickets, "c", cfg(**over), fake)
        kinds = {"tools" if c["tools"] else c["response_format"]["json_schema"]["name"] for c in fake.chat_inputs}
        assert {"tools", "method_c_consolidation"} <= kinds
        for c in fake.chat_inputs:
            assert GRANULARITY_PARAGRAPHS[level] in c["messages"][0]["content"]
        sent = "\n".join(fake.all_input_texts())
        for col in ["Typ", "Numer I/SR", "Zgłoszenie", "Status", "Grupa odpowiedzialna"]:
            for value in tickets[col].dropna().astype(str).unique():
                assert value not in sent, f"{col} value {value!r} leaked"
        for pii in ["jan.kowalski@firma.pl", "601 234 567", "85010112345"]:
            assert pii not in sent
        assert "jan.kowalski@firma.pl" in result.tickets.loc[1, "Opis"]


def test_tools_use_openai_function_calling_shape(tickets):
    fake = FakeBackend()
    run_method(tickets, "c", cfg(), fake)
    tools = next(c["tools"] for c in fake.chat_inputs if c["tools"])
    assert {t["function"]["name"] for t in tools} == {
        "search_similar_tickets", "get_problem", "assign_to_problem", "create_problem"}
    for t in tools:
        assert t["type"] == "function"
        params = t["function"]["parameters"]
        assert params["type"] == "object" and set(params["required"]) <= set(params["properties"])
    msgs = agent_requests(fake, "Zrywa połączenie VPN")[-1]
    assistant = next(m for m in msgs if m["role"] == "assistant")
    call = assistant["tool_calls"][0]
    assert call["type"] == "function" and json.loads(call["function"]["arguments"])
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == call["id"] and isinstance(tool_msg["content"], str)


def test_output_format_matches_method_a(tickets, tmp_path):
    from problem_clustering import write_outputs

    fake = FakeBackend()
    c = run_method(tickets, "c", cfg(), fake)
    a = run_method(tickets, "a", cfg(), FakeBackend())
    assert list(c.assignments.columns) == list(a.assignments.columns)
    assert list(c.catalogue.columns) == list(a.catalogue.columns)
    assert c.catalogue["problem_id"].iloc[-1] == "X-0000"
    paths = write_outputs(c, None, tmp_path)
    assert re.fullmatch(r"\d{6}_\d{4}_c\.xlsx", paths["xlsx"].name)
    assert c.usage.llm_calls == fake.chat_calls > 0
    assert c.usage.embedding_calls == fake.embed_calls > 0
