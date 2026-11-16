import pytest

from problem_clustering.backend import ChatReply, FakeBackend, json_schema_format, tool_reply

TOOLS = [{"type": "function", "function": {"name": "get_problem", "parameters": {"type": "object"}}}]
MSG = [{"role": "user", "content": "cześć"}]


def test_queue_then_rules_then_defaults_and_recording():
    fake = FakeBackend()
    fake.on("verify", {"one_problem": False})
    fake.queue(tool_reply(("get_problem", {"problem_id": "C-0001"})))

    first = fake.chat(MSG, tools=TOOLS)
    assert first.finish_reason == "tool_calls"
    assert first.tool_calls[0].name == "get_problem"
    assert first.tool_calls[0].args() == {"problem_id": "C-0001"}

    second = fake.chat(MSG, response_format=json_schema_format("verify", {}))
    assert second.json() == {"one_problem": False}
    assert second.usage.prompt_tokens > 0 and second.usage.completion_tokens > 0

    assert fake.chat_calls == 2
    assert fake.chat_inputs[0]["tools"] == TOOLS


def test_predicate_rule_with_callable_reply():
    fake = FakeBackend().on(lambda req: "kamera" in req.last_user, lambda req: ChatReply(content="ok"))
    assert fake.chat([{"role": "user", "content": "kamera"}]).content == "ok"


def test_unscripted_request_fails_loudly():
    with pytest.raises(LookupError):
        FakeBackend().chat(MSG, response_format=json_schema_format("nobody_scripted_this", {}))


def test_embeddings_are_deterministic_unit_vectors_and_recorded():
    fake = FakeBackend()
    a = fake.embed(["VPN rozłącza się", "drukarka nie drukuje"]).vectors
    b = FakeBackend().embed(["VPN rozłącza się", "drukarka nie drukuje"]).vectors
    assert (a == b).all()
    assert abs((a[0] ** 2).sum() - 1) < 1e-9
    assert fake.embed_calls == 1 and fake.embed_inputs == ["VPN rozłącza się", "drukarka nie drukuje"]
