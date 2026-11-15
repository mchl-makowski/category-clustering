"""Deterministic offline backend for tests and dry runs.

- Embeddings: hashed bag of crude word stems, so lexically similar Tickets end up close.
- Chat: replies are chosen, in order, from
    1. the FIFO queue (`fake.queue(...)`),
    2. per-instance rules (`fake.on(match, reply)`, last added wins),
    3. `DEFAULT_HANDLERS` keyed by the `response_format` json_schema name
       (or "tools" when tools are passed and no schema is set).
  No match raises LookupError so a missing script fails loudly.
- `fake.fail_next(n, exc=None)`: next n calls raise (default TransientBackendError).
- Records every input: `embed_inputs`, `chat_inputs`, counters `embed_calls`, `chat_calls`.

A reply may be a `ChatReply`, a dict/list (serialised as JSON content), a str
(content), or a callable `(ChatRequest) -> any of those`. Use `tool_reply(...)`
for tool-call replies, `content_filtered()` for a reply the Azure content filter
blocks (raises ContentFilteredError, like the Azure backend).

Extension: a ticket that adds a new LLM step registers a default reply with
`@default_handler("<schema name>")` (in this file, append-only) so the full
pipeline still runs offline with no per-test scripting.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .base import ChatReply, ContentFilteredError, EmbeddingReply, ToolCall, TransientBackendError, Usage

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class ChatRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    response_format: dict[str, Any] | None
    model: str | None

    @property
    def schema_name(self) -> str | None:
        rf = self.response_format or {}
        return (rf.get("json_schema") or {}).get("name")

    @property
    def tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools or []]

    @property
    def text(self) -> str:
        return "\n".join(_message_text(m) for m in self.messages)

    @property
    def last_user(self) -> str:
        users = [m for m in self.messages if m.get("role") == "user"]
        return _message_text(users[-1]) if users else ""


Reply = ChatReply | dict | list | str | Callable[[ChatRequest], Any]
Match = str | Callable[[ChatRequest], bool]

DEFAULT_HANDLERS: dict[str, Callable[[ChatRequest], Any]] = {}


def default_handler(key: str):
    def register(fn):
        DEFAULT_HANDLERS[key] = fn
        return fn

    return register


def tool_reply(*calls: tuple[str, dict[str, Any]]) -> ChatReply:
    """A reply asking for tool calls: `tool_reply(("get_problem", {"problem_id": "C-0001"}))`."""
    return ChatReply(
        content=None,
        tool_calls=[ToolCall(id=f"call_{i}", name=n, arguments=json.dumps(a, ensure_ascii=False))
                    for i, (n, a) in enumerate(calls)],
        finish_reason="tool_calls",
    )


def content_filtered() -> ChatReply:
    """A reply blocked by the Azure content filter; the Fake raises ContentFilteredError for it, as Azure does."""
    return ChatReply(content=None, finish_reason="content_filter")


def _message_text(m: dict[str, Any]) -> str:
    parts = [str(m.get("content") or "")]
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        parts.append(str(fn.get("arguments", "")))
    return "\n".join(p for p in parts if p)


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def stems(text: str) -> list[str]:
    """Crude Polish-ish stemming: lowercase words of 3+ chars cut to 6 chars."""
    return [w[:6] for w in _tokens(text) if len(w) >= 3]


class FakeBackend:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.embed_inputs: list[str] = []
        self.chat_inputs: list[dict[str, Any]] = []
        self.embed_calls = 0
        self.chat_calls = 0
        self._queue: list[Reply] = []
        self._rules: list[tuple[Match, Reply]] = []
        self._failures: list[Exception] = []
        self._lock = threading.RLock()  # calls may arrive from parallel_map threads

    # --- scripting -------------------------------------------------------
    def queue(self, *replies: Reply) -> FakeBackend:
        self._queue.extend(replies)
        return self

    def on(self, match: Match, reply: Reply) -> FakeBackend:
        """`match` = json_schema name, or predicate over ChatRequest. Later rules take precedence."""
        self._rules.insert(0, (match, reply))
        return self

    def fail_next(self, n: int, exc: Exception | None = None) -> FakeBackend:
        """The next `n` calls (embed or chat) raise `exc` (default TransientBackendError). Still counted."""
        self._failures.extend([exc or TransientBackendError("fake transient failure")] * n)
        return self

    def _maybe_fail(self) -> None:
        with self._lock:
            exc = self._failures.pop(0) if self._failures else None
        if exc is not None:
            raise exc

    def all_input_texts(self) -> list[str]:
        """Every string sent to the model: embed inputs, message contents, tool-call args."""
        out = list(self.embed_inputs)
        for call in self.chat_inputs:
            out.extend(_message_text(m) for m in call["messages"])
        return out

    # --- Backend protocol ------------------------------------------------
    def embed(self, texts: list[str]) -> EmbeddingReply:
        with self._lock:
            self.embed_calls += 1
        self._maybe_fail()
        with self._lock:
            self.embed_inputs.extend(texts)
        vectors = np.vstack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))
        return EmbeddingReply(vectors=vectors, usage=Usage(prompt_tokens=sum(len(_tokens(t)) for t in texts)))

    def chat(self, messages, tools=None, response_format=None, model=None) -> ChatReply:
        with self._lock:
            self.chat_calls += 1
        self._maybe_fail()
        req = ChatRequest(messages, tools, response_format, model)
        with self._lock:
            self.chat_inputs.append(
                {"messages": messages, "tools": tools, "response_format": response_format, "model": model}
            )
            picked = self._pick(req)
        reply = self._coerce(picked, req)
        if reply.finish_reason == "content_filter":
            raise ContentFilteredError("fake content filter")
        reply.usage = Usage(
            prompt_tokens=len(_tokens(req.text)) + 1,
            completion_tokens=len(_tokens(reply.content or ""))
            + sum(len(_tokens(tc.arguments)) for tc in reply.tool_calls) + 1,
        )
        return reply

    # --- internals -------------------------------------------------------
    def _pick(self, req: ChatRequest) -> Reply:
        if self._queue:
            return self._queue.pop(0)
        for match, reply in self._rules:
            if (match == req.schema_name) if isinstance(match, str) else match(req):
                return reply
        key = req.schema_name or ("tools" if req.tools else None)
        if key in DEFAULT_HANDLERS:
            return DEFAULT_HANDLERS[key]
        raise LookupError(f"FakeBackend has no reply for schema={req.schema_name!r} tools={req.tool_names}")

    @staticmethod
    def _coerce(reply: Reply, req: ChatRequest) -> ChatReply:
        if callable(reply) and not isinstance(reply, ChatReply):
            reply = reply(req)
        if isinstance(reply, ChatReply):
            return ChatReply(reply.content, list(reply.tool_calls), Usage(), reply.finish_reason)
        if isinstance(reply, (dict, list)):
            return ChatReply(content=json.dumps(reply, ensure_ascii=False))
        return ChatReply(content=str(reply))

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for s in stems(text) or [text]:
            h = int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")
            v[h % self.dim] += 1.0
        return v / np.linalg.norm(v)


# --- default replies (one per LLM step; append new steps below) -----------

STOPWORDS = {"się", "nie", "jest", "przy", "oraz", "proszę", "problem", "zgłoszenia", "brak", "kilka"}


@default_handler("problem_naming")
def _naming(req: ChatRequest) -> dict:
    tickets = [line[2:] for line in req.last_user.splitlines() if line.startswith("- ")]
    counts = Counter(w for t in tickets for w in set(_tokens(t)) if len(w) >= 4 and w not in STOPWORDS)
    top = [w for w, _ in counts.most_common(3)] or ["inne"]
    return {"name": "Problem: " + " ".join(top), "description": "Zgłoszenia dotyczące: " + ", ".join(top) + "."}


FAKE_AGENT_ASSIGN_SIMILARITY = 0.5


@default_handler("tools")
def _method_c_agent(req: ChatRequest) -> ChatReply:
    """Method C agent: search once, then assign to the top hit if similar enough, else create."""
    first_user = next((m for m in req.messages if m.get("role") == "user"), {})
    ticket = _message_text(first_user).split("\n", 1)[-1].strip()
    tool_results = [m for m in req.messages if m.get("role") == "tool"]
    if not tool_results:
        return tool_reply(("search_similar_tickets", {"query": ticket, "k": 5}))
    results = json.loads(tool_results[-1]["content"]).get("results") or []
    if results and results[0]["similarity"] >= FAKE_AGENT_ASSIGN_SIMILARITY:
        return tool_reply(("assign_to_problem", {"problem_id": results[0]["problem_id"],
                                                 "confidence_level": "wysokie"}))
    title = ticket.splitlines()[0] if ticket else "inne"
    return tool_reply(("create_problem", {"name": title, "description": f"Zgłoszenia: {title}.",
                                          "confidence_level": "średnie"}))


@default_handler("method_c_consolidation")
def _method_c_consolidation(req: ChatRequest) -> dict:
    return {"merges": []}


@default_handler("cluster_verification")
def _cluster_verification(req: ChatRequest) -> dict:
    """Method B: accept every cluster as one Problem (stable after one iteration)."""
    return {"one_problem": True, "reason": "Fake: domyślnie jeden Problem."}


@default_handler("problem_merge")
def _problem_merge(req: ChatRequest) -> dict:
    """Method B: reject every merge."""
    return {"same_problem": False, "reason": "Fake: domyślnie różne Problemy."}


@default_handler("symptom_summary")
def _symptom_summary(req: ChatRequest) -> dict:
    """Echo the Ticket text flattened to one lowercase line (keeps lexical grouping offline)."""
    text = req.last_user.split("\n", 1)[-1]
    return {"summary": " ".join(_tokens(text))}


# --- Judge (#7) -------------------------------------------------------------

def _judge_bullets(text: str) -> list[str]:
    return [line[2:] for line in text.splitlines() if line.startswith("- ")]


def _judge_stems(text: str) -> set[str]:
    return {s for s in stems(text) if s not in STOPWORDS}


@default_handler("judge_mixed")
def _judge_mixed(req: ChatRequest) -> dict:
    """Mixed when some sampled Ticket shares no word stem with any other sampled Ticket."""
    sets = [_judge_stems(t) for t in _judge_bullets(req.last_user)]
    lonely = [i for i, s in enumerate(sets) if not any(s & o for j, o in enumerate(sets) if j != i)]
    if lonely:
        return {"mixed": True, "reason": f"Zgłoszenie {lonely[0] + 1} opisuje inny objaw."}
    return {"mixed": False, "reason": "Wszystkie zgłoszenia opisują ten sam objaw."}


@default_handler("judge_split")
def _judge_split(req: ChatRequest) -> dict:
    """Same Problem when the two samples' stem sets overlap by Jaccard >= 0.5."""
    first, _, second = req.last_user.partition("Problem 2:")
    a = set().union(*[_judge_stems(t) for t in _judge_bullets(first)])
    b = set().union(*[_judge_stems(t) for t in _judge_bullets(second)])
    if len(a & b) / max(len(a | b), 1) >= 0.5:
        return {"same_problem": True, "reason": "Oba Problemy opisują ten sam objaw."}
    return {"same_problem": False, "reason": "Problemy opisują różne objawy."}
