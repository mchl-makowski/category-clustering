"""Method C: Ticket-by-Ticket tool-calling agent with lookup and periodic consolidation.

Tickets are processed in file order. For each one the agent may call
`search_similar_tickets` / `get_problem` and must finish with a terminal
`assign_to_problem` or `create_problem`. Every tool call (and every reply with
no tool call) counts toward `method_c.max_tool_calls`; when the budget runs out
without a valid decision the Ticket gets a new, unnamed Problem (`zgadywane`)
that the shared naming step names at the end.

Invalid tool arguments (unknown problem_id, bad Confidence Level, bad JSON) are
answered with an `{"error": ...}` tool result so the agent can correct itself.

Every `consolidate_every` Tickets and at the end, Problems with close centroids
are grouped (agglomerative on centroids) and the LLM decides which of them are
duplicates; merged Problems' Tickets move to the surviving Problem.

Problem ids used inside the run (`C-0001`, …, creation order) are provisional;
`run_method` reassigns final ids by size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..backend import ContentFilteredError, ToolCall, json_schema_format
from ..clustering import agglomerative, centroids, groups_from_labels, unit_rows
from ..prompts import (
    AGENT_NUDGE, CONSOLIDATION_SCHEMA, CONSOLIDATION_SCHEMA_NAME, agent_messages, agent_tools,
    consolidation_messages,
)
from ..result import CONFIDENCE_LEVELS, ProblemDraft
from . import MethodContext

FALLBACK_LEVEL = "zgadywane"
_TICKET_SNIPPET = 500  # chars of a Ticket text returned by tools


@dataclass
class _Problem:
    pid: str
    name: str | None
    description: str | None
    members: list[int] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)


@dataclass
class _Decision:
    level: str
    problem_id: str | None = None  # assign
    name: str | None = None  # create
    description: str | None = None


class _Agent:
    def __init__(self, ctx: MethodContext):
        self.ctx = ctx
        self.cfg = ctx.config.method_c
        self.texts = ctx.texts
        vectors = ctx.embeddings()
        self.unit = unit_rows(vectors)
        self.pos = {row: i for i, row in enumerate(ctx.texts.index)}
        self.problems: dict[str, _Problem] = {}
        self.row_problem: dict[int, str] = {}
        self.classified: list[int] = []
        self._next = 1
        self._consolidated_signatures: set[tuple] = set()
        self.tools = agent_tools(list(CONFIDENCE_LEVELS), self.cfg.search_max_k)
        self.stats = {"assigned": 0, "created": 0, "fallback": 0, "tool_errors": 0, "merged": 0}

    # --- catalogue -------------------------------------------------------
    def _new_problem(self, name: str | None, description: str | None) -> _Problem:
        pid = f"C-{self._next:04d}"
        self._next += 1
        self.problems[pid] = _Problem(pid, name, description)
        return self.problems[pid]

    def _add(self, problem: _Problem, row: int, level: str) -> None:
        problem.members.append(row)
        problem.levels.append(level)
        self.row_problem[row] = problem.pid
        self.classified.append(row)

    def _display_name(self, p: _Problem) -> str:
        return p.name or self._snippet(p.members[0]).splitlines()[0]

    def _snippet(self, row: int) -> str:
        return self.texts[row][:_TICKET_SNIPPET]

    # --- tools -----------------------------------------------------------
    def _search(self, args: dict) -> dict:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "Pusty parametr query."}
        try:
            k = int(args.get("k") or self.cfg.search_k)
        except (TypeError, ValueError):
            k = self.cfg.search_k
        k = max(1, min(k, self.cfg.search_max_k))
        if not self.classified:
            return {"results": [], "note": "Brak sklasyfikowanych zgłoszeń; katalog jest pusty."}
        q = unit_rows(self.ctx.embed([query]))[0]
        idx = [self.pos[r] for r in self.classified]
        sims = self.unit[idx] @ q
        order = np.argsort(-sims, kind="stable")[:k]
        results = []
        for o in order:
            row = self.classified[o]
            p = self.problems[self.row_problem[row]]
            results.append({"problem_id": p.pid, "problem_name": self._display_name(p),
                            "similarity": round(float(sims[o]), 3), "ticket": self._snippet(row)})
        return {"results": results}

    def _get_problem(self, args: dict) -> dict:
        p = self.problems.get(str(args.get("problem_id") or "").strip())
        if p is None:
            return self._unknown_problem(args.get("problem_id"))
        return {"problem_id": p.pid, "name": self._display_name(p), "description": p.description or "",
                "ticket_count": len(p.members),
                "example_tickets": [self._snippet(r) for r in p.members[: self.cfg.example_tickets]]}

    def _unknown_problem(self, pid) -> dict:
        return {"error": f"Nieznany problem_id {pid!r}. Użyj problem_id zwróconego przez search_similar_tickets."}

    @staticmethod
    def _level(args: dict) -> str | None:
        level = str(args.get("confidence_level") or "").strip().lower()
        return level if level in CONFIDENCE_LEVELS else None

    def _level_error(self, args: dict) -> dict:
        return {"error": f"Niepoprawny confidence_level {args.get('confidence_level')!r}. "
                         f"Dozwolone: {', '.join(CONFIDENCE_LEVELS)}."}

    def _execute(self, call: ToolCall) -> tuple[dict, _Decision | None]:
        try:
            args = call.args()
            if not isinstance(args, dict):
                raise ValueError
        except (ValueError, TypeError):
            return {"error": "Argumenty muszą być obiektem JSON."}, None
        if call.name == "search_similar_tickets":
            return self._search(args), None
        if call.name == "get_problem":
            return self._get_problem(args), None
        if call.name == "assign_to_problem":
            pid = str(args.get("problem_id") or "").strip()
            if pid not in self.problems:
                return self._unknown_problem(args.get("problem_id")), None
            level = self._level(args)
            if level is None:
                return self._level_error(args), None
            return {"ok": True}, _Decision(level, problem_id=pid)
        if call.name == "create_problem":
            name = str(args.get("name") or "").strip()
            description = str(args.get("description") or "").strip()
            if not name or not description:
                return {"error": "Parametry name i description nie mogą być puste."}, None
            level = self._level(args)
            if level is None:
                return self._level_error(args), None
            return {"ok": True}, _Decision(level, name=name, description=description)
        return {"error": f"Nieznane narzędzie {call.name!r}."}, None

    # --- per Ticket ------------------------------------------------------
    def classify(self, row: int) -> None:
        budget = self.cfg.max_tool_calls
        messages = agent_messages(self.ctx.config.granularity, self.texts[row], len(self.problems), budget)
        used = 0
        decision = None
        while used < budget and decision is None:
            # snapshot: the conversation grows after the call; backends/caches/recorders may keep the list
            try:
                reply = self.ctx.backend.chat(list(messages), tools=self.tools)
            except ContentFilteredError:
                self.ctx.log(f"method_c: row {row} agent reply blocked by content filter")
                break
            if not reply.tool_calls:
                used += 1
                messages.append({"role": "assistant", "content": reply.content or ""})
                messages.append({"role": "user", "content": AGENT_NUDGE})
                continue
            batch = reply.tool_calls[: budget - used]
            messages.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments}} for tc in batch],
            })
            for tc in batch:
                used += 1
                result, decision = self._execute(tc)
                if decision is not None:
                    break
                if "error" in result:
                    self.stats["tool_errors"] += 1
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})

        if decision is None:
            self.stats["fallback"] += 1
            self._add(self._new_problem(None, None), row, FALLBACK_LEVEL)
            self.ctx.log(f"method_c: row {row} no decision after {used} tool calls -> new Problem")
        elif decision.problem_id is not None:
            self.stats["assigned"] += 1
            self._add(self.problems[decision.problem_id], row, decision.level)
        else:
            self.stats["created"] += 1
            self._add(self._new_problem(decision.name, decision.description), row, decision.level)

    # --- consolidation ---------------------------------------------------
    def consolidate(self, processed: int) -> None:
        live = list(self.problems.values())
        merges_before = self.stats["merged"]
        if len(live) >= 2:
            cents = centroids([self.unit[[self.pos[r] for r in p.members]] for p in live])
            labels = agglomerative(cents, self.cfg.consolidate_threshold, "average")
            size = max(2, self.cfg.consolidate_group_size)
            for positions in groups_from_labels(labels):
                group = [live[i] for i in positions]
                for start in range(0, len(group), size):
                    chunk = [p for p in group[start: start + size] if p.pid in self.problems]
                    if len(chunk) >= 2:
                        self._consolidate_chunk(chunk)
        self.ctx.log(f"method_c: consolidation after {processed} Tickets: "
                     f"{self.stats['merged'] - merges_before} Problems merged, {len(self.problems)} remain")

    def _consolidate_chunk(self, chunk: list[_Problem]) -> None:
        signature = tuple(sorted((p.pid, len(p.members)) for p in chunk))
        if signature in self._consolidated_signatures:
            return
        self._consolidated_signatures.add(signature)
        payload = [{"problem_id": p.pid, "name": self._display_name(p), "description": p.description or "",
                    "ticket_count": len(p.members),
                    "examples": [self.texts[r] for r in p.members[: self.cfg.example_tickets]]} for p in chunk]
        fmt = json_schema_format(CONSOLIDATION_SCHEMA_NAME, CONSOLIDATION_SCHEMA)
        try:
            reply = self.ctx.backend.chat(consolidation_messages(self.ctx.config.granularity, payload),
                                          response_format=fmt)
        except ContentFilteredError:
            self.ctx.log("method_c: consolidation blocked by content filter, skipped")
            return
        try:
            merges = reply.json()["merges"]
        except (ValueError, TypeError, KeyError):
            self.ctx.log("method_c: consolidation reply not valid JSON, skipped")
            return
        allowed = {p.pid for p in chunk}
        for merge in merges if isinstance(merges, list) else []:
            if not isinstance(merge, dict):
                continue
            ids = []
            for pid in merge.get("problem_ids") or []:
                pid = str(pid).strip()
                if pid in allowed and pid in self.problems and pid not in ids:
                    ids.append(pid)
            if len(ids) < 2:
                continue
            ids.sort(key=lambda pid: int(pid.split("-")[1]))
            self._merge(ids, str(merge.get("name") or "").strip() or None,
                        str(merge.get("description") or "").strip() or None)

    def _merge(self, ids: list[str], name: str | None, description: str | None) -> None:
        survivor = self.problems[ids[0]]
        for pid in ids[1:]:
            other = self.problems.pop(pid)
            survivor.members.extend(other.members)
            survivor.levels.extend(other.levels)
            for row in other.members:
                self.row_problem[row] = survivor.pid
        survivor.name, survivor.description = name, description
        self.stats["merged"] += len(ids) - 1
        self.ctx.log(f"method_c: merged {', '.join(ids)} -> {survivor.pid}")

    def drafts(self) -> list[ProblemDraft]:
        return [ProblemDraft(members=list(p.members), confidences=[CONFIDENCE_LEVELS[lv] for lv in p.levels],
                             confidence_levels=list(p.levels), name=p.name, description=p.description)
                for p in self.problems.values()]


def run(ctx: MethodContext) -> list[ProblemDraft]:
    cfg = ctx.config.method_c
    agent = _Agent(ctx)
    since = 0
    for n, row in enumerate(ctx.texts.index, start=1):
        agent.classify(row)
        since += 1
        if cfg.consolidate_every > 0 and n % cfg.consolidate_every == 0:
            agent.consolidate(n)
            since = 0
    if since:
        agent.consolidate(len(ctx.texts))
    ctx.log("method_c: " + " ".join(f"{k}={v}" for k, v in agent.stats.items()))
    return agent.drafts()
