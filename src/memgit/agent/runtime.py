"""``MemoryAgent`` — the tool-calling loop, and the turn-to-commit contract.

This is the one module in ``memgit.agent`` that touches ``Repository``. It
owns four things:

1. **The loop.** A hand-written ``while stop_reason == "tool_use"`` over
   :class:`~memgit.agent.client.LLMClient`, not the SDK's beta
   ``tool_runner`` — no beta dependency, and every step is visible code
   rather than something a fake has to reverse-engineer. :func:`run_tool_loop`
   is that loop pulled out as a free function, independent of ``Repository``,
   so slice 6's replay engine can run the exact same tool-calling behavior
   against a state it never intends to commit.
2. **The commit protocol.** :meth:`Repository.commit` takes the *whole* fact
   set, never a delta (the no-index decision in PLAN.md), so one turn
   mutates a working copy of the state the model was shown and hands the
   complete result to ``commit`` once the loop ends — never per tool call,
   so an API failure mid-turn leaves memory untouched rather than
   half-written. **This working copy always starts as the full ``before``
   state, in retrieval mode or not** — retrieval only narrows what gets
   *rendered* into the prompt, never what a turn commits. Seeding ``working``
   from a retrieved subset instead would mean a turn that only saw 8 of 200
   facts silently commits a memory with 192 of them deleted.
3. **What "memory" means across turns.** The chat transcript
   (``messages``) lives only for the lifetime of one :class:`MemoryAgent`
   and is never committed — it is scratch, not memory. What a turn commits
   is read back from ``HEAD`` at the start of the *next* turn, which is what
   lets a fact remembered in one process reach a fact recalled in the next.
4. **Retrieval mode selection.** Below ``config["retrieval"]["full_below"]``
   facts (default 64), a turn injects the full state, exactly as slice 5
   did. At or above it, a turn retrieves the top-k facts most relevant to
   the query and offers the ``recall`` tool for the rest — the two are the
   same flag (:func:`~memgit.agent.tools.tool_schemas`'s ``recall``
   parameter), since under full injection there is nothing left to recall.
   ``recall`` searches the turn's committed ``before`` state, not the
   mutating ``working`` copy — searching ``working`` would let the model
   "recall" a belief it invented earlier in the same turn, which was never
   part of any real commit.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from memgit.agent.client import AgentError, LLMClient, default_client
from memgit.agent.prompt import build_system_prompt
from memgit.agent.tools import (
    RecallCall,
    ToolCallError,
    TOOL_SCHEMAS,
    apply_forget,
    apply_remember,
    decode_forget,
    decode_recall,
    decode_remember,
    tool_schemas,
)
from memgit.core.cardinality import CardinalityMap
from memgit.core.repository import Repository
from memgit.core.state import MemoryState
from memgit.retrieval.rank import Retriever

__all__ = ["MemoryAgent", "TurnResult", "ToolCallRecord", "AgentError", "run_tool_loop"]

_MESSAGE_LIMIT = 72
_DEFAULT_FULL_BELOW = 64


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """One tool call this turn made, and whether it succeeded."""

    name: str
    input: Mapping[str, Any]
    ok: bool


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What one :meth:`MemoryAgent.turn` did.

    Attributes:
        commit: The new commit's hash, or ``None`` when the turn changed
            nothing and ``record_empty`` was not set — see
            :class:`MemoryAgent`.
    """

    reply: str
    commit: str | None
    before: MemoryState
    after: MemoryState
    tool_calls: tuple[ToolCallRecord, ...]
    stop_reason: str


def run_tool_loop(
    client: LLMClient,
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    working: MemoryState,
    cardinality: CardinalityMap,
    *,
    source: str,
    max_tool_rounds: int,
    retriever: Retriever | None = None,
    recall_state: MemoryState | None = None,
    as_of: datetime | None = None,
) -> tuple[Any, MemoryState, list[ToolCallRecord]]:
    """Run one turn's tool-calling loop to completion.

    Mutates ``messages`` in place (appending each round, same as
    :meth:`MemoryAgent.turn` always has) and returns the final response, the
    resulting :class:`MemoryState`, and every tool call made along the way.
    Independent of ``Repository`` — nothing here commits anything — which is
    exactly what lets slice 6's replay engine reuse it against a state it
    only intends to explore.

    Args:
        retriever: If given, offers the ``recall`` tool this turn
            (:func:`~memgit.agent.tools.tool_schemas`'s ``recall`` flag
            tracks ``retriever is not None``). ``None`` reproduces slice
            5/6's exact write-only tool surface.
        recall_state: The state ``recall`` searches — the turn's committed
            ``before``, never ``working``, so a call within a turn can never
            "recall" a belief the model only just invented. Required when
            ``retriever`` is given.
        as_of: The moment retrieval ranks against. Required when
            ``retriever`` is given; deliberately never defaulted to
            ``datetime.now()`` here, so every call into this loop stays
            reproducible.

    Raises:
        AgentError: the model refused, or the loop exceeded
            ``max_tool_rounds``.
    """
    tool_calls: list[ToolCallRecord] = []
    rounds = 0
    schemas = list(tool_schemas(recall=retriever is not None))

    while True:
        response = client.create_message(system=system, messages=messages, tools=schemas)

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = getattr(details, "category", None)
            explanation = getattr(details, "explanation", None)
            raise AgentError(f"the model refused ({category}): {explanation}")

        if response.stop_reason != "tool_use":
            break

        rounds += 1
        if rounds > max_tool_rounds:
            raise AgentError(f"tool-calling loop exceeded {max_tool_rounds} rounds")

        messages.append({"role": "assistant", "content": response.content})
        results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            record, result_block, working = _apply_tool_call(
                block,
                working,
                cardinality,
                source=source,
                retriever=retriever,
                recall_state=recall_state,
                as_of=as_of,
            )
            tool_calls.append(record)
            results.append(result_block)
        messages.append({"role": "user", "content": results})

    messages.append({"role": "assistant", "content": response.content})
    return response, working, tool_calls


def _apply_tool_call(
    block: Any,
    working: MemoryState,
    cardinality: CardinalityMap,
    *,
    source: str,
    retriever: Retriever | None = None,
    recall_state: MemoryState | None = None,
    as_of: datetime | None = None,
) -> tuple[ToolCallRecord, dict[str, Any], MemoryState]:
    """Decode and apply one ``tool_use`` block; never raises.

    A ``ToolCallError`` (an invalid fact, an unknown tool name) becomes
    an ``is_error`` tool result instead of aborting the turn, so the
    model gets a chance to correct itself on the next round.
    """
    try:
        if block.name == "remember":
            call = decode_remember(block.input, source=source)
            working = apply_remember(working, call, cardinality)
            content = f"remembered {call.fact.subject} {call.fact.predicate} {call.fact.object}"
        elif block.name == "forget":
            call = decode_forget(block.input)
            working = apply_forget(working, call)
            target = call.key[0] + " " + call.key[1]
            content = f"forgot {target}" + (f"={call.object}" if call.object else " (all values)")
        elif block.name == "recall" and retriever is not None:
            assert recall_state is not None and as_of is not None
            call = decode_recall(block.input)
            result = retriever.retrieve(
                recall_state,
                call.query,
                k=call.limit,
                as_of=as_of,
                subjects={call.subject} if call.subject is not None else None,
            )
            content = result.render()
        else:
            raise ToolCallError(f"unknown tool {block.name!r}")
    except ToolCallError as exc:
        result_block = {"type": "tool_result", "tool_use_id": block.id, "content": str(exc), "is_error": True}
        return ToolCallRecord(name=block.name, input=block.input, ok=False), result_block, working

    result_block = {"type": "tool_result", "tool_use_id": block.id, "content": content}
    return ToolCallRecord(name=block.name, input=block.input, ok=True), result_block, working


def _summarize(user_message: str, *, limit: int = _MESSAGE_LIMIT) -> str:
    """A deterministic one-line commit message, never model-authored.

    A model-authored message can disagree with what the diff actually
    shows; a terse, derived one cannot.
    """
    text = " ".join(user_message.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return f"turn: {text}"


class MemoryAgent:
    """Runs one Claude turn against the memory at ``HEAD``.

    Args:
        repo: The repository to read memory from and commit turns to.
        client: An :class:`~memgit.agent.client.LLMClient`. Defaults to a
            real :class:`~memgit.agent.client.AnthropicClient`, built lazily
            so constructing a :class:`MemoryAgent` in a test never requires
            the ``agent`` extra.
        model: Recorded in commit metadata; does not itself select which
            model ``client`` talks to (that is ``client``'s job).
        max_tool_rounds: Caps the tool-calling loop — exceeding it raises
            :class:`AgentError` rather than looping forever.
        author: The commit author. Defaults to ``f"agent:{model}"``.
        record_empty: Commit even when a turn remembers nothing. Off by
            default — see :meth:`turn`.
    """

    def __init__(
        self,
        repo: Repository,
        *,
        client: LLMClient | None = None,
        model: str = "claude-opus-5",
        max_tool_rounds: int = 8,
        author: str | None = None,
        record_empty: bool = False,
    ) -> None:
        self.repo = repo
        self.client = client if client is not None else default_client(model=model)
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.author = author or f"agent:{model}"
        self.record_empty = record_empty
        self._session_id = uuid.uuid4().hex[:12]
        self._turn = 0
        self._messages: list[dict[str, Any]] = []

    @property
    def transcript(self) -> tuple[dict[str, Any], ...]:
        """This session's chat history so far. Not memory — see the module docstring."""
        return tuple(self._messages)

    def turn(self, user_message: str) -> TurnResult:
        """Run one turn: read memory, call the model, apply tool calls, commit.

        A turn that remembers nothing produces ``TurnResult(commit=None)``
        unless ``record_empty`` was set at construction — see
        :class:`~memgit.core.repository.EmptyCommitError`'s own docstring,
        which calls recording no-op turns "a deliberate choice for that
        caller to make," not a default.

        Raises:
            AgentError: the model refused, or the tool-calling loop exceeded
                ``max_tool_rounds``.
        """
        self._turn += 1
        head = self.repo.head_commit()
        before = self.repo.state("HEAD") if head is not None else MemoryState.empty()
        cardinality = self.repo.cardinality()

        full_below = self.repo.config().get("retrieval", {}).get("full_below", _DEFAULT_FULL_BELOW)
        retrieval_mode = len(before) >= full_below
        retriever = self.repo.retriever() if retrieval_mode else None
        as_of = datetime.now(timezone.utc)
        retrieved = retriever.retrieve(before, user_message, as_of=as_of) if retriever is not None else None

        system = build_system_prompt(before, cardinality, head=head, retrieved=retrieved)

        self._messages.append({"role": "user", "content": user_message})
        source = f"session:{self._session_id}/turn:{self._turn}"
        response, working, tool_calls = run_tool_loop(
            self.client,
            system,
            self._messages,
            before,
            cardinality,
            source=source,
            max_tool_rounds=self.max_tool_rounds,
            retriever=retriever,
            recall_state=before,
            as_of=as_of,
        )
        reply = "".join(block.text for block in response.content if block.type == "text")

        changed = working.tree != before.tree
        commit_hash: str | None = None
        after = before
        if changed or self.record_empty:
            if retrieved is not None:
                retrieval_metadata: dict[str, Any] = {
                    "mode": "top_k",
                    "k": len(retrieved.facts),
                    "embedder": retrieved.embedder,
                    "injected": [r.fact.hash for r in retrieved.facts],
                    "as_of": retrieved.as_of,
                }
            else:
                retrieval_metadata = {"mode": "full"}
            metadata = {
                "query": user_message,
                "model": self.model,
                "session": self._session_id,
                "turn": self._turn,
                "stop_reason": response.stop_reason,
                "tool_calls": [{"name": tc.name, "ok": tc.ok} for tc in tool_calls],
                "retrieval": retrieval_metadata,
            }
            commit_hash = self.repo.commit(
                working.facts,
                _summarize(user_message),
                author=self.author,
                metadata=metadata,
                allow_empty=self.record_empty,
            )
            after = self.repo.state(commit_hash)

        return TurnResult(
            reply=reply,
            commit=commit_hash,
            before=before,
            after=after,
            tool_calls=tuple(tool_calls),
            stop_reason=response.stop_reason,
        )
