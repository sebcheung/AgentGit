"""``MemoryAgent`` — the tool-calling loop, and the turn-to-commit contract.

This is the one module in ``memgit.agent`` that touches ``Repository``. It
owns three things:

1. **The loop.** A hand-written ``while stop_reason == "tool_use"`` over
   :class:`~memgit.agent.client.LLMClient`, not the SDK's beta
   ``tool_runner`` — no beta dependency, and every step is visible code
   rather than something a fake has to reverse-engineer.
2. **The commit protocol.** :meth:`Repository.commit` takes the *whole* fact
   set, never a delta (the no-index decision in PLAN.md), so one turn
   mutates a working copy of the state the model was shown and hands the
   complete result to ``commit`` once the loop ends — never per tool call,
   so an API failure mid-turn leaves memory untouched rather than
   half-written.
3. **What "memory" means across turns.** The chat transcript
   (``messages``) lives only for the lifetime of one :class:`MemoryAgent`
   and is never committed — it is scratch, not memory. What a turn commits
   is read back from ``HEAD`` at the start of the *next* turn, which is what
   lets a fact remembered in one process reach a fact recalled in the next.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from memgit.agent.client import AgentError, LLMClient, default_client
from memgit.agent.prompt import build_system_prompt
from memgit.agent.tools import (
    ForgetCall,
    RememberCall,
    ToolCallError,
    TOOL_SCHEMAS,
    decode_forget,
    decode_remember,
)
from memgit.core.cardinality import CardinalityMap
from memgit.core.repository import Repository
from memgit.core.state import MemoryState

__all__ = ["MemoryAgent", "TurnResult", "ToolCallRecord", "AgentError"]

_MESSAGE_LIMIT = 72


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


def _merge_remember(state: MemoryState, call: RememberCall, cardinality: CardinalityMap) -> MemoryState:
    """Fold one ``remember`` onto ``state``, per the repo's cardinality map.

    The exact same triple already present is a reaffirmation (replace, so
    the diff engine reports ``reaffirmed`` rather than a spurious
    duplicate). Otherwise: a ``single`` predicate replaces whatever was at
    the key (``contradicted``); a ``multi`` predicate adds alongside
    (``value_added``). Consulting the cardinality map here does not violate
    "a diff is a question you ask, not data you store" — the runtime is a
    client of that repo-local lens exactly as ``memgit diff`` is, and
    nothing about this merge is itself hashed or committed.
    """
    fact = call.fact
    existing = state.get(*fact.key)
    same_triple = tuple(f for f in existing if f.triple == fact.triple)
    if same_triple:
        dropped = {f.hash for f in same_triple}
        kept = tuple(f for f in state.facts if f.hash not in dropped)
    elif cardinality.is_multi(fact.predicate):
        kept = state.facts
    else:
        kept = tuple(f for f in state.facts if f.key != fact.key)
    return MemoryState.from_facts((*kept, fact))


def _merge_forget(state: MemoryState, call: ForgetCall) -> MemoryState:
    """Fold one ``forget`` onto ``state``."""
    if call.object is None:
        return state.without(call.key)
    subject, predicate = call.key
    result = state
    for fact in state.get(subject, predicate):
        if fact.object == call.object:
            result = result.without_fact(fact.hash)
    return result


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
        system = build_system_prompt(before, cardinality, head=head)

        self._messages.append({"role": "user", "content": user_message})
        working = before
        tool_calls: list[ToolCallRecord] = []
        rounds = 0

        while True:
            response = self.client.create_message(
                system=system, messages=self._messages, tools=list(TOOL_SCHEMAS)
            )

            if response.stop_reason == "refusal":
                details = response.stop_details
                category = getattr(details, "category", None)
                explanation = getattr(details, "explanation", None)
                raise AgentError(f"the model refused ({category}): {explanation}")

            if response.stop_reason != "tool_use":
                break

            rounds += 1
            if rounds > self.max_tool_rounds:
                raise AgentError(f"tool-calling loop exceeded {self.max_tool_rounds} rounds")

            self._messages.append({"role": "assistant", "content": response.content})
            results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                record, result_block, working = self._apply_tool_call(block, working, cardinality)
                tool_calls.append(record)
                results.append(result_block)
            self._messages.append({"role": "user", "content": results})

        self._messages.append({"role": "assistant", "content": response.content})
        reply = "".join(block.text for block in response.content if block.type == "text")

        changed = working.tree != before.tree
        commit_hash: str | None = None
        after = before
        if changed or self.record_empty:
            metadata = {
                "query": user_message,
                "model": self.model,
                "session": self._session_id,
                "turn": self._turn,
                "stop_reason": response.stop_reason,
                "tool_calls": [{"name": tc.name, "ok": tc.ok} for tc in tool_calls],
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

    def _apply_tool_call(
        self, block: Any, working: MemoryState, cardinality: CardinalityMap
    ) -> tuple[ToolCallRecord, dict[str, Any], MemoryState]:
        """Decode and apply one ``tool_use`` block; never raises.

        A ``ToolCallError`` (an invalid fact, an unknown tool name) becomes
        an ``is_error`` tool result instead of aborting the turn, so the
        model gets a chance to correct itself on the next round.
        """
        source = f"session:{self._session_id}/turn:{self._turn}"
        try:
            if block.name == "remember":
                call = decode_remember(block.input, source=source)
                working = _merge_remember(working, call, cardinality)
                content = f"remembered {call.fact.subject} {call.fact.predicate} {call.fact.object}"
            elif block.name == "forget":
                call = decode_forget(block.input)
                working = _merge_forget(working, call)
                target = call.key[0] + " " + call.key[1]
                content = f"forgot {target}" + (f"={call.object}" if call.object else " (all values)")
            else:
                raise ToolCallError(f"unknown tool {block.name!r}")
        except ToolCallError as exc:
            result = {"type": "tool_result", "tool_use_id": block.id, "content": str(exc), "is_error": True}
            return ToolCallRecord(name=block.name, input=block.input, ok=False), result, working

        result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
        return ToolCallRecord(name=block.name, input=block.input, ok=True), result, working
