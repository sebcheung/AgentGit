"""The tool surface an agent uses to touch memory: ``remember``, ``forget``, ``recall``.

``remember`` is the locked-in decision in PLAN.md — "Agent calls
``remember(...)`` deliberately as a tool: structured at birth, no
extraction step, no parsing noise." ``forget`` earns its place alongside it
because without it a belief can be *revised* (a ``remember`` at a
``single`` key contradicts the old value) but never *retracted* — and
``removed``/``value_removed`` are two of the diff engine's eight change
kinds. A tool surface that can never emit them would leave a quarter of the
taxonomy dead code on the agent path.

One tool deliberately **not** here:

- ``reaffirm`` — a ``remember`` of the same triple at a new confidence
  already *is* a reaffirmation; :meth:`Fact.reaffirm` and the diff engine's
  ``reaffirmed`` kind handle it without the model ever having to choose
  between two verbs for one intention.

``recall`` earns its place as of slice 7: full-context injection is no
longer the only mode (see ``prompt.py``), and once a turn only sees the
top-k most relevant facts, the model needs a second read path for the rest
of what it knows. ``runtime.py`` offers ``recall`` *only* in retrieval mode
— under full injection there is nothing to recall, and offering it anyway
would let the model act on facts it was shown twice under two different
guises, which is not the leak PLAN.md's retrieval-layer decision warns
about, but is exactly as pointless.

This module has no dependency on ``Repository`` or an LLM client — only on
:class:`~memgit.core.fact.Fact`, :class:`~memgit.core.state.MemoryState`, and
:class:`~memgit.core.cardinality.CardinalityMap`, all of them core and
offline. That is deliberate: slice 8's MCP server is documented in PLAN.md as
"moved up — the tool-call fact-write decision makes this nearly the same
code as slice 4," and this module — decoders *and* the :func:`apply_remember`
/ :func:`apply_forget` folds below — is the code it shares with
``agent/runtime.py``. ``recall``'s *decoder* lives here for the same reason;
the retrieval it triggers is the runtime's (or the MCP server's) job, not
this module's.

``apply_remember``/``apply_forget`` used to be private to ``runtime.py``
(``_merge_remember``/``_merge_forget``). They moved here, unchanged in
behavior, because ``decode_X`` (tool input → a typed call) and ``apply_X``
(a typed call → a new :class:`~memgit.core.state.MemoryState`) are two
halves of one contract — the same one this module's own docstring claims for
the MCP server above. Splitting them across ``agent/tools.py`` and
``agent/runtime.py`` would leave the cardinality-map write rule with two
places to go stale independently the first time it changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memgit.core.cardinality import CardinalityMap
from memgit.core.fact import Fact, FactKey
from memgit.core.state import MemoryState

__all__ = [
    "REMEMBER_SCHEMA",
    "FORGET_SCHEMA",
    "RECALL_SCHEMA",
    "TOOL_SCHEMAS",
    "WRITE_TOOL_SCHEMAS",
    "RememberCall",
    "ForgetCall",
    "RecallCall",
    "ToolCallError",
    "decode_remember",
    "decode_forget",
    "decode_recall",
    "apply_remember",
    "apply_forget",
    "tool_schemas",
]

REMEMBER_SCHEMA: dict[str, Any] = {
    "name": "remember",
    "description": (
        "Record one durable fact into long-term memory as a "
        "(subject, predicate, object) triple. Call this once per fact — call "
        "it several times in one turn to record several facts. Reuse an "
        "existing subject or predicate exactly when one already applies: "
        "(subject, predicate) is how two commits get compared, so a new "
        "spelling of an existing concept is invisible to that comparison. "
        "Record only durable, reusable facts — never the content of this "
        "conversation or an intermediate result."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "What the claim is about, e.g. 'user'.",
            },
            "predicate": {
                "type": "string",
                "description": "The relation asserted, e.g. 'prefers_language'.",
            },
            "object": {
                "type": "string",
                "description": "The value of the relation, e.g. 'Python'.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "How sure you are, honestly, in [0.0, 1.0]. 1.0 means the "
                    "user asserted this directly."
                ),
            },
            "source_text": {
                "type": "string",
                "description": "The original claim this triple was derived from, verbatim.",
            },
        },
        "required": ["subject", "predicate", "object", "confidence", "source_text"],
        "additionalProperties": False,
    },
}

FORGET_SCHEMA: dict[str, Any] = {
    "name": "forget",
    "description": (
        "Retract a belief that is no longer true. Use this only for "
        "retraction, not revision — if a belief was replaced by a new value, "
        "call remember with the new value instead, since forget would erase "
        "the fact that a revision happened at all."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "predicate": {"type": "string"},
            "object": {
                "type": ["string", "null"],
                "description": (
                    "Retract only this value at (subject, predicate) — the "
                    "multi-valued case. null retracts every value there."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Why this is no longer true.",
            },
        },
        "required": ["subject", "predicate", "object", "reason"],
        "additionalProperties": False,
    },
}

RECALL_SCHEMA: dict[str, Any] = {
    "name": "recall",
    "description": (
        "Search your long-term memory for facts relevant to a query. The "
        "facts shown in your system prompt are only the subset most "
        "relevant to this turn — use recall before concluding you have no "
        "belief about something, or before contradicting a key you can see "
        "in the key inventory but wasn't shown in full. Read-only: it never "
        "changes memory."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, in natural language.",
            },
            "subject": {
                "type": ["string", "null"],
                "description": "Restrict the search to this subject, or null to search every subject.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
                "description": "How many facts to return.",
            },
        },
        "required": ["query", "subject", "limit"],
        "additionalProperties": False,
    },
}

WRITE_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (REMEMBER_SCHEMA, FORGET_SCHEMA)
TOOL_SCHEMAS: tuple[dict[str, Any], ...] = WRITE_TOOL_SCHEMAS


def tool_schemas(*, recall: bool) -> tuple[dict[str, Any], ...]:
    """The tool schemas to offer this turn.

    ``recall`` should track whether the prompt is in retrieval mode
    (``runtime.py`` passes ``retriever is not None``) — offering it under
    full injection would give the model a redundant way to see what it was
    already shown in full.
    """
    return (*WRITE_TOOL_SCHEMAS, RECALL_SCHEMA) if recall else WRITE_TOOL_SCHEMAS


class ToolCallError(ValueError):
    """A tool call's input could not be turned into a memory mutation.

    Caught by the runtime and turned into an ``is_error`` tool result rather
    than aborting the turn — the message is written for a human by
    :class:`~memgit.core.fact.Fact`'s own validation, and reads fine to a
    model asked to try again.
    """


@dataclass(frozen=True, slots=True)
class RememberCall:
    """A decoded ``remember`` call: the fact it asks to add or reaffirm."""

    fact: Fact


@dataclass(frozen=True, slots=True)
class ForgetCall:
    """A decoded ``forget`` call.

    Attributes:
        key: The ``(subject, predicate)`` to retract from.
        object: The single value to retract, or ``None`` to retract every
            value at ``key`` — the same distinction
            :meth:`MemoryState.without_fact` vs. :meth:`MemoryState.without`
            already draws.
        reason: Why, for the record.
    """

    key: FactKey
    object: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class RecallCall:
    """A decoded ``recall`` call: what to search for, and how narrowly.

    Attributes:
        subject: Restrict the search to this subject, or ``None`` for every
            subject in scope.
        limit: How many facts the caller asked for.
    """

    query: str
    subject: str | None
    limit: int


def decode_remember(payload: dict[str, Any], *, source: str) -> RememberCall:
    """Turn a ``remember`` tool call's input into a :class:`RememberCall`.

    Args:
        payload: The tool call's already-JSON-parsed input.
        source: Where this call came from, e.g. ``"session:.../turn:3"`` —
            supplied by the runtime, never by the model itself: provenance
            of *who is talking to the store* is not the model's to author.

    Raises:
        ToolCallError: a required key is missing, or the resulting
            :class:`Fact` fails its own validation.
    """
    try:
        subject = payload["subject"]
        predicate = payload["predicate"]
        object_ = payload["object"]
        confidence = payload["confidence"]
        source_text = payload["source_text"]
    except KeyError as exc:
        raise ToolCallError(f"remember: missing required field {exc.args[0]!r}") from exc

    try:
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=object_,
            confidence=confidence,
            source=source,
            source_text=source_text,
        )
    except (TypeError, ValueError) as exc:
        raise ToolCallError(f"remember: {exc}") from exc

    return RememberCall(fact=fact)


def decode_forget(payload: dict[str, Any]) -> ForgetCall:
    """Turn a ``forget`` tool call's input into a :class:`ForgetCall`.

    Raises:
        ToolCallError: a required key is missing.
    """
    try:
        subject = payload["subject"]
        predicate = payload["predicate"]
        reason = payload["reason"]
    except KeyError as exc:
        raise ToolCallError(f"forget: missing required field {exc.args[0]!r}") from exc

    if not isinstance(subject, str) or not subject.strip():
        raise ToolCallError("forget: subject must be a non-empty string")
    if not isinstance(predicate, str) or not predicate.strip():
        raise ToolCallError("forget: predicate must be a non-empty string")

    object_ = payload.get("object")
    if object_ is not None and not isinstance(object_, str):
        raise ToolCallError("forget: object must be a string or null")

    return ForgetCall(key=(subject, predicate), object=object_, reason=reason)


def decode_recall(payload: dict[str, Any]) -> RecallCall:
    """Turn a ``recall`` tool call's input into a :class:`RecallCall`.

    Raises:
        ToolCallError: a required key is missing, or a field has the wrong
            shape.
    """
    try:
        query = payload["query"]
        subject = payload["subject"]
        limit = payload["limit"]
    except KeyError as exc:
        raise ToolCallError(f"recall: missing required field {exc.args[0]!r}") from exc

    if not isinstance(query, str) or not query.strip():
        raise ToolCallError("recall: query must be a non-empty string")
    if subject is not None and not isinstance(subject, str):
        raise ToolCallError("recall: subject must be a string or null")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ToolCallError("recall: limit must be a positive integer")

    return RecallCall(query=query, subject=subject, limit=limit)


def apply_remember(state: MemoryState, call: RememberCall, cardinality: CardinalityMap) -> MemoryState:
    """Fold one ``remember`` onto ``state``, per the repo's cardinality map.

    The exact same triple already present is a reaffirmation (replace, so
    the diff engine reports ``reaffirmed`` rather than a spurious
    duplicate). Otherwise: a ``single`` predicate replaces whatever was at
    the key (``contradicted``); a ``multi`` predicate adds alongside
    (``value_added``). Consulting the cardinality map here does not violate
    "a diff is a question you ask, not data you store" — a caller applying
    this fold is a client of that repo-local lens exactly as ``memgit diff``
    is, and nothing about this merge is itself hashed or committed.
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


def apply_forget(state: MemoryState, call: ForgetCall) -> MemoryState:
    """Fold one ``forget`` onto ``state``."""
    if call.object is None:
        return state.without(call.key)
    subject, predicate = call.key
    result = state
    for fact in state.get(subject, predicate):
        if fact.object == call.object:
            result = result.without_fact(fact.hash)
    return result
