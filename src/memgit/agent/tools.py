"""The tool surface an agent uses to write memory: ``remember`` and ``forget``.

Exactly these two, not more. ``remember`` is the locked-in decision in
PLAN.md — "Agent calls ``remember(...)`` deliberately as a tool: structured
at birth, no extraction step, no parsing noise." ``forget`` earns its place
alongside it because without it a belief can be *revised* (a ``remember`` at
a ``single`` key contradicts the old value) but never *retracted* — and
``removed``/``value_removed`` are two of the diff engine's eight change
kinds. A tool surface that can never emit them would leave a quarter of the
taxonomy dead code on the agent path.

Two tools deliberately **not** here:

- ``reaffirm`` — a ``remember`` of the same triple at a new confidence
  already *is* a reaffirmation; :meth:`Fact.reaffirm` and the diff engine's
  ``reaffirmed`` kind handle it without the model ever having to choose
  between two verbs for one intention.
- ``recall`` — the whole memory state is already injected into the system
  prompt in this slice (see ``prompt.py``). A second read path would let the
  model act on facts it was never shown, which is exactly the kind of leak
  PLAN.md's retrieval-layer decision warns against. It earns its place once
  slice 7 replaces full injection with scoped retrieval.

This module has no dependency on ``Repository``, an LLM client, or the
network — only on :class:`~memgit.core.fact.Fact`. That is deliberate:
slice 8's MCP server is documented in PLAN.md as "moved up — the tool-call
fact-write decision makes this nearly the same code as slice 4," and this is
the code it shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memgit.core.fact import Fact, FactKey

__all__ = [
    "REMEMBER_SCHEMA",
    "FORGET_SCHEMA",
    "TOOL_SCHEMAS",
    "RememberCall",
    "ForgetCall",
    "ToolCallError",
    "decode_remember",
    "decode_forget",
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

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (REMEMBER_SCHEMA, FORGET_SCHEMA)


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
