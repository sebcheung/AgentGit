"""System-prompt construction: the frozen preamble and the volatile memory block.

``MemoryState.render()`` is documented in ``state.py`` as "the context block
slice 5 injects verbatim" — this module is that injection, plus the two
pieces of framing a raw render doesn't carry on its own: what the agent is
for, and which predicates the repository has declared ``multi`` (so the
model knows a second value at a key coexists rather than overwrites).

The prompt is two ``system`` blocks, not one, split at a cache breakpoint:
static instructions first (``cache_control: ephemeral``), the rendered
memory state after. The state changes every turn — a fact ``remember``-ed in
turn *n* must be visible in turn *n+1*'s prompt, since memory crossing a
process boundary through commits rather than through the chat transcript is
the whole point (see ``runtime.py``) — so it has to sit after the last
breakpoint, or every turn would invalidate the cached prefix for nothing.

**Slice 7 adds a third, optional mode: scoped retrieval.** Pass a
``retrieved`` :class:`~memgit.retrieval.rank.RetrievalResult` to replace full
injection with the top-k facts most relevant to this turn's query — three
blocks instead of two, still after the cache breakpoint. Retrieval's real
danger is not a missing fact, it's *silent predicate drift*: a model that
only sees a subset will invent a near-duplicate predicate for a belief it
already holds but wasn't shown, and that lands in the diff engine as the
taxonomy's most reassuring label (``added``, "an unrelated new fact") rather
than the drift it actually is. The fix is a full ``(subject, predicate)`` key
inventory, always present alongside the retrieved subset — cheap
(``MemoryState.keys()`` is already computed) and it is what lets the model
recognize "I have a belief about this, I just wasn't shown it" instead of
inventing a new key.

``retrieved=None`` (the default) reproduces slice 5's exact two-block output,
byte for byte — every existing prompt test is a regression lock on that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from memgit.core.cardinality import CardinalityMap
from memgit.core.state import MemoryState

if TYPE_CHECKING:
    from memgit.retrieval.rank import RetrievalResult

__all__ = ["build_system_prompt"]

_INSTRUCTIONS = """\
You are an agent whose long-term memory is version-controlled. Every fact \
you record with `remember` becomes part of a commit a human can read, diff, \
and roll back later — so record durable, reusable facts about the user or \
the world, never the content of this conversation or an intermediate \
result.

State your confidence honestly rather than defaulting to certainty, and \
quote the user's own words into `source_text` verbatim.

Reuse an existing subject or predicate exactly when one already applies — \
`(subject, predicate)` is the key a diff lines facts up by, so a new \
spelling of an existing concept is invisible to that comparison. The memory \
below groups facts by subject; check it before inventing a new predicate.

Use `forget` only to retract a belief that is no longer true, not to revise \
one — a revision is a new `remember` at the same key."""

_RETRIEVAL_NOTE = """\
The facts below are only the subset most relevant to this turn, not \
everything you know — the key inventory lists every (subject, predicate) \
you have ever recorded, even keys not shown in full below. Absence from the \
facts block does not mean you have no belief about something: call `recall` \
before concluding that, and before contradicting a key listed in the \
inventory."""


def build_system_prompt(
    state: MemoryState,
    cardinality: CardinalityMap,
    *,
    head: str | None,
    retrieved: RetrievalResult | None = None,
) -> list[dict[str, Any]]:
    """Build the system prompt for one turn — two blocks, or three under retrieval.

    Args:
        state: The memory state at ``HEAD`` this turn will read from. Always
            used for the key inventory under retrieval; used for full
            injection when ``retrieved`` is ``None``.
        cardinality: The repository's single/multi map — declared here so
            the model knows which predicates coexist rather than overwrite.
        head: The commit the state came from, for each block's own label
            (``None`` on an unborn repository).
        retrieved: If given, replace full-state injection with this
            pre-computed top-k result and add a full key-inventory block
            between the instructions and the retrieved facts. ``None``
            (the default) reproduces the pre-slice-7 two-block prompt
            exactly.
    """
    multi = [predicate for predicate, kind in cardinality.declared() if kind == "multi"]
    cardinality_line = (
        f"Predicates that hold several coexisting values at once: {', '.join(multi)}."
        if multi
        else "No predicate is declared to hold several coexisting values; a second "
        "value at any key reads as a contradiction of the first."
    )

    label = f"HEAD ({head[:8]})" if head else "HEAD (no commits yet)"

    instructions_text = _INSTRUCTIONS
    if retrieved is not None:
        instructions_text = f"{instructions_text}\n\n{_RETRIEVAL_NOTE}"

    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{instructions_text}\n\n{cardinality_line}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    if retrieved is None:
        rendered = state.render()
        memory_block = (
            f"# Memory at {label}\n{rendered}" if rendered else f"# Memory at {label}\nNo memories recorded yet."
        )
        blocks.append({"type": "text", "text": memory_block})
        return blocks

    keys = state.keys()
    if keys:
        inventory_lines = "\n".join(f"  {subject} {predicate}" for subject, predicate in keys)
    else:
        inventory_lines = "No memories recorded yet."
    blocks.append(
        {"type": "text", "text": f"# Every known (subject, predicate) key at {label}\n{inventory_lines}"}
    )

    header = f"# Memory relevant to this turn — {len(retrieved.facts)} of {retrieved.candidates} fact(s) at {label}"
    blocks.append({"type": "text", "text": f"{header}\n{retrieved.render()}"})

    return blocks
