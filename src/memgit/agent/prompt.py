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
"""

from __future__ import annotations

from typing import Any

from memgit.core.cardinality import CardinalityMap
from memgit.core.state import MemoryState

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


def build_system_prompt(
    state: MemoryState, cardinality: CardinalityMap, *, head: str | None
) -> list[dict[str, Any]]:
    """Build the two-block system prompt for one turn.

    Args:
        state: The memory state at ``HEAD`` this turn will read from.
        cardinality: The repository's single/multi map — declared here so
            the model knows which predicates coexist rather than overwrite.
        head: The commit the state came from, for the block's own label
            (``None`` on an unborn repository).
    """
    multi = [predicate for predicate, kind in cardinality.declared() if kind == "multi"]
    cardinality_line = (
        f"Predicates that hold several coexisting values at once: {', '.join(multi)}."
        if multi
        else "No predicate is declared to hold several coexisting values; a second "
        "value at any key reads as a contradiction of the first."
    )

    label = f"HEAD ({head[:8]})" if head else "HEAD (no commits yet)"
    rendered = state.render()
    memory_block = (
        f"# Memory at {label}\n{rendered}" if rendered else f"# Memory at {label}\nNo memories recorded yet."
    )

    return [
        {
            "type": "text",
            "text": f"{_INSTRUCTIONS}\n\n{cardinality_line}",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": memory_block},
    ]
