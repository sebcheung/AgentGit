"""The diff engine — comparing two memory states.

Everything here is written against plain :class:`~memgit.core.tree.Tree`
objects, exactly the pattern ``graph.py`` uses for ``CommitReader``: the
algorithm never touches a filesystem, so it stays testable against synthetic
dicts. ``Repository`` is the only thing that knows a tree actually comes from
``.memgit/objects``.

The comparison starts with one merge-join over the two trees' sorted entries
(``_merge_join``) — the linear algorithm ``tree.py`` promises the diff engine.
Everything else in this module, including the full semantic diff added
alongside it, is built on top of that single join so there is exactly one way
keys get aligned.

``diff_hashes`` is the cheap half: a hash-only comparison. It can say a key
was added, removed, or "something changed" — but it cannot tell a
contradiction from a reaffirmation, since that distinction depends on
``Fact.object``, not on the hash. No fact is ever read to produce it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterator

from memgit.core.fact import FactKey
from memgit.core.tree import Tree

__all__ = [
    "HashKeyChange",
    "HashKeyEntry",
    "HashDiff",
    "diff_hashes",
]


# ---------------------------------------------------------------------------
# The merge-join
# ---------------------------------------------------------------------------


def _merge_join(
    before: Tree, after: Tree
) -> Iterator[tuple[FactKey, tuple[str, ...] | None, tuple[str, ...] | None]]:
    """Yield every key in either tree, exactly once, in sorted order.

    Linear in ``len(before) + len(after)``: one pass over both trees' already
    -sorted entries, no dict built, no per-key allocation beyond the yielded
    tuple. This is the merge-join ``tree.py`` promises the diff engine.

    The comparison ``ka < kb`` below must be the identical ordering
    ``Tree.from_entries`` used (``sorted(grouped.items())`` over
    ``(subject, predicate)`` tuples) — it is, since both are plain Python
    tuple ordering over the same pair. That equivalence is not cosmetic: a
    concatenated ``f"{subject}\\t{predicate}"`` comparison would order
    ``("a", "b/c")`` and ``("a/b", "c")`` differently and silently mis-align
    two trees at that key.
    """
    a, b = before.entries, after.entries
    i = j = 0
    while i < len(a) and j < len(b):
        ka = (a[i][0], a[i][1])
        kb = (b[j][0], b[j][1])
        if ka == kb:
            yield ka, a[i][2], b[j][2]
            i += 1
            j += 1
        elif ka < kb:
            yield ka, a[i][2], None
            i += 1
        else:
            yield kb, None, b[j][2]
            j += 1
    while i < len(a):
        yield (a[i][0], a[i][1]), a[i][2], None
        i += 1
    while j < len(b):
        yield (b[j][0], b[j][1]), None, b[j][2]
        j += 1


# ---------------------------------------------------------------------------
# Hash-only diff
# ---------------------------------------------------------------------------


class HashKeyChange(StrEnum):
    """What changed at a key, without reading any fact — see module docstring."""

    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


HashKeyEntry = tuple[FactKey, HashKeyChange, tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class HashDiff:
    """The cheap comparison: which keys differ, without saying why.

    Powers callers that don't need the semantic detail: ``--name-only``, a
    dashboard's "how many keys did this commit touch" over hundreds of
    commits, and slice 4's checkout deciding what to materialize.
    """

    entries: tuple[HashKeyEntry, ...]

    def changed_keys(self) -> tuple[FactKey, ...]:
        return tuple(key for key, kind, _b, _a in self.entries if kind != HashKeyChange.UNCHANGED)

    def __iter__(self) -> Iterator[HashKeyEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def diff_hashes(before: Tree, after: Tree) -> HashDiff:
    """Compare two trees by hash alone. Reads no facts."""
    entries: list[HashKeyEntry] = []
    for key, before_hashes, after_hashes in _merge_join(before, after):
        b = before_hashes or ()
        a = after_hashes or ()
        if before_hashes is None:
            kind = HashKeyChange.ADDED
        elif after_hashes is None:
            kind = HashKeyChange.REMOVED
        elif b == a:
            kind = HashKeyChange.UNCHANGED
        else:
            kind = HashKeyChange.CHANGED
        entries.append((key, kind, b, a))
    return HashDiff(tuple(entries))
