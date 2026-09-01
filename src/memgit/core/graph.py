"""Traversal over the commit DAG.

Everything in this module is written against a plain ``CommitReader``
callable — "give me the commit for this hash" — rather than against
``Repository``. That keeps the trickiest algorithms in the project testable
against synthetic dicts, with no filesystem involved at all, which matters
because this is exactly the module future slices will keep extending.

``walk`` orders commits most-recent-first by comparing ``committed_at``,
mirroring git's ``--date-order``. That ordering has a known gap, and it is
worth stating plainly rather than pretending otherwise: date order does not
*guarantee* a parent is emitted after its child if two commits' clocks are
skewed or their timestamps tie. Git has the identical problem and answers it
with a separate ``--topo-order`` pass when it matters. Here, every timestamp
comes from one process on one machine, so skew is close to impossible, and
ties are broken on hash to keep the order at least deterministic — which is
what the test suite actually needs. A Kahn-style topological pass is the
honest fix if the dashboard's graph rendering ever needs a hard guarantee;
it is noted here, not built.

``merge_base`` (lowest common ancestor) is deliberately not in this module
yet. Nothing before the diff engine (slice 3, where "diff between two
branches" is the feature that needs it) calls it, and a *correct*
multi-candidate LCA over a DAG wants generation numbers or careful
multi-source painting — code that is hard to trust without the branchy
fixtures that don't exist until there's a reason to merge something. The one
thing this slice owes that future work is that ``Commit.parents`` is already
a list.
"""

from __future__ import annotations

import heapq
from typing import Callable, Iterable, Iterator

from memgit.core.commit import Commit

__all__ = ["CommitReader", "walk", "ancestors", "is_ancestor"]

CommitReader = Callable[[str], Commit]


class _Newest:
    """Wraps a timestamp so it sorts backwards in ``heapq``'s min-heap.

    ``heapq`` only ever gives you the smallest item; getting "most recent
    first" out of it means the comparison itself has to be inverted, which a
    numeric negation can't do for an ISO-8601 string. This is the smallest
    way to say "greater timestamps compare as smaller" without reformatting
    the timestamp itself.
    """

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __lt__(self, other: "_Newest") -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Newest) and self.value == other.value


def walk(
    start: str | Iterable[str],
    read: CommitReader,
    *,
    limit: int | None = None,
) -> Iterator[tuple[str, Commit]]:
    """Yield ``(hash, commit)`` pairs reachable from ``start``, most-recent-first.

    Lazy: nothing beyond what has already been yielded (plus its immediate
    parents) is read via ``read``. A diamond — two branches merging back into
    a shared ancestor — is visited exactly once. ``limit`` stops both the
    yielding and the reading; asking for one commit does not walk the whole
    history first.

    Args:
        start: A single starting hash, or several (e.g. a merge's parents).
        read: Loads a :class:`~memgit.core.commit.Commit` given its hash.
        limit: Stop after yielding this many commits. ``None`` for all of them.
    """
    starts = [start] if isinstance(start, str) else list(start)

    # Most-recent-first via a min-heap on (_Newest(committed_at), hash): the
    # timestamp comparison is inverted by _Newest, and the hash is the
    # deterministic tie-break the module docstring's clock-skew caveat
    # depends on.
    heap: list[tuple[_Newest, str]] = []
    seen: set[str] = set()
    loaded: dict[str, Commit] = {}

    def push(commit_hash: str) -> None:
        if commit_hash in seen:
            return
        seen.add(commit_hash)
        commit = read(commit_hash)
        loaded[commit_hash] = commit
        heapq.heappush(heap, (_Newest(commit.committed_at), commit_hash))

    for commit_hash in starts:
        push(commit_hash)

    yielded = 0
    while heap and (limit is None or yielded < limit):
        _key, commit_hash = heapq.heappop(heap)
        commit = loaded.pop(commit_hash)
        yield commit_hash, commit
        yielded += 1
        for parent in commit.parents:
            push(parent)


def ancestors(start: str, read: CommitReader) -> set[str]:
    """Every commit reachable from ``start``, including ``start`` itself."""
    return {commit_hash for commit_hash, _commit in walk(start, read)}


def is_ancestor(candidate: str, of: str, read: CommitReader) -> bool:
    """Whether ``candidate`` is ``of`` itself or one of its ancestors."""
    return candidate in ancestors(of, read)
