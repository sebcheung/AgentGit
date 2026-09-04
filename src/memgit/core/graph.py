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

``merge_base``/``merge_bases`` implement the lowest common ancestor(s) via
two-source painting (git's ``paint_down_to_common``), because the diff engine
(slice 3) needs a fork point to answer "what did *this* branch do?" without
also reporting facts the other branch simply hasn't received yet. Two honest
gaps carry over from ``walk``'s:

**Clock skew hits termination, not just ordering.** The stale-queue
termination below is a date-ordered heap, so it inherits ``walk``'s caveat:
skewed timestamps could in principle stop the walk early. Git had the
identical bug and fixed it with generation numbers; the same mitigation
applies here (one process, one machine), and the same honest fix is named,
not built.

**Multi-candidate bases are returned, not resolved.** On a criss-cross
history, more than one commit can be a valid merge base with neither
provably better than the other. ``merge_bases`` returns all survivors;
``merge_base`` picks one deterministically by date. Git's answer for a
criss-cross is to synthesize a virtual base by recursively merging the
candidates — that needs a merge algorithm this project has no slice for, so
the ambiguity is surfaced to the caller instead of quietly resolved.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable, Iterator

from memgit.core.commit import Commit

__all__ = ["CommitReader", "ancestors", "is_ancestor", "merge_base", "merge_bases", "walk"]

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

    def __lt__(self, other: _Newest) -> bool:
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


_REACH_A = 1
_REACH_B = 2
_RESULT = 4
_STALE = 8


def merge_bases(a: str, b: str, read: CommitReader) -> tuple[str, ...]:
    """Every lowest common ancestor of ``a`` and ``b``, newest-first.

    Two-source painting: walk both histories at once in date order, flagging
    each commit with which side(s) can reach it. The first commit reachable
    from both sides is a merge base; ``_STALE`` propagates from it to its own
    ancestors so they are not reported too. The walk stops as soon as every
    commit still queued is stale — the laziness property that keeps this from
    reading the whole history on a shallow fork.

    A final reduce pass drops any candidate that is itself an ancestor of
    another candidate — a backstop for graphs stale-propagation alone
    wouldn't cleanly resolve, turning "plausible" into "correct".

    Two independently-rooted histories are a legitimate state, not an error:
    this returns ``()`` for them.
    """
    flags: dict[str, int] = {}
    loaded: dict[str, Commit] = {}
    heap: list[tuple[_Newest, str]] = []
    queued: set[str] = set()

    def push(commit_hash: str, flag: int) -> None:
        flags[commit_hash] = flags.get(commit_hash, 0) | flag
        if commit_hash in queued:
            return
        queued.add(commit_hash)
        commit = read(commit_hash)
        loaded[commit_hash] = commit
        heapq.heappush(heap, (_Newest(commit.committed_at), commit_hash))

    push(a, _REACH_A)
    push(b, _REACH_B)

    candidates: list[str] = []
    while heap:
        if all(flags[h] & _STALE for _key, h in heap):
            break

        _key, commit_hash = heapq.heappop(heap)
        queued.discard(commit_hash)
        commit = loaded[commit_hash]
        own_flags = flags[commit_hash]

        is_candidate = (
            own_flags & _REACH_A and own_flags & _REACH_B and not own_flags & (_RESULT | _STALE)
        )
        if is_candidate:
            own_flags |= _RESULT | _STALE
            flags[commit_hash] = own_flags
            candidates.append(commit_hash)

        propagate = own_flags & (_REACH_A | _REACH_B | _STALE)
        for parent in commit.parents:
            push(parent, propagate)

    reduced = [
        h for h in candidates if not any(other != h and is_ancestor(h, other, read) for other in candidates)
    ]
    reduced.sort(key=lambda h: (loaded[h].committed_at, h), reverse=True)
    return tuple(reduced)


def merge_base(a: str, b: str, read: CommitReader) -> str | None:
    """One lowest common ancestor of ``a`` and ``b``, chosen deterministically.

    ``None`` if the two histories share no common ancestor. On a criss-cross
    history where more than one base exists, this picks the newest by date
    (hash tie-broken) — see the module docstring for why that ambiguity is
    surfaced rather than resolved.
    """
    bases = merge_bases(a, b, read)
    return bases[0] if bases else None
