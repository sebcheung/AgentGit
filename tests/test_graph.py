"""Tests for commit-graph traversal, against synthetic dict fixtures.

No filesystem and no Repository involved — walk/ancestors/is_ancestor only
depend on a CommitReader callable, and these tests hold it to that. Grouped
by the properties that carry the design: ordering, termination on cycles-free
DAGs (a diamond visited once), laziness (limit actually stops reading), and
ancestry queries.
"""

from __future__ import annotations

import pytest

from memgit.core.commit import Commit
from memgit.core.graph import ancestors, is_ancestor, walk

EMPTY_TREE = "0" * 64


def commit_hash(commit: Commit) -> str:
    return commit.hash


def build_chain(n: int, start_time: str = "2026-01-01T00:00:00+00:00") -> tuple[dict[str, Commit], list[str]]:
    """A linear chain of n commits, oldest first; returns (store, hashes oldest->newest)."""
    from datetime import datetime, timedelta, timezone

    store: dict[str, Commit] = {}
    hashes: list[str] = []
    parent: tuple[str, ...] = ()
    base = datetime.fromisoformat(start_time)
    for i in range(n):
        ts = (base + timedelta(minutes=i)).isoformat()
        commit = Commit(tree=EMPTY_TREE, parents=parent, message=f"commit {i}", committed_at=ts)
        h = commit.hash
        store[h] = commit
        hashes.append(h)
        parent = (h,)
    return store, hashes


class CountingReader:
    """Wraps a dict-backed reader and counts calls, to prove laziness."""

    def __init__(self, store: dict[str, Commit]) -> None:
        self._store = store
        self.calls = 0

    def __call__(self, commit_hash: str) -> Commit:
        self.calls += 1
        return self._store[commit_hash]


class TestOrdering:
    def test_linear_chain_is_reverse_chronological(self):
        store, hashes = build_chain(4)
        read = CountingReader(store)
        walked = [h for h, _c in walk(hashes[-1], read)]
        assert walked == list(reversed(hashes))

    def test_deterministic_under_tied_timestamps(self):
        tie = "2026-01-01T00:00:00+00:00"
        c1 = Commit(tree=EMPTY_TREE, parents=(), message="a", committed_at=tie)
        c2 = Commit(tree=EMPTY_TREE, parents=(), message="b", committed_at=tie)
        store = {c1.hash: c1, c2.hash: c2}
        read = CountingReader(store)

        order_1 = [h for h, _c in walk([c1.hash, c2.hash], read)]
        order_2 = [h for h, _c in walk([c1.hash, c2.hash], read)]
        # Whichever direction the hash tie-break goes, it must be the same
        # every time — that determinism, not a specific direction, is the
        # property under test.
        assert order_1 == order_2
        assert set(order_1) == {c1.hash, c2.hash}


class TestDiamond:
    def test_each_commit_visited_exactly_once(self):
        root = Commit(tree=EMPTY_TREE, parents=(), message="root", committed_at="2026-01-01T00:00:00+00:00")
        left = Commit(
            tree=EMPTY_TREE, parents=(root.hash,), message="left", committed_at="2026-01-01T00:01:00+00:00"
        )
        right = Commit(
            tree=EMPTY_TREE, parents=(root.hash,), message="right", committed_at="2026-01-01T00:01:00+00:00"
        )
        merge = Commit(
            tree=EMPTY_TREE,
            parents=(left.hash, right.hash),
            message="merge",
            committed_at="2026-01-01T00:02:00+00:00",
        )
        store = {c.hash: c for c in (root, left, right, merge)}
        read = CountingReader(store)

        walked = [h for h, _c in walk(merge.hash, read)]
        assert sorted(walked) == sorted(store.keys())
        assert len(walked) == len(set(walked))
        assert walked[0] == merge.hash
        assert walked[-1] == root.hash


class TestLaziness:
    def test_limit_stops_yielding_early(self):
        store, hashes = build_chain(10)
        read = CountingReader(store)
        walked = [h for h, _c in walk(hashes[-1], read, limit=3)]
        assert len(walked) == 3
        assert walked == list(reversed(hashes))[:3]

    def test_limit_stops_reading_early(self):
        """The property that proves walk is lazy, not just truncated output."""
        store, hashes = build_chain(10)
        read = CountingReader(store)
        list(walk(hashes[-1], read, limit=3))
        # Each commit is read exactly once when pushed onto the heap; asking
        # for 3 of 10 must not have read anywhere near all 10.
        assert read.calls <= 4

    def test_each_commit_is_read_exactly_once(self):
        store, hashes = build_chain(5)
        read = CountingReader(store)
        list(walk(hashes[-1], read))
        assert read.calls == 5


class TestMissingParent:
    def test_raises_rather_than_yielding_a_half_graph(self):
        commit = Commit(
            tree=EMPTY_TREE, parents=("f" * 64,), message="orphaned", committed_at="2026-01-01T00:00:00+00:00"
        )
        store = {commit.hash: commit}
        read = CountingReader(store)
        with pytest.raises(KeyError):
            list(walk(commit.hash, read))


class TestAncestry:
    def test_ancestors_includes_self(self):
        store, hashes = build_chain(3)
        read = CountingReader(store)
        assert hashes[-1] in ancestors(hashes[-1], read)

    def test_ancestors_includes_every_earlier_commit(self):
        store, hashes = build_chain(3)
        read = CountingReader(store)
        assert ancestors(hashes[-1], read) == set(hashes)

    def test_is_ancestor_true_for_an_earlier_commit(self):
        store, hashes = build_chain(3)
        read = CountingReader(store)
        assert is_ancestor(hashes[0], hashes[-1], read)

    def test_is_ancestor_false_for_a_later_commit(self):
        store, hashes = build_chain(3)
        read = CountingReader(store)
        assert not is_ancestor(hashes[-1], hashes[0], read)

    def test_is_ancestor_true_for_self(self):
        store, hashes = build_chain(1)
        read = CountingReader(store)
        assert is_ancestor(hashes[0], hashes[0], read)
