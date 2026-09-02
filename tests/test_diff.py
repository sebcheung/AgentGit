"""Tests for the diff engine, against synthetic dict fixtures.

No filesystem: diff_trees/diff_hashes only need a Tree and a FactReader
callable, and these tests hold them to that — the direct analogue of
test_graph.py's approach to walk(). Grouped by the properties that carry the
design: the merge-join's totality and ordering, the classification rules
under both cardinalities, precedence when several things happen to one key at
once, duplicate-object shadowing, the empty-tree baseline, and laziness (an
unchanged key costs zero fact reads).
"""

from __future__ import annotations

import pytest

from memgit.core.cardinality import CardinalityMap
from memgit.core.diff import (
    ChangeKind,
    ValueKind,
    diff_hashes,
    diff_trees,
)
from memgit.core.fact import Fact
from memgit.core.tree import EMPTY_TREE_HASH, Tree


def make_fact(**overrides) -> Fact:
    defaults = dict(
        subject="user",
        predicate="prefers_language",
        object="Python",
        confidence=0.9,
        asserted_at="2026-08-11T12:00:00+00:00",
    )
    defaults.update(overrides)
    return Fact(**defaults)


class CountingFactReader:
    """Wraps a dict-backed reader and counts calls, to prove laziness."""

    def __init__(self, facts: dict[str, Fact]) -> None:
        self._facts = facts
        self.calls = 0

    def __call__(self, fact_hash: str) -> Fact:
        self.calls += 1
        return self._facts[fact_hash]


def tree_and_reader(*facts: Fact) -> tuple[Tree, CountingFactReader]:
    return Tree.from_facts(facts), CountingFactReader({f.hash: f for f in facts})


class TestMergeJoin:
    def test_every_key_from_either_tree_appears_exactly_once(self):
        a = make_fact(subject="user", predicate="a", object="1")
        b = make_fact(subject="user", predicate="b", object="1")
        c = make_fact(subject="user", predicate="c", object="1")
        before = Tree.from_facts([a, b])
        after = Tree.from_facts([b, c])
        reader = CountingFactReader({f.hash: f for f in (a, b, c)})

        result = diff_trees(before, after, reader, include_unchanged=True)
        assert {kd.key for kd in result.keys} == {("user", "a"), ("user", "b"), ("user", "c")}

    def test_output_order_matches_sorted_union_of_keys(self):
        a = make_fact(subject="z", predicate="p", object="1")
        b = make_fact(subject="a", predicate="p", object="1")
        c = make_fact(subject="m", predicate="p", object="1")
        before = Tree.from_facts([a, c])
        after = Tree.from_facts([b, c])
        reader = CountingFactReader({f.hash: f for f in (a, b, c)})

        result = diff_trees(before, after, reader, include_unchanged=True)
        assert [kd.key for kd in result.keys] == sorted(kd.key for kd in result.keys)

    def test_one_side_empty_all_added(self):
        facts = [make_fact(subject="a", predicate="p", object="1"), make_fact(subject="b", predicate="p", object="1")]
        after, reader = tree_and_reader(*facts)
        result = diff_trees(Tree(()), after, reader)
        assert all(kd.kind == ChangeKind.ADDED for kd in result.keys)

    def test_one_side_empty_all_removed(self):
        facts = [make_fact(subject="a", predicate="p", object="1"), make_fact(subject="b", predicate="p", object="1")]
        before, reader = tree_and_reader(*facts)
        result = diff_trees(before, Tree(()), reader)
        assert all(kd.kind == ChangeKind.REMOVED for kd in result.keys)

    def test_both_empty_no_entries(self):
        result = diff_trees(Tree(()), Tree(()), lambda h: (_ for _ in ()).throw(AssertionError("no reads")))
        assert result.is_empty

    def test_sort_key_is_tuple_ordering_not_concatenation(self):
        # ("a", "b/c") vs ("a/b", "c"): concatenating with "/" would tie or
        # mis-order these; tuple ordering (matching Tree.from_entries) must
        # keep them distinct and in the tuple-ordered position.
        left = make_fact(subject="a", predicate="b/c", object="1")
        right = make_fact(subject="a/b", predicate="c", object="1")
        before, reader = tree_and_reader(left)
        after = Tree.from_facts([right])
        combined_reader = CountingFactReader({left.hash: left, right.hash: right})

        result = diff_trees(before, after, combined_reader, include_unchanged=True)
        assert [kd.key for kd in result.keys] == sorted([("a", "b/c"), ("a/b", "c")])

    def test_diff_hashes_reads_no_facts(self):
        before, _ = tree_and_reader(make_fact(object="Python"))
        after, _ = tree_and_reader(make_fact(object="Rust"))
        result = diff_hashes(before, after)
        assert len(result) == 1


class TestEmptyTreeDiff:
    def test_empty_before_all_added(self):
        fact = make_fact()
        after = Tree.from_facts([fact])
        reader = CountingFactReader({fact.hash: fact})
        result = diff_trees(Tree(()), after, reader)
        assert result.keys[0].kind == ChangeKind.ADDED

    def test_empty_after_all_removed(self):
        fact = make_fact()
        before = Tree.from_facts([fact])
        reader = CountingFactReader({fact.hash: fact})
        result = diff_trees(before, Tree(()), reader)
        assert result.keys[0].kind == ChangeKind.REMOVED

    def test_both_empty_is_empty(self):
        result = diff_trees(Tree(()), Tree(()), lambda h: (_ for _ in ()).throw(AssertionError()))
        assert result.is_empty

    def test_empty_tree_hash_matches_constant(self):
        assert Tree(()).hash == EMPTY_TREE_HASH


