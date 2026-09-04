"""Tests for the Tree — a snapshot of a memory state.

Three properties carry the design, one class each: canonicality (the same
state hashes the same way no matter how it was built), multimap tolerance
(a key can hold more than one fact hash, which is what protects slice 3's
cardinality work), and structural sharing (two states differing by one fact
share every other entry — the dedup claim this project is partly about).
Serialization strictness and single-key mutation get their own classes too.
"""

from __future__ import annotations

import pytest

from memgit.core.fact import Fact
from memgit.core.store import ObjectStore
from memgit.core.tree import EMPTY_TREE_HASH, Tree


@pytest.fixture
def store(tmp_path) -> ObjectStore:
    return ObjectStore(tmp_path / "objects")


def make_fact(**overrides) -> Fact:
    defaults = {
        "subject": "user",
        "predicate": "prefers_language",
        "object": "Python",
        "confidence": 0.9,
        "asserted_at": "2026-08-11T12:00:00+00:00",
    }
    defaults.update(overrides)
    return Fact(**defaults)


class TestCanonicality:
    def test_same_facts_different_insertion_order_hash_the_same(self):
        a = make_fact(subject="user", predicate="likes", object="chess")
        b = make_fact(subject="user", predicate="likes", object="go")
        c = make_fact(subject="project:memgit", predicate="language", object="Python")

        assert Tree.from_facts([a, b, c]).hash == Tree.from_facts([c, b, a]).hash

    def test_entries_are_sorted_by_subject_then_predicate(self):
        tree = Tree.from_facts(
            [
                make_fact(subject="z", predicate="a", object="1"),
                make_fact(subject="a", predicate="z", object="1"),
                make_fact(subject="a", predicate="a", object="1"),
            ]
        )
        keys = [(s, p) for s, p, _ in tree]
        assert keys == sorted(keys)

    def test_hashes_within_a_key_are_sorted(self):
        a = make_fact(subject="user", predicate="likes", object="chess")
        b = make_fact(subject="user", predicate="likes", object="go")
        tree = Tree.from_facts([a, b])
        (_, _, hashes) = next(iter(tree))
        assert hashes == tuple(sorted(hashes))

    def test_empty_tree_hash_is_stable(self):
        assert Tree(()).hash == EMPTY_TREE_HASH
        assert len(Tree(())) == 0


class TestMultimapTolerance:
    """Regression coverage for the decision that protects slice 3: a key can
    hold more than one fact hash, and the tree does not silently collapse it."""

    def test_two_facts_sharing_a_key_both_survive(self):
        chess = make_fact(subject="user", predicate="likes", object="chess")
        go = make_fact(subject="user", predicate="likes", object="go")
        tree = Tree.from_facts([chess, go])

        assert len(tree) == 1
        by_key = tree.by_key()
        assert set(by_key[("user", "likes")]) == {chess.hash, go.hash}

    def test_from_entries_merges_entries_sharing_a_key(self):
        chess = make_fact(subject="user", predicate="likes", object="chess")
        go = make_fact(subject="user", predicate="likes", object="go")
        tree = Tree.from_entries(
            [
                ("user", "likes", chess.hash),
                ("user", "likes", go.hash),
            ]
        )
        assert len(tree) == 1
        assert set(tree.by_key()[("user", "likes")]) == {chess.hash, go.hash}

    def test_exact_duplicates_collapse(self):
        fact = make_fact()
        tree = Tree.from_facts([fact, fact, fact])
        assert tree.by_key()[fact.key] == (fact.hash,)


class TestStructuralSharing:
    def test_two_trees_differing_by_one_fact_share_every_other_hash(self):
        shared_a = make_fact(subject="user", predicate="timezone", object="EST")
        shared_b = make_fact(subject="project:memgit", predicate="language", object="Python")

        tree_1 = Tree.from_facts([shared_a, shared_b, make_fact(object="Python")])
        tree_2 = Tree.from_facts([shared_a, shared_b, make_fact(object="Rust")])

        keys_1 = tree_1.by_key()
        keys_2 = tree_2.by_key()
        assert keys_1[shared_a.key] == keys_2[shared_a.key]
        assert keys_1[shared_b.key] == keys_2[shared_b.key]
        assert keys_1[("user", "prefers_language")] != keys_2[("user", "prefers_language")]


class TestSerialization:
    def test_round_trips_through_to_dict_and_from_dict(self):
        tree = Tree.from_facts([make_fact(), make_fact(subject="x", predicate="y", object="z")])
        assert Tree.from_dict(tree.to_dict()) == tree

    def test_write_then_read_preserves_hash(self, store):
        tree = Tree.from_facts([make_fact()])
        tree_hash = tree.write(store)
        assert Tree.read(store, tree_hash) == tree
        assert tree_hash == tree.hash

    def test_load_materializes_every_referenced_fact(self, store):
        fact = make_fact()
        store.put(fact.to_dict())
        tree = Tree.from_facts([fact])
        assert tree.load(store) == [fact]

    def test_from_dict_rejects_non_tree_type(self):
        with pytest.raises(ValueError, match="expected a tree object"):
            Tree.from_dict({"type": "commit", "entries": []})

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown keys"):
            Tree.from_dict({"type": "tree", "entries": [], "extra": 1})

    def test_from_dict_rejects_missing_entries(self):
        with pytest.raises(ValueError, match="missing 'entries'"):
            Tree.from_dict({"type": "tree"})

    def test_from_dict_rejects_malformed_entry_shape(self):
        with pytest.raises(ValueError, match="malformed tree entry"):
            Tree.from_dict({"type": "tree", "entries": [["user", "likes"]]})

    def test_from_entries_rejects_non_hex_hash(self):
        with pytest.raises(ValueError, match="malformed fact hash"):
            Tree.from_entries([("user", "likes", "not-a-hash")])

    def test_from_entries_rejects_empty_subject(self):
        with pytest.raises(ValueError, match="subject"):
            Tree.from_entries([("", "likes", "a" * 64)])

    def test_from_entries_rejects_empty_hash_list(self):
        with pytest.raises(ValueError, match="no fact hashes"):
            Tree.from_entries([("user", "likes", [])])

    def test_to_dict_entries_are_json_native_lists(self):
        tree = Tree.from_facts([make_fact()])
        payload = tree.to_dict()
        assert isinstance(payload["entries"], list)
        assert all(isinstance(e, list) for e in payload["entries"])


class TestSingleKeyMutation:
    def test_with_key_replaces_only_that_key(self):
        tree = Tree.from_facts([make_fact(), make_fact(subject="x", predicate="y", object="z")])
        replaced = tree.with_key(("user", "prefers_language"), "b" * 64)
        assert replaced.by_key()[("user", "prefers_language")] == ("b" * 64,)
        assert replaced.by_key()[("x", "y")] == tree.by_key()[("x", "y")]

    def test_without_key_drops_it_and_nothing_else(self):
        tree = Tree.from_facts([make_fact(), make_fact(subject="x", predicate="y", object="z")])
        dropped = tree.without_key(("user", "prefers_language"))
        assert ("user", "prefers_language") not in dropped
        assert ("x", "y") in dropped
        assert len(dropped) == 1


class TestLookups:
    def test_contains_checks_key_not_hash(self):
        tree = Tree.from_facts([make_fact()])
        assert ("user", "prefers_language") in tree
        assert ("user", "nope") not in tree

    def test_fact_hashes_returns_every_hash_sorted(self):
        a = make_fact(subject="user", predicate="likes", object="chess")
        b = make_fact(subject="user", predicate="likes", object="go")
        tree = Tree.from_facts([a, b])
        assert tree.fact_hashes() == tuple(sorted([a.hash, b.hash]))

    def test_subjects_returns_distinct_subjects_sorted(self):
        tree = Tree.from_facts(
            [make_fact(subject="b"), make_fact(subject="a"), make_fact(subject="a", predicate="x")]
        )
        assert tree.subjects() == ("a", "b")
