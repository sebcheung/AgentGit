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
    defaults = {
        "subject": "user",
        "predicate": "prefers_language",
        "object": "Python",
        "confidence": 0.9,
        "asserted_at": "2026-08-11T12:00:00+00:00",
    }
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
        before, _reader = tree_and_reader(left)
        after = Tree.from_facts([right])
        combined_reader = CountingFactReader({left.hash: left, right.hash: right})

        result = diff_trees(before, after, combined_reader, include_unchanged=True)
        assert [kd.key for kd in result.keys] == sorted([("a", "b/c"), ("a/b", "c")])

    def test_diff_hashes_reads_no_facts(self):
        before, _ = tree_and_reader(make_fact(object="Python"))
        after, _ = tree_and_reader(make_fact(object="Rust"))
        result = diff_hashes(before, after)
        assert len(result) == 1


class TestSingleValuedClassification:
    def test_new_key_is_added(self):
        after, reader = tree_and_reader(make_fact())
        result = diff_trees(Tree(()), after, reader)
        assert result.keys[0].kind == ChangeKind.ADDED

    def test_vanished_key_is_removed(self):
        before, reader = tree_and_reader(make_fact())
        result = diff_trees(before, Tree(()), reader)
        assert result.keys[0].kind == ChangeKind.REMOVED

    def test_identical_hash_lists_are_unchanged_with_zero_reads(self):
        fact = make_fact()
        tree = Tree.from_facts([fact])
        reader = CountingFactReader({fact.hash: fact})
        result = diff_trees(tree, tree, reader)
        assert result.is_empty
        assert reader.calls == 0

    def test_reaffirmed_with_higher_confidence(self):
        before_fact = make_fact(confidence=0.70)
        after_fact = make_fact(confidence=0.95)
        before = Tree.from_facts([before_fact])
        after = Tree.from_facts([after_fact])
        reader = CountingFactReader({before_fact.hash: before_fact, after_fact.hash: after_fact})

        result = diff_trees(before, after, reader)
        kd = result.keys[0]
        assert kd.kind == ChangeKind.REAFFIRMED
        value = kd.changed_values[0]
        assert value.kind == ValueKind.REAFFIRMED
        assert value.confidence_delta == pytest.approx(0.25)
        assert value.is_strengthened

    def test_reaffirmed_when_only_asserted_at_differs(self):
        before_fact = make_fact(asserted_at="2026-01-01T00:00:00+00:00")
        after_fact = make_fact(asserted_at="2026-02-01T00:00:00+00:00")
        before = Tree.from_facts([before_fact])
        after = Tree.from_facts([after_fact])
        reader = CountingFactReader({before_fact.hash: before_fact, after_fact.hash: after_fact})

        result = diff_trees(before, after, reader)
        kd = result.keys[0]
        assert kd.kind == ChangeKind.REAFFIRMED
        assert kd.changed_values[0].confidence_delta == 0.0

    def test_reaffirmed_when_only_source_text_differs(self):
        before_fact = make_fact(source_text="original")
        after_fact = make_fact(source_text="rephrased")
        before = Tree.from_facts([before_fact])
        after = Tree.from_facts([after_fact])
        reader = CountingFactReader({before_fact.hash: before_fact, after_fact.hash: after_fact})

        result = diff_trees(before, after, reader)
        assert result.keys[0].kind == ChangeKind.REAFFIRMED

    def test_replaced_value_is_contradicted(self):
        vim = make_fact(predicate="favorite_editor", object="vim")
        neovim = make_fact(predicate="favorite_editor", object="neovim")
        before = Tree.from_facts([vim])
        after = Tree.from_facts([neovim])
        reader = CountingFactReader({vim.hash: vim, neovim.hash: neovim})

        result = diff_trees(before, after, reader)
        kd = result.keys[0]
        assert kd.kind == ChangeKind.CONTRADICTED
        kinds = {v.kind for v in kd.values}
        assert kinds == {ValueKind.REMOVED, ValueKind.ADDED}

    def test_rival_value_added_is_also_contradicted_and_a_violation(self):
        vim = make_fact(predicate="favorite_editor", object="vim")
        neovim = make_fact(predicate="favorite_editor", object="neovim")
        before = Tree.from_facts([vim])
        after = Tree.from_entries([("user", "favorite_editor", [vim.hash, neovim.hash])])
        reader = CountingFactReader({vim.hash: vim, neovim.hash: neovim})

        result = diff_trees(before, after, reader)
        kd = result.keys[0]
        assert kd.kind == ChangeKind.CONTRADICTED
        assert len(result.violations) == 1
        assert result.violations[0].side == "after"
        assert set(result.violations[0].objects) == {"vim", "neovim"}

    def test_narrowing_two_values_to_one_is_value_removed_and_a_violation(self):
        vim = make_fact(predicate="favorite_editor", object="vim")
        neovim = make_fact(predicate="favorite_editor", object="neovim")
        before = Tree.from_entries([("user", "favorite_editor", [vim.hash, neovim.hash])])
        after = Tree.from_facts([vim])
        reader = CountingFactReader({vim.hash: vim, neovim.hash: neovim})

        result = diff_trees(before, after, reader)
        kd = result.keys[0]
        assert kd.kind == ChangeKind.VALUE_REMOVED
        assert len(result.violations) == 1
        assert result.violations[0].side == "before"


class TestMultiValuedClassification:
    MULTI = CardinalityMap({"likes": "multi"})

    def test_gained_value_is_value_added(self):
        chess = make_fact(predicate="likes", object="chess")
        go = make_fact(predicate="likes", object="go")
        before = Tree.from_facts([chess])
        after = Tree.from_facts([chess, go])
        reader = CountingFactReader({chess.hash: chess, go.hash: go})

        result = diff_trees(before, after, reader, cardinality=self.MULTI)
        kd = result.keys[0]
        assert kd.kind == ChangeKind.VALUE_ADDED
        assert not result.violations

    def test_lost_value_is_value_removed(self):
        chess = make_fact(predicate="likes", object="chess")
        go = make_fact(predicate="likes", object="go")
        before = Tree.from_facts([chess, go])
        after = Tree.from_facts([chess])
        reader = CountingFactReader({chess.hash: chess, go.hash: go})

        result = diff_trees(before, after, reader, cardinality=self.MULTI)
        assert result.keys[0].kind == ChangeKind.VALUE_REMOVED

    def test_gained_and_lost_is_mixed(self):
        chess = make_fact(predicate="likes", object="chess")
        go = make_fact(predicate="likes", object="go")
        rust = make_fact(predicate="likes", object="rust")
        before = Tree.from_facts([chess, go])
        after = Tree.from_facts([chess, rust])
        reader = CountingFactReader({chess.hash: chess, go.hash: go, rust.hash: rust})

        result = diff_trees(before, after, reader, cardinality=self.MULTI)
        kd = result.keys[0]
        assert kd.kind == ChangeKind.MIXED
        kinds = {v.kind for v in kd.changed_values}
        assert kinds == {ValueKind.ADDED, ValueKind.REMOVED}

    def test_value_replaced_entirely_is_mixed_not_contradicted(self):
        go = make_fact(predicate="likes", object="go")
        rust = make_fact(predicate="likes", object="rust")
        before = Tree.from_facts([go])
        after = Tree.from_facts([rust])
        reader = CountingFactReader({go.hash: go, rust.hash: rust})

        result = diff_trees(before, after, reader, cardinality=self.MULTI)
        assert result.keys[0].kind == ChangeKind.MIXED

    def test_one_of_two_reaffirmed_key_is_reaffirmed(self):
        chess = make_fact(predicate="likes", object="chess", confidence=0.5)
        chess2 = make_fact(predicate="likes", object="chess", confidence=0.9)
        go = make_fact(predicate="likes", object="go")
        before = Tree.from_facts([chess, go])
        after = Tree.from_facts([chess2, go])
        reader = CountingFactReader({chess.hash: chess, chess2.hash: chess2, go.hash: go})

        result = diff_trees(before, after, reader, cardinality=self.MULTI)
        kd = result.keys[0]
        assert kd.kind == ChangeKind.REAFFIRMED
        kinds = {v.kind for v in kd.values}
        assert kinds == {ValueKind.REAFFIRMED, ValueKind.UNCHANGED}

    def test_cardinality_is_the_only_variable_pinning_test(self):
        """Same two trees, both cardinalities: kind differs, values don't."""
        go = make_fact(predicate="likes", object="go")
        rust = make_fact(predicate="likes", object="rust")
        before = Tree.from_facts([go])
        after_single = Tree.from_entries([("user", "likes", [go.hash, rust.hash])])
        reader = CountingFactReader({go.hash: go, rust.hash: rust})

        single_result = diff_trees(before, after_single, reader, cardinality=CardinalityMap.default_map())
        multi_result = diff_trees(before, after_single, reader, cardinality=self.MULTI)

        assert single_result.keys[0].kind == ChangeKind.CONTRADICTED
        assert multi_result.keys[0].kind == ChangeKind.VALUE_ADDED
        assert single_result.keys[0].values == multi_result.keys[0].values


class TestMixedAndPrecedence:
    def test_add_remove_and_reaffirm_at_one_multi_key(self):
        chess = make_fact(predicate="likes", object="chess", confidence=0.5)
        chess2 = make_fact(predicate="likes", object="chess", confidence=0.9)
        go = make_fact(predicate="likes", object="go")
        rust = make_fact(predicate="likes", object="rust")
        before = Tree.from_facts([chess, go])
        after = Tree.from_facts([chess2, rust])
        reader = CountingFactReader({f.hash: f for f in (chess, chess2, go, rust)})

        result = diff_trees(before, after, reader, cardinality=CardinalityMap({"likes": "multi"}))
        kd = result.keys[0]
        assert kd.kind == ChangeKind.MIXED
        kinds = {v.kind for v in kd.values}
        assert kinds == {ValueKind.REAFFIRMED, ValueKind.ADDED, ValueKind.REMOVED}

    def test_add_and_reaffirm_at_single_key_is_contradicted_not_mixed(self):
        vim = make_fact(predicate="favorite_editor", object="vim", confidence=0.5)
        vim2 = make_fact(predicate="favorite_editor", object="vim", confidence=0.9)
        neovim = make_fact(predicate="favorite_editor", object="neovim")
        before = Tree.from_facts([vim])
        after = Tree.from_entries([("user", "favorite_editor", [vim2.hash, neovim.hash])])
        reader = CountingFactReader({f.hash: f for f in (vim, vim2, neovim)})

        result = diff_trees(before, after, reader)
        assert result.keys[0].kind == ChangeKind.CONTRADICTED

    def test_values_are_sorted_by_object(self):
        chess = make_fact(predicate="likes", object="chess")
        go = make_fact(predicate="likes", object="go")
        before = Tree.from_facts([])
        after = Tree.from_facts([go, chess])
        reader = CountingFactReader({chess.hash: chess, go.hash: go})

        result = diff_trees(before, after, reader)
        objects = [v.object for v in result.keys[0].values]
        assert objects == sorted(objects)


class TestShadowedDuplicates:
    def test_same_object_twice_is_not_a_violation(self):
        older = make_fact(object="Python", asserted_at="2026-01-01T00:00:00+00:00", confidence=0.5)
        newer = make_fact(object="Python", asserted_at="2026-02-01T00:00:00+00:00", confidence=0.9)
        after = Tree.from_entries([("user", "prefers_language", [older.hash, newer.hash])])
        reader = CountingFactReader({older.hash: older, newer.hash: newer})

        result = diff_trees(Tree(()), after, reader)
        assert not result.violations
        kd = result.keys[0]
        assert kd.shadowed_after == (older,)
        assert kd.values[0].after == newer

    def test_tie_on_asserted_at_breaks_on_confidence(self):
        low = make_fact(object="Python", confidence=0.3, asserted_at="2026-01-01T00:00:00+00:00")
        high = make_fact(object="Python", confidence=0.8, asserted_at="2026-01-01T00:00:00+00:00")
        after = Tree.from_entries([("user", "prefers_language", [low.hash, high.hash])])
        reader = CountingFactReader({low.hash: low, high.hash: high})

        result = diff_trees(Tree(()), after, reader)
        assert result.keys[0].values[0].after == high
        assert result.keys[0].shadowed_after == (low,)

    def test_tie_on_both_is_stable_across_runs(self):
        a = make_fact(object="Python", confidence=0.5, asserted_at="2026-01-01T00:00:00+00:00")
        b = make_fact(object="Python", confidence=0.5, asserted_at="2026-01-01T00:00:00+00:00", source="x")
        after = Tree.from_entries([("user", "prefers_language", [a.hash, b.hash])])
        reader = CountingFactReader({a.hash: a, b.hash: b})

        first = diff_trees(Tree(()), after, reader).keys[0].values[0].after
        second = diff_trees(Tree(()), after, reader).keys[0].values[0].after
        assert first == second


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


class TestLaziness:
    def test_unchanged_keys_cost_no_fact_reads_at_scale(self):
        facts = [make_fact(subject=f"s{i}", predicate="p", object=str(i)) for i in range(1000)]
        before = Tree.from_facts(facts)
        changed = make_fact(subject="s0", predicate="p", object="changed")
        after_facts = [*facts[1:], changed]
        after = Tree.from_facts(after_facts)
        reader = CountingFactReader({f.hash: f for f in [*facts, changed]})

        result = diff_trees(before, after, reader)
        assert len(result.keys) == 1
        assert reader.calls <= 4

    def test_include_unchanged_includes_all_keys_with_still_no_extra_reads(self):
        """The identical-hash short circuit means unchanged keys never cost a
        fact read regardless of the flag; ``include_unchanged`` only changes
        which keys make it into the result."""
        facts = [make_fact(subject=f"s{i}", predicate="p", object=str(i)) for i in range(5)]
        tree = Tree.from_facts(facts)
        reader = CountingFactReader({f.hash: f for f in facts})

        assert diff_trees(tree, tree, reader, include_unchanged=False).is_empty
        result = diff_trees(tree, tree, reader, include_unchanged=True)
        assert len(result.keys) == 5
        assert reader.calls == 0

    def test_diff_hashes_reads_nothing(self):
        facts = [make_fact(subject=f"s{i}", predicate="p", object=str(i)) for i in range(5)]
        before = Tree.from_facts(facts)
        after = Tree.from_facts(facts[1:])
        diff_hashes(before, after)  # would raise if it tried to read a fact


class TestStatAndSerialization:
    def test_stat_agrees_with_keys(self):
        chess = make_fact(predicate="likes", object="chess")
        vim = make_fact(predicate="favorite_editor", object="vim")
        neovim = make_fact(predicate="favorite_editor", object="neovim")
        before = Tree.from_facts([vim])
        after = Tree.from_facts([chess, neovim])
        reader = CountingFactReader({f.hash: f for f in (chess, vim, neovim)})

        result = diff_trees(before, after, reader)
        stat = result.stat()
        assert stat.keys_changed == len(result.keys)
        assert sum(stat.by_kind.values()) == len(result.keys)

    def test_to_dict_survives_json_round_trip(self):
        import json

        fact = make_fact()
        after = Tree.from_facts([fact])
        reader = CountingFactReader({fact.hash: fact})
        result = diff_trees(Tree(()), after, reader)
        json.dumps(result.to_dict())

    def test_to_dict_embeds_the_cardinality_map_used(self):
        mapping = CardinalityMap({"likes": "multi"})
        result = diff_trees(Tree(()), Tree(()), lambda h: (_ for _ in ()).throw(AssertionError()), cardinality=mapping)
        assert result.to_dict()["cardinality"] == mapping.to_dict()
