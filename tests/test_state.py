"""Tests for MemoryState: materialization, lookup, derivation, and rendering.

Facts are built directly (no store, no repository) since MemoryState only
needs a FactReader-shaped callable, matching the pattern diff.py's tests use.
"""

from __future__ import annotations

from memgit.core.fact import Fact
from memgit.core.state import MemoryState
from memgit.core.tree import EMPTY_TREE_HASH, Tree


def fact(subject, predicate, obj, **kw) -> Fact:
    return Fact(subject=subject, predicate=predicate, object=obj, **kw)


def reader(facts: list[Fact]):
    by_hash = {f.hash: f for f in facts}
    return by_hash.__getitem__


class TestConstruction:
    def test_empty_has_no_facts_and_the_empty_tree_hash(self):
        state = MemoryState.empty()
        assert len(state) == 0
        assert state.tree == EMPTY_TREE_HASH
        assert state.commit is None

    def test_from_facts_builds_the_matching_tree_hash(self):
        facts = [fact("user", "likes", "chess")]
        state = MemoryState.from_facts(facts)
        assert state.tree == Tree.from_facts(facts).hash

    def test_from_tree_materializes_every_referenced_fact(self):
        facts = [fact("user", "likes", "chess"), fact("user", "likes", "go")]
        tree = Tree.from_facts(facts)
        state = MemoryState.from_tree(tree, reader(facts))
        assert len(state) == 2
        assert state.tree == tree.hash

    def test_from_tree_carries_commit_provenance(self):
        facts = [fact("user", "likes", "chess")]
        tree = Tree.from_facts(facts)
        state = MemoryState.from_tree(tree, reader(facts), commit="c" * 64)
        assert state.commit == "c" * 64

    def test_empty_tree_materializes_to_no_facts(self):
        state = MemoryState.from_tree(Tree(()), reader([]))
        assert len(state) == 0
        assert state.tree == EMPTY_TREE_HASH


class TestCollection:
    def test_len_counts_facts_not_keys(self):
        facts = [fact("user", "likes", "chess"), fact("user", "likes", "go")]
        state = MemoryState.from_facts(facts)
        assert len(state) == 2
        assert len(state.keys()) == 1

    def test_iteration_yields_every_fact(self):
        facts = [fact("user", "likes", "chess"), fact("project", "name", "memgit")]
        state = MemoryState.from_facts(facts)
        assert set(state) == set(facts)

    def test_contains_checks_the_key_not_the_object_value(self):
        state = MemoryState.from_facts([fact("user", "likes", "chess")])
        assert ("user", "likes") in state
        assert ("user", "dislikes") not in state

    def test_subjects_are_sorted_and_deduplicated(self):
        facts = [fact("b", "p", "x"), fact("a", "p", "y"), fact("a", "q", "z")]
        state = MemoryState.from_facts(facts)
        assert state.subjects() == ("a", "b")

    def test_keys_are_sorted_and_deduplicated(self):
        facts = [fact("user", "likes", "chess"), fact("user", "likes", "go")]
        state = MemoryState.from_facts(facts)
        assert state.keys() == (("user", "likes"),)


class TestLookup:
    def test_get_returns_every_fact_at_a_multi_valued_key(self):
        facts = [fact("user", "likes", "chess"), fact("user", "likes", "go")]
        state = MemoryState.from_facts(facts)
        assert set(state.get("user", "likes")) == set(facts)

    def test_get_on_a_missing_key_is_empty(self):
        state = MemoryState.from_facts([fact("user", "likes", "chess")])
        assert state.get("user", "dislikes") == ()

    def test_one_returns_none_on_a_missing_key(self):
        state = MemoryState.empty()
        assert state.one("user", "likes") is None

    def test_one_picks_highest_confidence(self):
        low = fact("user", "likes", "chess", confidence=0.3)
        high = fact("user", "likes", "go", confidence=0.9)
        state = MemoryState.from_facts([low, high])
        assert state.one("user", "likes") == high

    def test_one_breaks_confidence_ties_by_most_recent(self):
        older = fact("user", "likes", "chess", confidence=0.5, asserted_at="2026-01-01T00:00:00+00:00")
        newer = fact("user", "likes", "go", confidence=0.5, asserted_at="2026-01-02T00:00:00+00:00")
        state = MemoryState.from_facts([older, newer])
        assert state.one("user", "likes") == newer

    def test_one_breaks_remaining_ties_by_hash_deterministically(self):
        a = fact(
            "user", "likes", "chess", confidence=0.5, asserted_at="2026-01-01T00:00:00+00:00"
        )
        b = fact(
            "user", "likes", "go", confidence=0.5, asserted_at="2026-01-01T00:00:00+00:00"
        )
        expected = max((a, b), key=lambda f: f.hash)
        state = MemoryState.from_facts([a, b])
        assert state.one("user", "likes") == expected

    def test_by_subject(self):
        facts = [fact("user", "likes", "chess"), fact("project", "name", "memgit")]
        state = MemoryState.from_facts(facts)
        assert state.by_subject("user") == (facts[0],)

    def test_by_predicate(self):
        facts = [fact("user", "likes", "chess"), fact("project", "likes", "python")]
        state = MemoryState.from_facts(facts)
        assert set(state.by_predicate("likes")) == set(facts)


class TestDerivation:
    def test_filter_by_min_confidence(self):
        low = fact("user", "likes", "chess", confidence=0.2)
        high = fact("user", "likes", "go", confidence=0.8)
        state = MemoryState.from_facts([low, high])
        filtered = state.filter(min_confidence=0.5)
        assert filtered.facts == (high,)

    def test_filter_by_subjects(self):
        facts = [fact("user", "likes", "chess"), fact("project", "name", "memgit")]
        state = MemoryState.from_facts(facts)
        filtered = state.filter(subjects={"user"})
        assert filtered.facts == (facts[0],)

    def test_filter_by_predicates(self):
        facts = [fact("user", "likes", "chess"), fact("user", "dislikes", "checkers")]
        state = MemoryState.from_facts(facts)
        filtered = state.filter(predicates={"likes"})
        assert filtered.facts == (facts[0],)

    def test_filter_drops_commit_provenance(self):
        facts = [fact("user", "likes", "chess")]
        state = MemoryState.from_facts(facts, commit="c" * 64)
        assert state.filter(min_confidence=0.0).commit is None

    def test_without_removes_every_value_at_a_key(self):
        facts = [fact("user", "likes", "chess"), fact("user", "likes", "go"), fact("project", "name", "x")]
        state = MemoryState.from_facts(facts)
        result = state.without(("user", "likes"))
        assert result.facts == (facts[2],)

    def test_without_drops_commit_provenance(self):
        state = MemoryState.from_facts([fact("user", "likes", "chess")], commit="c" * 64)
        assert state.without(("user", "likes")).commit is None

    def test_without_fact_removes_only_that_hash(self):
        chess = fact("user", "likes", "chess")
        go = fact("user", "likes", "go")
        state = MemoryState.from_facts([chess, go])
        result = state.without_fact(chess.hash)
        assert result.facts == (go,)


class TestToTree:
    def test_round_trips_through_from_tree(self):
        facts = [fact("user", "likes", "chess"), fact("project", "name", "memgit")]
        tree = Tree.from_facts(facts)
        state = MemoryState.from_tree(tree, reader(facts))
        assert state.to_tree().hash == tree.hash

    def test_to_tree_of_empty_state_is_the_empty_tree(self):
        assert MemoryState.empty().to_tree().hash == EMPTY_TREE_HASH


class TestRender:
    def test_is_deterministic_across_calls(self):
        facts = [fact("b", "p", "x"), fact("a", "p", "y")]
        state = MemoryState.from_facts(facts)
        assert state.render() == state.render()

    def test_is_stable_under_insertion_order(self):
        facts = [fact("b", "p", "x"), fact("a", "p", "y")]
        state1 = MemoryState.from_facts(facts)
        state2 = MemoryState.from_facts(list(reversed(facts)))
        assert state1.render() == state2.render()

    def test_groups_by_subject(self):
        facts = [fact("user", "likes", "chess"), fact("project", "name", "memgit")]
        state = MemoryState.from_facts(facts)
        rendered = state.render()
        assert "user:" in rendered
        assert "project:" in rendered

    def test_full_confidence_has_no_suffix(self):
        state = MemoryState.from_facts([fact("user", "likes", "chess", confidence=1.0)])
        assert "(" not in state.render()

    def test_partial_confidence_is_shown(self):
        state = MemoryState.from_facts([fact("user", "likes", "chess", confidence=0.42)])
        assert "0.42" in state.render()

    def test_max_facts_truncates(self):
        facts = [fact("user", "likes", "chess"), fact("user", "likes", "go")]
        state = MemoryState.from_facts(facts)
        truncated = state.render(max_facts=1)
        full = state.render()
        assert len(truncated) < len(full)

    def test_empty_state_renders_to_empty_string(self):
        assert MemoryState.empty().render() == ""


class TestToDict:
    def test_has_no_type_tag(self):
        state = MemoryState.from_facts([fact("user", "likes", "chess")])
        assert "type" not in state.to_dict()

    def test_omits_commit_when_unset(self):
        state = MemoryState.from_facts([fact("user", "likes", "chess")])
        assert "commit" not in state.to_dict()

    def test_includes_commit_when_set(self):
        state = MemoryState.from_facts([fact("user", "likes", "chess")], commit="c" * 64)
        assert state.to_dict()["commit"] == "c" * 64
