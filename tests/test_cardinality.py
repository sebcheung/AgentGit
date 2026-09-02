"""Tests for CardinalityMap — the diff engine's single/multi schema.

Grouped by the properties that carry the design: the default lens (every
predicate is 'single' unless declared or the map's own default is flipped),
strict serialization matching the rest of the codebase, and that derivation
methods (`with_predicate`/`without_predicate`) return new maps rather than
mutating.
"""

from __future__ import annotations

import pytest

from memgit.core.cardinality import CardinalityMap


class TestDefaults:
    def test_default_map_is_all_single(self):
        mapping = CardinalityMap.default_map()
        assert mapping.default == "single"
        assert mapping["anything"] == "single"
        assert not mapping.is_multi("anything")

    def test_declared_predicate_overrides_default(self):
        mapping = CardinalityMap({"likes": "multi"})
        assert mapping["likes"] == "multi"
        assert mapping.is_multi("likes")
        assert mapping["favorite_editor"] == "single"

    def test_map_default_can_be_flipped_to_multi(self):
        mapping = CardinalityMap({"favorite_editor": "single"}, default="multi")
        assert mapping["likes"] == "multi"
        assert mapping["favorite_editor"] == "single"

    def test_declared_returns_sorted_pairs(self):
        mapping = CardinalityMap({"z": "multi", "a": "single"})
        assert mapping.declared() == (("a", "single"), ("z", "multi"))


class TestValidation:
    def test_rejects_bad_top_level_default(self):
        with pytest.raises(ValueError, match="default"):
            CardinalityMap(default="sometimes")

    def test_rejects_bad_predicate_cardinality(self):
        with pytest.raises(ValueError, match="likes"):
            CardinalityMap({"likes": "several"})

    def test_rejects_empty_predicate(self):
        with pytest.raises(ValueError, match="non-empty"):
            CardinalityMap({"": "multi"})


class TestSerialization:
    def test_round_trips(self):
        mapping = CardinalityMap({"likes": "multi", "knows": "multi"}, default="single")
        restored = CardinalityMap.from_dict(mapping.to_dict())
        assert restored == mapping

    def test_to_dict_predicates_are_sorted(self):
        mapping = CardinalityMap({"z": "multi", "a": "multi"})
        assert list(mapping.to_dict()["predicates"]) == ["a", "z"]

    def test_from_dict_defaults_missing_fields(self):
        mapping = CardinalityMap.from_dict({})
        assert mapping == CardinalityMap.default_map()

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown keys"):
            CardinalityMap.from_dict({"predicates": {}, "extra": 1})

    def test_from_dict_rejects_unknown_version(self):
        with pytest.raises(ValueError, match="version"):
            CardinalityMap.from_dict({"version": 99})

    def test_from_dict_rejects_bad_cardinality_naming_the_predicate(self):
        with pytest.raises(ValueError, match="likes"):
            CardinalityMap.from_dict({"predicates": {"likes": "lots"}})


class TestDerivation:
    def test_with_predicate_returns_new_map(self):
        original = CardinalityMap.default_map()
        updated = original.with_predicate("likes", "multi")
        assert original["likes"] == "single"
        assert updated["likes"] == "multi"

    def test_without_predicate_reverts_to_default(self):
        mapping = CardinalityMap({"likes": "multi"})
        reverted = mapping.without_predicate("likes")
        assert reverted["likes"] == "single"
        assert mapping["likes"] == "multi"

    def test_without_predicate_is_a_noop_when_absent(self):
        mapping = CardinalityMap.default_map()
        assert mapping.without_predicate("likes") == mapping
