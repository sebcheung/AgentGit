"""Tests for DecayPolicy — exponential half-life over asserted_at.

Grouped like ``test_cardinality.py``: defaults, math, validation,
serialization, derivation — plus the property that makes component 9 need no
new machinery at all, reaffirmation resetting decay.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memgit.core.decay import DecayPolicy
from memgit.core.fact import Fact

NOW = datetime(2026, 9, 3, tzinfo=UTC)


def _fact(predicate="prefers_language", confidence=1.0, age_days=0.0, **kwargs):
    asserted = (NOW - timedelta(days=age_days)).isoformat()
    return Fact(subject="user", predicate=predicate, object="Python",
                confidence=confidence, asserted_at=asserted, **kwargs)


class TestDefaults:
    def test_default_policy_has_a_180_day_half_life(self):
        policy = DecayPolicy.default_map()
        assert policy.default == 180.0
        assert policy.half_life("anything") == 180.0

    def test_declared_predicate_overrides_default(self):
        policy = DecayPolicy({"born_in": None})
        assert policy.half_life("born_in") is None
        assert policy.half_life("prefers_language") == 180.0

    def test_declared_returns_sorted_pairs(self):
        policy = DecayPolicy({"z": 10.0, "a": 5.0})
        assert policy.declared() == (("a", 5.0), ("z", 10.0))


class TestMath:
    def test_no_age_no_decay(self):
        policy = DecayPolicy(default=180.0)
        fact = _fact(confidence=1.0, age_days=0.0)
        assert policy.decayed(fact, as_of=NOW) == pytest.approx(1.0)

    def test_one_half_life_halves_confidence(self):
        policy = DecayPolicy(default=100.0)
        fact = _fact(confidence=0.8, age_days=100.0)
        assert policy.decayed(fact, as_of=NOW) == pytest.approx(0.4)

    def test_two_half_lives_quarters_confidence(self):
        policy = DecayPolicy(default=100.0)
        fact = _fact(confidence=0.8, age_days=200.0)
        assert policy.decayed(fact, as_of=NOW) == pytest.approx(0.2)

    def test_null_half_life_never_decays(self):
        policy = DecayPolicy({"born_in": None})
        fact = _fact(predicate="born_in", confidence=1.0, age_days=10_000.0)
        assert policy.decayed(fact, as_of=NOW) == pytest.approx(1.0)

    def test_per_predicate_override_beats_default(self):
        policy = DecayPolicy({"prefers_language": 10.0}, default=1000.0)
        fact = _fact(confidence=1.0, age_days=10.0)
        assert policy.decayed(fact, as_of=NOW) == pytest.approx(0.5)

    def test_floor_clamps_the_minimum(self):
        policy = DecayPolicy(default=1.0, floor=0.1)
        fact = _fact(confidence=1.0, age_days=1000.0)
        assert policy.decayed(fact, as_of=NOW) == pytest.approx(0.1)

    def test_negative_age_clamps_to_zero(self):
        policy = DecayPolicy(default=100.0)
        fact = _fact(confidence=0.9, age_days=-10.0)  # asserted "in the future"
        assert policy.age_days(fact, as_of=NOW) == 0.0
        assert policy.decayed(fact, as_of=NOW) == pytest.approx(0.9)

    def test_unparseable_asserted_at_returns_stored_confidence(self):
        fact = Fact(subject="user", predicate="prefers_language", object="Python",
                     confidence=0.7, asserted_at="not-a-timestamp")
        policy = DecayPolicy(default=1.0)
        assert policy.decayed(fact, as_of=NOW) == 0.7

    def test_monotone_decreasing_in_age(self):
        policy = DecayPolicy(default=50.0)
        earlier = policy.decayed(_fact(age_days=5.0), as_of=NOW)
        later = policy.decayed(_fact(age_days=50.0), as_of=NOW)
        assert later < earlier

    def test_reaffirmation_resets_decay(self):
        """The property that makes component 9 need no new machinery: a
        fresh `reaffirm()` bumps asserted_at, so the same fact is young
        again."""
        policy = DecayPolicy(default=10.0)
        stale = _fact(confidence=1.0, age_days=100.0)
        assert policy.decayed(stale, as_of=NOW) < 0.01

        fresh = stale.reaffirm(asserted_at=NOW.isoformat())
        assert policy.decayed(fresh, as_of=NOW) == pytest.approx(1.0)


class TestValidation:
    def test_rejects_non_positive_default(self):
        with pytest.raises(ValueError, match="default"):
            DecayPolicy(default=0.0)

    def test_rejects_non_positive_half_life(self):
        with pytest.raises(ValueError, match="prefers_language"):
            DecayPolicy({"prefers_language": -5.0})

    def test_rejects_empty_predicate(self):
        with pytest.raises(ValueError, match="non-empty"):
            DecayPolicy({"": 10.0})

    def test_rejects_bad_floor(self):
        with pytest.raises(ValueError, match="floor"):
            DecayPolicy(floor=1.5)


class TestSerialization:
    def test_round_trips(self):
        policy = DecayPolicy({"likes": 30.0, "born_in": None}, default=180.0, floor=0.05)
        restored = DecayPolicy.from_dict(policy.to_dict())
        assert restored == policy

    def test_from_dict_defaults_missing_fields(self):
        policy = DecayPolicy.from_dict({})
        assert policy == DecayPolicy.default_map()

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="unknown keys"):
            DecayPolicy.from_dict({"half_lives": {}, "extra": 1})

    def test_from_dict_rejects_unknown_version(self):
        with pytest.raises(ValueError, match="version"):
            DecayPolicy.from_dict({"version": 99})


class TestDerivation:
    def test_with_half_life_returns_new_policy(self):
        original = DecayPolicy.default_map()
        updated = original.with_half_life("likes", 30.0)
        assert original.half_life("likes") == 180.0
        assert updated.half_life("likes") == 30.0

    def test_without_half_life_reverts_to_default(self):
        policy = DecayPolicy({"likes": 30.0})
        reverted = policy.without_half_life("likes")
        assert reverted.half_life("likes") == 180.0
        assert policy.half_life("likes") == 30.0

    def test_without_half_life_is_a_noop_when_absent(self):
        policy = DecayPolicy.default_map()
        assert policy.without_half_life("likes") == policy
