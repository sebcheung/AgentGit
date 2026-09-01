"""Tests for the Fact schema and canonical serialization.

The property under test throughout is: **the same logical fact always produces
the same hash, and different logical facts never collide.** Everything
downstream — dedup, diffing, ablation — assumes it.
"""

from __future__ import annotations

import json

import pytest

from memgit.core.fact import Fact, canonical_json, hash_payload


def make_fact(**overrides) -> Fact:
    """A fact with all fields populated, for tests that vary one at a time."""
    defaults = dict(
        subject="user",
        predicate="prefers_language",
        object="Python",
        confidence=0.9,
        asserted_at="2026-08-11T12:00:00+00:00",
        source="session:42/turn:7",
        source_text="I mostly write Python these days",
    )
    defaults.update(overrides)
    return Fact(**defaults)


# -- canonical_json ------------------------------------------------------


class TestCanonicalJSON:
    def test_key_order_does_not_affect_output(self):
        """Insertion order is not part of a value, so it must not change bytes."""
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})

    def test_no_incidental_whitespace(self):
        assert canonical_json({"a": 1, "b": 2}) == b'{"a":1,"b":2}'

    def test_non_ascii_is_preserved_not_escaped(self):
        """UTF-8 in, UTF-8 out — one encoding, not a \\uXXXX escape soup."""
        raw = canonical_json({"name": "Ali Ünal"})
        assert "Ünal" in raw.decode("utf-8")
        assert b"\\u" not in raw

    def test_nested_structures_are_sorted_at_every_level(self):
        a = {"outer": {"z": 1, "a": 2}}
        b = {"outer": {"a": 2, "z": 1}}
        assert canonical_json(a) == canonical_json(b)

    def test_rejects_nan_and_infinity(self):
        """NaN/Infinity are not JSON and do not round-trip through other parsers."""
        with pytest.raises(ValueError):
            canonical_json({"x": float("nan")})
        with pytest.raises(ValueError):
            canonical_json({"x": float("inf")})

    def test_output_is_parseable_json(self):
        payload = {"b": [3, 1, 2], "a": {"nested": True}}
        assert json.loads(canonical_json(payload).decode("utf-8")) == payload


class TestHashPayload:
    def test_is_deterministic_across_calls(self):
        payload = {"subject": "user", "predicate": "likes"}
        assert hash_payload(payload) == hash_payload(payload)

    def test_is_sha256_shaped(self):
        digest = hash_payload({"a": 1})
        assert len(digest) == 64
        int(digest, 16)  # raises if not hex

    def test_differing_payloads_differ(self):
        assert hash_payload({"a": 1}) != hash_payload({"a": 2})


# -- Fact construction and validation ------------------------------------


class TestFactValidation:
    @pytest.mark.parametrize("field_name", ["subject", "predicate", "object"])
    def test_rejects_empty_triple_fields(self, field_name):
        """An empty key field would corrupt the diff engine's alignment."""
        with pytest.raises(ValueError, match="non-empty"):
            make_fact(**{field_name: ""})

    @pytest.mark.parametrize("field_name", ["subject", "predicate", "object"])
    def test_rejects_whitespace_only_triple_fields(self, field_name):
        with pytest.raises(ValueError, match="non-empty"):
            make_fact(**{field_name: "   "})

    @pytest.mark.parametrize("field_name", ["subject", "predicate", "object"])
    def test_rejects_non_string_triple_fields(self, field_name):
        with pytest.raises(TypeError):
            make_fact(**{field_name: 42})

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_rejects_out_of_range_confidence(self, bad):
        with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
            make_fact(confidence=bad)

    @pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
    def test_accepts_boundary_confidence(self, ok):
        assert make_fact(confidence=ok).confidence == ok

    def test_rejects_bool_confidence(self):
        """bool is an int subclass in Python; True as a confidence is a bug."""
        with pytest.raises(TypeError):
            make_fact(confidence=True)

    def test_is_immutable(self):
        """Facts are values addressed by content — mutation would desync them."""
        fact = make_fact()
        with pytest.raises(AttributeError):
            fact.object = "Rust"  # type: ignore[misc]

    def test_defaults_asserted_at_to_now(self):
        fact = Fact(subject="user", predicate="likes", object="tea")
        assert fact.asserted_at.endswith("+00:00")


class TestFactIdentity:
    def test_key_is_subject_predicate(self):
        assert make_fact().key == ("user", "prefers_language")

    def test_triple_ignores_metadata(self):
        assert make_fact().triple == ("user", "prefers_language", "Python")

    def test_same_fields_produce_same_hash(self):
        assert make_fact().hash == make_fact().hash

    def test_construction_order_does_not_affect_hash(self):
        a = Fact(subject="user", predicate="likes", object="tea", asserted_at="2026-01-01T00:00:00+00:00")
        b = Fact(object="tea", predicate="likes", subject="user", asserted_at="2026-01-01T00:00:00+00:00")
        assert a.hash == b.hash

    def test_different_object_changes_hash(self):
        assert make_fact(object="Python").hash != make_fact(object="Rust").hash

    def test_different_confidence_changes_hash(self):
        """Deliberate: this is what makes a strengthened belief visible in a diff."""
        assert make_fact(confidence=0.5).hash != make_fact(confidence=0.9).hash

    def test_different_timestamp_changes_hash(self):
        a = make_fact(asserted_at="2026-01-01T00:00:00+00:00")
        b = make_fact(asserted_at="2026-06-01T00:00:00+00:00")
        assert a.hash != b.hash

    def test_facts_sharing_a_key_can_have_different_hashes(self):
        """Semantic identity and content identity are deliberately distinct."""
        a = make_fact(object="Python")
        b = make_fact(object="Rust")
        assert a.key == b.key
        assert a.hash != b.hash

    def test_unset_optional_field_hashes_same_as_explicit_none(self):
        """One value, one hash — regardless of how the fact was constructed."""
        a = Fact(
            subject="user", predicate="likes", object="tea",
            asserted_at="2026-01-01T00:00:00+00:00",
        )
        b = Fact(
            subject="user", predicate="likes", object="tea",
            asserted_at="2026-01-01T00:00:00+00:00",
            source=None, source_text=None,
        )
        assert a.hash == b.hash


class TestFactSerialization:
    def test_round_trips_through_dict(self):
        fact = make_fact()
        assert Fact.from_dict(fact.to_dict()) == fact

    def test_round_trip_preserves_hash(self):
        fact = make_fact()
        assert Fact.from_dict(fact.to_dict()).hash == fact.hash

    def test_round_trips_without_optional_fields(self):
        fact = Fact(
            subject="user", predicate="likes", object="tea",
            asserted_at="2026-01-01T00:00:00+00:00",
        )
        assert Fact.from_dict(fact.to_dict()) == fact

    def test_to_dict_omits_unset_optional_fields(self):
        payload = Fact(
            subject="user", predicate="likes", object="tea",
            asserted_at="2026-01-01T00:00:00+00:00",
        ).to_dict()
        assert "source" not in payload
        assert "source_text" not in payload

    def test_to_dict_is_tagged_with_a_type(self):
        """The store holds facts, trees, and commits — objects must be self-describing."""
        assert make_fact().to_dict()["type"] == "fact"

    def test_from_dict_rejects_other_object_types(self):
        with pytest.raises(ValueError, match="expected a fact"):
            Fact.from_dict({"type": "commit", "subject": "x"})


class TestReaffirm:
    def test_preserves_the_claim(self):
        original = make_fact()
        again = original.reaffirm()
        assert again.triple == original.triple
        assert again.key == original.key

    def test_produces_a_new_hash(self):
        """Otherwise reaffirmation would be invisible to the diff engine."""
        original = make_fact()
        again = original.reaffirm(asserted_at="2026-09-01T00:00:00+00:00")
        assert again.hash != original.hash

    def test_can_raise_confidence(self):
        again = make_fact(confidence=0.5).reaffirm(confidence=0.95)
        assert again.confidence == 0.95

    def test_carries_metadata_forward_when_not_overridden(self):
        original = make_fact()
        again = original.reaffirm()
        assert again.source == original.source
        assert again.source_text == original.source_text

    def test_leaves_the_original_untouched(self):
        original = make_fact(confidence=0.5)
        original.reaffirm(confidence=0.99)
        assert original.confidence == 0.5
