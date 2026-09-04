"""Tests for the content-addressable object store.

Four properties carry the design, and each gets its own class below:
dedup (identical content is stored once), integrity (corruption is caught on
read), immutability (an object's address always describes its contents), and
safety (a hostile hash string cannot escape the objects directory).
"""

from __future__ import annotations

import zlib

import pytest

from memgit.core.fact import Fact
from memgit.core.store import (
    AmbiguousPrefixError,
    CorruptObjectError,
    ObjectNotFoundError,
    ObjectStore,
    hash_object,
    is_object_hash,
)


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


class TestRoundTrip:
    def test_stores_and_retrieves_a_payload(self, store):
        obj_hash = store.put({"type": "fact", "subject": "user"})
        assert store.get(obj_hash) == {"type": "fact", "subject": "user"}

    def test_stores_and_retrieves_a_fact(self, store):
        fact = make_fact()
        obj_hash = store.put(fact.to_dict())
        assert Fact.from_dict(store.get(obj_hash)) == fact

    def test_put_returns_the_facts_own_hash(self, store):
        """The store's address and Fact.hash must agree, or lookups by fact break."""
        fact = make_fact()
        assert store.put(fact.to_dict()) == fact.hash

    def test_handles_non_ascii(self, store):
        obj_hash = store.put({"note": "café ☕ 日本語"})
        assert store.get(obj_hash)["note"] == "café ☕ 日本語"

    def test_handles_nested_structures(self, store):
        payload = {"a": [1, 2, {"b": {"c": [True, None]}}]}
        assert store.get(store.put(payload)) == payload

    def test_creates_the_objects_directory_on_demand(self, tmp_path):
        store = ObjectStore(tmp_path / "does" / "not" / "exist" / "objects")
        obj_hash = store.put({"a": 1})
        assert store.get(obj_hash) == {"a": 1}


class TestDeduplication:
    def test_identical_content_stored_once(self, store):
        for _ in range(10):
            store.put({"type": "fact", "subject": "user"})
        assert store.count() == 1

    def test_identical_content_yields_the_same_hash(self, store):
        assert store.put({"a": 1, "b": 2}) == store.put({"b": 2, "a": 1})

    def test_reasserting_a_fact_unchanged_costs_nothing(self, store):
        """The property that makes carrying facts across commits nearly free."""
        fact = make_fact()
        store.put(fact.to_dict())
        size_before = store.size_on_disk()

        for _ in range(50):
            store.put(fact.to_dict())

        assert store.count() == 1
        assert store.size_on_disk() == size_before

    def test_a_changed_fact_is_a_new_object(self, store):
        store.put(make_fact(object="Python").to_dict())
        store.put(make_fact(object="Rust").to_dict())
        assert store.count() == 2

    def test_reaffirmation_is_a_new_object(self, store):
        """Same claim, new assertion — the diff engine needs to see both."""
        fact = make_fact()
        store.put(fact.to_dict())
        store.put(fact.reaffirm(asserted_at="2026-12-01T00:00:00+00:00").to_dict())
        assert store.count() == 2


class TestLayout:
    def test_uses_two_character_fanout_directories(self, store):
        """256 buckets keeps directories small enough for the filesystem."""
        obj_hash = store.put({"a": 1})
        expected = store.objects_dir / obj_hash[:2] / obj_hash[2:]
        assert expected.exists()

    def test_stored_bytes_are_compressed(self, store):
        obj_hash = store.put({"padding": "x" * 5000})
        on_disk = (store.objects_dir / obj_hash[:2] / obj_hash[2:]).read_bytes()
        assert len(on_disk) < 5000

    def test_stored_bytes_decompress_to_canonical_json(self, store):
        obj_hash = store.put({"b": 2, "a": 1})
        on_disk = (store.objects_dir / obj_hash[:2] / obj_hash[2:]).read_bytes()
        assert zlib.decompress(on_disk) == b'{"a":1,"b":2}'

    def test_leaves_no_temp_files_behind(self, store):
        store.put({"a": 1})
        assert list(store.objects_dir.rglob("*.tmp")) == []


class TestIntegrity:
    def test_get_raises_on_missing_object(self, store):
        with pytest.raises(ObjectNotFoundError):
            store.get("a" * 64)

    def test_detects_tampered_contents(self, store):
        """The whole point of content addressing: edits are self-announcing."""
        obj_hash = store.put({"balance": 100})
        path = store.objects_dir / obj_hash[:2] / obj_hash[2:]
        path.write_bytes(zlib.compress(b'{"balance":999999}'))

        with pytest.raises(CorruptObjectError, match="do not match address"):
            store.get(obj_hash)

    def test_detects_undecompressable_data(self, store):
        obj_hash = store.put({"a": 1})
        path = store.objects_dir / obj_hash[:2] / obj_hash[2:]
        path.write_bytes(b"this is not zlib data")

        with pytest.raises(CorruptObjectError, match="could not be decompressed"):
            store.get(obj_hash)

    def test_detects_truncation(self, store):
        obj_hash = store.put({"padding": "x" * 1000})
        path = store.objects_dir / obj_hash[:2] / obj_hash[2:]
        path.write_bytes(path.read_bytes()[:10])

        with pytest.raises(CorruptObjectError):
            store.get(obj_hash)

    def test_verify_returns_empty_for_a_healthy_store(self, store):
        for i in range(5):
            store.put({"n": i})
        assert store.verify() == []

    def test_verify_reports_the_broken_object(self, store):
        good = store.put({"n": 1})
        bad = store.put({"n": 2})
        (store.objects_dir / bad[:2] / bad[2:]).write_bytes(b"garbage")

        assert store.verify() == [bad]
        assert good not in store.verify()


class TestHashValidation:
    """A hash reaches the filesystem, so it is untrusted input."""

    @pytest.mark.parametrize(
        "attack",
        [
            "../../../etc/passwd",
            "..",
            "/absolute/path",
            "a" * 63,
            "a" * 65,
            "z" * 64,  # right length, not hex
        ],
    )
    def test_rejects_malformed_hashes(self, store, attack):
        with pytest.raises((ValueError, TypeError)):
            store.get(attack)

    def test_rejects_non_string_hashes(self, store):
        with pytest.raises(TypeError):
            store.get(12345)  # type: ignore[arg-type]

    def test_contains_is_false_rather_than_raising_on_junk(self, store):
        """Membership is a question, not an assertion — junk means 'no'."""
        assert store.contains("../../etc/passwd") is False
        assert store.contains("nonsense") is False


class TestIntrospection:
    def test_contains_reflects_presence(self, store):
        obj_hash = store.put({"a": 1})
        assert store.contains(obj_hash)
        assert obj_hash in store
        assert "b" * 64 not in store

    def test_count_of_an_empty_store_is_zero(self, store):
        assert store.count() == 0

    def test_iter_hashes_yields_every_stored_object(self, store):
        written = {store.put({"n": i}) for i in range(20)}
        assert set(store.iter_hashes()) == written

    def test_size_on_disk_grows_with_distinct_objects(self, store):
        store.put({"n": 0})
        first = store.size_on_disk()
        store.put({"padding": "y" * 2000})
        assert store.size_on_disk() > first

    def test_size_on_disk_of_an_empty_store_is_zero(self, store):
        assert store.size_on_disk() == 0


class TestIsObjectHash:
    @pytest.mark.parametrize("value", ["a" * 64, "0123456789abcdef" * 4, "F" * 64])
    def test_accepts_64_hex_characters(self, value):
        assert is_object_hash(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "a" * 63,
            "a" * 65,
            "z" * 64,
            "",
            "../../etc/passwd",
            12345,
            None,
        ],
    )
    def test_rejects_anything_else(self, value):
        assert is_object_hash(value) is False


class TestResolvePrefix:
    def test_resolves_a_unique_prefix(self, store):
        obj_hash = store.put({"a": 1})
        assert store.resolve_prefix(obj_hash[:8]) == obj_hash

    def test_a_full_hash_resolves_to_itself(self, store):
        obj_hash = store.put({"a": 1})
        assert store.resolve_prefix(obj_hash) == obj_hash

    def test_raises_on_no_match(self, store):
        store.put({"a": 1})
        with pytest.raises(ObjectNotFoundError):
            store.resolve_prefix("deadbeef")

    def test_raises_on_full_hash_with_no_match(self, store):
        with pytest.raises(ObjectNotFoundError):
            store.resolve_prefix("a" * 64)

    def test_raises_on_ambiguous_prefix(self, store, monkeypatch):
        """Two objects sharing a prefix is astronomically unlikely with real
        hashes, so the ambiguity path is tested by monkeypatching iter_hashes
        rather than trying to mine a SHA-256 collision."""
        obj_hash = store.put({"a": 1})
        other = obj_hash[:8] + ("0" if obj_hash[8] != "0" else "1") + obj_hash[9:]
        monkeypatch.setattr(store, "iter_hashes", lambda: iter([obj_hash, other]))
        with pytest.raises(AmbiguousPrefixError):
            store.resolve_prefix(obj_hash[:8])

    def test_rejects_prefix_shorter_than_min_len(self, store):
        with pytest.raises(ValueError, match="at least"):
            store.resolve_prefix("abc")

    def test_rejects_non_hex_prefix(self, store):
        with pytest.raises(ValueError, match="hexadecimal"):
            store.resolve_prefix("zzzz")


class TestHashObject:
    def test_matches_what_put_would_store(self, store):
        payload = {"type": "fact", "subject": "user"}
        assert hash_object(payload) == store.put(payload)

    def test_does_not_write_anything(self, store):
        hash_object({"a": 1})
        assert store.count() == 0
