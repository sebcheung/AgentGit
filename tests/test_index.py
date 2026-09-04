"""Tests for VectorIndex — the fact_hash -> vector cache.

Covers round-tripping, idempotence, cross-embedder isolation, and the two
crash-safety properties: the pack is written before the index, and a
truncated pack or an out-of-range index entry is detected rather than
silently misread.
"""

from __future__ import annotations

import struct

import pytest

from memgit.retrieval.index import VectorIndex, VectorIndexError


def _index(tmp_path, *, embedder_id="hash-v1/4", dim=4):
    return VectorIndex.open(tmp_path / "embeddings", embedder_id=embedder_id, dim=dim)


class TestRoundTrip:
    def test_put_then_get(self, tmp_path):
        index = _index(tmp_path)
        vector = (0.1, 0.2, 0.3, 0.4)
        index.put("abc123", vector)
        assert index.get("abc123") == pytest.approx(vector)

    def test_missing_key_returns_none(self, tmp_path):
        index = _index(tmp_path)
        assert index.get("nope") is None
        assert "nope" not in index

    def test_contains_after_put(self, tmp_path):
        index = _index(tmp_path)
        index.put("abc123", (0.0, 0.0, 0.0, 1.0))
        assert "abc123" in index

    def test_put_same_hash_twice_appends_nothing(self, tmp_path):
        index = _index(tmp_path)
        index.put("abc123", (0.0, 0.0, 0.0, 1.0))
        index.put("abc123", (9.0, 9.0, 9.0, 9.0))  # different vector, ignored
        assert index.get("abc123") == pytest.approx((0.0, 0.0, 0.0, 1.0))
        pack = tmp_path / "embeddings" / "hash-v1_4" / "vectors.pack"
        assert pack.stat().st_size == 4 * 4  # one record only

    def test_force_overwrites_the_cached_vector(self, tmp_path):
        index = _index(tmp_path)
        index.put("abc123", (0.0, 0.0, 0.0, 1.0))
        index.put("abc123", (1.0, 0.0, 0.0, 0.0), force=True)
        assert index.get("abc123") == pytest.approx((1.0, 0.0, 0.0, 0.0))

    def test_persists_across_instances(self, tmp_path):
        first = _index(tmp_path)
        first.put("abc123", (0.1, 0.2, 0.3, 0.4))
        second = _index(tmp_path)
        assert second.get("abc123") == pytest.approx((0.1, 0.2, 0.3, 0.4))

    def test_len_counts_records(self, tmp_path):
        index = _index(tmp_path)
        index.put("a", (0.0,) * 4)
        index.put("b", (1.0,) * 4)
        assert len(index) == 2

    def test_keys_lists_every_cached_hash(self, tmp_path):
        index = _index(tmp_path)
        index.put("a", (0.0,) * 4)
        index.put("b", (1.0,) * 4)
        assert set(index.keys()) == {"a", "b"}


class TestValidation:
    def test_dim_mismatch_rejected(self, tmp_path):
        index = _index(tmp_path, dim=4)
        with pytest.raises(VectorIndexError):
            index.put("abc123", (0.1, 0.2, 0.3))


class TestIsolation:
    def test_two_embedder_ids_get_separate_directories(self, tmp_path):
        one = _index(tmp_path, embedder_id="hash-v1/4", dim=4)
        two = _index(tmp_path, embedder_id="hash-v1/8", dim=8)
        one.put("shared-hash", (1.0, 0.0, 0.0, 0.0))
        two.put("shared-hash", (0.0,) * 8)
        assert one.get("shared-hash") == pytest.approx((1.0, 0.0, 0.0, 0.0))
        assert two.get("shared-hash") == pytest.approx((0.0,) * 8)

    def test_reopening_with_a_different_embedder_id_at_the_same_path_raises(self, tmp_path):
        base = tmp_path / "embeddings" / "hash-v1_4"
        one = VectorIndex(base, embedder_id="hash-v1/4", dim=4)
        one.put("a", (1.0, 0.0, 0.0, 0.0))
        mismatched = VectorIndex(base, embedder_id="a-different-embedder", dim=4)
        with pytest.raises(VectorIndexError):
            mismatched.get("a")


class TestCorruption:
    def test_truncated_pack_detected(self, tmp_path):
        index = _index(tmp_path)
        index.put("abc123", (0.1, 0.2, 0.3, 0.4))
        pack = tmp_path / "embeddings" / "hash-v1_4" / "vectors.pack"
        pack.write_bytes(pack.read_bytes()[:4])  # truncate mid-record
        with pytest.raises(VectorIndexError):
            index.get("abc123")

    def test_index_entry_past_eof_detected(self, tmp_path):
        index = _index(tmp_path)
        index.put("abc123", (0.1, 0.2, 0.3, 0.4))
        index_path = tmp_path / "embeddings" / "hash-v1_4" / "index.json"
        payload = index_path.read_text(encoding="utf-8")
        payload = payload.replace('"abc123": 0', '"abc123": 99')
        index_path.write_text(payload, encoding="utf-8")

        fresh = _index(tmp_path)
        with pytest.raises(VectorIndexError):
            fresh.get("abc123")

    def test_atomic_rewrite_leaves_no_lock_file(self, tmp_path):
        index = _index(tmp_path)
        index.put("abc123", (0.1, 0.2, 0.3, 0.4))
        lock = tmp_path / "embeddings" / "hash-v1_4" / "index.json.lock"
        assert not lock.exists()

    def test_orphan_pack_bytes_without_an_index_entry_are_invisible(self, tmp_path):
        """The crash-ordering property: append pack bytes without updating
        the index (simulating a crash between the two writes), and confirm
        the orphan record is simply not there — garbage, not corruption."""
        index = _index(tmp_path)
        index.path.mkdir(parents=True, exist_ok=True)
        with (index.path / "vectors.pack").open("ab") as handle:
            handle.write(struct.pack("<4f", 9.0, 9.0, 9.0, 9.0))

        fresh = _index(tmp_path)
        assert len(fresh) == 0
        assert fresh.get("phantom") is None

        # A real put still works and lands after the orphan bytes.
        fresh.put("abc123", (0.1, 0.2, 0.3, 0.4))
        assert fresh.get("abc123") == pytest.approx((0.1, 0.2, 0.3, 0.4))
