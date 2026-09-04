"""Tests for the Embedder seam and HashingEmbedder, the zero-dep default.

The load-bearing property this module protects: ``HashingEmbedder`` must be
deterministic *across processes*, not just within one — it uses ``sha256``
rather than Python's per-process-salted ``hash()`` specifically so a vector
computed today matches one computed tomorrow in a different interpreter.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from memgit.core.fact import Fact
from memgit.retrieval.embed import (
    Embedder,
    HashingEmbedder,
    UnknownEmbedderError,
    default_embedder,
    embed_text,
)


def _fact(**overrides):
    defaults = dict(subject="user", predicate="prefers_language", object="Python", confidence=0.9)
    defaults.update(overrides)
    return Fact(**defaults)


class TestProtocol:
    def test_hashing_embedder_satisfies_embedder(self):
        assert isinstance(HashingEmbedder(), Embedder)


class TestHashingEmbedder:
    def test_deterministic_across_calls(self):
        embedder = HashingEmbedder()
        assert embedder.embed("hello world") == embedder.embed("hello world")

    def test_different_text_different_vector(self):
        embedder = HashingEmbedder()
        assert embedder.embed("hello world") != embedder.embed("goodbye world")

    def test_dim_is_honored(self):
        embedder = HashingEmbedder(dim=64)
        assert embedder.dim == 64
        assert len(embedder.embed("anything")) == 64

    def test_vector_is_l2_normalized(self):
        embedder = HashingEmbedder()
        vector = embedder.embed("a reasonably long piece of text to embed")
        norm = math.sqrt(sum(v * v for v in vector))
        assert norm == pytest.approx(1.0)

    def test_empty_text_is_the_zero_vector_not_a_crash(self):
        embedder = HashingEmbedder()
        vector = embedder.embed("")
        assert all(v == 0.0 for v in vector)

    def test_embed_batch_matches_embed_one_by_one(self):
        embedder = HashingEmbedder()
        texts = ["one", "two", "three"]
        assert embedder.embed_batch(texts) == tuple(embedder.embed(t) for t in texts)

    def test_id_reflects_dimension(self):
        assert HashingEmbedder(dim=64).id != HashingEmbedder(dim=256).id
        assert HashingEmbedder(dim=128).id == HashingEmbedder(dim=128).id

    def test_rejects_non_positive_dim(self):
        with pytest.raises(Exception):
            HashingEmbedder(dim=0)

    def test_deterministic_across_processes(self):
        """sha256, not hash(), is what makes this true — hash() is salted
        per process and would silently break a persisted vector index."""
        script = (
            "from memgit.retrieval.embed import HashingEmbedder;"
            "print(HashingEmbedder().embed('deterministic please'))"
        )
        results = []
        for seed in ("0", "1", "12345"):
            env = {"PYTHONHASHSEED": seed}
            import os

            full_env = dict(os.environ)
            full_env.update(env)
            proc = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, env=full_env
            )
            assert proc.returncode == 0, proc.stderr
            results.append(proc.stdout.strip())
        assert len(set(results)) == 1


class TestEmbedText:
    def test_splits_predicate_underscores(self):
        text = embed_text(_fact(predicate="prefers_language"))
        assert "prefers language" in text
        assert "prefers_language" not in text

    def test_includes_source_text_when_present(self):
        text = embed_text(_fact(source_text="I really love Python"))
        assert "I really love Python" in text

    def test_omits_source_text_when_absent(self):
        text = embed_text(_fact())
        assert text.count("\n") == 0

    def test_includes_subject_and_object(self):
        text = embed_text(_fact())
        assert "user" in text
        assert "Python" in text


class TestDefaultEmbedder:
    def test_no_config_is_the_hashing_default(self):
        embedder = default_embedder(None)
        assert isinstance(embedder, HashingEmbedder)
        assert embedder.dim == 256

    def test_explicit_hash_spec_selects_dimension(self):
        embedder = default_embedder({"embedder": "hash-v1/64"})
        assert isinstance(embedder, HashingEmbedder)
        assert embedder.dim == 64

    def test_unknown_embedder_raises(self):
        with pytest.raises(UnknownEmbedderError):
            default_embedder({"embedder": "some-real-model/384"})

    def test_malformed_hash_spec_raises(self):
        with pytest.raises(UnknownEmbedderError):
            default_embedder({"embedder": "hash-v1/not-a-number"})
