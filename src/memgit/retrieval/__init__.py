"""Slice 7 — scoped retrieval.

Embeds facts, retrieves the top-k most relevant to a query out of a
:class:`~memgit.core.state.MemoryState`, and blends similarity with decayed
confidence (:mod:`memgit.core.decay`) into one ranking. Commit/branch
scoping falls out of retrieving from a already-materialized ``MemoryState``
rather than a separate per-commit index — see
:mod:`memgit.retrieval.rank`'s module docstring.

A separate top-level package from ``memgit.core``, mirroring
``memgit.replay``: the optional real embedding backend this seam allows for
later is a network/model dependency, and ``memgit.core.*`` stays import-clean
per PLAN.md's repo conventions.
"""

from __future__ import annotations

from memgit.retrieval.embed import Embedder, HashingEmbedder, default_embedder, embed_text
from memgit.retrieval.index import VectorIndex, VectorIndexError
from memgit.retrieval.rank import RetrievalResult, Retrieved, Retriever, cosine

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "default_embedder",
    "embed_text",
    "VectorIndex",
    "VectorIndexError",
    "RetrievalResult",
    "Retrieved",
    "Retriever",
    "cosine",
]
