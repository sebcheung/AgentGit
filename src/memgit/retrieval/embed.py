"""The ``Embedder`` seam: text in, a vector out.

Shaped exactly like :class:`~memgit.agent.client.LLMClient` — a
``@runtime_checkable`` one-method Protocol, so a test fake satisfies it
structurally without inheriting from it, and a real backend drops in behind
the same seam later without any caller changing.

``Embedder.id`` is load-bearing, not decorative: it is the *only* thing that
invalidates a cached vector (see ``retrieval/index.py``), and it namespaces
where those vectors are stored on disk. Two embedders that produce
different vectors must never share an ``id``.

**This module ships exactly one implementation: :class:`HashingEmbedder`.**
It is lexical, not semantic — deterministic feature hashing over tokens and
character n-grams, zero dependencies, fully offline. It will not match
``"favourite editor"`` to ``"preferred text editor"``; it is honest about
being a bag-of-hashed-features baseline, not a stand-in for a real sentence
embedding model.

A real backend (``sentence-transformers``, or a hosted embedding API) is a
deliberate non-goal of this slice: the former pulls torch and a multi-GB
dependency footprint into a project whose only other runtime dependency is
``typer``; the latter means onboarding a second API key and a second network
surface — Anthropic has no first-party embeddings endpoint, verified against
the installed SDK — with a network call sitting in the fact-write path. Both
plug in behind :class:`Embedder` without touching anything else in
``memgit.retrieval`` or ``memgit.agent`` if a later slice wants them; Voyage
AI (Anthropic's own documented recommendation) is the natural drop-in.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from memgit.core.fact import Fact

__all__ = ["Embedder", "EmbedderError", "HashingEmbedder", "UnknownEmbedderError", "default_embedder", "embed_text"]


class EmbedderError(Exception):
    """Raised when an embedder cannot be constructed or used."""


class UnknownEmbedderError(EmbedderError):
    """Raised by :func:`default_embedder` for an unrecognized embedder id."""


@runtime_checkable
class Embedder(Protocol):
    """What retrieval needs from an embedding backend: text in, a vector out.

    Both :class:`HashingEmbedder` and a test's scripted fake satisfy this
    structurally — ``isinstance(x, Embedder)`` works on either without
    either one inheriting from it, the same contract
    :class:`~memgit.agent.client.LLMClient` establishes for the LLM seam.
    """

    @property
    def id(self) -> str:
        """A stable identifier for this embedder and its configuration.

        The only thing that invalidates a cached vector, and the string
        that namespaces the on-disk vector index. Must change whenever the
        vectors it produces would no longer compare meaningfully against
        ones already stored — a version bump, a changed dimension, a
        different underlying model.
        """
        ...

    @property
    def dim(self) -> int:
        """The dimensionality of vectors this embedder produces."""
        ...

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed one piece of text into an L2-normalized vector of length :attr:`dim`."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed several texts at once, in order. A default loop is a valid implementation."""
        ...


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Word tokens plus character 3-grams, lowercased.

    Character n-grams give the hashing trick some resilience to minor
    spelling/pluralization differences that a pure word-level bag-of-words
    would miss entirely — still lexical, not semantic, but a little less
    brittle for a zero-dependency default.
    """
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    tokens = list(words)
    for word in words:
        padded = f"^{word}$"
        for i in range(len(padded) - 2):
            tokens.append(padded[i : i + 3])
    return tokens


class HashingEmbedder:
    """A deterministic, zero-dependency, offline embedder.

    Feature-hashes each token into one of :attr:`dim` buckets using
    ``sha256`` — never Python's built-in ``hash()``, which is salted per
    process and would make vectors differ across runs, a silent
    cross-process correctness bug for anything persisted to disk. The sign
    of each contribution is likewise derived from the hash, and the result
    is L2-normalized so cosine similarity behaves as a bounded, comparable
    score.
    """

    _PREFIX = "hash-v1"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise EmbedderError(f"dim must be positive, got {dim!r}")
        self._dim = dim

    @property
    def id(self) -> str:
        """See :attr:`Embedder.id` — encodes the scheme and dimension."""
        return f"{self._PREFIX}/{self._dim}"

    @property
    def dim(self) -> int:
        """See :attr:`Embedder.dim`."""
        return self._dim

    def embed(self, text: str) -> tuple[float, ...]:
        """See :meth:`Embedder.embed` — feature-hashes tokens into buckets."""
        vector = [0.0] * self._dim
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(v / norm for v in vector)

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """See :meth:`Embedder.embed_batch`."""
        return tuple(self.embed(text) for text in texts)

    def __repr__(self) -> str:
        return f"HashingEmbedder(dim={self._dim})"


def embed_text(fact: Fact) -> str:
    """The text a fact embeds as — shared by every embedder, every backend.

    Splits the predicate's underscores into words (``prefers_language`` ->
    ``prefers language``): otherwise it is one opaque token to any
    embedder, lexical or semantic. Includes ``source_text`` when present —
    the richest natural-language signal available, and the query side of a
    retrieval is natural language too.
    """
    predicate_words = fact.predicate.replace("_", " ")
    text = f"{fact.subject} {predicate_words} {fact.object}"
    if fact.source_text:
        text = f"{text}\n{fact.source_text}"
    return text


def default_embedder(config: dict[str, Any] | None = None) -> Embedder:
    """The repo's configured embedder, or :class:`HashingEmbedder` if unset.

    Mirrors :func:`~memgit.agent.client.default_client`'s shape: a factory
    that raises a clear, catchable error for a bad configuration instead of
    letting an import error surface from somewhere unexpected.

    Args:
        config: The ``"retrieval"`` section of ``Repository.config()``, or
            ``None``. ``{"embedder": "hash-v1/256"}`` selects the hashing
            embedder at that dimension explicitly; an absent or empty config
            uses the default dimension.

    Raises:
        UnknownEmbedderError: ``config["embedder"]`` doesn't name a known
            embedder.
    """
    spec = (config or {}).get("embedder")
    if spec is None:
        return HashingEmbedder()

    prefix, _, rest = spec.partition("/")
    if prefix == "hash-v1":
        try:
            dim = int(rest) if rest else 256
        except ValueError:
            raise UnknownEmbedderError(f"invalid hashing embedder spec: {spec!r}") from None
        return HashingEmbedder(dim=dim)

    raise UnknownEmbedderError(
        f"unknown embedder {spec!r}: no backend besides 'hash-v1/<dim>' is built in"
    )
