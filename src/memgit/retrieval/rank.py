"""Ranking and scoping: ``Retriever`` turns a query into the top-k relevant facts.

**Commit/branch scoping is not something this module implements — it falls
out of the signature.** :meth:`Retriever.retrieve` takes a
:class:`~memgit.core.state.MemoryState`, never a
:class:`~memgit.core.repository.Repository` and never a revision string. It
iterates ``state.facts`` and nothing else, so there is no code path by which
a fact absent from that state can appear in the result — even if that
fact's vector is already sitting in the on-disk index from a later commit or
a different branch. ``Repository.state(rev)`` already materializes exactly
the right fact set (and is already tested); building a second,
commit-aware index here would mean re-implementing that guarantee instead of
inheriting it, with a real chance of getting it wrong.

**Scoring blends similarity with decayed confidence, linearly, not
multiplicatively:** ``score = (1 - w) * similarity + w * decayed_confidence``.
A product would let a fact that has decayed toward zero vanish from the
results even when it is the *only* relevant belief in scope — exactly the
stale belief a debugging tool exists to surface. Linear degrades gracefully
at both ends of ``w`` (0 = pure similarity, 1 = pure decayed confidence),
which is also what makes both extremes easy to test independently.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime

from memgit.core.decay import DecayPolicy
from memgit.core.fact import Fact
from memgit.core.state import MemoryState
from memgit.retrieval.embed import Embedder, embed_text
from memgit.retrieval.index import VectorIndex

__all__ = ["RetrievalResult", "Retrieved", "Retriever", "cosine"]

_DEFAULT_CONFIDENCE_WEIGHT = 0.25


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity between two vectors, not assumed pre-normalized.

    Returns ``0.0`` for a zero vector on either side rather than dividing by
    zero — an empty-text embedding (see ``HashingEmbedder``) is a legitimate
    input, not an error.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True, slots=True)
class Retrieved:
    """One fact in a :class:`RetrievalResult`, with its scoring breakdown."""

    fact: Fact
    score: float
    similarity: float
    confidence: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What one :meth:`Retriever.retrieve` call produced.

    Attributes:
        commit: Echoed from the state's own ``commit`` field, purely as
            provenance — never used to enforce scoping, since a legitimately
            derived state (an ablated one, see ``state.py``) has
            ``commit=None`` and must still be retrievable against.
        candidates: How many facts were in scope before ranking and
            filtering — what lets a caller say "8 of 214 facts", the same
            auditability instinct as a diff embedding the cardinality map it
            used.
    """

    query: str
    facts: tuple[Retrieved, ...]
    commit: str | None
    candidates: int
    embedder: str
    as_of: str

    def render(self) -> str:
        """A deterministic, human- and prompt-readable rendering of this result."""
        if not self.facts:
            return "No matching facts."
        lines = []
        for retrieved in self.facts:
            fact = retrieved.fact
            lines.append(
                f"  {fact.subject} {fact.predicate} {fact.object} "
                f"(score={retrieved.score:.2f}, sim={retrieved.similarity:.2f}, "
                f"conf={retrieved.confidence:.2f})"
            )
        return "\n".join(lines)


class Retriever:
    """Ranks a :class:`MemoryState`'s facts against a query.

    Args:
        index: Where fact vectors are cached. A cache miss is embedded and
            written back on the spot — retrieval is never *wrong* just
            because ``memgit embed`` hasn't been run, only slower the first
            time.
        embedder: Must be the same embedder ``index`` was opened for; the
            caller (typically :meth:`Repository.retriever`) is responsible
            for that pairing.
        policy: The decay policy blended into ranking.
        confidence_weight: How much decayed confidence counts against raw
            similarity, in ``[0, 1]``. Default ``0.25``: relevance
            dominates, decayed confidence breaks ties and demotes stale
            beliefs without hiding them.
    """

    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        policy: DecayPolicy,
        *,
        confidence_weight: float = _DEFAULT_CONFIDENCE_WEIGHT,
    ) -> None:
        if not 0.0 <= confidence_weight <= 1.0:
            raise ValueError(f"confidence_weight must be within [0.0, 1.0], got {confidence_weight!r}")
        self.index = index
        self.embedder = embedder
        self.policy = policy
        self.confidence_weight = confidence_weight

    def _vector_for(self, fact: Fact) -> tuple[float, ...]:
        cached = self.index.get(fact.hash)
        if cached is not None:
            return cached
        vector = self.embedder.embed(embed_text(fact))
        self.index.put(fact.hash, vector)
        # Read back through the same float32 pack encoding a cache hit
        # would use, rather than returning the just-computed float64
        # vector directly — otherwise a fact's similarity score would
        # depend on whether this call happened to be the one that warmed
        # the cache, which is exactly the kind of nondeterminism a
        # debugging tool cannot afford.
        readback = self.index.get(fact.hash)
        assert readback is not None, "just-written vector must be readable back"
        return readback

    def retrieve(
        self,
        state: MemoryState,
        query: str,
        *,
        k: int = 8,
        as_of: datetime,
        subjects: Collection[str] | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """Rank ``state``'s facts against ``query`` and return the top ``k``.

        The candidate set is exactly ``state.facts`` (optionally narrowed by
        ``subjects``) — see the module docstring for why that, not a
        separate index, is the scoping guarantee.
        """
        candidates = tuple(
            fact for fact in state.facts if subjects is None or fact.subject in subjects
        )
        query_vector = self.embedder.embed(query)

        scored: list[Retrieved] = []
        for fact in candidates:
            similarity = max(0.0, cosine(query_vector, self._vector_for(fact)))
            confidence = self.policy.decayed(fact, as_of=as_of)
            score = (1 - self.confidence_weight) * similarity + self.confidence_weight * confidence
            scored.append(Retrieved(fact=fact, score=score, similarity=similarity, confidence=confidence))

        scored = [r for r in scored if r.score >= min_score]
        scored.sort(key=lambda r: (-r.score, -r.confidence, r.fact.hash))

        return RetrievalResult(
            query=query,
            facts=tuple(scored[:k]),
            commit=state.commit,
            candidates=len(candidates),
            embedder=self.embedder.id,
            as_of=as_of.isoformat(),
        )
