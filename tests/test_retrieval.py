"""Tests for the ranking function and the commit/branch scoping guarantee.

The two scoping tests are the headline of this module: they prove a fact is
unreachable from ``Retriever.retrieve`` unless it is in the ``MemoryState``
passed in, *even when its vector is already cached* — the property the
whole retrieval design leans on instead of building a second, commit-aware
index.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memgit.core.decay import DecayPolicy
from memgit.core.fact import Fact
from memgit.core.state import MemoryState
from memgit.retrieval.embed import HashingEmbedder
from memgit.retrieval.index import VectorIndex
from memgit.retrieval.rank import Retriever, cosine

NOW = datetime(2026, 9, 3, tzinfo=UTC)


def _fact(subject="user", predicate="prefers_language", obj="Python", **kwargs):
    defaults = {"confidence": 1.0, "asserted_at": NOW.isoformat()}
    defaults.update(kwargs)
    return Fact(subject=subject, predicate=predicate, object=obj, **defaults)


def _retriever(tmp_path, *, confidence_weight=0.25, dim=64):
    embedder = HashingEmbedder(dim=dim)
    index = VectorIndex.open(tmp_path / "embeddings", embedder_id=embedder.id, dim=dim)
    return Retriever(index, embedder, DecayPolicy.default_map(), confidence_weight=confidence_weight)


class TestCosine:
    def test_identical_vectors_are_similarity_one(self):
        assert cosine((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert cosine((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_opposite_vectors_are_negative_one(self):
        assert cosine((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)

    def test_zero_vector_is_zero_not_a_crash(self):
        assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0


class TestBlend:
    def test_pure_similarity_at_w_zero(self, tmp_path):
        retriever = _retriever(tmp_path, confidence_weight=0.0)
        state = MemoryState.from_facts([_fact(confidence=0.01)])
        result = retriever.retrieve(state, "prefers language Python", as_of=NOW)
        assert result.facts[0].score == pytest.approx(result.facts[0].similarity)

    def test_pure_confidence_at_w_one(self, tmp_path):
        retriever = _retriever(tmp_path, confidence_weight=1.0)
        state = MemoryState.from_facts([_fact(confidence=0.42)])
        result = retriever.retrieve(state, "something totally unrelated", as_of=NOW)
        assert result.facts[0].score == pytest.approx(result.facts[0].confidence)
        assert result.facts[0].confidence == pytest.approx(0.42)


class TestRankingBehavior:
    def test_top_k_truncates(self, tmp_path):
        retriever = _retriever(tmp_path)
        facts = [_fact(obj=f"lang{i}") for i in range(10)]
        state = MemoryState.from_facts(facts)
        result = retriever.retrieve(state, "prefers language", k=3, as_of=NOW)
        assert len(result.facts) == 3
        assert result.candidates == 10

    def test_min_score_filters(self, tmp_path):
        retriever = _retriever(tmp_path, confidence_weight=0.0)
        state = MemoryState.from_facts([_fact()])
        result = retriever.retrieve(state, "completely different topic entirely", min_score=0.99, as_of=NOW)
        assert result.facts == ()

    def test_subject_filter_narrows_candidates(self, tmp_path):
        retriever = _retriever(tmp_path)
        facts = [_fact(subject="user"), _fact(subject="project:memgit")]
        state = MemoryState.from_facts(facts)
        result = retriever.retrieve(state, "prefers language", subjects={"user"}, as_of=NOW)
        assert result.candidates == 1
        assert all(r.fact.subject == "user" for r in result.facts)

    def test_deterministic_tie_break(self, tmp_path):
        retriever = _retriever(tmp_path, confidence_weight=1.0)  # force ties via identical confidence
        facts = [_fact(obj="A", confidence=0.5), _fact(obj="B", confidence=0.5)]
        state = MemoryState.from_facts(facts)
        first = retriever.retrieve(state, "query", as_of=NOW)
        second = retriever.retrieve(state, "query", as_of=NOW)
        assert [r.fact.hash for r in first.facts] == [r.fact.hash for r in second.facts]

    def test_lazy_embed_on_miss_populates_the_index(self, tmp_path):
        retriever = _retriever(tmp_path)
        fact = _fact()
        assert retriever.index.get(fact.hash) is None
        retriever.retrieve(MemoryState.from_facts([fact]), "prefers language", as_of=NOW)
        assert retriever.index.get(fact.hash) is not None

    def test_result_echoes_commit_and_candidate_count(self, tmp_path):
        retriever = _retriever(tmp_path)
        state = MemoryState(facts=(_fact(),), tree="deadbeef", commit="abc123")
        result = retriever.retrieve(state, "prefers language", as_of=NOW)
        assert result.commit == "abc123"
        assert result.candidates == 1
        assert result.embedder == retriever.embedder.id

    def test_retrieval_against_an_ablated_state_with_no_commit_works(self, tmp_path):
        retriever = _retriever(tmp_path)
        state = MemoryState.from_facts([_fact()])  # commit=None, as any derived state is
        assert state.commit is None
        result = retriever.retrieve(state, "prefers language", as_of=NOW)
        assert result.commit is None
        assert len(result.facts) == 1


class TestCommitAndBranchScoping:
    def test_a_later_commits_fact_is_never_returned_even_if_already_indexed(self, tmp_path):
        retriever = _retriever(tmp_path)
        early_fact = _fact(obj="Python")
        later_fact = _fact(predicate="favorite_editor", obj="Vim")

        # Index both vectors up front, as `memgit embed --all` would after
        # both commits exist -- the index alone must not leak the later one.
        retriever._vector_for(early_fact)
        retriever._vector_for(later_fact)

        state_at_commit_a = MemoryState.from_facts([early_fact])
        result = retriever.retrieve(state_at_commit_a, "editor vim", as_of=NOW)

        assert later_fact.hash not in {r.fact.hash for r in result.facts}
        assert result.candidates == 1

    def test_no_cross_branch_leak(self, tmp_path):
        retriever = _retriever(tmp_path)
        branch_a_fact = _fact(predicate="favorite_editor", obj="Vim")
        branch_b_fact = _fact(predicate="favorite_editor", obj="Emacs")

        retriever._vector_for(branch_a_fact)
        retriever._vector_for(branch_b_fact)

        state_a = MemoryState.from_facts([branch_a_fact])
        state_b = MemoryState.from_facts([branch_b_fact])

        result_a = retriever.retrieve(state_a, "favorite editor", as_of=NOW)
        result_b = retriever.retrieve(state_b, "favorite editor", as_of=NOW)

        assert {r.fact.hash for r in result_a.facts} == {branch_a_fact.hash}
        assert {r.fact.hash for r in result_b.facts} == {branch_b_fact.hash}


class TestValidation:
    def test_rejects_bad_confidence_weight(self, tmp_path):
        with pytest.raises(ValueError):
            _retriever(tmp_path, confidence_weight=1.5)


class TestRender:
    def test_empty_result_renders_a_clear_message(self, tmp_path):
        retriever = _retriever(tmp_path)
        result = retriever.retrieve(MemoryState.empty(), "anything", as_of=NOW)
        assert result.render() == "No matching facts."

    def test_nonempty_result_mentions_the_fact(self, tmp_path):
        retriever = _retriever(tmp_path)
        state = MemoryState.from_facts([_fact()])
        result = retriever.retrieve(state, "prefers language", as_of=NOW)
        assert "prefers_language" in result.render()
        assert "Python" in result.render()
