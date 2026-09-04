"""Tests for the two-block system prompt.

Pinning the cache breakpoint's placement matters more than it looks: block 0
must be stable across states so its ``cache_control`` breakpoint keeps
paying off turn after turn, and block 1 must carry everything that changes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from memgit.agent.prompt import build_system_prompt
from memgit.core.cardinality import CardinalityMap
from memgit.core.decay import DecayPolicy
from memgit.core.fact import Fact
from memgit.core.state import MemoryState
from memgit.retrieval.embed import HashingEmbedder
from memgit.retrieval.index import VectorIndex
from memgit.retrieval.rank import Retriever

NOW = datetime(2026, 9, 3, tzinfo=timezone.utc)


def fact(**overrides) -> Fact:
    defaults = dict(subject="user", predicate="prefers_language", object="Python", confidence=0.9)
    defaults.update(overrides)
    return Fact(**defaults)


def _retrieve(tmp_path, state, query="language", **kwargs):
    embedder = HashingEmbedder(dim=32)
    index = VectorIndex.open(tmp_path / "embeddings", embedder_id=embedder.id, dim=32)
    retriever = Retriever(index, embedder, DecayPolicy.default_map())
    return retriever.retrieve(state, query, as_of=NOW, **kwargs)


class TestBuildSystemPrompt:
    def test_returns_exactly_two_blocks(self):
        prompt = build_system_prompt(MemoryState.empty(), CardinalityMap.default_map(), head=None)
        assert len(prompt) == 2

    def test_only_the_first_block_carries_the_cache_breakpoint(self):
        prompt = build_system_prompt(MemoryState.empty(), CardinalityMap.default_map(), head=None)
        assert prompt[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in prompt[1]

    def test_first_block_is_identical_across_different_states(self):
        empty_prompt = build_system_prompt(MemoryState.empty(), CardinalityMap.default_map(), head=None)
        full_prompt = build_system_prompt(
            MemoryState.from_facts([fact()]), CardinalityMap.default_map(), head="deadbeef" * 5
        )
        assert empty_prompt[0]["text"] == full_prompt[0]["text"]

    def test_empty_state_renders_an_explicit_no_memories_line(self):
        prompt = build_system_prompt(MemoryState.empty(), CardinalityMap.default_map(), head=None)
        assert "No memories recorded yet" in prompt[1]["text"]

    def test_nonempty_state_renders_verbatim(self):
        state = MemoryState.from_facts([fact()])
        prompt = build_system_prompt(state, CardinalityMap.default_map(), head="deadbeef" * 5)
        assert state.render() in prompt[1]["text"]
        assert "deadbeef" in prompt[1]["text"]

    def test_multi_predicates_are_named_in_the_first_block(self):
        cardinality = CardinalityMap.default_map().with_predicate("likes", "multi")
        prompt = build_system_prompt(MemoryState.empty(), cardinality, head=None)
        assert "likes" in prompt[0]["text"]

    def test_no_multi_predicates_says_so(self):
        prompt = build_system_prompt(MemoryState.empty(), CardinalityMap.default_map(), head=None)
        assert "contradiction" in prompt[0]["text"]


class TestRetrievalMode:
    def test_retrieved_none_is_byte_identical_to_full_injection(self, tmp_path):
        state = MemoryState.from_facts([fact()])
        without = build_system_prompt(state, CardinalityMap.default_map(), head="deadbeef" * 5)
        explicit_none = build_system_prompt(
            state, CardinalityMap.default_map(), head="deadbeef" * 5, retrieved=None
        )
        assert without == explicit_none

    def test_three_blocks_under_retrieval(self, tmp_path):
        state = MemoryState.from_facts([fact()])
        result = _retrieve(tmp_path, state)
        prompt = build_system_prompt(state, CardinalityMap.default_map(), head="deadbeef" * 5, retrieved=result)
        assert len(prompt) == 3

    def test_first_block_still_carries_the_only_cache_breakpoint(self, tmp_path):
        state = MemoryState.from_facts([fact()])
        result = _retrieve(tmp_path, state)
        prompt = build_system_prompt(state, CardinalityMap.default_map(), head="deadbeef" * 5, retrieved=result)
        assert prompt[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in prompt[1]
        assert "cache_control" not in prompt[2]

    def test_key_inventory_lists_every_key_not_just_the_retrieved_ones(self, tmp_path):
        state = MemoryState.from_facts(
            [fact(), fact(predicate="favorite_editor", object="Vim")]
        )
        # Retrieve with k=1 so only one of the two facts is in the retrieved block.
        result = _retrieve(tmp_path, state, query="prefers language", k=1)
        prompt = build_system_prompt(state, CardinalityMap.default_map(), head="deadbeef" * 5, retrieved=result)
        assert "prefers_language" in prompt[1]["text"]
        assert "favorite_editor" in prompt[1]["text"]

    def test_retrieved_block_shows_the_n_of_m_header(self, tmp_path):
        state = MemoryState.from_facts([fact(), fact(predicate="favorite_editor", object="Vim")])
        result = _retrieve(tmp_path, state, query="prefers language", k=1)
        prompt = build_system_prompt(state, CardinalityMap.default_map(), head="deadbeef" * 5, retrieved=result)
        assert "1 of 2 fact(s)" in prompt[2]["text"]

    def test_subset_warning_appears_only_in_retrieval_mode(self, tmp_path):
        state = MemoryState.from_facts([fact()])
        result = _retrieve(tmp_path, state)
        full = build_system_prompt(state, CardinalityMap.default_map(), head="deadbeef" * 5)
        retrieval = build_system_prompt(
            state, CardinalityMap.default_map(), head="deadbeef" * 5, retrieved=result
        )
        assert "only the subset most relevant" not in full[0]["text"]
        assert "only the subset most relevant" in retrieval[0]["text"]

    def test_empty_state_key_inventory_says_so(self, tmp_path):
        result = _retrieve(tmp_path, MemoryState.empty())
        prompt = build_system_prompt(MemoryState.empty(), CardinalityMap.default_map(), head=None, retrieved=result)
        assert "No memories recorded yet" in prompt[1]["text"]

    def test_retrieved_block_renders_the_facts(self, tmp_path):
        state = MemoryState.from_facts([fact()])
        result = _retrieve(tmp_path, state)
        prompt = build_system_prompt(state, CardinalityMap.default_map(), head="deadbeef" * 5, retrieved=result)
        assert "prefers_language" in prompt[2]["text"]
        assert "Python" in prompt[2]["text"]
