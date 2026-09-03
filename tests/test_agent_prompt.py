"""Tests for the two-block system prompt.

Pinning the cache breakpoint's placement matters more than it looks: block 0
must be stable across states so its ``cache_control`` breakpoint keeps
paying off turn after turn, and block 1 must carry everything that changes.
"""

from __future__ import annotations

from memgit.agent.prompt import build_system_prompt
from memgit.core.cardinality import CardinalityMap
from memgit.core.fact import Fact
from memgit.core.state import MemoryState


def fact(**overrides) -> Fact:
    defaults = dict(subject="user", predicate="prefers_language", object="Python", confidence=0.9)
    defaults.update(overrides)
    return Fact(**defaults)


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
