"""Tests for MemoryAgent: the tool loop and the turn-to-commit contract.

Uses the ScriptedClient fake from conftest.py — zero network. Grouped by the
properties the design turns on: no-op turns don't commit, a committing turn
lands the right facts with the right provenance, tool calls are batched and
recovered from correctly, and memory (not the transcript) is what crosses
from one turn to the next.
"""

from __future__ import annotations

import pytest

from conftest import ScriptedClient, refusal_message, text_message, tool_use_message
from memgit.agent.client import AgentError
from memgit.agent.runtime import MemoryAgent
from memgit.core.fact import Fact


class TestNoOpTurns:
    def test_a_turn_with_no_tool_calls_does_not_commit(self, agent_repo):
        client = ScriptedClient(responses=[text_message("Hi there!")])
        agent = MemoryAgent(agent_repo, client=client)

        result = agent.turn("hello")

        assert result.commit is None
        assert result.reply == "Hi there!"
        assert agent_repo.head_commit() is None

    def test_record_empty_commits_anyway(self, agent_repo):
        client = ScriptedClient(responses=[text_message("Hi there!")])
        agent = MemoryAgent(agent_repo, client=client, record_empty=True)

        result = agent.turn("hello")

        assert result.commit is not None
        assert agent_repo.head_commit() == result.commit


class TestRememberCommits:
    def test_one_remember_lands_one_commit_with_the_fact(self, agent_repo):
        call = ("call_1", "remember", {
            "subject": "user", "predicate": "prefers_language", "object": "Python",
            "confidence": 0.9, "source_text": "I like Python.",
        })
        client = ScriptedClient(responses=[
            tool_use_message(call),
            text_message("Noted."),
        ])
        agent = MemoryAgent(agent_repo, client=client)

        result = agent.turn("I like Python.")

        assert result.commit is not None
        assert agent_repo.head_commit() == result.commit
        state = agent_repo.state()
        fact = state.one("user", "prefers_language")
        assert fact is not None
        assert fact.object == "Python"
        assert fact.source_text == "I like Python."

    def test_commit_metadata_is_json_safe_and_carries_the_query(self, agent_repo):
        call = ("call_1", "remember", {
            "subject": "user", "predicate": "prefers_language", "object": "Python",
            "confidence": 0.9, "source_text": "I like Python.",
        })
        client = ScriptedClient(responses=[tool_use_message(call), text_message("Noted.")])
        agent = MemoryAgent(agent_repo, client=client)

        result = agent.turn("I like Python.")

        commit = agent_repo.read_commit(result.commit)
        assert commit.metadata["query"] == "I like Python."
        assert commit.metadata["tool_calls"] == [{"name": "remember", "ok": True}]

    def test_commit_author_is_derived_from_the_model(self, agent_repo):
        call = ("call_1", "remember", {
            "subject": "user", "predicate": "prefers_language", "object": "Python",
            "confidence": 0.9, "source_text": "x",
        })
        client = ScriptedClient(responses=[tool_use_message(call), text_message("Noted.")])
        agent = MemoryAgent(agent_repo, client=client, model="claude-opus-5")

        result = agent.turn("I like Python.")

        commit = agent_repo.read_commit(result.commit)
        assert commit.author == "agent:claude-opus-5"

    def test_parallel_tool_calls_are_batched_into_one_user_message(self, agent_repo):
        calls = (
            ("call_1", "remember", {
                "subject": "user", "predicate": "prefers_language", "object": "Python",
                "confidence": 0.9, "source_text": "x",
            }),
            ("call_2", "remember", {
                "subject": "user", "predicate": "timezone", "object": "Europe/Berlin",
                "confidence": 0.8, "source_text": "y",
            }),
        )
        client = ScriptedClient(responses=[tool_use_message(*calls), text_message("Noted.")])
        agent = MemoryAgent(agent_repo, client=client)

        agent.turn("two facts at once")

        second_request = client.requests[1]
        tool_result_messages = [m for m in second_request["messages"] if m["role"] == "user"][1:]
        assert len(tool_result_messages) == 1
        assert len(tool_result_messages[0]["content"]) == 2

    def test_two_tool_rounds_before_a_final_reply(self, agent_repo):
        first_call = ("call_1", "remember", {
            "subject": "user", "predicate": "likes", "object": "chess",
            "confidence": 0.9, "source_text": "x",
        })
        second_call = ("call_2", "remember", {
            "subject": "user", "predicate": "likes", "object": "go",
            "confidence": 0.9, "source_text": "y",
        })
        client = ScriptedClient(responses=[
            tool_use_message(first_call),
            tool_use_message(second_call),
            text_message("Got both."),
        ])
        cardinality_path = agent_repo.memgit_dir / "cardinality.json"
        cardinality_path.write_text('{"version": 1, "default": "single", "predicates": {"likes": "multi"}}')
        agent = MemoryAgent(agent_repo, client=client)

        result = agent.turn("I like chess and go.")

        assert result.reply == "Got both."
        state = agent_repo.state()
        assert {f.object for f in state.get("user", "likes")} == {"chess", "go"}


class TestForget:
    def test_forget_removes_a_previously_remembered_fact(self, agent_repo):
        remember_call = ("call_1", "remember", {
            "subject": "user", "predicate": "has_pet", "object": "dog",
            "confidence": 1.0, "source_text": "I have a dog.",
        })
        forget_call = ("call_2", "forget", {
            "subject": "user", "predicate": "has_pet", "object": None, "reason": "gave the dog away",
        })
        client = ScriptedClient(responses=[
            tool_use_message(remember_call), text_message("Noted."),
            tool_use_message(forget_call), text_message("Understood."),
        ])
        agent = MemoryAgent(agent_repo, client=client)
        agent.turn("I have a dog.")

        agent.turn("I gave the dog away.")

        assert agent_repo.state().get("user", "has_pet") == ()


class TestInvalidToolCallsRecover:
    def test_invalid_fact_gets_an_error_result_and_the_turn_continues(self, agent_repo):
        bad_call = ("call_1", "remember", {
            "subject": "user", "predicate": "confidence_test", "object": "x",
            "confidence": 5.0, "source_text": "y",
        })
        good_call = ("call_2", "remember", {
            "subject": "user", "predicate": "confidence_test", "object": "x",
            "confidence": 0.5, "source_text": "y",
        })
        client = ScriptedClient(responses=[
            tool_use_message(bad_call),
            tool_use_message(good_call),
            text_message("Fixed it."),
        ])
        agent = MemoryAgent(agent_repo, client=client)

        result = agent.turn("remember this")

        assert result.commit is not None
        second_request = client.requests[1]
        tool_result_message = [m for m in second_request["messages"] if m["role"] == "user"][-1]
        assert tool_result_message["content"][0]["is_error"] is True
        assert agent_repo.state().one("user", "confidence_test").confidence == 0.5

    def test_unknown_tool_name_gets_an_error_result(self, agent_repo):
        bad_call = ("call_1", "mystery_tool", {})
        client = ScriptedClient(responses=[tool_use_message(bad_call), text_message("ok")])
        agent = MemoryAgent(agent_repo, client=client)

        result = agent.turn("do something odd")

        assert result.commit is None


class TestFailureModes:
    def test_refusal_raises_and_commits_nothing(self, agent_repo):
        client = ScriptedClient(responses=[refusal_message(category="other")])
        agent = MemoryAgent(agent_repo, client=client)

        with pytest.raises(AgentError):
            agent.turn("do something disallowed")

        assert agent_repo.head_commit() is None

    def test_tool_loop_limit_raises(self, agent_repo):
        call = ("call_1", "remember", {
            "subject": "user", "predicate": "likes", "object": "chess",
            "confidence": 0.9, "source_text": "x",
        })
        client = ScriptedClient(responses=[tool_use_message(call) for _ in range(5)])
        agent = MemoryAgent(agent_repo, client=client, max_tool_rounds=2)

        with pytest.raises(AgentError):
            agent.turn("keep going forever")


class TestMemoryCrossesTurnsTranscriptDoesNot:
    def test_turn_two_prompt_contains_turn_one_fact(self, agent_repo):
        remember_call = ("call_1", "remember", {
            "subject": "user", "predicate": "prefers_language", "object": "Python",
            "confidence": 0.9, "source_text": "x",
        })
        client = ScriptedClient(responses=[
            tool_use_message(remember_call), text_message("Noted."),
            text_message("You prefer Python."),
        ])
        agent = MemoryAgent(agent_repo, client=client)

        agent.turn("I like Python.")
        agent.turn("What do I prefer?")

        third_request = client.requests[2]
        assert "Python" in third_request["system"][1]["text"]

    def test_transcript_grows_across_turns_within_one_session(self, agent_repo):
        client = ScriptedClient(responses=[text_message("hi"), text_message("hi again")])
        agent = MemoryAgent(agent_repo, client=client)

        agent.turn("first")
        agent.turn("second")

        roles_and_texts = [(m["role"], m.get("content")) for m in agent.transcript]
        assert any(role == "user" and content == "first" for role, content in roles_and_texts)
        assert any(role == "user" and content == "second" for role, content in roles_and_texts)


def _seed_many_facts(repo, count: int) -> None:
    """Commit ``count`` distinct facts directly, bypassing the agent — used
    to push a repo's fact count over the retrieval threshold without
    scripting one tool call per fact."""
    facts = [
        Fact(subject="user", predicate=f"fact_{i}", object=f"value_{i}", confidence=0.9)
        for i in range(count)
    ]
    repo.commit(facts, "seed many facts", author="test")


class TestRetrievalMode:
    def test_below_threshold_uses_full_injection_and_offers_no_recall(self, agent_repo):
        client = ScriptedClient(responses=[text_message("hi")])
        agent = MemoryAgent(agent_repo, client=client)

        agent.turn("hello")

        tool_names = {schema["name"] for schema in client.requests[0]["tools"]}
        assert "recall" not in tool_names
        assert len(client.requests[0]["system"]) == 2

    def test_at_or_above_threshold_offers_recall_and_uses_three_blocks(self, agent_repo):
        _seed_many_facts(agent_repo, 64)
        client = ScriptedClient(responses=[text_message("hi")])
        agent = MemoryAgent(agent_repo, client=client)

        agent.turn("what do you know about fact_5?")

        tool_names = {schema["name"] for schema in client.requests[0]["tools"]}
        assert "recall" in tool_names
        assert len(client.requests[0]["system"]) == 3

    def test_recall_tool_call_leaves_working_state_unchanged_and_commits_nothing(self, agent_repo):
        _seed_many_facts(agent_repo, 64)
        call = ("call_1", "recall", {"query": "fact_5", "subject": None, "limit": 3})
        client = ScriptedClient(responses=[tool_use_message(call), text_message("Found it.")])
        agent = MemoryAgent(agent_repo, client=client)

        result = agent.turn("what do you know about fact_5?")

        assert result.commit is None  # recall is read-only

    def test_recall_call_while_retrieval_off_is_an_error_the_model_can_recover_from(self, agent_repo):
        # Force a `recall` call even though the repo is small (full-injection
        # mode) -- simulates a model calling a tool it wasn't offered.
        call = ("call_1", "recall", {"query": "anything", "subject": None, "limit": 3})
        client = ScriptedClient(responses=[tool_use_message(call), text_message("ok")])
        agent = MemoryAgent(agent_repo, client=client)

        agent.turn("hello")

        second_request = client.requests[1]
        tool_result_message = [m for m in second_request["messages"] if m["role"] == "user"][-1]
        assert tool_result_message["content"][0]["is_error"] is True

    def test_recall_searches_before_not_working_within_the_same_turn(self, agent_repo):
        """A fact just `remember`-ed this turn should not be recall-able
        within that same turn -- recall only sees committed history."""
        remember_call = ("call_1", "remember", {
            "subject": "user", "predicate": "brand_new_fact", "object": "just now",
            "confidence": 0.9, "source_text": "x",
        })
        _seed_many_facts(agent_repo, 64)
        recall_call = ("call_2", "recall", {"query": "brand_new_fact just now", "subject": None, "limit": 5})
        client = ScriptedClient(responses=[
            tool_use_message(remember_call),
            tool_use_message(recall_call),
            text_message("done"),
        ])
        agent = MemoryAgent(agent_repo, client=client)

        agent.turn("remember something and then recall it")

        third_request = client.requests[2]
        recall_result = [m for m in third_request["messages"] if m["role"] == "user"][-1]
        assert "brand_new_fact" not in recall_result["content"][0]["content"]

    def test_a_turn_that_retrieves_a_subset_still_commits_the_full_state(self, agent_repo):
        """The mass-deletion guard: retrieval must never narrow what a turn
        commits, only what gets rendered into the prompt."""
        _seed_many_facts(agent_repo, 64)
        remember_call = ("call_1", "remember", {
            "subject": "user", "predicate": "one_more_fact", "object": "added this turn",
            "confidence": 0.9, "source_text": "x",
        })
        client = ScriptedClient(responses=[tool_use_message(remember_call), text_message("Noted.")])
        agent = MemoryAgent(agent_repo, client=client)

        result = agent.turn("remember one more thing")

        assert result.commit is not None
        state = agent_repo.state()
        assert len(state) == 65  # the original 64, untouched, plus the new one

    def test_commit_metadata_records_full_mode(self, agent_repo):
        client = ScriptedClient(responses=[text_message("hi")])
        agent = MemoryAgent(agent_repo, client=client, record_empty=True)

        result = agent.turn("hello")

        commit = agent_repo.read_commit(result.commit)
        assert commit.metadata["retrieval"] == {"mode": "full"}

    def test_commit_metadata_records_top_k_mode_and_injected_hashes(self, agent_repo):
        _seed_many_facts(agent_repo, 64)
        client = ScriptedClient(responses=[text_message("hi")])
        agent = MemoryAgent(agent_repo, client=client, record_empty=True)

        result = agent.turn("what do you know about fact_5?")

        commit = agent_repo.read_commit(result.commit)
        retrieval_meta = commit.metadata["retrieval"]
        assert retrieval_meta["mode"] == "top_k"
        assert isinstance(retrieval_meta["injected"], list)
        assert retrieval_meta["embedder"].startswith("hash-v1/")

