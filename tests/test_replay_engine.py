"""Tests for the replay / ablation engine (slice 6).

Uses the ScriptedClient fake from conftest.py — zero network. Grouped by the
properties the design turns on: a replay never touches ``Repository.commit``,
an ablated state really is missing just the one belief, and the same query
against the same state deterministically reuses the ablated conversation's
own isolation (no bleed-through from the baseline run).
"""

from __future__ import annotations

from conftest import ScriptedClient, text_message, tool_use_message
from memgit.core.fact import Fact
from memgit.core.state import MemoryState
from memgit.replay.engine import ablate_and_replay, replay_query


def _remember_fact(**overrides) -> Fact:
    fields = {
        "subject": "user",
        "predicate": "prefers_language",
        "object": "Python",
        "confidence": 1.0,
        "source": "test",
        "source_text": "I like Python.",
    }
    fields.update(overrides)
    return Fact(**fields)


class TestReplayQuery:
    def test_returns_the_models_reply(self, agent_repo):
        client = ScriptedClient(responses=[text_message("You prefer Python.")])
        state = MemoryState.from_facts([_remember_fact()])
        outcome = replay_query(state, "what do I prefer?", agent_repo.cardinality(), client=client)

        assert outcome.reply == "You prefer Python."
        assert outcome.stop_reason == "end_turn"

    def test_never_touches_the_repository(self, agent_repo):
        client = ScriptedClient(responses=[text_message("noted")])
        state = MemoryState.from_facts([_remember_fact()])

        replay_query(state, "anything", agent_repo.cardinality(), client=client)

        assert agent_repo.head_commit() is None

    def test_each_call_is_a_fresh_conversation(self, agent_repo):
        client = ScriptedClient(responses=[text_message("a"), text_message("b")])
        state = MemoryState.from_facts([_remember_fact()])

        replay_query(state, "first", agent_repo.cardinality(), client=client)
        replay_query(state, "second", agent_repo.cardinality(), client=client)

        second_request = client.requests[1]
        assert len(second_request["messages"]) == 1  # only this call's user turn, no bleed from the first
        assert second_request["messages"][0]["content"] == "second"

    def test_tool_calls_are_applied_to_the_returned_state_not_committed(self, agent_repo):
        call = ("call_1", "remember", {
            "subject": "user", "predicate": "has_pet", "object": "cat",
            "confidence": 0.9, "source_text": "I have a cat.",
        })
        client = ScriptedClient(responses=[tool_use_message(call), text_message("noted")])
        state = MemoryState.empty()

        outcome = replay_query(state, "I have a cat.", agent_repo.cardinality(), client=client)

        assert outcome.state.one("user", "has_pet").object == "cat"
        assert agent_repo.head_commit() is None


class TestAblateAndReplay:
    def _committed_repo(self, agent_repo):
        agent_repo.commit(
            [_remember_fact(), _remember_fact(predicate="has_pet", object="dog", source_text="I have a dog.")],
            "seed",
            author="test",
        )
        return agent_repo

    def test_baseline_sees_the_fact_ablated_run_does_not(self, agent_repo):
        repo = self._committed_repo(agent_repo)
        client = ScriptedClient(
            responses=[text_message("You prefer Python."), text_message("I don't know.")]
        )

        result = ablate_and_replay(
            repo, "HEAD", ("user", "prefers_language"), "what do I prefer?", client=client
        )

        baseline_system = client.requests[0]["system"][1]["text"]
        ablated_system = client.requests[1]["system"][1]["text"]
        assert "Python" in baseline_system
        assert "Python" not in ablated_system

    def test_changed_reflects_whether_the_reply_differed(self, agent_repo):
        repo = self._committed_repo(agent_repo)
        client = ScriptedClient(
            responses=[text_message("You prefer Python."), text_message("I don't know.")]
        )

        result = ablate_and_replay(
            repo, "HEAD", ("user", "prefers_language"), "what do I prefer?", client=client
        )

        assert result.changed is True
        assert result.baseline.reply == "You prefer Python."
        assert result.ablated.reply == "I don't know."

    def test_identical_replies_report_unchanged(self, agent_repo):
        repo = self._committed_repo(agent_repo)
        client = ScriptedClient(
            responses=[text_message("I have a dog."), text_message("I have a dog.")]
        )

        result = ablate_and_replay(
            repo, "HEAD", ("user", "prefers_language"), "what pet do I have?", client=client
        )

        assert result.changed is False

    def test_never_commits_or_moves_head(self, agent_repo):
        repo = self._committed_repo(agent_repo)
        head_before = repo.head_commit()
        client = ScriptedClient(responses=[text_message("a"), text_message("b")])

        ablate_and_replay(repo, "HEAD", ("user", "prefers_language"), "q", client=client)

        assert repo.head_commit() == head_before

    def test_ablating_by_fact_hash_removes_only_that_value(self, agent_repo):
        repo = agent_repo
        repo.commit(
            [
                _remember_fact(predicate="likes", object="chess", source_text="I like chess."),
                _remember_fact(predicate="likes", object="go", source_text="I like go."),
            ],
            "seed",
            author="test",
        )
        cardinality_path = repo.memgit_dir / "cardinality.json"
        cardinality_path.write_text('{"version": 1, "default": "single", "predicates": {"likes": "multi"}}')

        state = repo.state()
        chess_fact = next(f for f in state.get("user", "likes") if f.object == "chess")

        client = ScriptedClient(responses=[text_message("chess and go"), text_message("go")])
        result = ablate_and_replay(
            repo,
            "HEAD",
            ("user", "likes"),
            "what do I like?",
            client=client,
            fact_hash=chess_fact.hash,
        )

        ablated_system = client.requests[1]["system"][1]["text"]
        assert "go" in ablated_system
        assert "chess" not in ablated_system
        assert result.fact_hash == chess_fact.hash
