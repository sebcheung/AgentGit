"""Tests for the replay / ablation engine (slice 6).

Uses the ScriptedClient fake from conftest.py — zero network. Grouped by the
properties the design turns on: a replay never touches ``Repository.commit``,
an ablated state really is missing just the one belief, and the same query
against the same state deterministically reuses the ablated conversation's
own isolation (no bleed-through from the baseline run).
"""

from __future__ import annotations

from datetime import UTC

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

        ablate_and_replay(
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


def _seed_many_facts(repo, count: int) -> None:
    facts = [
        Fact(subject="user", predicate=f"fact_{i}", object=f"value_{i}", confidence=0.9)
        for i in range(count)
    ]
    repo.commit(facts, "seed many facts", author="test")


class TestPinnedRetrieval:
    """Above the retrieval threshold, both sides' prompts must be built from
    one shared, pre-computed retrieval — never two independent ones — or
    ablation stops being a one-variable experiment."""

    def _committed_repo_with_many_facts(self, agent_repo):
        _seed_many_facts(agent_repo, 64)
        agent_repo.commit(
            [*agent_repo.state().facts, _remember_fact()],
            "add the ablation target",
            author="test",
        )
        return agent_repo

    def test_retrieval_mode_engages_above_the_threshold(self, agent_repo):
        repo = self._committed_repo_with_many_facts(agent_repo)
        client = ScriptedClient(
            responses=[text_message("You prefer Python."), text_message("I don't know.")]
        )

        result = ablate_and_replay(
            repo, "HEAD", ("user", "prefers_language"), "what do I prefer?", client=client
        )

        assert result.retrieved is not None
        assert len(client.requests[0]["system"]) == 3  # key inventory + retrieved block

    def test_ablated_prompt_is_the_pinned_set_minus_the_fact_no_backfill(self, agent_repo):
        repo = self._committed_repo_with_many_facts(agent_repo)
        client = ScriptedClient(
            responses=[text_message("You prefer Python."), text_message("I don't know.")]
        )

        result = ablate_and_replay(
            repo, "HEAD", ("user", "prefers_language"), "what do I prefer?", client=client, k=64
        )

        baseline_hashes = {r.fact.hash for r in result.retrieved.facts}
        ablated_fact = next(f for f in repo.state().facts if f.key == ("user", "prefers_language"))
        assert ablated_fact.hash in baseline_hashes  # k=64 covers every fact, so it must be present

        ablated_system_text = client.requests[1]["system"][2]["text"]
        assert "prefers_language Python" not in ablated_system_text

        # No backfill: the ablated block has exactly one fewer entry than
        # the baseline block, never the same count with a replacement.
        baseline_system_text = client.requests[0]["system"][2]["text"]
        baseline_lines = [line for line in baseline_system_text.splitlines() if line.strip()]
        ablated_lines = [line for line in ablated_system_text.splitlines() if line.strip()]
        assert len(ablated_lines) == len(baseline_lines) - 1

    def test_fixed_as_of_makes_retrieval_reproducible(self, agent_repo):
        from datetime import datetime

        repo = self._committed_repo_with_many_facts(agent_repo)
        moment = datetime(2027, 1, 1, tzinfo=UTC)
        client = ScriptedClient(
            responses=[text_message("a"), text_message("b")]
        )

        result = ablate_and_replay(
            repo, "HEAD", ("user", "prefers_language"), "q", client=client, as_of=moment
        )

        assert result.retrieved.as_of == moment.isoformat()

    def test_full_injection_forced_off_via_retrieve_flag(self, agent_repo):
        repo = self._committed_repo_with_many_facts(agent_repo)
        client = ScriptedClient(responses=[text_message("a"), text_message("b")])

        result = ablate_and_replay(
            repo, "HEAD", ("user", "prefers_language"), "q", client=client, retrieve=False
        )

        assert result.retrieved is None
        assert len(client.requests[0]["system"]) == 2
