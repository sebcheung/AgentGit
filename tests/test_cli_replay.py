"""Tests for `memgit replay` (slice 6).

A sibling of ``test_cli_agent.py``: same ``cli._agent_client`` monkeypatch
seam, zero network. Seeds a commit via ``memgit ask`` (already covered by
``test_cli_agent.py``) rather than reaching into ``Repository`` directly, so
these tests only exercise CLI surface.
"""

from __future__ import annotations

import json

import pytest
from conftest import ScriptedClient, text_message, tool_use_message
from typer.testing import CliRunner

import memgit.cli as cli
from memgit.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _patch_client(monkeypatch, client: ScriptedClient) -> None:
    monkeypatch.setattr(cli, "_agent_client", lambda model, **kwargs: client)


def _seed(monkeypatch) -> None:
    runner.invoke(app, ["init"])
    call = ("call_1", "remember", {
        "subject": "user", "predicate": "prefers_language", "object": "Python",
        "confidence": 0.9, "source_text": "I like Python.",
    })
    _patch_client(monkeypatch, ScriptedClient(responses=[tool_use_message(call), text_message("Noted.")]))
    runner.invoke(app, ["ask", "I like Python."])


class TestReplay:
    def test_prints_baseline_and_ablated_replies(self, monkeypatch):
        _seed(monkeypatch)
        _patch_client(
            monkeypatch,
            ScriptedClient(responses=[text_message("You prefer Python."), text_message("I don't know.")]),
        )

        result = runner.invoke(
            app, ["replay", "HEAD", "user", "prefers_language", "what do I prefer?"]
        )

        assert result.exit_code == 0
        assert "baseline> You prefer Python." in result.output
        assert "ablated>  I don't know." in result.output

    def test_json_output_reports_changed(self, monkeypatch):
        _seed(monkeypatch)
        _patch_client(
            monkeypatch,
            ScriptedClient(responses=[text_message("You prefer Python."), text_message("I don't know.")]),
        )

        result = runner.invoke(
            app, ["replay", "HEAD", "user", "prefers_language", "what do I prefer?", "--json"]
        )

        assert result.exit_code == 0
        assert '"changed": true' in result.output

    def test_never_creates_a_commit(self, monkeypatch):
        _seed(monkeypatch)
        head_before = runner.invoke(app, ["rev-parse", "HEAD"]).output.strip()
        _patch_client(
            monkeypatch,
            ScriptedClient(responses=[text_message("a"), text_message("b")]),
        )

        runner.invoke(app, ["replay", "HEAD", "user", "prefers_language", "q"])

        head_after = runner.invoke(app, ["rev-parse", "HEAD"]).output.strip()
        assert head_after == head_before

    def test_unknown_revision_fails_cleanly(self, monkeypatch):
        _seed(monkeypatch)
        _patch_client(monkeypatch, ScriptedClient(responses=[]))

        result = runner.invoke(
            app, ["replay", "not-a-real-revision", "user", "prefers_language", "q"]
        )

        assert result.exit_code == 1

    def test_outside_a_repository_fails_cleanly(self, monkeypatch):
        _patch_client(monkeypatch, ScriptedClient(responses=[]))

        result = runner.invoke(app, ["replay", "HEAD", "user", "prefers_language", "q"])

        assert result.exit_code == 1

    def test_agent_error_from_the_client_exits_nonzero(self, monkeypatch):
        from memgit.agent.client import AgentError

        _seed(monkeypatch)

        def _raise(model, **kwargs):
            raise AgentError("no API key configured")

        monkeypatch.setattr(cli, "_agent_client", _raise)

        result = runner.invoke(app, ["replay", "HEAD", "user", "prefers_language", "q"])

        assert result.exit_code == 1
        assert "no API key configured" in result.output


class TestReplayRetrievalMode:
    def _seed_above_threshold(self, monkeypatch):
        import json as _json

        runner.invoke(app, ["init"])
        facts = [
            {
                "type": "fact", "subject": "user", "predicate": f"fact_{i}", "object": f"value_{i}",
                "confidence": 0.9, "asserted_at": "2026-09-01T00:00:00+00:00",
            }
            for i in range(64)
        ]
        facts.append({
            "type": "fact", "subject": "user", "predicate": "prefers_language", "object": "Python",
            "confidence": 0.9, "asserted_at": "2026-09-01T00:00:00+00:00",
        })
        with open("many_facts.json", "w", encoding="utf-8") as handle:
            handle.write(_json.dumps(facts))
        result = runner.invoke(app, ["commit", "-m", "seed", "--file", "many_facts.json"])
        assert result.exit_code == 0, result.output

    def test_json_output_includes_retrieval_provenance(self, monkeypatch):
        self._seed_above_threshold(monkeypatch)
        _patch_client(
            monkeypatch,
            ScriptedClient(responses=[text_message("You prefer Python."), text_message("I don't know.")]),
        )

        result = runner.invoke(
            app, ["replay", "HEAD", "user", "prefers_language", "what do I prefer?", "--json"]
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "retrieval" in payload
        assert payload["retrieval"]["embedder"].startswith("hash-v1/")

    def test_human_output_notes_pinned_retrieval(self, monkeypatch):
        self._seed_above_threshold(monkeypatch)
        _patch_client(
            monkeypatch,
            ScriptedClient(responses=[text_message("a"), text_message("b")]),
        )

        result = runner.invoke(app, ["replay", "HEAD", "user", "prefers_language", "q"])

        assert result.exit_code == 0
        assert "pinned for both sides" in result.output
