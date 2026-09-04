"""Tests for `memgit ask` and `memgit chat`.

A sibling of ``test_cli.py`` rather than a merge into it: this file's whole
premise is monkeypatching ``cli._agent_client`` with the scripted fake, which
``test_cli.py``'s existing tests have no reason to know about. Zero network —
the fake never imports ``anthropic``.
"""

from __future__ import annotations

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
    monkeypatch.setattr(cli, "_agent_client", lambda model: client)


class TestAsk:
    def test_no_tool_calls_prints_reply_and_no_commit_marker(self, monkeypatch, tmp_path):
        runner.invoke(app, ["init"])
        _patch_client(monkeypatch, ScriptedClient(responses=[text_message("Hello yourself.")]))

        result = runner.invoke(app, ["ask", "hi"])

        assert result.exit_code == 0
        assert "Hello yourself." in result.output
        assert "no memory change" in result.output

    def test_remember_lands_a_commit_and_is_visible_in_state(self, monkeypatch, tmp_path):
        runner.invoke(app, ["init"])
        call = ("call_1", "remember", {
            "subject": "user", "predicate": "prefers_language", "object": "Python",
            "confidence": 0.9, "source_text": "I like Python.",
        })
        _patch_client(monkeypatch, ScriptedClient(responses=[tool_use_message(call), text_message("Noted.")]))

        result = runner.invoke(app, ["ask", "I like Python."])
        assert result.exit_code == 0

        state_result = runner.invoke(app, ["state"])
        assert "prefers_language Python" in state_result.output

    def test_json_output_carries_reply_and_commit(self, monkeypatch, tmp_path):
        runner.invoke(app, ["init"])
        _patch_client(monkeypatch, ScriptedClient(responses=[text_message("hi")]))

        result = runner.invoke(app, ["ask", "hi", "--json"])

        assert result.exit_code == 0
        assert '"reply": "hi"' in result.output
        assert '"commit": null' in result.output

    def test_outside_a_repository_fails_cleanly(self, monkeypatch, tmp_path):
        _patch_client(monkeypatch, ScriptedClient(responses=[text_message("hi")]))

        result = runner.invoke(app, ["ask", "hi"])

        assert result.exit_code == 1

    def test_agent_error_from_the_client_exits_nonzero(self, monkeypatch, tmp_path):
        from memgit.agent.client import AgentError

        runner.invoke(app, ["init"])

        def _raise(model):
            raise AgentError("no API key configured")

        monkeypatch.setattr(cli, "_agent_client", _raise)

        result = runner.invoke(app, ["ask", "hi"])

        assert result.exit_code == 1
        assert "no API key configured" in result.output


class TestChat:
    def test_quit_exits_immediately(self, monkeypatch, tmp_path):
        runner.invoke(app, ["init"])
        _patch_client(monkeypatch, ScriptedClient(responses=[]))

        result = runner.invoke(app, ["chat"], input="/quit\n")

        assert result.exit_code == 0

    def test_state_command_shows_memory_without_spending_a_turn(self, monkeypatch, tmp_path):
        runner.invoke(app, ["init"])
        client = ScriptedClient(responses=[])
        _patch_client(monkeypatch, client)

        result = runner.invoke(app, ["chat"], input="/state\n/quit\n")

        assert result.exit_code == 0
        assert client.requests == []

    def test_one_turn_then_quit(self, monkeypatch, tmp_path):
        runner.invoke(app, ["init"])
        _patch_client(monkeypatch, ScriptedClient(responses=[text_message("Hi!")]))

        result = runner.invoke(app, ["chat"], input="hello\n/quit\n")

        assert result.exit_code == 0
        assert "Hi!" in result.output

    def test_eof_exits_cleanly(self, monkeypatch, tmp_path):
        runner.invoke(app, ["init"])
        _patch_client(monkeypatch, ScriptedClient(responses=[]))

        result = runner.invoke(app, ["chat"], input="")

        assert result.exit_code == 0
