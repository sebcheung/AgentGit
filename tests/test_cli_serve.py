"""Tests for `memgit serve`'s own CLI-level error handling (slice 8).

The actual MCP server behavior is covered by ``test_mcp_server.py`` via the
SDK's in-memory transport; these only exercise the CLI wrapper around it —
argument validation and the friendly error when the `mcp` extra is missing.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from memgit.cli import app
from memgit.core.fact import Fact
from memgit.core.repository import Repository

runner = CliRunner()


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _seed_repo(tmp_path) -> Repository:
    repo = Repository.init(tmp_path)
    repo.commit(
        [Fact(subject="user", predicate="prefers_language", object="Python", confidence=0.9)], "seed"
    )
    return repo


class TestServe:
    def test_bad_transport_fails_fast(self, tmp_path):
        _seed_repo(tmp_path)
        result = runner.invoke(app, ["serve", "--transport", "sse"])
        assert result.exit_code != 0
        assert "stdio" in result.output

    def test_missing_mcp_extra_is_a_friendly_error(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "memgit.mcp.server" or name.startswith("memgit.mcp"):
                raise ImportError("simulated: mcp extra not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code != 0
        assert "mcp" in result.output.lower()
