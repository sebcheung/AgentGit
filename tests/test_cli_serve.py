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


class TestServeAuth:
    def test_refuses_non_loopback_host_with_no_api_key(self, tmp_path):
        _seed_repo(tmp_path)
        result = runner.invoke(app, ["serve", "--transport", "streamable-http", "--host", "0.0.0.0"])
        assert result.exit_code != 0
        assert "refusing to bind" in result.output

    def test_insecure_allows_non_loopback_host_with_no_key(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        import memgit.mcp.server as mcp_server_module

        calls = {}

        def fake_run(repo, *, transport, host, port, api_key=None):
            calls["host"] = host
            calls["api_key"] = api_key

        monkeypatch.setattr(mcp_server_module, "run", fake_run)

        result = runner.invoke(
            app, ["serve", "--transport", "streamable-http", "--host", "0.0.0.0", "--insecure"]
        )
        assert result.exit_code == 0, result.output
        assert calls["host"] == "0.0.0.0"
        assert calls["api_key"] is None

    def test_api_key_flag_reaches_run(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        import memgit.mcp.server as mcp_server_module

        calls = {}

        def fake_run(repo, *, transport, host, port, api_key=None):
            calls["api_key"] = api_key

        monkeypatch.setattr(mcp_server_module, "run", fake_run)

        result = runner.invoke(app, ["serve", "--transport", "streamable-http", "--api-key", "secret"])
        assert result.exit_code == 0, result.output
        assert calls["api_key"] == "secret"

    def test_env_var_is_used_when_no_flag_given(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.setenv("MEMGIT_API_KEY", "from-env")
        import memgit.mcp.server as mcp_server_module

        calls = {}

        def fake_run(repo, *, transport, host, port, api_key=None):
            calls["api_key"] = api_key

        monkeypatch.setattr(mcp_server_module, "run", fake_run)

        result = runner.invoke(app, ["serve", "--transport", "streamable-http"])
        assert result.exit_code == 0, result.output
        assert calls["api_key"] == "from-env"

    def test_stdio_ignores_the_loopback_guard(self, tmp_path, monkeypatch):
        # stdio has no host/port to guard against -- --host is meaningless
        # to it, so no key should ever be required for it.
        _seed_repo(tmp_path)
        import memgit.mcp.server as mcp_server_module

        monkeypatch.setattr(mcp_server_module, "run", lambda *a, **k: None)

        result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])
        assert result.exit_code == 0, result.output
