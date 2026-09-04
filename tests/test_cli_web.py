"""Tests for `memgit serve-web`'s own CLI-level wrapper (slice 10).

The actual dashboard/API behavior is covered by tests/test_api_*.py against
`create_app` directly; these only exercise the CLI wrapper around it --
argument pass-through and the friendly error when the `web` extra is
missing -- mirroring test_cli_serve.py's coverage of `memgit serve`.
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


class TestServeWeb:
    def test_missing_web_extra_is_a_friendly_error(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "memgit.api.app" or name.startswith("memgit.api") or name == "uvicorn":
                raise ImportError("simulated: web extra not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = runner.invoke(app, ["serve-web"])
        assert result.exit_code != 0
        assert "web" in result.output.lower()

    def test_fails_outside_a_repository(self, tmp_path):
        result = runner.invoke(app, ["serve-web"])
        assert result.exit_code != 0

    def test_host_port_and_model_pass_through(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        calls = {}

        class _FakeUvicorn:
            @staticmethod
            def run(web_app, *, host, port):
                calls["web_app"] = web_app
                calls["host"] = host
                calls["port"] = port

        monkeypatch.setitem(__import__("sys").modules, "uvicorn", _FakeUvicorn)

        result = runner.invoke(
            app,
            [
                "serve-web",
                "--host",
                "0.0.0.0",
                "--port",
                "9999",
                "--model",
                "claude-fable-5-1",
                "--insecure",
            ],
        )

        assert result.exit_code == 0, result.output
        assert calls["host"] == "0.0.0.0"
        assert calls["port"] == 9999
        assert calls["web_app"].state.model == "claude-fable-5-1"

    def test_refuses_non_loopback_host_with_no_api_key(self, tmp_path):
        _seed_repo(tmp_path)
        result = runner.invoke(app, ["serve-web", "--host", "0.0.0.0"])
        assert result.exit_code != 0
        assert "refusing to bind" in result.output

    def test_insecure_allows_non_loopback_host_with_a_warning(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)

        class _FakeUvicorn:
            @staticmethod
            def run(web_app, *, host, port):
                pass

        monkeypatch.setitem(__import__("sys").modules, "uvicorn", _FakeUvicorn)

        result = runner.invoke(app, ["serve-web", "--host", "0.0.0.0", "--insecure"])
        assert result.exit_code == 0, result.output
        assert "WARNING" in result.output

    def test_api_key_flag_reaches_the_app_state(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        calls = {}

        class _FakeUvicorn:
            @staticmethod
            def run(web_app, *, host, port):
                calls["web_app"] = web_app

        monkeypatch.setitem(__import__("sys").modules, "uvicorn", _FakeUvicorn)

        result = runner.invoke(app, ["serve-web", "--api-key", "secret"])
        assert result.exit_code == 0, result.output
        assert calls["web_app"].state.api_key == "secret"

    def test_env_var_is_used_when_no_flag_given(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.setenv("MEMGIT_API_KEY", "from-env")
        calls = {}

        class _FakeUvicorn:
            @staticmethod
            def run(web_app, *, host, port):
                calls["web_app"] = web_app

        monkeypatch.setitem(__import__("sys").modules, "uvicorn", _FakeUvicorn)

        result = runner.invoke(app, ["serve-web"])
        assert result.exit_code == 0, result.output
        assert calls["web_app"].state.api_key == "from-env"

    def test_a_configured_key_permits_a_non_loopback_bind(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)

        class _FakeUvicorn:
            @staticmethod
            def run(web_app, *, host, port):
                pass

        monkeypatch.setitem(__import__("sys").modules, "uvicorn", _FakeUvicorn)

        result = runner.invoke(app, ["serve-web", "--host", "0.0.0.0", "--api-key", "secret"])
        assert result.exit_code == 0, result.output
