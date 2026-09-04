"""Tests for `memgit project`/`blame`/`stats`: the CLI wrapper around memgit.pg.

Uses a real SQLite file as $DATABASE_URL for the end-to-end tests -- the
projection's dialect-aware upserts already support it (see
project.py's ``_insert_or_ignore``), so this exercises the full CLI path
with no real Postgres server needed.
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


def _database_url(tmp_path) -> str:
    # In production, `alembic upgrade head` (or the compose stack's
    # `migrate` service) creates the schema before anything calls `memgit
    # project`; this stands in for that one-time step so the CLI test can
    # focus on project/blame/stats themselves.
    from sqlalchemy import create_engine

    from memgit.pg.schema import metadata

    url = f"sqlite:///{tmp_path}/pg.db"
    metadata.create_all(create_engine(url))
    return url


class TestMissingDatabaseUrl:
    def test_project_fails_without_database_url(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = runner.invoke(app, ["project"])
        assert result.exit_code != 0
        assert "DATABASE_URL" in result.output

    def test_blame_fails_without_database_url(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = runner.invoke(app, ["blame", "user", "prefers_language"])
        assert result.exit_code != 0
        assert "DATABASE_URL" in result.output

    def test_stats_fails_without_database_url(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = runner.invoke(app, ["stats"])
        assert result.exit_code != 0
        assert "DATABASE_URL" in result.output


class TestMissingPgExtra:
    def test_missing_pg_extra_is_a_friendly_error(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.setenv("DATABASE_URL", _database_url(tmp_path))
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "memgit.pg.engine" or name.startswith("memgit.pg"):
                raise ImportError("simulated: pg extra not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = runner.invoke(app, ["project"])
        assert result.exit_code != 0
        assert "pg" in result.output.lower()


class TestEndToEnd:
    def test_project_then_blame(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.setenv("DATABASE_URL", _database_url(tmp_path))

        project_result = runner.invoke(app, ["project"])
        assert project_result.exit_code == 0, project_result.output
        assert "1 commit(s)" in project_result.output

        blame_result = runner.invoke(app, ["blame", "user", "prefers_language"])
        assert blame_result.exit_code == 0, blame_result.output
        assert "+" in blame_result.output

    def test_blame_with_no_history_says_so(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.setenv("DATABASE_URL", _database_url(tmp_path))
        runner.invoke(app, ["project"])

        result = runner.invoke(app, ["blame", "nobody", "nothing"])

        assert result.exit_code == 0, result.output
        assert "no history" in result.output.lower()

    def test_stats_reports_dedup_ratio(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.setenv("DATABASE_URL", _database_url(tmp_path))
        runner.invoke(app, ["project"])

        result = runner.invoke(app, ["stats"])

        assert result.exit_code == 0, result.output
        assert "dedup ratio" in result.output.lower()

    def test_rerunning_project_reports_zero_added(self, tmp_path, monkeypatch):
        _seed_repo(tmp_path)
        monkeypatch.setenv("DATABASE_URL", _database_url(tmp_path))
        runner.invoke(app, ["project"])

        result = runner.invoke(app, ["project"])

        assert result.exit_code == 0, result.output
        assert "0 commit(s)" in result.output
