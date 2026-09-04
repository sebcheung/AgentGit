"""Tests for the `memgit staging` CLI surface (slice 8).

Exercises the CLI's own translation into exit codes and output — the actual
staging mechanics are covered end to end in ``test_staging.py``, so these
tests only need enough setup to exist a staging area, not to explore its
overlay/conflict behavior again.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memgit.cli import app
from memgit.core.fact import Fact
from memgit.core.repository import Repository
from memgit.core.staging import session_key

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


class TestStagingListEmpty:
    def test_no_open_areas(self, tmp_path):
        _seed_repo(tmp_path)
        result = runner.invoke(app, ["staging"])
        assert result.exit_code == 0
        assert "no open staging areas" in result.output


class TestStagingLifecycle:
    def _stage_one(self, repo: Repository, session_id: str = "s1") -> str:
        area = repo.open_staging(session_id)
        repo.stage(
            session_id,
            [Fact(subject="user", predicate="prefers_language", object="Rust", confidence=0.9)],
            based_on=area.tree,
        )
        return session_key(session_id)

    def test_list_shows_the_open_area(self, tmp_path):
        repo = _seed_repo(tmp_path)
        key = self._stage_one(repo)
        result = runner.invoke(app, ["staging"])
        assert result.exit_code == 0
        assert key in result.output

    def test_show_renders_the_staged_state(self, tmp_path):
        repo = _seed_repo(tmp_path)
        key = self._stage_one(repo)
        result = runner.invoke(app, ["staging", "show", key])
        assert result.exit_code == 0
        assert "Rust" in result.output

    def test_show_unknown_key_fails(self, tmp_path):
        _seed_repo(tmp_path)
        result = runner.invoke(app, ["staging", "show", "deadbeef00000000"])
        assert result.exit_code != 0

    def test_diff_shows_the_pending_change(self, tmp_path):
        repo = _seed_repo(tmp_path)
        key = self._stage_one(repo)
        result = runner.invoke(app, ["staging", "diff", key])
        assert result.exit_code == 0
        assert "prefers_language" in result.output

    def test_status_reports_open_staging_areas(self, tmp_path):
        repo = _seed_repo(tmp_path)
        self._stage_one(repo)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "1 open staging area" in result.output

    def test_commit_seals_and_removes_the_area(self, tmp_path):
        repo = _seed_repo(tmp_path)
        key = self._stage_one(repo)
        result = runner.invoke(app, ["staging", "commit", key, "-m", "sealed"])
        assert result.exit_code == 0

        reopened = Repository.open(repo.memgit_dir)
        assert reopened.staging_area(key) is None
        assert reopened.state("HEAD").get("user", "prefers_language")[0].object == "Rust"

    def test_drop_discards_without_committing(self, tmp_path):
        repo = _seed_repo(tmp_path)
        key = self._stage_one(repo)
        result = runner.invoke(app, ["staging", "drop", key])
        assert result.exit_code == 0

        reopened = Repository.open(repo.memgit_dir)
        assert reopened.staging_area(key) is None
        assert reopened.state("HEAD").get("user", "prefers_language")[0].object == "Python"

    def test_drop_unknown_key_fails(self, tmp_path):
        _seed_repo(tmp_path)
        result = runner.invoke(app, ["staging", "drop", "deadbeef00000000"])
        assert result.exit_code != 0
