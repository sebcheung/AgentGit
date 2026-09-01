"""Tests for the CLI's translation of core exceptions into exit codes.

Deliberately the one test module that doesn't map 1:1 onto a single core
module: ``cli.py`` is the only place a Repository/Tree/Commit exception
becomes an exit code and a stderr message, and that translation was
previously untested. Covers the happy path end to end, then one case per
error a user is likely to hit.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memgit.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


FACTS = json.dumps(
    [
        {
            "type": "fact",
            "subject": "user",
            "predicate": "prefers_language",
            "object": "Python",
            "confidence": 0.9,
            "asserted_at": "2026-09-01T00:00:00+00:00",
        }
    ]
)


def _init_and_commit(tmp_path, message="seed"):
    (tmp_path / "facts.json").write_text(FACTS, encoding="utf-8")
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["commit", "-m", message, "--file", "facts.json"])
    assert result.exit_code == 0, result.output
    return result.output.strip()


class TestHappyPath:
    def test_init_creates_a_repository(self, tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (tmp_path / ".memgit").is_dir()

    def test_commit_prints_a_hash(self, tmp_path):
        runner.invoke(app, ["init"])
        (tmp_path / "facts.json").write_text(FACTS, encoding="utf-8")
        result = runner.invoke(app, ["commit", "-m", "seed", "--file", "facts.json"])
        assert result.exit_code == 0
        assert len(result.output.strip()) == 64

    def test_log_shows_the_commit(self, tmp_path):
        commit_hash = _init_and_commit(tmp_path)
        result = runner.invoke(app, ["log", "--oneline"])
        assert result.exit_code == 0
        assert commit_hash[:8] in result.output
        assert "seed" in result.output

    def test_show_lists_the_committed_fact(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["show", "HEAD"])
        assert result.exit_code == 0
        assert "prefers_language" in result.output

    def test_ls_tree_prints_tab_separated_entries(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["ls-tree", "HEAD"])
        assert result.exit_code == 0
        assert "user\tprefers_language\t" in result.output

    def test_branch_creates_and_lists(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["branch", "experiment"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["branch"])
        assert result.exit_code == 0
        assert "* main" in result.output
        assert "experiment" in result.output

    def test_rev_parse_resolves_head(self, tmp_path):
        commit_hash = _init_and_commit(tmp_path)
        result = runner.invoke(app, ["rev-parse", "HEAD"])
        assert result.exit_code == 0
        assert result.output.strip() == commit_hash

    def test_status_reports_branch_and_fact_count(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "On branch main" in result.output
        assert "1 fact(s)" in result.output

    def test_fsck_reports_all_objects_verified(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["fsck"])
        assert result.exit_code == 0
        assert "all objects verified" in result.output


class TestErrors:
    def test_commands_fail_outside_a_repository(self, tmp_path):
        result = runner.invoke(app, ["log"])
        assert result.exit_code == 1
        assert "not a memgit repository" in result.output

    def test_reinit_fails(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1

    def test_unknown_revision_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["log", "does-not-exist"])
        assert result.exit_code == 1
        assert "unknown revision" in result.output

    def test_unknown_object_fails(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["cat-file", "a" * 64])
        assert result.exit_code == 1
        assert "object not found" in result.output

    def test_bad_json_file_fails(self, tmp_path):
        runner.invoke(app, ["init"])
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        result = runner.invoke(app, ["commit", "-m", "x", "--file", "bad.json"])
        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_empty_commit_fails_without_allow_empty(self, tmp_path):
        _init_and_commit(tmp_path)
        (tmp_path / "facts.json").write_text(FACTS, encoding="utf-8")
        result = runner.invoke(app, ["commit", "-m", "no-op", "--file", "facts.json"])
        assert result.exit_code == 1

    def test_deleting_current_branch_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["branch", "-d", "main"])
        assert result.exit_code == 1
