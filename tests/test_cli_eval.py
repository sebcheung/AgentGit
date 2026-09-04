"""Tests for the `memgit eval` CLI surface."""

from __future__ import annotations

import json

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
    repo.commit([Fact(subject="user", predicate="city", object="Boston")], "seed")
    return repo


def _write_suite(repo: Repository, cases: list[dict]) -> None:
    eval_dir = repo.memgit_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    (eval_dir / "basics.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")


class TestEvalNoSuites:
    def test_no_eval_directory(self, tmp_path):
        _seed_repo(tmp_path)
        result = runner.invoke(app, ["eval"])
        assert result.exit_code == 0
        assert "no eval cases declared" in result.output


class TestEvalRun:
    def test_all_pass(self, tmp_path):
        repo = _seed_repo(tmp_path)
        _write_suite(
            repo,
            [{"id": "knows-city", "checks": [{"check": "value_is", "subject": "user", "predicate": "city", "object": "Boston"}]}],
        )
        result = runner.invoke(app, ["eval"])
        assert result.exit_code == 0
        assert "PASS  knows-city" in result.output
        assert "1/1 passed" in result.output

    def test_failure_exits_nonzero(self, tmp_path):
        repo = _seed_repo(tmp_path)
        _write_suite(
            repo,
            [{"id": "knows-city", "checks": [{"check": "value_is", "subject": "user", "predicate": "city", "object": "Berlin"}]}],
        )
        result = runner.invoke(app, ["eval"])
        assert result.exit_code == 1
        assert "FAIL  knows-city" in result.output

    def test_json_output(self, tmp_path):
        repo = _seed_repo(tmp_path)
        _write_suite(
            repo,
            [{"id": "knows-city", "checks": [{"check": "value_is", "subject": "user", "predicate": "city", "object": "Boston"}]}],
        )
        result = runner.invoke(app, ["eval", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload[0]["cases"][0]["case_id"] == "knows-city"
        assert payload[0]["cases"][0]["ok"] is True

    def test_malformed_suite_fails_cleanly(self, tmp_path):
        repo = _seed_repo(tmp_path)
        eval_dir = repo.memgit_dir / "eval"
        eval_dir.mkdir()
        (eval_dir / "broken.json").write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["eval"])
        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_unresolvable_rev_fails_cleanly(self, tmp_path):
        repo = _seed_repo(tmp_path)
        _write_suite(repo, [{"id": "c1", "checks": [{"check": "no_violations"}]}])
        result = runner.invoke(app, ["eval", "nonexistent-rev"])
        assert result.exit_code == 1

    def test_custom_suite_path(self, tmp_path):
        _seed_repo(tmp_path)
        suite_path = tmp_path / "custom.json"
        suite_path.write_text(
            json.dumps({"cases": [{"id": "c1", "checks": [{"check": "key_exists", "subject": "user", "predicate": "city"}]}]})
        )
        result = runner.invoke(app, ["eval", "--suite", str(suite_path)])
        assert result.exit_code == 0
        assert "PASS  c1" in result.output


class TestEvalSince:
    def test_reports_first_failing_commit(self, tmp_path):
        repo = _seed_repo(tmp_path)
        good = repo.head_commit()
        bad = repo.commit([Fact(subject="user", predicate="city", object="Berlin")], "moved")
        _write_suite(
            repo,
            [{"id": "knows-city", "checks": [{"check": "value_is", "subject": "user", "predicate": "city", "object": "Boston"}]}],
        )
        result = runner.invoke(app, ["eval", "--since", good])
        assert result.exit_code == 1
        assert "knows-city" in result.output
        assert bad[:8] in result.output

    def test_no_regressions(self, tmp_path):
        repo = _seed_repo(tmp_path)
        good = repo.head_commit()
        repo.commit([Fact(subject="user", predicate="city", object="Boston")], "reaffirm")
        _write_suite(
            repo,
            [{"id": "knows-city", "checks": [{"check": "value_is", "subject": "user", "predicate": "city", "object": "Boston"}]}],
        )
        result = runner.invoke(app, ["eval", "--since", good])
        assert result.exit_code == 0
        assert "no regressions" in result.output
