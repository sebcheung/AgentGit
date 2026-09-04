"""Tests for `memgit recall` (slice 7).

A sibling of ``test_cli_agent.py``/``test_cli_replay.py`` in spirit, but
``recall`` never touches an LLM client — it only exercises
``Repository.retriever()`` and the ranking path, so these tests use a plain
``CliRunner`` with no client monkeypatch.
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
        },
        {
            "type": "fact",
            "subject": "user",
            "predicate": "favorite_editor",
            "object": "Vim",
            "confidence": 0.8,
            "asserted_at": "2026-09-01T00:00:00+00:00",
        },
    ]
)


def _seed(tmp_path) -> None:
    runner.invoke(app, ["init"])
    (tmp_path / "facts.json").write_text(FACTS, encoding="utf-8")
    result = runner.invoke(app, ["commit", "-m", "seed", "--file", "facts.json"])
    assert result.exit_code == 0, result.output


class TestRecall:
    def test_returns_matching_facts(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "what language does the user prefer"])
        assert result.exit_code == 0
        assert "prefers_language" in result.output

    def test_reports_scope_header(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "language"])
        assert result.exit_code == 0
        assert "of 2 fact(s) at HEAD" in result.output

    def test_limit_narrows_results(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "user beliefs", "-k", "1"])
        assert result.exit_code == 0
        assert "1 of 2 fact(s)" in result.output

    def test_subject_filters(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "anything", "--subject", "nonexistent-subject"])
        assert result.exit_code == 0
        assert "0 of 0 fact(s)" in result.output

    def test_json_output_includes_scores(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "editor", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["candidates"] == 2
        assert all("score" in fact for fact in payload["facts"])

    def test_as_of_is_deterministic(self, tmp_path):
        _seed(tmp_path)
        first = runner.invoke(app, ["recall", "language", "--as-of", "2027-01-01T00:00:00+00:00", "--json"])
        second = runner.invoke(app, ["recall", "language", "--as-of", "2027-01-01T00:00:00+00:00", "--json"])
        assert json.loads(first.output) == json.loads(second.output)

    def test_no_decay_ranks_on_similarity_only(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "language", "--no-decay", "--json"])
        assert result.exit_code == 0

    def test_unknown_revision_fails(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "language", "does-not-exist"])
        assert result.exit_code == 1

    def test_bad_as_of_fails(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "language", "--as-of", "not-a-timestamp"])
        assert result.exit_code == 1

    def test_min_score_can_exclude_everything(self, tmp_path):
        _seed(tmp_path)
        result = runner.invoke(app, ["recall", "completely unrelated nonsense topic", "--min-score", "0.999"])
        assert result.exit_code == 0
        assert "No matching facts." in result.output

    def test_scoped_to_revision_never_leaks_a_later_commit(self, tmp_path):
        runner.invoke(app, ["init"])
        (tmp_path / "first.json").write_text(
            json.dumps(
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
            ),
            encoding="utf-8",
        )
        runner.invoke(app, ["commit", "-m", "first", "--file", "first.json"])
        first_head = runner.invoke(app, ["rev-parse", "HEAD"]).output.strip()

        (tmp_path / "second.json").write_text(
            json.dumps(
                [
                    {
                        "type": "fact",
                        "subject": "user",
                        "predicate": "prefers_language",
                        "object": "Python",
                        "confidence": 0.9,
                        "asserted_at": "2026-09-01T00:00:00+00:00",
                    },
                    {
                        "type": "fact",
                        "subject": "user",
                        "predicate": "favorite_editor",
                        "object": "Vim",
                        "confidence": 0.8,
                        "asserted_at": "2026-09-02T00:00:00+00:00",
                    },
                ]
            ),
            encoding="utf-8",
        )
        runner.invoke(app, ["commit", "-m", "second", "--file", "second.json"])

        # Warm the index against HEAD (which has both facts) before querying
        # the earlier commit -- proves the scope guarantee isn't just "we
        # never embedded the later fact".
        runner.invoke(app, ["embed"])

        result = runner.invoke(app, ["recall", "editor", first_head, "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["candidates"] == 1
        assert all(fact["predicate"] != "favorite_editor" for fact in payload["facts"])
