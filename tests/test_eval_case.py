"""Tests for eval case/suite parsing and discovery (`memgit.eval.case`)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memgit.core.diff import ChangeKind
from memgit.eval.case import (
    ConfidenceAtLeast,
    DiffKind,
    EvalCase,
    EvalFormatError,
    EvalSuite,
    KeyAbsent,
    KeyExists,
    NoViolations,
    Recalls,
    ValueIs,
    load_suites,
)


class TestCheckDecoding:
    def test_key_exists(self):
        case = EvalCase.from_dict({"id": "c1", "checks": [{"check": "key_exists", "subject": "user", "predicate": "city"}]})
        assert case.checks == (KeyExists(subject="user", predicate="city"),)

    def test_key_absent(self):
        case = EvalCase.from_dict({"id": "c1", "checks": [{"check": "key_absent", "subject": "user", "predicate": "city"}]})
        assert case.checks == (KeyAbsent(subject="user", predicate="city"),)

    def test_value_is(self):
        payload = {"check": "value_is", "subject": "user", "predicate": "city", "object": "Boston"}
        case = EvalCase.from_dict({"id": "c1", "checks": [payload]})
        assert case.checks == (ValueIs(subject="user", predicate="city", object="Boston"),)

    def test_confidence_at_least(self):
        payload = {"check": "confidence_at_least", "subject": "user", "predicate": "city", "min": 0.5}
        case = EvalCase.from_dict({"id": "c1", "checks": [payload]})
        assert case.checks == (ConfidenceAtLeast(subject="user", predicate="city", min=0.5),)

    def test_diff_kind(self):
        payload = {
            "check": "diff_kind",
            "subject": "user",
            "predicate": "city",
            "kinds": ["contradicted", "added"],
        }
        case = EvalCase.from_dict({"id": "c1", "checks": [payload]})
        assert case.checks == (
            DiffKind(subject="user", predicate="city", kinds=(ChangeKind.CONTRADICTED, ChangeKind.ADDED)),
        )

    def test_no_violations(self):
        case = EvalCase.from_dict({"id": "c1", "checks": [{"check": "no_violations"}]})
        assert case.checks == (NoViolations(),)

    def test_recalls(self):
        payload = {
            "check": "recalls",
            "query": "what should I order",
            "subject": "user",
            "predicate": "allergies",
            "as_of": "2026-01-01T00:00:00+00:00",
            "k": 3,
        }
        case = EvalCase.from_dict({"id": "c1", "checks": [payload]})
        (check,) = case.checks
        assert isinstance(check, Recalls)
        assert check.k == 3
        assert check.as_of == datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_recalls_defaults_k(self):
        payload = {
            "check": "recalls",
            "query": "q",
            "subject": "user",
            "predicate": "allergies",
            "as_of": "2026-01-01T00:00:00+00:00",
        }
        case = EvalCase.from_dict({"id": "c1", "checks": [payload]})
        assert case.checks[0].k == 8


class TestCheckDecodingErrors:
    def test_unknown_check_kind(self):
        with pytest.raises(EvalFormatError, match="unknown check kind"):
            EvalCase.from_dict({"id": "c1", "checks": [{"check": "nonsense"}]})

    def test_missing_check_key(self):
        with pytest.raises(EvalFormatError, match="missing its 'check' kind"):
            EvalCase.from_dict({"id": "c1", "checks": [{"subject": "user"}]})

    def test_missing_required_field(self):
        with pytest.raises(EvalFormatError, match="missing required keys"):
            EvalCase.from_dict({"id": "c1", "checks": [{"check": "key_exists", "subject": "user"}]})

    def test_unknown_field(self):
        payload = {"check": "key_exists", "subject": "user", "predicate": "city", "surprise": 1}
        with pytest.raises(EvalFormatError, match="unknown keys"):
            EvalCase.from_dict({"id": "c1", "checks": [payload]})

    def test_unknown_change_kind(self):
        payload = {"check": "diff_kind", "subject": "user", "predicate": "city", "kinds": ["bogus"]}
        with pytest.raises(EvalFormatError, match="invalid 'kinds' entry"):
            EvalCase.from_dict({"id": "c1", "checks": [payload]})

    def test_invalid_as_of(self):
        payload = {
            "check": "recalls",
            "query": "q",
            "subject": "user",
            "predicate": "allergies",
            "as_of": "not-a-date",
        }
        with pytest.raises(EvalFormatError, match="invalid 'as_of'"):
            EvalCase.from_dict({"id": "c1", "checks": [payload]})


class TestEvalCase:
    def test_requires_id(self):
        with pytest.raises(EvalFormatError, match="non-empty string"):
            EvalCase.from_dict({"checks": [{"check": "no_violations"}]})

    def test_requires_nonempty_checks(self):
        with pytest.raises(EvalFormatError, match="non-empty 'checks'"):
            EvalCase.from_dict({"id": "c1", "checks": []})

    def test_unknown_case_key(self):
        with pytest.raises(EvalFormatError, match="unknown keys"):
            EvalCase.from_dict({"id": "c1", "checks": [{"check": "no_violations"}], "surprise": 1})

    def test_description_is_optional(self):
        case = EvalCase.from_dict(
            {"id": "c1", "description": "sanity", "checks": [{"check": "no_violations"}]}
        )
        assert case.description == "sanity"


class TestEvalSuite:
    def test_from_dict(self):
        suite = EvalSuite.from_dict(
            "basics",
            {"cases": [{"id": "c1", "checks": [{"check": "no_violations"}]}]},
        )
        assert suite.name == "basics"
        assert len(suite) == 1
        assert [c.id for c in suite] == ["c1"]

    def test_duplicate_case_id_rejected(self):
        payload = {
            "cases": [
                {"id": "c1", "checks": [{"check": "no_violations"}]},
                {"id": "c1", "checks": [{"check": "no_violations"}]},
            ]
        }
        with pytest.raises(EvalFormatError, match="duplicate case id"):
            EvalSuite.from_dict("basics", payload)

    def test_unsupported_version_rejected(self):
        with pytest.raises(EvalFormatError, match="unsupported version"):
            EvalSuite.from_dict("basics", {"version": 99, "cases": []})

    def test_missing_cases_rejected(self):
        with pytest.raises(EvalFormatError, match="'cases' list"):
            EvalSuite.from_dict("basics", {})


class TestLoadSuites:
    def test_missing_default_directory_is_empty(self, agent_repo):
        assert load_suites(agent_repo) == ()

    def test_loads_every_json_file_sorted(self, agent_repo):
        eval_dir = agent_repo.memgit_dir / "eval"
        eval_dir.mkdir()
        (eval_dir / "b.json").write_text(json.dumps({"cases": []}), encoding="utf-8")
        (eval_dir / "a.json").write_text(json.dumps({"cases": []}), encoding="utf-8")

        suites = load_suites(agent_repo)
        assert [s.name for s in suites] == ["a", "b"]

    def test_explicit_file_path(self, agent_repo, tmp_path):
        path = tmp_path / "custom.json"
        path.write_text(json.dumps({"cases": [{"id": "c1", "checks": [{"check": "no_violations"}]}]}))
        (suite,) = load_suites(agent_repo, path)
        assert suite.name == "custom"

    def test_explicit_directory_path(self, agent_repo, tmp_path):
        other = tmp_path / "suites"
        other.mkdir()
        (other / "s.json").write_text(json.dumps({"cases": []}))
        suites = load_suites(agent_repo, other)
        assert [s.name for s in suites] == ["s"]

    def test_missing_explicit_path_raises(self, agent_repo, tmp_path):
        with pytest.raises(EvalFormatError, match="no such suite"):
            load_suites(agent_repo, tmp_path / "nope")

    def test_invalid_json_raises(self, agent_repo):
        eval_dir = agent_repo.memgit_dir / "eval"
        eval_dir.mkdir()
        (eval_dir / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(EvalFormatError, match="not valid JSON"):
            load_suites(agent_repo)
