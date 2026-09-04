"""Tests for the eval runner (`memgit.eval.runner`)."""

from __future__ import annotations

from datetime import datetime, timezone

from memgit.core.diff import ChangeKind
from memgit.core.fact import Fact
from memgit.eval.case import (
    ConfidenceAtLeast,
    DiffKind,
    EvalCase,
    KeyAbsent,
    KeyExists,
    NoViolations,
    Recalls,
    ValueIs,
)
from memgit.eval.runner import run_case, run_suite
from memgit.eval.case import EvalSuite


def _case(*checks) -> EvalCase:
    return EvalCase(id="c", checks=tuple(checks))


class TestKeyExists:
    def test_pass(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "seed")
        result = run_case(agent_repo, _case(KeyExists("user", "city")))
        assert result.ok

    def test_fail(self, agent_repo):
        agent_repo.commit([], "seed", allow_empty=True)
        result = run_case(agent_repo, _case(KeyExists("user", "city")))
        assert not result.ok
        assert "found none" in result.failures[0].detail


class TestKeyAbsent:
    def test_pass(self, agent_repo):
        agent_repo.commit([], "seed", allow_empty=True)
        result = run_case(agent_repo, _case(KeyAbsent("user", "city")))
        assert result.ok

    def test_fail(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "seed")
        result = run_case(agent_repo, _case(KeyAbsent("user", "city")))
        assert not result.ok


class TestValueIs:
    def test_pass(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "seed")
        result = run_case(agent_repo, _case(ValueIs("user", "city", "Boston")))
        assert result.ok

    def test_fail_reports_actual_objects(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Berlin")], "seed")
        result = run_case(agent_repo, _case(ValueIs("user", "city", "Boston")))
        assert not result.ok
        assert "Berlin" in result.failures[0].detail


class TestConfidenceAtLeast:
    def test_pass(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston", confidence=0.9)], "seed")
        result = run_case(agent_repo, _case(ConfidenceAtLeast("user", "city", 0.5)))
        assert result.ok

    def test_fail_on_low_confidence(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston", confidence=0.2)], "seed")
        result = run_case(agent_repo, _case(ConfidenceAtLeast("user", "city", 0.5)))
        assert not result.ok

    def test_fail_when_key_absent(self, agent_repo):
        agent_repo.commit([], "seed", allow_empty=True)
        result = run_case(agent_repo, _case(ConfidenceAtLeast("user", "city", 0.5)))
        assert not result.ok
        assert "no fact" in result.failures[0].detail

    def test_ignores_decay(self, agent_repo):
        """Stored confidence, never decayed -- per the decay blast-radius row."""
        agent_repo.set_decay(agent_repo.decay().with_half_life("city", 1))
        agent_repo.commit(
            [Fact(subject="user", predicate="city", object="Boston", confidence=1.0, asserted_at="2000-01-01T00:00:00+00:00")],
            "seed",
        )
        result = run_case(agent_repo, _case(ConfidenceAtLeast("user", "city", 0.99)))
        assert result.ok


class TestDiffKind:
    def test_pass(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "add")
        result = run_case(agent_repo, _case(DiffKind("user", "city", (ChangeKind.ADDED,))))
        assert result.ok

    def test_fail_reports_actual_kind(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "add")
        result = run_case(agent_repo, _case(DiffKind("user", "city", (ChangeKind.CONTRADICTED,))))
        assert not result.ok
        assert "added" in result.failures[0].detail

    def test_unchanged_key_when_absent_from_diff(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "add")
        result = run_case(agent_repo, _case(DiffKind("user", "other", (ChangeKind.UNCHANGED,))))
        assert result.ok


class TestNoViolations:
    def test_pass(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "add")
        result = run_case(agent_repo, _case(NoViolations()))
        assert result.ok

    def test_fail_on_single_cardinality_violation(self, agent_repo):
        agent_repo.commit(
            [
                Fact(subject="user", predicate="city", object="Boston"),
                Fact(subject="user", predicate="city", object="Berlin"),
            ],
            "add",
        )
        result = run_case(agent_repo, _case(NoViolations()))
        assert not result.ok


class TestRecalls:
    def test_pass(self, agent_repo):
        agent_repo.commit(
            [Fact(subject="user", predicate="allergies", object="peanuts", source_text="I'm allergic to peanuts")],
            "add",
        )
        check = Recalls(
            query="what should I avoid eating",
            subject="user",
            predicate="allergies",
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            k=8,
        )
        result = run_case(agent_repo, _case(check))
        assert result.ok

    def test_fail_when_not_in_topk(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "add")
        check = Recalls(
            query="anything",
            subject="user",
            predicate="allergies",
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            k=8,
        )
        result = run_case(agent_repo, _case(check))
        assert not result.ok


class TestRunSuite:
    def test_reports_every_case(self, agent_repo):
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "seed")
        suite = EvalSuite(
            name="s",
            cases=(
                _case(KeyExists("user", "city")),
                _case(KeyExists("user", "missing")),
            ),
        )
        result = run_suite(agent_repo, suite)
        assert not result.ok
        assert len(result.cases) == 2
        assert len(result.failed) == 1

    def test_resolves_rev_once(self, agent_repo):
        commit_hash = agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "seed")
        suite = EvalSuite(name="s", cases=(_case(KeyExists("user", "city")),))
        result = run_suite(agent_repo, suite, "HEAD")
        assert result.rev == commit_hash
