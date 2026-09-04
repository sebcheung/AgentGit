"""Tests for the eval sweep (`memgit.eval.sweep`)."""

from __future__ import annotations

import pytest

from memgit.core.fact import Fact
from memgit.core.repository import RevisionNotFoundError
from memgit.eval.case import EvalCase, EvalSuite, ValueIs
from memgit.eval.sweep import sweep


def _suite(*cases: EvalCase) -> EvalSuite:
    return EvalSuite(name="s", cases=cases)


class TestSweep:
    def test_finds_first_breaking_commit(self, agent_repo):
        commit_a = agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "a")
        agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "b")
        commit_c = agent_repo.commit([Fact(subject="user", predicate="city", object="Berlin")], "c")

        case = EvalCase(id="knows-city", checks=(ValueIs("user", "city", "Boston"),))
        (failure,) = sweep(agent_repo, _suite(case), since=commit_a, until=commit_c)

        assert failure.case_id == "knows-city"
        assert failure.commit == commit_c
        assert "Berlin" in failure.message

    def test_case_never_failing_is_omitted(self, agent_repo):
        commit_a = agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "a")
        commit_b = agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "b")

        case = EvalCase(id="knows-city", checks=(ValueIs("user", "city", "Boston"),))
        assert sweep(agent_repo, _suite(case), since=commit_a, until=commit_b) == ()

    def test_failing_at_the_oldest_commit_in_range(self, agent_repo):
        commit_a = agent_repo.commit([Fact(subject="user", predicate="city", object="Berlin")], "a")
        commit_b = agent_repo.commit([Fact(subject="user", predicate="city", object="Berlin")], "b")

        case = EvalCase(id="knows-city", checks=(ValueIs("user", "city", "Boston"),))
        (failure,) = sweep(agent_repo, _suite(case), since=commit_a, until=commit_b)
        assert failure.commit == commit_a

    def test_reports_first_failure_only_once(self, agent_repo):
        commit_a = agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "a")
        commit_b = agent_repo.commit([Fact(subject="user", predicate="city", object="Berlin")], "b")
        commit_c = agent_repo.commit([Fact(subject="user", predicate="city", object="Berlin")], "c")

        case = EvalCase(id="knows-city", checks=(ValueIs("user", "city", "Boston"),))
        (failure,) = sweep(agent_repo, _suite(case), since=commit_a, until=commit_c)
        assert failure.commit == commit_b

    def test_since_not_an_ancestor_raises(self, agent_repo):
        commit_a = agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "a")
        later_commit = agent_repo.commit([Fact(subject="user", predicate="city", object="Boston")], "b")
        case = EvalCase(id="knows-city", checks=(ValueIs("user", "city", "Boston"),))
        with pytest.raises(RevisionNotFoundError):
            sweep(agent_repo, _suite(case), since=later_commit, until=commit_a)

    def test_preserves_suite_case_order(self, agent_repo):
        commit_a = agent_repo.commit([Fact(subject="user", predicate="city", object="Berlin")], "a")

        first = EvalCase(id="b-case", checks=(ValueIs("user", "city", "Boston"),))
        second = EvalCase(id="a-case", checks=(ValueIs("user", "city", "Boston"),))
        failures = sweep(agent_repo, _suite(first, second), since=commit_a, until=commit_a)
        assert [f.case_id for f in failures] == ["b-case", "a-case"]
