"""Tests for blame/dedup_stats/contains -- each validated against a full CAS walk on the same repo.

The projection is a derived read-model; the point of that design is that
every answer it gives can be checked against the object store that
actually owns the data. So every test here builds a small history, asks
the projection a question, and asks the CAS the same question by hand
(``core.graph.walk``/``core.graph.is_ancestor``/``Repository.diff``) to
confirm they agree.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine

from memgit.core.fact import Fact
from memgit.core.graph import is_ancestor, walk
from memgit.core.repository import Repository
from memgit.pg import schema
from memgit.pg.project import project
from memgit.pg.queries import blame, contains, dedup_stats


def make_fact(**overrides):
    defaults = {"subject": "user", "predicate": "prefers_language", "object": "Python", "confidence": 0.9}
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository.init(tmp_path)


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    schema.metadata.create_all(eng)
    return eng


def _cas_key_history(repo: Repository, subject: str, predicate: str) -> list[tuple[str, str, str]]:
    """Every ``(commit_hash, op, fact_hash)`` event at this key, oldest first -- blame's ground truth."""
    commits = list(walk(list(repo.branches().values()), repo.read_commit))
    commits.sort(key=lambda pair: pair[1].committed_at)
    events: list[tuple[str, str, str]] = []
    for commit_hash, commit in commits:
        parent = commit.parents[0] if commit.parents else None
        result = repo.diff(before=parent, after=commit_hash)
        commit_events: list[tuple[str, str, str]] = []
        for kd in result.keys:
            if kd.subject != subject or kd.predicate != predicate:
                continue
            for value in kd.changed_values:
                if value.before is not None:
                    commit_events.append((commit_hash, "-", value.before.hash))
                if value.after is not None:
                    commit_events.append((commit_hash, "+", value.after.hash))
        # Matches blame()'s own ORDER BY (committed_at, commit_hash, op):
        # within one commit, '+' sorts before '-' lexically.
        commit_events.sort(key=lambda e: e[1])
        events.extend(commit_events)
    return events


class TestBlame:
    def test_matches_a_full_cas_walk(self, repo, engine):
        repo.commit([make_fact(confidence=0.5)], "seed")
        repo.commit([make_fact(confidence=0.9)], "reaffirm")
        repo.commit([make_fact(subject="other", predicate="x", object="y")], "unrelated")
        project(repo, engine)

        expected = _cas_key_history(repo, "user", "prefers_language")
        actual = [(e.commit_hash, e.op, e.fact_hash) for e in blame(engine, "user", "prefers_language")]

        assert actual == expected

    def test_no_history_is_an_empty_tuple(self, repo, engine):
        repo.commit([make_fact()], "seed")
        project(repo, engine)
        assert blame(engine, "nobody", "nothing") == ()

    def test_ordered_oldest_first(self, repo, engine):
        repo.commit([make_fact(confidence=0.1)], "seed")
        repo.commit([make_fact(confidence=0.5)], "step2")
        repo.commit([make_fact(confidence=0.9)], "step3")
        project(repo, engine)

        timestamps = [e.committed_at for e in blame(engine, "user", "prefers_language")]

        assert timestamps == sorted(timestamps)


class TestDedupStats:
    def test_reused_fact_raises_the_ratio(self, repo, engine):
        fact = make_fact()
        repo.commit([fact], "seed")
        repo.commit([fact, make_fact(subject="other", predicate="x", object="y")], "add-more")
        project(repo, engine)

        stats = dedup_stats(engine)

        assert stats.distinct_facts == 2
        assert stats.total_tree_entries == 3
        assert stats.ratio == pytest.approx(1.5)

    def test_empty_projection_has_a_zero_ratio(self, repo, engine):
        stats = dedup_stats(engine)
        assert stats.distinct_facts == 0
        assert stats.ratio == 0.0


class TestContains:
    def test_matches_is_ancestor_on_a_forked_history(self, repo, engine):
        first = repo.commit([make_fact()], "seed")
        repo.create_branch("experiment")
        second = repo.commit([make_fact(predicate="current_project", object="memgit")], "main-only")
        repo.refs.detach_head(repo.resolve("experiment"))
        third = repo.commit([make_fact(predicate="timezone", object="UTC")], "experiment-only")
        repo.create_branch("experiment-tip", at="HEAD")
        project(repo, engine)

        pairs = [(first, second), (first, third), (second, third), (third, second), (first, first)]
        for ancestor, descendant in pairs:
            assert contains(engine, ancestor=ancestor, descendant=descendant) == is_ancestor(
                ancestor, of=descendant, read=repo.read_commit
            )

    def test_a_commit_contains_itself(self, repo, engine):
        head = repo.commit([make_fact()], "seed")
        project(repo, engine)
        assert contains(engine, ancestor=head, descendant=head) is True

    def test_unrelated_roots_do_not_contain_each_other(self, repo, engine):
        first = repo.commit([make_fact()], "seed")
        repo.create_branch("first-branch", at=first)
        second = repo.commit([make_fact(subject="other", predicate="x", object="y")], "second root", parents=())
        project(repo, engine)

        assert contains(engine, ancestor=first, descendant=second) is False
        assert contains(engine, ancestor=second, descendant=first) is False
