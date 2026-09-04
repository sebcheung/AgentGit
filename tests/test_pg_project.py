"""Tests for the incremental projector: full rebuild matches the commit graph, re-running is a no-op.

SQLite in-memory throughout -- no network, part of the default suite. Every
assertion checks the projection's shape against what ``repo.commit()``/
``repo.create_branch()`` actually built, not against a second implementation
of the same logic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import create_engine, select

from memgit.core.commit import Commit
from memgit.core.fact import Fact
from memgit.core.repository import Repository
from memgit.pg import schema
from memgit.pg.project import project


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


class TestFullProjection:
    def test_projects_every_commit(self, repo, engine):
        repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "reaffirm")
        result = project(repo, engine)
        assert result.commits_added == 2
        with engine.connect() as conn:
            rows = conn.execute(select(schema.commits)).all()
        assert len(rows) == 2

    def test_records_commit_parents(self, repo, engine):
        first = repo.commit([make_fact()], "seed")
        second = repo.commit([make_fact(object="Rust")], "reaffirm")
        project(repo, engine)
        with engine.connect() as conn:
            rows = conn.execute(
                select(schema.commit_parents).where(schema.commit_parents.c.child_hash == second)
            ).all()
        assert len(rows) == 1
        assert rows[0].parent_hash == first
        assert rows[0].ordinal == 0

    def test_root_commit_has_no_parent_rows(self, repo, engine):
        first = repo.commit([make_fact()], "seed")
        project(repo, engine)
        with engine.connect() as conn:
            rows = conn.execute(
                select(schema.commit_parents).where(schema.commit_parents.c.child_hash == first)
            ).all()
        assert rows == []

    def test_projects_a_forked_branch_topology(self, repo, engine):
        repo.commit([make_fact()], "seed")
        repo.create_branch("experiment")
        repo.commit([make_fact(predicate="current_project", object="memgit")], "main-only")
        repo.refs.detach_head(repo.resolve("experiment"))
        repo.commit([make_fact(predicate="timezone", object="UTC")], "experiment-only")
        repo.create_branch("experiment-tip", at="HEAD")

        result = project(repo, engine)

        assert result.commits_added == 3

    def test_records_a_multi_parent_merge_commit(self, repo, engine):
        first = repo.commit([make_fact()], "seed")
        repo.create_branch("side")
        second = repo.commit([make_fact(predicate="current_project", object="memgit")], "main-only")
        repo.refs.detach_head(first)
        third = repo.commit([make_fact(predicate="timezone", object="UTC")], "side-only")

        merge_tree_hash = repo.read_commit(second).tree
        merge = Commit(tree=merge_tree_hash, parents=(second, third), message="merge", author="test")
        merge_hash = merge.write(repo.store)
        repo.refs.write_ref("refs/heads/main", merge_hash, op="commit")

        project(repo, engine)

        with engine.connect() as conn:
            rows = conn.execute(
                select(schema.commit_parents)
                .where(schema.commit_parents.c.child_hash == merge_hash)
                .order_by(schema.commit_parents.c.ordinal)
            ).all()
        assert [r.parent_hash for r in rows] == [second, third]

    def test_records_facts_and_tree_entries(self, repo, engine):
        repo.commit([make_fact()], "seed")
        project(repo, engine)
        with engine.connect() as conn:
            facts = conn.execute(select(schema.facts)).all()
            entries = conn.execute(select(schema.tree_entries)).all()
        assert len(facts) == 1
        assert len(entries) == 1

    def test_dedups_a_fact_reused_across_commits(self, repo, engine):
        fact = make_fact()
        repo.commit([fact], "seed")
        repo.commit([fact, make_fact(subject="other", predicate="x", object="y")], "add-more")

        project(repo, engine)

        with engine.connect() as conn:
            facts = conn.execute(select(schema.facts)).all()
            entries = conn.execute(select(schema.tree_entries)).all()
        assert len(facts) == 2
        assert len(entries) == 3

    def test_records_key_deltas_for_a_reaffirmation(self, repo, engine):
        repo.commit([make_fact(confidence=0.5)], "seed")
        second = repo.commit([make_fact(confidence=0.9)], "reaffirm")
        project(repo, engine)
        with engine.connect() as conn:
            rows = conn.execute(
                select(schema.key_deltas).where(schema.key_deltas.c.commit_hash == second)
            ).all()
        assert sorted(r.op for r in rows) == ["+", "-"]

    def test_records_branch_refs(self, repo, engine):
        head = repo.commit([make_fact()], "seed")
        project(repo, engine)
        with engine.connect() as conn:
            row = conn.execute(select(schema.refs).where(schema.refs.c.name == "main")).first()
        assert row is not None
        assert row.target_hash == head


class TestIdempotency:
    def test_rerunning_on_an_unmoved_repo_adds_nothing(self, repo, engine):
        repo.commit([make_fact()], "seed")
        first = project(repo, engine)
        second = project(repo, engine)
        assert first.commits_added == 1
        assert second.commits_added == 0

    def test_rerunning_after_a_new_commit_only_adds_the_new_one(self, repo, engine):
        repo.commit([make_fact()], "seed")
        project(repo, engine)
        repo.commit([make_fact(object="Rust")], "reaffirm")
        result = project(repo, engine)
        assert result.commits_added == 1

    def test_ref_target_updates_on_rerun(self, repo, engine):
        repo.commit([make_fact()], "seed")
        project(repo, engine)
        second = repo.commit([make_fact(object="Rust")], "reaffirm")
        project(repo, engine)
        with engine.connect() as conn:
            row = conn.execute(select(schema.refs).where(schema.refs.c.name == "main")).first()
        assert row.target_hash == second


class TestEmptyRepository:
    def test_unborn_repo_projects_nothing(self, repo, engine):
        result = project(repo, engine)
        assert result.commits_added == 0
        assert result.facts_added == 0
