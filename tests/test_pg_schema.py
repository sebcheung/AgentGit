"""Tests for the projection schema: create_all against SQLite, structural assertions.

SQLite, not Postgres, because the schema is written to be dialect-portable
(``JSON().with_variant(JSONB, "postgresql")``, ``TIMESTAMP(timezone=True)``)
-- this is what lets these tests run in the default suite, no network, no
``$DATABASE_URL``. Only the Alembic upgrade/downgrade round trip and the
autogenerate-drift guard (``test_pg_migrations.py``) need a real server.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy import Engine, create_engine, inspect

from memgit.pg.schema import metadata, projection


@pytest.fixture
def engine() -> Engine:
    eng = create_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    return eng


def test_every_table_is_created(engine: Engine) -> None:
    tables = set(inspect(engine).get_table_names())
    assert tables == {
        "commits",
        "commit_parents",
        "refs",
        "facts",
        "tree_entries",
        "key_deltas",
        "projection",
    }


def test_commits_primary_key_is_hash(engine: Engine) -> None:
    pk = inspect(engine).get_pk_constraint("commits")
    assert pk["constrained_columns"] == ["hash"]


def test_commit_parents_primary_key_is_composite(engine: Engine) -> None:
    pk = inspect(engine).get_pk_constraint("commit_parents")
    assert set(pk["constrained_columns"]) == {"child_hash", "ordinal"}


def test_commit_parents_has_a_foreign_key_to_commits(engine: Engine) -> None:
    fks = inspect(engine).get_foreign_keys("commit_parents")
    assert any(fk["referred_table"] == "commits" for fk in fks)


def test_tree_entries_has_a_foreign_key_to_facts(engine: Engine) -> None:
    fks = inspect(engine).get_foreign_keys("tree_entries")
    assert any(fk["referred_table"] == "facts" for fk in fks)


def test_key_deltas_has_a_foreign_key_to_commits(engine: Engine) -> None:
    fks = inspect(engine).get_foreign_keys("key_deltas")
    assert any(fk["referred_table"] == "commits" for fk in fks)


def test_key_deltas_is_indexed_for_blame(engine: Engine) -> None:
    indexes = inspect(engine).get_indexes("key_deltas")
    names = {tuple(idx["column_names"]) for idx in indexes}
    assert ("subject", "predicate", "commit_hash") in names


def test_facts_is_indexed_by_key(engine: Engine) -> None:
    indexes = inspect(engine).get_indexes("facts")
    names = {tuple(idx["column_names"]) for idx in indexes}
    assert ("subject", "predicate") in names


def test_commit_parents_is_indexed_by_parent_hash(engine: Engine) -> None:
    indexes = inspect(engine).get_indexes("commit_parents")
    names = {tuple(idx["column_names"]) for idx in indexes}
    assert ("parent_hash",) in names


def test_projection_accepts_its_singleton_row(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            projection.insert().values(
                id=1, format_version=1, projected_refs={}, projected_at=datetime.now(UTC)
            )
        )
    with engine.connect() as conn:
        row = conn.execute(projection.select()).fetchone()
    assert row is not None
    assert row.id == 1


def test_projection_rejects_a_second_row(engine: Engine) -> None:
    from sqlalchemy.exc import IntegrityError

    with engine.begin() as conn:
        conn.execute(
            projection.insert().values(
                id=1, format_version=1, projected_refs={}, projected_at=datetime.now(UTC)
            )
        )
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            projection.insert().values(
                id=2, format_version=1, projected_refs={}, projected_at=datetime.now(UTC)
            )
        )
