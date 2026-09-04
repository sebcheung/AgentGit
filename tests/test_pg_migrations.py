"""Tests that need a real Postgres server: the upgrade/downgrade round trip and the drift guard.

Excluded from the default run by the ``postgres`` marker (see
pyproject.toml), mirroring ``live``'s own opt-in shape. Run explicitly
with::

    DATABASE_URL=postgresql+psycopg://user:pass@localhost/memgit \\
        py -m uv run pytest -m postgres --all-extras

The drift guard is what keeps ``schema.py`` and the migrations honest with
each other over time: if a future change edits one without a matching
change to the other, ``alembic revision --autogenerate`` against head would
produce a non-empty diff. This test fails loudly the moment that happens,
instead of the drift being discovered the next time someone actually runs
autogenerate by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("alembic")

pytestmark = pytest.mark.postgres

if not os.environ.get("DATABASE_URL"):
    pytest.skip("DATABASE_URL not set", allow_module_level=True)

from alembic.autogenerate import compare_metadata
from alembic.command import downgrade, upgrade
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from memgit.pg.schema import metadata

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TABLE_NAMES = {"commits", "commit_parents", "refs", "facts", "tree_entries", "key_deltas", "projection"}


@pytest.fixture
def alembic_config() -> Config:
    config = Config(str(_REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_REPO_ROOT / "migrations"))
    return config


@pytest.fixture(autouse=True)
def _downgrade_after(alembic_config: Config):
    yield
    downgrade(alembic_config, "base")


def test_upgrade_creates_every_table(alembic_config: Config) -> None:
    upgrade(alembic_config, "head")
    engine = create_engine(os.environ["DATABASE_URL"])
    tables = set(inspect(engine).get_table_names())
    assert _TABLE_NAMES <= tables


def test_downgrade_removes_every_table(alembic_config: Config) -> None:
    upgrade(alembic_config, "head")
    downgrade(alembic_config, "base")
    engine = create_engine(os.environ["DATABASE_URL"])
    tables = set(inspect(engine).get_table_names())
    assert not (_TABLE_NAMES & tables)


def test_autogenerate_produces_no_diff_at_head(alembic_config: Config) -> None:
    upgrade(alembic_config, "head")
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        migration_context = MigrationContext.configure(conn)
        diff = compare_metadata(migration_context, metadata)
    assert diff == [], f"schema.py and the migrations have drifted: {diff}"
