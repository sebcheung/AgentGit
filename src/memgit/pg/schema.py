"""The projection's tables — SQLAlchemy Core, not the ORM.

Core, not declarative models, because nothing here is an object with
behavior: every row is a denormalized fact about a commit the CAS already
has full custody of. A row is data to query, not an entity to load, mutate,
and save back — the ORM's whole reason to exist doesn't apply.

Seven tables, each earning its place against one thing the filesystem CAS
cannot do cheaply:

- ``commits``/``commit_parents``/``refs`` mirror the commit graph with real
  foreign keys, so ancestry containment (:func:`memgit.pg.queries.contains`)
  is a recursive CTE instead of a Python-side BFS through zlib-inflated
  objects.
- ``facts``/``tree_entries`` mirror one tree's worth of (subject, predicate)
  -> fact-hash mappings per commit, which is what makes
  :func:`memgit.pg.queries.dedup_stats` (distinct fact objects vs. total
  tree-entry rows) a pair of ``COUNT(*)`` queries instead of a full-history
  walk.
- ``key_deltas`` is the one table with no CAS analogue at all: a per-commit,
  per-key, value-level add/remove log, indexed for
  :func:`memgit.pg.queries.blame`. It stores raw ``(+fact_hash)``/
  ``(-fact_hash)`` evidence, never a :class:`~memgit.core.diff.ChangeKind` --
  ``ChangeKind`` is a function of the *uncommitted*, repo-local
  :class:`~memgit.core.cardinality.CardinalityMap`, so materializing one
  into a table would freeze one reading of a question the cardinality map
  itself says has no stored answer. ``blame`` applies the *current* map to
  this raw evidence at read time, exactly as ``memgit diff`` does.
- ``projection`` is a singleton watermark row: how far
  :func:`memgit.pg.project.project` has walked, so re-running it is a
  no-op on an unmoved repository.
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON, TIMESTAMP, Boolean

__all__ = ["metadata"]

metadata = MetaData()

# `JSON().with_variant(JSONB, "postgresql")` and `TIMESTAMP(timezone=True)`
# keep every table dialect-portable: `tests/test_pg_schema.py` runs this
# same schema against SQLite in-memory, no network, in the default CI job.
# Only the Alembic upgrade/downgrade round trip and the autogenerate-drift
# guard (`tests/test_pg_migrations.py`) need a real Postgres server.
_JsonType = JSON().with_variant(JSONB, "postgresql")

commits = Table(
    "commits",
    metadata,
    Column("hash", String(64), primary_key=True),
    Column("tree", String(64), nullable=False),
    Column("message", Text, nullable=False),
    Column("summary", Text, nullable=False),
    Column("author", String(255), nullable=False),
    Column("committed_at", TIMESTAMP(timezone=True), nullable=False),
    Column("is_root", Boolean, nullable=False),
    Column("is_merge", Boolean, nullable=False),
    Column("metadata", _JsonType, nullable=True),
    Index("ix_commits_committed_at", "committed_at"),
)

# The real DAG edges, as real foreign keys -- `parent_hash` is not itself an
# FK to `commits.hash` because a projection walk can insert a commit before
# it has walked as far as one of its parents; only the child side needs to
# exist for the row to make sense at insert time.
commit_parents = Table(
    "commit_parents",
    metadata,
    Column("child_hash", String(64), ForeignKey("commits.hash", ondelete="CASCADE"), primary_key=True),
    Column("ordinal", SmallInteger, primary_key=True),
    Column("parent_hash", String(64), nullable=False),
    Index("ix_commit_parents_parent_hash", "parent_hash"),
)

refs = Table(
    "refs",
    metadata,
    Column("name", String(255), primary_key=True),
    Column("kind", String(32), nullable=False),
    Column("target_hash", String(64), nullable=False),
)

facts = Table(
    "facts",
    metadata,
    Column("hash", String(64), primary_key=True),
    Column("subject", Text, nullable=False),
    Column("predicate", Text, nullable=False),
    Column("object", Text, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("asserted_at", TIMESTAMP(timezone=True), nullable=False),
    Column("source", Text, nullable=True),
    Column("source_text", Text, nullable=True),
    Index("ix_facts_subject_predicate", "subject", "predicate"),
)

tree_entries = Table(
    "tree_entries",
    metadata,
    Column("tree_hash", String(64), primary_key=True),
    Column("subject", Text, primary_key=True),
    Column("predicate", Text, primary_key=True),
    Column("ordinal", SmallInteger, primary_key=True),
    Column("fact_hash", String(64), ForeignKey("facts.hash"), nullable=False),
)

key_deltas = Table(
    "key_deltas",
    metadata,
    Column("commit_hash", String(64), ForeignKey("commits.hash", ondelete="CASCADE"), primary_key=True),
    Column("subject", Text, primary_key=True),
    Column("predicate", Text, primary_key=True),
    Column("fact_hash", String(64), primary_key=True),
    Column("op", String(1), primary_key=True),
    CheckConstraint("op IN ('+', '-')", name="ck_key_deltas_op"),
    Index("ix_key_deltas_subject_predicate_commit", "subject", "predicate", "commit_hash"),
)

# A singleton: exactly one row, `id=1`, holding how far `project()` has
# walked. Never a stored answer to a *query* -- see `dedup_stats`/`blame`,
# which always compute fresh from the tables above -- only a watermark for
# the projector's own incremental-vs-rebuild decision.
projection = Table(
    "projection",
    metadata,
    Column("id", SmallInteger, primary_key=True),
    CheckConstraint("id = 1", name="ck_projection_singleton"),
    Column("format_version", Integer, nullable=False),
    Column("projected_refs", _JsonType, nullable=False),
    Column("projected_at", TIMESTAMP(timezone=True), nullable=False),
)
