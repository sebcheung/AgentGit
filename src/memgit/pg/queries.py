"""The queries the projection exists to answer -- all read-only.

Every function here has a twin assertion in ``tests/test_pg_queries.py``
that validates its result against a full CAS walk (``core.graph.walk`` plus
``Repository.diff``) on the same repository. That is the whole point of
this being a *derived* read-model: an answer here is only trustworthy
because it can be checked against the object store that actually owns the
data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Engine, func, select, text

from memgit.pg import schema

__all__ = ["BlameEntry", "DedupStats", "blame", "contains", "dedup_stats"]


@dataclass(frozen=True, slots=True)
class BlameEntry:
    """One commit's effect on one ``(subject, predicate)`` key."""

    commit_hash: str
    op: Literal["+", "-"]
    fact_hash: str
    committed_at: str


def blame(engine: Engine, subject: str, predicate: str) -> tuple[BlameEntry, ...]:
    """Every commit that touched ``(subject, predicate)``, oldest first, with what it did.

    An index seek on ``key_deltas``, joined to ``commits`` only for
    ordering -- the query the filesystem CAS answers by walking every
    commit and diffing each against its first parent (see ``memgit
    blame``'s CLI help for the O(commits) alternative this replaces).
    """
    stmt = (
        select(
            schema.key_deltas.c.commit_hash,
            schema.key_deltas.c.op,
            schema.key_deltas.c.fact_hash,
            schema.commits.c.committed_at,
        )
        .join(schema.commits, schema.commits.c.hash == schema.key_deltas.c.commit_hash)
        .where(schema.key_deltas.c.subject == subject, schema.key_deltas.c.predicate == predicate)
        .order_by(schema.commits.c.committed_at, schema.key_deltas.c.commit_hash, schema.key_deltas.c.op)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).all()
    return tuple(
        BlameEntry(
            commit_hash=row.commit_hash,
            op=row.op,
            fact_hash=row.fact_hash,
            committed_at=row.committed_at.isoformat(),
        )
        for row in rows
    )


@dataclass(frozen=True, slots=True)
class DedupStats:
    """How much the content-addressable store's deduplication is actually buying."""

    distinct_facts: int
    total_tree_entries: int

    @property
    def ratio(self) -> float:
        """Tree entries per distinct fact — 1.0 means no reuse at all yet."""
        if self.distinct_facts == 0:
            return 0.0
        return self.total_tree_entries / self.distinct_facts


def dedup_stats(engine: Engine) -> DedupStats:
    """Distinct fact objects vs. total tree-entry rows across every projected commit.

    A pair of ``COUNT(*)`` queries against tables the filesystem CAS has no
    equivalent index for -- answering this from the object store directly
    means materializing every tree in history and counting by hand.
    """
    with engine.connect() as conn:
        distinct_facts = conn.execute(select(func.count()).select_from(schema.facts)).scalar_one()
        total_tree_entries = conn.execute(select(func.count()).select_from(schema.tree_entries)).scalar_one()
    return DedupStats(distinct_facts=distinct_facts, total_tree_entries=total_tree_entries)


# Both dialects the projector supports handle a plain recursive CTE with
# identical syntax, so one query string serves either -- no
# `postgresql.insert`/`sqlite.insert`-style branch is needed here, unlike
# `project.py`'s upserts.
_CONTAINS_SQL = text(
    """
    WITH RECURSIVE ancestry(hash) AS (
        SELECT :descendant
        UNION
        SELECT cp.parent_hash
        FROM commit_parents cp
        JOIN ancestry a ON cp.child_hash = a.hash
    )
    SELECT 1 FROM ancestry WHERE hash = :ancestor
    """
)


def contains(engine: Engine, *, ancestor: str, descendant: str) -> bool:
    """Whether ``ancestor`` is ``descendant`` itself or one of its ancestors.

    A recursive CTE over real foreign keys, over the same
    ``commit_parents`` table :func:`memgit.core.graph.is_ancestor` answers
    by walking the DAG in Python -- both are checked against each other in
    ``test_pg_queries.py``.
    """
    with engine.connect() as conn:
        row = conn.execute(_CONTAINS_SQL, {"descendant": descendant, "ancestor": ancestor}).first()
    return row is not None
