"""Walk the commit graph and upsert it into the projection.

``project()`` is the only place that writes to the tables in
:mod:`memgit.pg.schema`. Nothing in ``memgit.core`` calls it — a write hook
in :meth:`~memgit.core.repository.Repository.commit` would put a database
in the path of the offline core, which is exactly the discipline this
package exists to keep. Instead it is invoked by hand (``memgit project``)
or from an operator's own cron/CI step, and it can go stale between runs
with no correctness consequence: every query in :mod:`memgit.pg.queries` is
validated against a full CAS walk in its own tests, so a stale projection
is a slow answer, never a wrong one, once it does run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Connection, Engine, select
from sqlalchemy.dialects import postgresql, sqlite

from memgit.core.graph import walk
from memgit.core.repository import Repository
from memgit.pg import schema

__all__ = ["ProjectionResult", "project"]


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """What one :func:`project` call did."""

    commits_added: int
    facts_added: int


def _insert_or_ignore(engine: Engine, conn: Connection, table: Any, rows: list[dict[str, Any]]) -> None:
    """Insert ``rows``, silently skipping any that collide on a primary key.

    Needed because a fact hash (or, in principle, a tree hash) is
    content-addressed and legitimately reappears across many commits' trees
    -- that reuse is the dedup ratio :func:`memgit.pg.queries.dedup_stats`
    measures, not an error. Dialect-specific because "insert, ignore
    conflicts" has no portable SQL spelling; only SQLite and PostgreSQL are
    handled since those are the only two dialects this project's tests and
    deployment target actually exercise.
    """
    if not rows:
        return
    stmt: Any
    if engine.dialect.name == "postgresql":
        stmt = postgresql.insert(table).values(rows).on_conflict_do_nothing()
    elif engine.dialect.name == "sqlite":
        stmt = sqlite.insert(table).values(rows).on_conflict_do_nothing()
    else:
        raise NotImplementedError(f"unsupported dialect for the projection: {engine.dialect.name!r}")
    conn.execute(stmt)


def project(repo: Repository, engine: Engine) -> ProjectionResult:
    """Walk every commit reachable from every branch tip; upsert what's new.

    Idempotent: a commit already present in the ``commits`` table is never
    re-read or re-inserted, so running this again on an unmoved repository
    does no work beyond the walk itself and the branch-tip bookkeeping.
    Re-walks the full DAG from the tips on every call rather than resuming
    from a stored watermark -- the simpler, honestly-stated tradeoff for a
    projection sized for a demo repository, not a multi-million-commit one;
    ``graph.walk`` already deduplicates a diamond within one call, so the
    cost is one full history read, not one per unprojected commit.
    """
    with engine.begin() as conn:
        existing = {row.hash for row in conn.execute(select(schema.commits.c.hash))}

    starts = list(repo.branches().values())
    if not starts:
        return ProjectionResult(commits_added=0, facts_added=0)

    commits_added = 0
    facts_added = 0

    with engine.begin() as conn:
        for commit_hash, commit in walk(starts, repo.read_commit):
            if commit_hash in existing:
                continue

            conn.execute(
                schema.commits.insert().values(
                    hash=commit_hash,
                    tree=commit.tree,
                    message=commit.message,
                    summary=commit.summary,
                    author=commit.author,
                    committed_at=datetime.fromisoformat(commit.committed_at),
                    is_root=commit.is_root,
                    is_merge=commit.is_merge,
                    metadata=dict(commit.metadata) if commit.metadata is not None else None,
                )
            )
            if commit.parents:
                conn.execute(
                    schema.commit_parents.insert(),
                    [
                        {"child_hash": commit_hash, "ordinal": ordinal, "parent_hash": parent}
                        for ordinal, parent in enumerate(commit.parents)
                    ],
                )
            commits_added += 1

            tree = repo.read_tree(commit.tree)
            fact_rows: list[dict[str, Any]] = []
            tree_entry_rows: list[dict[str, Any]] = []
            seen_facts: set[str] = set()
            for subject, predicate, fact_hashes in tree.entries:
                for ordinal, fact_hash in enumerate(fact_hashes):
                    tree_entry_rows.append(
                        {
                            "tree_hash": commit.tree,
                            "subject": subject,
                            "predicate": predicate,
                            "ordinal": ordinal,
                            "fact_hash": fact_hash,
                        }
                    )
                    if fact_hash in seen_facts:
                        continue
                    seen_facts.add(fact_hash)
                    fact = repo.read_fact(fact_hash)
                    fact_rows.append(
                        {
                            "hash": fact_hash,
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.object,
                            "confidence": fact.confidence,
                            "asserted_at": datetime.fromisoformat(fact.asserted_at),
                            "source": fact.source,
                            "source_text": fact.source_text,
                        }
                    )
            _insert_or_ignore(engine, conn, schema.facts, fact_rows)
            facts_added += len(fact_rows)
            _insert_or_ignore(engine, conn, schema.tree_entries, tree_entry_rows)

            # `before=None` is `Repository.diff`'s own "this commit's first
            # parent, or the empty tree for a root" -- exactly the boundary
            # `key_deltas` needs. The cardinality map `diff` reads
            # internally only picks a *label* (`ChangeKind`) this table
            # never stores; `.values` -- which side of each object existed
            # on -- does not depend on it at all. See schema.py's
            # module docstring for why that label is deliberately absent
            # here.
            delta_rows: list[dict[str, Any]] = []
            for key_diff in repo.diff(after=commit_hash).keys:
                for value in key_diff.changed_values:
                    if value.before is not None:
                        delta_rows.append(
                            {
                                "commit_hash": commit_hash,
                                "subject": key_diff.subject,
                                "predicate": key_diff.predicate,
                                "fact_hash": value.before.hash,
                                "op": "-",
                            }
                        )
                    if value.after is not None:
                        delta_rows.append(
                            {
                                "commit_hash": commit_hash,
                                "subject": key_diff.subject,
                                "predicate": key_diff.predicate,
                                "fact_hash": value.after.hash,
                                "op": "+",
                            }
                        )
            if delta_rows:
                conn.execute(schema.key_deltas.insert(), delta_rows)

        refs = repo.branches()
        _insert_or_ignore(
            engine,
            conn,
            schema.refs,
            [{"name": name, "kind": "branch", "target_hash": target} for name, target in refs.items()],
        )
        for name, target in refs.items():
            conn.execute(schema.refs.update().where(schema.refs.c.name == name).values(target_hash=target))

        _insert_or_ignore(
            engine,
            conn,
            schema.projection,
            [{"id": 1, "format_version": 1, "projected_refs": refs, "projected_at": datetime.now(UTC)}],
        )
        conn.execute(
            schema.projection.update()
            .where(schema.projection.c.id == 1)
            .values(format_version=1, projected_refs=refs, projected_at=datetime.now(UTC))
        )

    return ProjectionResult(commits_added=commits_added, facts_added=facts_added)
