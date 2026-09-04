"""A Postgres read-model over the filesystem CAS -- a projection, never the source of truth.

A commit's identity is the SHA-256 of its own canonical JSON; a table
holding that same JSON is a second copy that can disagree with its own
hash the moment either one drifts. So the CAS in ``core/`` stays canonical,
and everything under this package exists to answer one query class the CAS
cannot answer cheaply: "which commits touched (subject, predicate), and
what did each one do to it" -- see :func:`memgit.pg.queries.blame`.

Nothing under ``memgit.core`` imports this package, and nothing in this
package is required for any existing command to keep working with no
database configured at all -- see ``tests/test_pg_imports.py``.
"""

from __future__ import annotations

__all__: list[str] = []
