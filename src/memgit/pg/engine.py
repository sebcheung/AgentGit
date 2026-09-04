"""Build the SQLAlchemy engine the projection reads and writes through.

A thin wrapper over ``create_engine``, not a class, matching the plain-
function seams the rest of this codebase already uses (``cli.py``'s
``_repo()``, ``api/deps.py``'s ``get_repo``) — there is exactly one engine
per process, built once, and nothing here needs state beyond that.
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine

__all__ = ["DATABASE_URL_ENV", "MissingDatabaseUrlError", "engine_from_env"]

DATABASE_URL_ENV = "DATABASE_URL"


class MissingDatabaseUrlError(Exception):
    """Raised when a command needing the projection has no ``$DATABASE_URL`` to connect with."""


def engine_from_env() -> Engine:
    """Build an engine from ``$DATABASE_URL``.

    Raises:
        MissingDatabaseUrlError: The environment variable is unset. Every
            caller of this function (``memgit project``/``blame``/``stats``)
            is the *only* part of MemGit that needs a database at all, so
            the error message is this function's one chance to say so
            clearly rather than let a bare ``KeyError`` or connection
            failure surface instead.
    """
    url = os.environ.get(DATABASE_URL_ENV)
    if not url:
        raise MissingDatabaseUrlError(
            f"${DATABASE_URL_ENV} is not set -- the projection needs a Postgres connection string "
            "(e.g. postgresql+psycopg://user:pass@localhost/memgit), or `docker compose up db`"
        )
    return create_engine(url)
