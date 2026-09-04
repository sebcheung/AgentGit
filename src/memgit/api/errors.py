"""Maps core exceptions to HTTP responses — once, for every route.

The status a given exception deserves is a property of the exception
*type*, not of the route it happened to surface in: ``RevisionNotFoundError``
means 404 everywhere. Centralizing that here, rather than repeating a
``try``/``except`` in every route the way ``cli.py`` does per command, means
the mapping can't drift between routes, and it also catches failures raised
inside a dependency (``deps.get_repo``, ``deps.get_llm_client``), which a
route-level ``try`` cannot reach. ``cli.py`` stays per-command because there
the *message* is the UX and differs per command; over HTTP there is exactly
one envelope. See PLAN.md's "Exception mapping" decision row.

The one exception kept local to its route is the ``ValueError`` raised by
``Repository.diff``'s ``use_merge_base`` guard — see ``read.py``. A *global*
``ValueError`` handler would turn every genuine bug in this package into a
polite 400, which is exactly how a broken service looks healthy.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from memgit.core.repository import NotARepositoryError, RevisionNotFoundError
from memgit.core.store import CorruptObjectError, ObjectNotFoundError

__all__ = ["LLMUnavailableError", "register_exception_handlers"]


class LLMUnavailableError(Exception):
    """The configured LLM client could not be constructed.

    Raised by :func:`memgit.api.deps.get_llm_client` when the ``agent``
    extra isn't installed or ``ANTHROPIC_API_KEY`` is missing — an
    operator-fixable 503, and deliberately a distinct type from
    :class:`~memgit.agent.client.AgentError`, which means a *live* call
    failed after the client was already built (a 502). Defined here, with
    no dependency on ``memgit.agent``, so this module stays importable on a
    ``web``-only install.
    """


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


def register_exception_handlers(app: FastAPI) -> None:
    """Install one handler per exception type. Called once, from ``create_app``."""

    @app.exception_handler(RevisionNotFoundError)
    async def _revision_not_found(request: Request, exc: RevisionNotFoundError) -> JSONResponse:
        return _error(404, "revision_not_found", str(exc))

    @app.exception_handler(NotARepositoryError)
    async def _not_a_repository(request: Request, exc: NotARepositoryError) -> JSONResponse:
        return _error(503, "repository_unavailable", str(exc))

    @app.exception_handler(ObjectNotFoundError)
    async def _object_not_found(request: Request, exc: ObjectNotFoundError) -> JSONResponse:
        return _error(500, "object_store_corrupt", str(exc))

    @app.exception_handler(CorruptObjectError)
    async def _corrupt_object(request: Request, exc: CorruptObjectError) -> JSONResponse:
        return _error(500, "object_store_corrupt", str(exc))

    @app.exception_handler(LLMUnavailableError)
    async def _llm_unavailable(request: Request, exc: LLMUnavailableError) -> JSONResponse:
        return _error(503, "llm_unavailable", str(exc))

    # Imported lazily so this module -- and therefore `memgit.api.app` --
    # never pulls in `memgit.agent` at import time. The class itself has no
    # external dependency; only the anthropic SDK it wraps does, and that is
    # imported lazily again, one layer further down, inside
    # `AnthropicClient` itself. So this costs nothing even on a `web`-only
    # install, and only runs once `create_app` (not a bare module import)
    # actually executes.
    from memgit.agent.client import AgentError

    @app.exception_handler(AgentError)
    async def _agent_error(request: Request, exc: AgentError) -> JSONResponse:
        return _error(502, "agent_error", str(exc))
