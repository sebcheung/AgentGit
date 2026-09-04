"""The FastAPI app factory — slice 10's REST layer and dashboard.

A factory, not a module-level ``app = FastAPI()``, mirroring
``mcp.server.build_server``'s split for the same reason: it is what lets a
test build an app over a ``tmp_path`` repository with no ``chdir``, and what
lets ``memgit serve-web`` fail fast — before uvicorn ever binds — if there
is no repository to serve.
"""

from __future__ import annotations

from importlib import resources

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from memgit import __version__
from memgit.api.errors import register_exception_handlers
from memgit.api.middleware import request_id_middleware
from memgit.api.read import router as read_router
from memgit.api.replay import router as replay_router
from memgit.core.repository import Repository

__all__ = ["create_app"]


def create_app(repo: Repository, *, model: str = "claude-opus-5", static: bool = True) -> FastAPI:
    """Build (but do not run) an app serving ``repo`` read-only, plus replay.

    Args:
        repo: The one repository this app serves. Discovered once, at
            ``memgit serve-web`` startup, and injected here rather than
            re-discovered per request — nothing about it is cached, so one
            instance still observes commits made by another process. See
            PLAN.md's "Repository lifetime" decision row.
        model: The model ``POST /api/replay`` replays with. Server-side,
            never a request field.
        static: Whether to mount the dashboard's static assets at ``/``.
            Tests that only exercise ``/api/*`` can skip it.
    """
    app = FastAPI(title="memgit", version=__version__)
    app.state.repo = repo
    app.state.model = model

    app.middleware("http")(request_id_middleware)
    register_exception_handlers(app)
    app.include_router(read_router, prefix="/api")
    app.include_router(replay_router, prefix="/api")

    if static:
        # `/api/*` is registered above, before this mount -- a Starlette
        # `Mount` at "/" matches every path, so the routers must come first
        # or the mount would shadow them.
        static_dir = resources.files("memgit.api") / "static"
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="dashboard")

    return app
