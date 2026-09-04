"""Read-only routes: repo, log, commits, state, diff, recall.

Every route is a plain, synchronous ``def`` — each does blocking filesystem
IO, and FastAPI runs sync routes in its threadpool; an ``async def`` here
would block the event loop for no benefit, since nothing in this module
ever awaits anything.
"""

from __future__ import annotations

from fastapi import APIRouter

from memgit import __version__
from memgit.api.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only — no repository probe. See PLAN.md's ``/health`` row."""
    return HealthResponse(status="ok", version=__version__)
