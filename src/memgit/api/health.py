"""Liveness — ``GET /api/health``, exempt from API-key auth.

Its own router, not a route in ``read.py``, because it must never require a
key: Docker's own ``HEALTHCHECK`` has no key to send, and a liveness probe
that 401s reads as "down" to whatever is polling it. ``deps.require_api_key``
is applied per included-router in ``app.py``, and this router is the one
deliberately left off that list.
"""

from __future__ import annotations

from fastapi import APIRouter

from memgit import __version__
from memgit.api.models import HealthResponse

__all__ = ["router"]

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only — no repository probe. See PLAN.md's ``/health`` row."""
    return HealthResponse(status="ok", version=__version__)
