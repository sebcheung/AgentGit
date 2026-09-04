"""Liveness and readiness — both exempt from API-key auth.

Their own router, not routes in ``read.py``, because they must never
require a key: Docker's own ``HEALTHCHECK`` has no key to send, and a
liveness or readiness probe that 401s reads as "down" to whatever is
polling it. ``deps.require_api_key`` is applied per included-router in
``app.py``, and this router is the one deliberately left off that list.

``/api/health`` and ``/api/ready`` are two endpoints, not one, because they
drive two different actions: an orchestrator *restarts* a process that
fails liveness, but only *depools* one that fails readiness. Conflating
them would restart a perfectly healthy process over a slow dependency --
exactly the failure mode a real readiness check exists to avoid.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from memgit import __version__
from memgit.api.deps import get_repo
from memgit.api.models import CheckResult, HealthResponse, ReadyResponse
from memgit.core.repository import Repository

__all__ = ["router"]

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only — no repository probe. See PLAN.md's ``/health`` row."""
    return HealthResponse(status="ok", version=__version__)


def _check_repository(repo: Repository) -> CheckResult:
    start = time.perf_counter()
    ok = repo.memgit_dir.is_dir()
    detail = "ok" if ok else f"{repo.memgit_dir} does not exist"
    return CheckResult(ok=ok, detail=detail, ms=round((time.perf_counter() - start) * 1000, 2))


def _check_object_store(repo: Repository) -> CheckResult:
    start = time.perf_counter()
    ok = repo.store.objects_dir.is_dir()
    detail = "ok" if ok else f"{repo.store.objects_dir} does not exist"
    return CheckResult(ok=ok, detail=detail, ms=round((time.perf_counter() - start) * 1000, 2))


# The registry future checks join -- a Postgres projection, once one
# exists, adds a "projection" entry here (a `SELECT 1` plus a
# watermark-staleness check) without changing anything about how `ready()`
# aggregates or reports the ones that already exist.
_CHECKS = (("repository", _check_repository), ("object_store", _check_object_store))


@router.get("/ready", response_model=ReadyResponse)
def ready(repo: Repository = Depends(get_repo)) -> JSONResponse:
    """Everything this process needs to actually serve a request, not just answer one.

    200 ``{"status": "ready", ...}`` when every check passes, 503
    ``{"status": "degraded", ...}`` otherwise -- with per-check detail
    either way, so a human (or a dashboard) can see which dependency is the
    problem without reading logs.
    """
    checks = {name: check_fn(repo) for name, check_fn in _CHECKS}
    payload = ReadyResponse(status="ready" if all(c.ok for c in checks.values()) else "degraded", checks=checks)
    status_code = 200 if payload.status == "ready" else 503
    return JSONResponse(status_code=status_code, content=payload.model_dump())
