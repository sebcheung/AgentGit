"""Per-request logging: propagate ``X-Request-ID`` and log one line per request.

Registered from ``api/app.py`` via ``app.middleware("http")(...)`` rather
than a decorator in this module, so this stays importable (and testable) on
its own without the app factory. Sets
:data:`~memgit.logging_config.request_id_var` for the duration of the
request, which is what lets any log line emitted deeper in the stack --
``deps.get_llm_client``, ``replay.py`` -- carry the same id without it being
threaded through every function signature in between.
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from memgit.logging_config import request_id_var

__all__ = ["request_id_middleware"]

logger = logging.getLogger("memgit.api")


async def request_id_middleware(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Echo (or mint) ``X-Request-ID``, then log method/path/status/duration."""
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = request_id_var.set(request_id)
    try:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "http.request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response
    finally:
        request_id_var.reset(token)
