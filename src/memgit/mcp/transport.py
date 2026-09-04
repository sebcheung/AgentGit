"""A pure-ASGI API-key gate for the MCP server's streamable-HTTP transport.

Framework-agnostic on purpose: :func:`~memgit.mcp.server.build_server`'s
``streamable_http_app()`` returns a Starlette app, but this module imports
neither Starlette nor FastAPI — a bare ASGI 3 callable wraps any of them,
and it's the shape ``mcp/server.py`` already avoids adding a ``web``-extra
dependency for.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from memgit.keyauth import check, extract_presented

__all__ = ["ApiKeyMiddleware"]

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_UNAUTHORIZED_BODY = b'{"error": "unauthorized", "detail": "missing or invalid API key"}'


class ApiKeyMiddleware:
    """401s an HTTP request that doesn't present ``key`` — a no-op wrapper when ``key`` is ``None``."""

    def __init__(self, app: ASGIApp, *, key: str | None) -> None:
        self._app = app
        self._key = key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """The ASGI 3 application callable."""
        if self._key is None or scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = {name.decode("latin-1"): value.decode("latin-1") for name, value in scope.get("headers", [])}
        if check(extract_presented(headers), self._key):
            await self._app(scope, receive, send)
            return

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})
