"""Tests for ApiKeyMiddleware: a bare ASGI callable, exercised with no port and no SDK.

Manually assembles ASGI scope/receive/send rather than going through
uvicorn or a real HTTP client -- the middleware's whole contract is "given
this scope, call through or 401", which is testable as plain async
function calls.
"""

from __future__ import annotations

import json

import pytest

from memgit.mcp.transport import ApiKeyMiddleware


def _http_scope(headers: dict[str, str]) -> dict:
    return {
        "type": "http",
        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()],
    }


async def _noop_receive() -> dict:
    return {"type": "http.request"}


class _RecordingApp:
    def __init__(self) -> None:
        self.called = False

    async def __call__(self, scope, receive, send) -> None:
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


class _SendRecorder:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def __call__(self, message: dict) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int | None:
        start = next((m for m in self.messages if m["type"] == "http.response.start"), None)
        return start["status"] if start else None

    @property
    def body(self) -> bytes:
        return b"".join(m.get("body", b"") for m in self.messages if m["type"] == "http.response.body")


@pytest.mark.anyio
class TestApiKeyMiddleware:
    async def test_no_key_configured_is_a_passthrough(self) -> None:
        inner = _RecordingApp()
        middleware = ApiKeyMiddleware(inner, key=None)
        send = _SendRecorder()

        await middleware(_http_scope({}), _noop_receive, send)

        assert inner.called is True
        assert send.status == 200

    async def test_missing_key_is_401(self) -> None:
        inner = _RecordingApp()
        middleware = ApiKeyMiddleware(inner, key="secret")
        send = _SendRecorder()

        await middleware(_http_scope({}), _noop_receive, send)

        assert inner.called is False
        assert send.status == 401
        assert json.loads(send.body)["error"] == "unauthorized"

    async def test_wrong_key_is_401(self) -> None:
        inner = _RecordingApp()
        middleware = ApiKeyMiddleware(inner, key="secret")
        send = _SendRecorder()

        await middleware(_http_scope({"x-api-key": "wrong"}), _noop_receive, send)

        assert inner.called is False
        assert send.status == 401

    async def test_correct_key_via_x_api_key_passes_through(self) -> None:
        inner = _RecordingApp()
        middleware = ApiKeyMiddleware(inner, key="secret")
        send = _SendRecorder()

        await middleware(_http_scope({"x-api-key": "secret"}), _noop_receive, send)

        assert inner.called is True
        assert send.status == 200

    async def test_correct_key_via_bearer_token_passes_through(self) -> None:
        inner = _RecordingApp()
        middleware = ApiKeyMiddleware(inner, key="secret")
        send = _SendRecorder()

        await middleware(_http_scope({"authorization": "Bearer secret"}), _noop_receive, send)

        assert inner.called is True
        assert send.status == 200

    async def test_non_http_scopes_always_pass_through(self) -> None:
        # Lifespan events (startup/shutdown) carry no headers to check, and
        # gating them would break the ASGI lifespan protocol entirely.
        inner = _RecordingApp()
        middleware = ApiKeyMiddleware(inner, key="secret")
        send = _SendRecorder()

        await middleware({"type": "lifespan"}, _noop_receive, send)

        assert inner.called is True
