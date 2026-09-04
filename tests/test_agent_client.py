"""Tests for ``AnthropicClient``'s exception wrapping.

The installed SDK validates credentials lazily, on the first request, not
at ``Anthropic()`` construction time -- a missing key or token surfaces as
a bare ``TypeError`` from ``_validate_headers``, not ``anthropic.APIError``,
so ``AnthropicClient.create_message`` has to catch it specifically to keep
"no API key configured" a clean :class:`AgentError` for every caller
(``cli.py``'s ``_fail``, the API layer's 502 handler) instead of a raw
traceback discovered only by actually trying to replay without one.
"""

from __future__ import annotations

import logging

import anthropic
import httpx
import pytest
from conftest import text_message

from memgit.agent.client import AgentError, AnthropicClient


class _FakeMessages:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def create(self, **kwargs):
        raise self._exc


class _FakeAnthropicClient:
    def __init__(self, exc: Exception) -> None:
        self.messages = _FakeMessages(exc)


def _make_client(exc: Exception) -> AnthropicClient:
    return AnthropicClient(client=_FakeAnthropicClient(exc))


def test_missing_credentials_is_wrapped_as_agent_error() -> None:
    exc = TypeError(
        '"Could not resolve authentication method. Expected one of api_key, '
        'auth_token, or credentials to be set. Or for one of the `X-Api-Key` '
        'or `Authorization` headers to be explicitly omitted"'
    )
    client = _make_client(exc)

    with pytest.raises(AgentError, match="authentication method"):
        client.create_message(system=[], messages=[], tools=[])


def test_unrelated_type_error_is_not_swallowed() -> None:
    exc = TypeError("some_kwarg() got an unexpected keyword argument 'foo'")
    client = _make_client(exc)

    with pytest.raises(TypeError, match="unexpected keyword"):
        client.create_message(system=[], messages=[], tools=[])


class _FakeSucceedingMessages:
    def __init__(self, message) -> None:
        self._message = message

    def create(self, **kwargs):
        return self._message


class _FakeSucceedingAnthropicClient:
    def __init__(self, message) -> None:
        self.messages = _FakeSucceedingMessages(message)


class TestLogging:
    """One log line per call: a request before, and a response or error after."""

    def test_logs_request_and_response_on_success(self, caplog: pytest.LogCaptureFixture) -> None:
        message = text_message("hi", stop_reason="end_turn")
        client = AnthropicClient(client=_FakeSucceedingAnthropicClient(message))

        with caplog.at_level(logging.INFO, logger="memgit.agent"):
            client.create_message(system=[], messages=[], tools=[])

        names = [r.message for r in caplog.records]
        assert "agent.request" in names
        assert "agent.response" in names
        response_record = next(r for r in caplog.records if r.message == "agent.response")
        assert response_record.stop_reason == "end_turn"
        assert response_record.input_tokens == 1
        assert response_record.output_tokens == 1
        assert response_record.duration_ms >= 0

    def test_logs_error_on_api_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        exc = anthropic.APIConnectionError(request=request)
        client = _make_client(exc)

        with caplog.at_level(logging.INFO, logger="memgit.agent"), pytest.raises(AgentError):
            client.create_message(system=[], messages=[], tools=[])

        names = [r.message for r in caplog.records]
        assert "agent.request" in names
        assert "agent.error" in names
