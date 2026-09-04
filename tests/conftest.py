"""Shared fixtures for the agent test suite: a scripted fake LLM client.

Lives in conftest.py, not a source module, since nothing under
``src/memgit`` needs it — this is purely test infrastructure, matching
PLAN.md's one-test-module-per-core-module convention by not becoming a
second module that convention would have to account for.

The fake builds real ``anthropic.types`` objects (``Message``,
``TextBlock``, ``ToolUseBlock``) via ``model_construct``, verified against
the installed SDK, rather than re-modelling them — the agent runtime never
learns it isn't talking to the real API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from memgit.core.repository import Repository


def text_message(text: str, *, stop_reason: str = "end_turn") -> Message:
    """An assistant turn with just a text reply — no tool calls."""
    return Message.model_construct(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=[TextBlock(type="text", text=text)],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_details=None,
    )


def tool_use_message(*calls: tuple[str, str, dict[str, Any]], text: str = "") -> Message:
    """An assistant turn asking for one or more tool calls.

    Args:
        calls: ``(tool_use_id, tool_name, input_dict)`` tuples — all in one
            assistant message, so tests can exercise parallel tool use.
    """
    content: list[Any] = []
    if text:
        content.append(TextBlock(type="text", text=text))
    for call_id, name, tool_input in calls:
        content.append(ToolUseBlock(type="tool_use", id=call_id, name=name, input=tool_input))
    return Message.model_construct(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=content,
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
        stop_details=None,
    )


def refusal_message(*, category: str = "other", explanation: str = "declined") -> Message:
    class _StopDetails:
        def __init__(self, category: str, explanation: str) -> None:
            self.category = category
            self.explanation = explanation

    return Message.model_construct(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=[],
        stop_reason="refusal",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=0),
        stop_details=_StopDetails(category, explanation),
    )


@dataclass
class ScriptedClient:
    """A fake :class:`~memgit.agent.client.LLMClient`: replays canned responses.

    One entry in ``responses`` is consumed per ``create_message`` call; every
    request it received is recorded in ``requests`` so a test can assert on
    exactly what the runtime sent — the shape of ``system``/``messages`` and
    whether tool results were batched into one user message.
    """

    responses: list[Message]
    requests: list[dict[str, Any]] = field(default_factory=list)

    def create_message(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Message:
        # Snapshot the list — the runtime keeps mutating its own `messages`
        # list in place across loop iterations, so a bare reference here
        # would make every recorded request look like the last one.
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.responses:
            raise AssertionError("ScriptedClient ran out of responses")
        return self.responses.pop(0)


@pytest.fixture
def agent_repo(tmp_path) -> Repository:
    return Repository.init(tmp_path)


@pytest.fixture
def anyio_backend() -> str:
    """Pins the ``mcp.Client`` in-memory transport's tests to asyncio.

    Required by ``pytest.mark.anyio`` tests (``test_mcp_server.py``); harmless
    for every other test, which never requests this fixture.
    """
    return "asyncio"
