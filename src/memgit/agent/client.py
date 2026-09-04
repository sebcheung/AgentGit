"""The seam between the agent runtime and the Anthropic API.

``memgit.core`` is import-clean — no network, no ``anthropic`` — and this
module is where that discipline extends one layer up: it is the *only*
module in ``memgit.agent`` that imports the SDK, and it imports it lazily,
so ``import memgit.agent`` still works in an environment that never
installed the ``agent`` extra.

``LLMClient`` is deliberately a one-method Protocol rather than a mirror of
``anthropic.Anthropic``'s nested ``client.messages.create`` shape. Nesting
would force a test fake to be two cooperating objects and would let the
runtime reach for SDK-only parameters (``model``, ``thinking``, ``betas``)
that it has no business knowing about. Flattened to
``create_message(*, system, messages, tools) -> Message``, the runtime's
request-shaping ends exactly where model choice, thinking mode, effort, and
fallback configuration begin — all of which live here, in
:class:`AnthropicClient`, and nowhere else.

The return type is left as the SDK's own ``anthropic.types.Message``, not a
re-modelled dataclass. Re-defining ``TextBlock``/``ToolUseBlock``/``Message``
is exactly the anti-pattern the SDK's own guidance warns against, and a test
fake can build the real type directly via ``Message.model_construct(...)`` —
verified against the installed SDK — so nothing is lost by keeping it.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from anthropic.types import Message

__all__ = ["AgentError", "AnthropicClient", "LLMClient", "default_client"]

logger = logging.getLogger("memgit.agent")


class AgentError(Exception):
    """Raised for anything that goes wrong talking to the model.

    The one exception type ``memgit.agent`` exposes across the client
    boundary — ``runtime.py`` and ``cli.py`` catch this and never need to
    import ``anthropic`` themselves to know what to catch.
    """


@runtime_checkable
class LLMClient(Protocol):
    """What the agent runtime needs from a model: one request, one response.

    Both :class:`AnthropicClient` and a test's scripted fake satisfy this
    structurally — ``isinstance(x, LLMClient)`` works on either without
    either one inheriting from it.
    """

    def create_message(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Message:
        """Send one request, get back one SDK-native ``Message``."""
        ...


class AnthropicClient:
    """Adapts ``anthropic.Anthropic`` to :class:`LLMClient`.

    Every Anthropic-specific request parameter lives here: the model name,
    adaptive thinking, effort, and ``max_tokens``. ``budget_tokens`` is
    deliberately absent — it is rejected outright on ``claude-opus-5``,
    which is the one model this project's "Decisions locked in" table
    commits to.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        max_tokens: int = 16000,
        client: Any | None = None,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        if client is None:
            import anthropic

            # The SDK already retries 408/409/429/5xx and connection errors
            # with exponential backoff, honoring `retry-after` -- a second
            # retry loop wrapped around this would multiply attempts (N x M
            # instead of N + M) and ignore `retry-after` entirely, turning
            # one 429 into a small self-inflicted DDoS. `max_retries`/
            # `timeout` configure the SDK's own mechanism; `client.py` only
            # adds observability around it (see `create_message`'s logging).
            client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
        # Typed `Any`, not the SDK's `Anthropic`: narrowing it further would
        # make mypy check `create_message`'s call below against the SDK's
        # own precise overloads, defeating the whole point of the flattened
        # dict-shaped `LLMClient` Protocol this class exists to satisfy (see
        # the module docstring).
        self._client: Any = client
        self._model = model
        self._max_tokens = max_tokens

    def create_message(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Message:
        """See :meth:`LLMClient.create_message`."""
        import anthropic

        logger.info("agent.request", extra={"model": self._model, "max_tokens": self._max_tokens})
        start = time.perf_counter()
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                tools=tools,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
            )
        except anthropic.APIError as exc:
            logger.warning("agent.error", extra={"model": self._model, "error": str(exc)})
            raise AgentError(str(exc)) from exc
        except TypeError as exc:
            # The installed SDK validates credentials lazily, on the first
            # request, not at `Anthropic()` construction time -- a missing
            # key or token surfaces here as a bare TypeError from
            # `_validate_headers`, not `anthropic.APIError`. Wrapping it
            # keeps the "no API key configured" case a clean AgentError for
            # every caller (cli.py's `_fail`, the API layer's 502 handler)
            # instead of a raw traceback discovered only by actually trying
            # to replay without one.
            if "authentication method" not in str(exc).lower():
                raise
            logger.warning("agent.error", extra={"model": self._model, "error": str(exc)})
            raise AgentError(str(exc)) from exc

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "agent.response",
            extra={
                "model": self._model,
                "duration_ms": round(duration_ms, 2),
                "stop_reason": message.stop_reason,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        )
        return message


def default_client(**kwargs: Any) -> AnthropicClient:
    """The CLI's factory for a real client — the one place that can fail fast.

    Raises:
        AgentError: the ``anthropic`` package (the ``agent`` extra) is not
            installed.
    """
    try:
        return AnthropicClient(**kwargs)
    except ImportError as exc:
        raise AgentError(
            "the agent runtime needs the anthropic SDK: "
            'install it with `py -m uv sync --all-extras --system-certs` '
            "or `pip install memgit[agent]`"
        ) from exc
