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

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from anthropic.types import Message

__all__ = ["LLMClient", "AnthropicClient", "AgentError", "default_client"]


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
    ) -> "Message": ...


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
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def create_message(
        self,
        *,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> "Message":
        import anthropic

        try:
            return self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                tools=tools,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
            )
        except anthropic.APIError as exc:
            raise AgentError(str(exc)) from exc


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
