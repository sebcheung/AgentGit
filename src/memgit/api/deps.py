"""FastAPI dependencies: the injected repository, LLM client, and locks.

Both dependencies are plain functions, not classes — the same shape as
``cli.py``'s ``_repo()``/``_agent_client()`` seams, and what lets a test
override either one with ``app.dependency_overrides[...]`` instead of
monkeypatching a private module attribute.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from fastapi import Request

from memgit.api.errors import LLMUnavailableError
from memgit.core.repository import Repository

if TYPE_CHECKING:
    from memgit.agent.client import LLMClient

__all__ = ["get_llm_client", "get_repo", "replay_semaphore", "retrieval_lock"]

# `VectorIndex` (retrieval/index.py) is single-writer by construction: a
# cache-miss write appends to `vectors.pack`, then read-modify-writes
# `index.json` through a *fixed* `index.json.lock` path. Every caller before
# this API layer was serial -- one CLI process, one stdio MCP loop -- so
# nothing before now could race on it. FastAPI runs sync routes in its
# threadpool, making this the project's first genuinely concurrent caller;
# two cold recalls racing can lose offset records to a last-writer-wins
# rewrite, or collide on the lock filename. Serializing retrieval here costs
# nothing for a single-user dashboard, and a warm cache (`memgit embed`)
# makes the locked section a pure read. See PLAN.md's "Vector-cache writes
# under concurrency" decision row.
retrieval_lock = threading.Lock()

# At most one replay in flight per process -- see replay.py. Two paid model
# calls behind an unauthenticated button a demo audience can spam.
replay_semaphore = threading.Semaphore(1)


def get_repo(request: Request) -> Repository:
    """The repository this app was built with — see ``app.create_app``."""
    return request.app.state.repo


def get_llm_client(request: Request) -> LLMClient:
    """The client ``POST /api/replay`` replays with, built fresh per request.

    Imports ``memgit.agent.client`` lazily so this module -- and therefore
    ``memgit.api.app`` -- never requires the ``agent`` extra merely to
    import. Constructing the client *here*, rather than inside the route,
    is what turns "the extra isn't installed" into a 503 with an install
    hint instead of a 500 with a traceback: the failure happens at a layer
    ``errors.py`` already maps.

    A plain function, not a class, so a test replaces it wholesale via
    ``app.dependency_overrides[get_llm_client] = lambda: scripted_client`` —
    the ASGI equivalent of ``cli.py``'s ``_agent_client`` monkeypatch seam.
    """
    from memgit.agent.client import AgentError, default_client

    try:
        return default_client(model=request.app.state.model)
    except AgentError as exc:
        raise LLMUnavailableError(str(exc)) from exc
