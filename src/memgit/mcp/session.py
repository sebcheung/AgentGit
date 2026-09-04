"""Mapping one MCP connection onto one :class:`~memgit.core.staging.StagingArea`.

The MCP Python SDK's ``Context`` exposes a ``session`` object and per-request
headers, but no ready-made "session id" property that means the same thing
across every transport — stdio and the in-memory test transport have no
concept of a session id at all (one connection is one process), while
Streamable HTTP mints a real ``Mcp-Session-Id`` and puts it on every request.
:func:`resolve_session_id` is the one place that reconciles the two: prefer
the transport's own id when there is one, otherwise fall back to a single id
generated once per server process — correct for stdio, since there is only
ever one client per process anyway.

Everything else in this module is a thin, typed wrapper around
:class:`~memgit.core.repository.Repository`'s own staging methods
(``open_staging``, ``stage``, ``seal_staging``) plus the write folds in
``agent.tools`` (``apply_remember``, ``apply_forget``) — the "nearly the same
code as slice 4" ``tools.py``'s docstring promises. It retries exactly once
on a losing compare-and-swap, per
:meth:`~memgit.core.repository.Repository.stage`'s own contract: a fold is a
pure function of ``(state, call)``, so redoing it against the freshly-read
current state is always correct, and a second consecutive loss almost
certainly means real, sustained contention rather than one unlucky
interleaving — so it is surfaced rather than retried forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memgit.agent.tools import (
    ForgetCall,
    RememberCall,
    apply_forget,
    apply_remember,
)
from memgit.core.repository import Repository
from memgit.core.staging import StagingConflictError
from memgit.core.state import MemoryState

__all__ = ["resolve_session_id", "staged_state", "remember", "forget", "seal"]


def resolve_session_id(ctx: Any, *, process_session_id: str) -> str:
    """The session id to stage writes under, for whichever transport ``ctx`` came from.

    ``ctx.headers`` is ``None`` on stdio and the in-memory test transport
    (there is no HTTP request to carry headers on); Streamable HTTP sets
    ``Mcp-Session-Id`` on every request once a session exists. Preferring the
    real header when present is what keeps two concurrent HTTP clients from
    ever sharing one staging area.
    """
    headers = getattr(ctx, "headers", None) or {}
    return headers.get("mcp-session-id", process_session_id)


def staged_state(repo: Repository, session_id: str) -> tuple[MemoryState, str]:
    """The session's current staged facts, and the tree hash they were read at.

    The tree hash is what a subsequent :meth:`remember`/:meth:`forget` call
    must pass back as ``based_on`` — see
    :meth:`~memgit.core.repository.Repository.stage` for why that has to be
    the value actually seen, not re-read.
    """
    area = repo.open_staging(session_id)
    state = MemoryState.from_tree(repo.read_tree(area.tree), repo.read_fact)
    return state, area.tree


def remember(repo: Repository, session_id: str, call: RememberCall) -> MemoryState:
    """Fold one ``remember`` onto ``session_id``'s staged state; return the result."""
    cardinality = repo.cardinality()
    state, based_on = staged_state(repo, session_id)
    folded = apply_remember(state, call, cardinality)
    try:
        area = repo.stage(session_id, folded.facts, based_on=based_on)
    except StagingConflictError:
        state, based_on = staged_state(repo, session_id)
        folded = apply_remember(state, call, cardinality)
        area = repo.stage(session_id, folded.facts, based_on=based_on)
    return MemoryState.from_tree(repo.read_tree(area.tree), repo.read_fact)


def forget(repo: Repository, session_id: str, call: ForgetCall) -> MemoryState:
    """Fold one ``forget`` onto ``session_id``'s staged state; return the result."""
    state, based_on = staged_state(repo, session_id)
    folded = apply_forget(state, call)
    try:
        area = repo.stage(session_id, folded.facts, based_on=based_on)
    except StagingConflictError:
        state, based_on = staged_state(repo, session_id)
        folded = apply_forget(state, call)
        area = repo.stage(session_id, folded.facts, based_on=based_on)
    return MemoryState.from_tree(repo.read_tree(area.tree), repo.read_fact)


def seal(
    repo: Repository, session_id: str, message: str, *, author: str
) -> str | None:
    """Seal ``session_id``'s staged writes into a commit; see
    :meth:`~memgit.core.repository.Repository.seal_staging` for the overlay
    semantics under a moved branch tip.
    """
    return repo.seal_staging(session_id, message, author=author)
