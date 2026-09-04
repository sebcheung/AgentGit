"""Slice 5 — the agent runtime.

Wires a real Claude agent to :mod:`memgit.core`: reads the memory state at
``HEAD``, exposes ``remember``/``forget`` as tools, and commits the result
after each turn. Deliberately outside ``memgit.core``, which stays import
clean — no network — as PLAN.md's repo conventions require; this package,
not core, is where ``anthropic`` gets imported, and only in
:mod:`memgit.agent.client`.
"""

from __future__ import annotations

from memgit.agent.client import AgentError
from memgit.agent.runtime import MemoryAgent, ToolCallRecord, TurnResult

__all__ = ["AgentError", "MemoryAgent", "ToolCallRecord", "TurnResult"]
