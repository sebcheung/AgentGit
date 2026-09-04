"""Slice 6 — the replay / ablation engine.

Causal attribution for a memory: replay the same query against a state and
against that same state with one belief removed, and report whether the
reply changed. See :mod:`memgit.replay.engine` for the implementation and
its "never commits" invariant.
"""

from __future__ import annotations

from memgit.replay.engine import AblationResult, ReplayOutcome, ablate_and_replay, replay_query

__all__ = ["AblationResult", "ReplayOutcome", "ablate_and_replay", "replay_query"]
