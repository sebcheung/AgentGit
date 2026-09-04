"""``POST /api/replay`` — the one write-shaped exception to a read-only API.

Structurally cannot write: :func:`~memgit.replay.engine.ablate_and_replay`
never calls ``Repository.commit`` (see ``replay/engine.py``'s module
docstring), so exposing it here does not reopen the "no unauthenticated
writes" decision the rest of this package holds to — see PLAN.md's "REST
surface" decision row.

The engine is imported lazily, inside the route, exactly as ``cli.py``'s
``replay_cmd`` imports it — so this module, and therefore ``memgit.api.app``,
never requires the ``agent`` extra merely to import.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException

from memgit.api.deps import get_llm_client, get_repo, replay_semaphore
from memgit.api.models import ReplayOutcomeModel, ReplayRequest, ReplayResponse, RetrievalInfoModel
from memgit.core.repository import Repository

if TYPE_CHECKING:
    from memgit.agent.client import LLMClient

router = APIRouter()


def _parse_as_of(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"as_of must be ISO-8601, got {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@router.post("/replay", response_model=ReplayResponse)
def replay(
    request: ReplayRequest,
    repo: Repository = Depends(get_repo),
    client: LLMClient = Depends(get_llm_client),
) -> ReplayResponse:
    """Ablate ``(subject, predicate)`` at ``rev`` and replay ``query`` with and without it.

    Never commits. At most one replay runs per process at a time — two paid model calls
    behind an unauthenticated button a demo audience can spam; see
    PLAN.md's "Replay's model and rate" decision row.
    """
    from memgit.replay.engine import ablate_and_replay

    if not replay_semaphore.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="a replay is already running; try again shortly")
    try:
        moment = _parse_as_of(request.as_of) if request.as_of is not None else None
        result = ablate_and_replay(
            repo,
            request.rev,
            (request.subject, request.predicate),
            request.query,
            client=client,
            fact_hash=request.fact_hash,
            retrieve=request.retrieve,
            k=request.k,
            as_of=moment,
        )
    finally:
        replay_semaphore.release()

    retrieval = None
    if result.retrieved is not None:
        retrieval = RetrievalInfoModel(
            k=len(result.retrieved.facts),
            candidates=result.retrieved.candidates,
            embedder=result.retrieved.embedder,
            as_of=result.retrieved.as_of,
            pinned=[r.fact.hash for r in result.retrieved.facts],
        )

    return ReplayResponse(
        query=result.query,
        key=list(result.key),
        fact_hash=result.fact_hash,
        baseline=ReplayOutcomeModel(
            reply=result.baseline.reply, stop_reason=result.baseline.stop_reason
        ),
        ablated=ReplayOutcomeModel(
            reply=result.ablated.reply, stop_reason=result.ablated.stop_reason
        ),
        changed=result.changed,
        retrieval=retrieval,
    )
