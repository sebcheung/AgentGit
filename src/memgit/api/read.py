"""Read-only routes: repo, log, commits, state, diff, recall.

Every route is a plain, synchronous ``def`` — each does blocking filesystem
IO, and FastAPI runs sync routes in its threadpool; an ``async def`` here
would block the event loop for no benefit, since nothing in this module
ever awaits anything.

Every route that takes a revision resolves it exactly once via
``repo.resolve`` and echoes both the raw input and the resolved hash back —
the same auditability instinct as a diff embedding the cardinality map it
used: two requests seconds apart stay comparable even if ``HEAD`` moved
between them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from memgit import __version__
from memgit.api.deps import get_repo, retrieval_lock
from memgit.api.models import (
    CommitDetail,
    CommitNode,
    DiffResponse,
    HeadModel,
    HealthResponse,
    LogResponse,
    RecalledFact,
    RecallResponse,
    RepoResponse,
    StateResponse,
)
from memgit.core.commit import Commit
from memgit.core.graph import walk
from memgit.core.repository import Repository

router = APIRouter()


def _parse_as_of(value: str) -> datetime:
    """Parse an ``as_of`` query param.

    A genuine client-input error, so this raises locally (400) rather than
    through the global handlers, matching the diff route's own local
    ``ValueError`` guard below.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"as_of must be ISO-8601, got {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _commit_node(commit_hash: str, commit: Commit) -> CommitNode:
    """Attaches the hash ``Commit.to_dict()`` deliberately omits.

    See PLAN.md's "Commit payloads over the wire" decision row: the hash
    can't live inside the payload it's computed from without being
    self-referential, so a wire commit is a view that names its own
    identity rather than the bare object.
    """
    return CommitNode(
        hash=commit_hash,
        tree=commit.tree,
        parents=list(commit.parents),
        message=commit.message,
        summary=commit.summary,
        author=commit.author,
        committed_at=commit.committed_at,
        metadata=dict(commit.metadata) if commit.metadata is not None else None,
        is_root=commit.is_root,
        is_merge=commit.is_merge,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness only — no repository probe. See PLAN.md's ``/health`` row."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/repo", response_model=RepoResponse)
def get_repository(repo: Repository = Depends(get_repo)) -> RepoResponse:
    """The repository's root, config, HEAD, and branch tips."""
    head = repo.head()
    config = repo.config()
    return RepoResponse(
        root=str(repo.root),
        default_branch=config.get("default_branch", "main"),
        format_version=config.get("format_version", Repository.FORMAT_VERSION),
        head=HeadModel(ref=head.ref, branch=head.branch, commit=head.commit, detached=head.is_detached),
        branches=repo.branches(),
        unborn=head.commit is None,
    )


@router.get("/log", response_model=LogResponse)
def get_log(
    rev: Annotated[str, Query()] = "HEAD",
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    all_branches: Annotated[bool, Query(alias="all")] = False,
    repo: Repository = Depends(get_repo),
) -> LogResponse:
    """Walk history with ``core.graph.walk``, never ``Repository.log()``.

    ``Repository.log()`` yields bare ``Commit`` objects with no hash attached
    (see ``_commit_node``). ``all=true`` walks every branch tip at once;
    ``walk`` already de-duplicates a diamond, so no manual merge is needed
    here.
    """
    if all_branches:
        starts = list(repo.branches().values())
        resolved: str | None = None
    elif rev == "HEAD" and repo.head_commit() is None:
        # An unborn repository (freshly `init`ed, no commits yet) is a
        # normal state, not an error -- see Repository.head()'s own
        # docstring. Resolving "HEAD" here would otherwise raise
        # RevisionNotFoundError for a repo that simply hasn't committed
        # anything, which is not what a 404 should mean.
        return LogResponse(rev=rev, resolved=None, all=False, commits=[])
    else:
        resolved = repo.resolve(rev)
        starts = [resolved]

    commits = [_commit_node(h, c) for h, c in walk(starts, repo.read_commit, limit=limit)]
    return LogResponse(rev=rev, resolved=resolved, all=all_branches, commits=commits)


@router.get("/commits/{rev:path}", response_model=CommitDetail)
def get_commit(rev: str, repo: Repository = Depends(get_repo)) -> CommitDetail:
    """One commit, plus its diff stat against its first parent."""
    resolved = repo.resolve(rev)
    commit = repo.read_commit(resolved)
    stat = repo.diff(after=resolved).stat().to_dict()
    node = _commit_node(resolved, commit)
    return CommitDetail(**node.model_dump(), stat=stat)


@router.get("/state", response_model=StateResponse)
def get_state(
    rev: Annotated[str, Query()] = "HEAD",
    subject: Annotated[str | None, Query()] = None,
    predicate: Annotated[str | None, Query()] = None,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    as_of: Annotated[str | None, Query()] = None,
    repo: Repository = Depends(get_repo),
) -> StateResponse:
    """Mirrors ``cli.py``'s ``state --json`` exactly, decay lens included.

    ``as_of`` never touches stored confidence, only adds a
    ``decayed_confidence`` field per fact.
    """
    resolved = repo.resolve(rev)
    memory = repo.state(resolved)

    if subject is not None or predicate is not None or min_confidence > 0.0:
        memory = memory.filter(
            min_confidence=min_confidence,
            subjects={subject} if subject is not None else None,
            predicates={predicate} if predicate is not None else None,
        )

    payload: dict[str, Any] = memory.to_dict()
    as_of_dt = _parse_as_of(as_of) if as_of is not None else None
    if as_of_dt is not None:
        policy = repo.decay()
        payload["as_of"] = as_of_dt.isoformat()
        for fact_payload, fact in zip(payload["facts"], memory.facts, strict=False):
            fact_payload["decayed_confidence"] = policy.decayed(fact, as_of=as_of_dt)

    return StateResponse(
        rev=rev, resolved=resolved, as_of=as_of_dt.isoformat() if as_of_dt else None, state=payload
    )


@router.get("/diff", response_model=DiffResponse)
def get_diff(
    before: Annotated[str | None, Query()] = None,
    after: Annotated[str, Query()] = "HEAD",
    include_unchanged: Annotated[bool, Query()] = False,
    use_merge_base: Annotated[bool, Query()] = False,
    repo: Repository = Depends(get_repo),
) -> DiffResponse:
    """``Diff.to_dict()`` passed through unmodified.

    See ``models.py``'s ``DiffResponse`` docstring for why. The one local
    ``ValueError`` guard below is deliberately *not* a global handler: see
    ``errors.py``'s module docstring for the argument.
    """
    try:
        result = repo.diff(
            before, after, include_unchanged=include_unchanged, use_merge_base=use_merge_base
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_after = repo.resolve(after)
    # Mirrors cli.py's own diff --json, which echoes the raw `before` input
    # (often None, for the implicit-first-parent default) rather than
    # re-deriving what it resolved to.
    resolved_before = repo.resolve(before) if before is not None else None

    return DiffResponse(
        before=before,
        after=after,
        resolved_before=resolved_before,
        resolved_after=resolved_after,
        use_merge_base=use_merge_base,
        diff=result.to_dict(),
    )


@router.get("/recall", response_model=RecallResponse)
def recall(
    query: Annotated[str, Query(min_length=1)],
    rev: Annotated[str, Query()] = "HEAD",
    k: Annotated[int, Query(ge=1, le=100)] = 8,
    subject: Annotated[str | None, Query()] = None,
    min_score: Annotated[float, Query(ge=0.0)] = 0.0,
    as_of: Annotated[str | None, Query()] = None,
    decay: Annotated[bool, Query()] = True,
    repo: Repository = Depends(get_repo),
) -> RecallResponse:
    """The same retrieval code path ``memgit recall`` and an agent turn use.

    Uses :class:`~memgit.retrieval.rank.Retriever`, scoped to ``rev`` by
    construction rather than by a filter that could leak a later commit or
    another branch. See ``deps.retrieval_lock`` for why this acquires a
    lock: a cache miss here writes to the vector index, and this API is the
    project's first concurrent caller of it.
    """
    resolved = repo.resolve(rev)
    state = repo.state(resolved)
    weight = 0.0 if not decay else None
    moment = _parse_as_of(as_of) if as_of is not None else datetime.now(UTC)

    with retrieval_lock:
        retriever = repo.retriever(confidence_weight=weight)
        result = retriever.retrieve(
            state,
            query,
            k=k,
            as_of=moment,
            subjects={subject} if subject is not None else None,
            min_score=min_score,
        )

    return RecallResponse(
        query=result.query,
        rev=rev,
        resolved=resolved,
        commit=result.commit,
        candidates=result.candidates,
        embedder=result.embedder,
        as_of=result.as_of,
        facts=[
            RecalledFact(
                subject=r.fact.subject,
                predicate=r.fact.predicate,
                object=r.fact.object,
                confidence=r.confidence,
                asserted_at=r.fact.asserted_at,
                source=r.fact.source,
                source_text=r.fact.source_text,
                score=r.score,
                similarity=r.similarity,
            )
            for r in result.facts
        ],
    )
