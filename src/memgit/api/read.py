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

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from memgit import __version__
from memgit.api.deps import get_repo
from memgit.api.models import CommitNode, HeadModel, HealthResponse, LogResponse, RepoResponse
from memgit.core.commit import Commit
from memgit.core.graph import walk
from memgit.core.repository import Repository

router = APIRouter()


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
    """Walk history with ``core.graph.walk`` — never ``Repository.log()``,
    which yields bare ``Commit`` objects with no hash attached (see
    ``_commit_node``). ``all=true`` walks every branch tip at once; ``walk``
    already de-duplicates a diamond, so no manual merge is needed here.
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
