"""Request and response models for the REST layer.

Three tiers, and the tiering is the design position (see PLAN.md's
"Pydantic vs. ``to_dict``" decision row):

1. Every *request* gets a model — query params via ``Query(...)`` bounds,
   and :class:`ReplayRequest`. Untrusted input; validation is the job.
2. Every *response this API composes* gets a model — the types with no
   ``to_dict`` of their own (``Head``, ``RetrievalResult``, ``AblationResult``,
   ``ReplayOutcome``), plus ``Commit``, whose ``to_dict`` deliberately omits
   its own hash.
3. ``Diff`` and ``MemoryState`` pass through as the dicts core already
   emits (:class:`DiffResponse.diff`, :class:`StateResponse.state`), inside
   a thin typed envelope — ``Diff.to_dict()`` is already this project's
   machine-readable diff contract, tested against directly by
   ``tests/test_diff.py`` and embedding the cardinality map it was computed
   under, so a Pydantic mirror of it would be a second, untested definition
   of one schema.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "ErrorResponse",
    "HealthResponse",
    "HeadModel",
    "RepoResponse",
    "CommitNode",
    "CommitDetail",
    "LogResponse",
    "StateResponse",
    "DiffResponse",
    "RecalledFact",
    "RecallResponse",
    "ReplayRequest",
    "ReplayOutcomeModel",
    "RetrievalInfoModel",
    "ReplayResponse",
]


class ErrorResponse(BaseModel):
    """The one error envelope every mapped exception renders as.

    ``detail`` matches FastAPI's own field name so a handwritten error and
    a framework-native 422 both have something at ``.detail`` for the
    dashboard to show; ``error`` is a stable machine-readable slug, since an
    exception class name is not a contract.
    """

    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    version: str


class HeadModel(BaseModel):
    """``Head`` has no ``to_dict`` of its own — see ``core/refs.py``."""

    ref: str | None
    branch: str | None
    commit: str | None
    detached: bool


class RepoResponse(BaseModel):
    root: str
    default_branch: str
    format_version: int
    head: HeadModel
    branches: dict[str, str]
    unborn: bool


class CommitNode(BaseModel):
    """A commit, as returned over the wire.

    ``Commit.to_dict()`` deliberately omits its own hash — the hash *is*
    ``hash_payload(self.to_dict())``, so it can't be inside the payload it's
    computed from without being self-referential. A wire commit is a view,
    not the object, and a view needs an identity: the API attaches ``hash``
    itself rather than ever relying on ``Repository.log()``, which yields
    bare ``Commit`` objects with no hash at all. See PLAN.md's "Commit
    payloads over the wire" decision row.
    """

    hash: str
    tree: str
    parents: list[str]
    message: str
    summary: str
    author: str
    committed_at: str
    metadata: dict[str, Any] | None = None
    is_root: bool
    is_merge: bool


class CommitDetail(CommitNode):
    stat: dict[str, Any]


class LogResponse(BaseModel):
    rev: str
    resolved: str | None
    all: bool
    commits: list[CommitNode]


class StateResponse(BaseModel):
    """``state`` is ``MemoryState.to_dict()``, passed through unmodified."""

    rev: str
    resolved: str
    as_of: str | None
    state: dict[str, Any]


class DiffResponse(BaseModel):
    """``diff`` is ``Diff.to_dict()``, passed through unmodified.

    Already embeds the cardinality map it was computed under — see
    ``core/diff.py``'s ``Diff.to_dict``.
    """

    before: str | None
    after: str
    resolved_before: str | None
    resolved_after: str
    use_merge_base: bool
    diff: dict[str, Any]


class RecalledFact(BaseModel):
    """One fact plus its retrieval scoring breakdown.

    ``confidence`` here is the *decayed* confidence at the retrieval's
    ``as_of`` moment, not stored confidence — mirroring ``cli.py``'s own
    ``recall --json`` payload, which spreads ``Retrieved.confidence`` over
    ``Fact.to_dict()``'s stored value for the same reason: what a caller
    sees here is what actually drove ranking.
    """

    subject: str
    predicate: str
    object: str
    confidence: float
    asserted_at: str
    source: str | None = None
    source_text: str | None = None
    score: float
    similarity: float


class RecallResponse(BaseModel):
    query: str
    rev: str
    resolved: str
    commit: str | None
    candidates: int
    embedder: str
    as_of: str
    facts: list[RecalledFact]


class ReplayRequest(BaseModel):
    """No ``model`` field — the model is server-side (``serve-web --model``).

    This endpoint is unauthenticated in this slice, so a request-chosen
    model would be a request-chosen bill. See PLAN.md's "Replay's model and
    rate" decision row.
    """

    rev: str = "HEAD"
    subject: str
    predicate: str
    query: str
    fact_hash: str | None = None
    k: int = Field(8, ge=1, le=100)
    retrieve: bool | None = None
    as_of: str | None = None


class ReplayOutcomeModel(BaseModel):
    reply: str
    stop_reason: str


class RetrievalInfoModel(BaseModel):
    k: int
    candidates: int
    embedder: str
    as_of: str
    pinned: list[str]


class ReplayResponse(BaseModel):
    query: str
    key: list[str]
    fact_hash: str | None
    baseline: ReplayOutcomeModel
    ablated: ReplayOutcomeModel
    changed: bool
    retrieval: RetrievalInfoModel | None = None
