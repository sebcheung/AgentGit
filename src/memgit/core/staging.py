"""Staging areas — durable, hashed write buffers that cross process boundaries.

``Repository.commit`` takes a whole fact set, never a delta, because there is
no working tree and no index (see PLAN.md's "no index" decision): "an index
is mutable, uncommitted, unhashed state — exactly the one thing this
project's premise says should always be diffable and reversible." That row
also names its own escape hatch for the one case it doesn't cover: "When
slice 5 needs staging across two process invocations, the answer is a ref
holding a tree hash, not git's binary index." Slice 8's MCP server is that
case — an external host calls ``remember``/``forget`` as isolated tool calls
with no observable turn boundary, so a memory state has to persist somewhere
between them.

A :class:`StagingArea` is exactly that escape hatch: two plain refs under
``refs/memgit/staging/<key>/`` — ``tree`` (the staged ``Tree``'s hash) and
``base`` (the commit the target branch was at when staging began). Neither
adjective in "mutable, uncommitted, unhashed" fully applies here — mutable
and uncommitted, yes, unavoidably; but **hashed**, always: every intermediate
staged tree is a real, content-addressed object with every fact already
written, exactly like a commit's tree, just without a ``Commit`` wrapped
around it yet. That is what keeps it diffable (``Repository.diff(branch,
"refs/memgit/staging/<key>/tree")`` works with no new code — the explicit
``before`` branch of ``diff`` never calls ``read_commit``) and reversible
(every movement is reflogged by the same :class:`~memgit.core.refs.RefStore`
hook every branch ref gets).

This module owns none of the *fold* logic — turning a decoded ``remember``/
``forget`` call into a new fact set is ``agent.tools.apply_remember`` /
``apply_forget``'s job, deliberately, so that this module (like ``tools.py``)
has no dependency on an LLM client or the MCP package and can be exercised
entirely from the CLI or a test. A caller reads :meth:`Repository.staging`
(or its 404-shaped ``None``), folds one more call onto its facts itself, and
writes the result back with :meth:`Repository.stage`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from memgit.core.fact import Fact, FactKey
from memgit.core.tree import EMPTY_TREE_HASH, Tree

if TYPE_CHECKING:
    from memgit.core.repository import Repository

__all__ = ["StagingArea", "session_key", "StagingConflictError"]

STAGING_PREFIX = "refs/memgit/staging"


class StagingConflictError(Exception):
    """Raised when a staging write loses a concurrent compare-and-swap.

    The caller (``mcp/session.py``) is the one place that knows how to redo
    the fold that produced the losing write — this module only knows how to
    detect the race, via :class:`~memgit.core.refs.RefStore`'s existing
    compare-and-swap, not how to resolve it.
    """


def session_key(session_id: str) -> str:
    """Map an external session id to the filesystem-safe key its staging
    area lives under.

    Never use a raw session id as a ref path component. A ref name only
    forbids ``../`` and a handful of other shapes (see
    ``refs.py``'s ``_validate_ref_name``) — it does not forbid ``/`` — so an
    attacker-influenced id (plausible for the tunneled HTTP transport
    PLAN.md's demo story leans on) could otherwise steer a ref write outside
    ``refs/memgit/staging/``. Hashing is deterministic — the same session id
    always maps to the same key, which is what lets a human echo a tool
    result's key straight into ``memgit staging show <key>`` — and every
    output already satisfies ``_ALLOWED_CHARS`` by construction, closing off
    the whole injection class rather than trying to enumerate it.
    """
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def _tree_ref(key: str) -> str:
    return f"{STAGING_PREFIX}/{key}/tree"


def _base_ref(key: str) -> str:
    return f"{STAGING_PREFIX}/{key}/base"


@dataclass(frozen=True, slots=True)
class StagingArea:
    """One session's in-progress, uncommitted memory state.

    Attributes:
        key: The hashed session key (see :func:`session_key`).
        tree: The staged ``Tree``'s hash — always a real object hash, even
            for an empty stage (``EMPTY_TREE_HASH``, which is never written
            to the store but is always readable, matching
            :meth:`~memgit.core.repository.Repository.read_tree`).
        base: The commit the target branch was at when staging began, or
            ``None`` if staging began on an unborn branch.
    """

    key: str
    tree: str
    base: str | None


def read_staging(repo: "Repository", key: str) -> StagingArea | None:
    """The staging area at ``key``, or ``None`` if nothing is staged there."""
    tree_hash = repo.refs.read_ref(_tree_ref(key))
    if tree_hash is None:
        return None
    base = repo.refs.read_ref(_base_ref(key))
    return StagingArea(key=key, tree=tree_hash, base=base)


def list_staging(repo: "Repository") -> tuple[StagingArea, ...]:
    """Every open staging area, sorted by key.

    A tree ref with no matching ``base`` ref is a staging area opened on an
    unborn branch — a normal state, not a sign of corruption (see
    :meth:`open_staging`).
    """
    trees = repo.refs.list_refs(f"{STAGING_PREFIX}/")
    areas = []
    for ref_name, tree_hash in sorted(trees.items()):
        if not ref_name.endswith("/tree"):
            continue
        key = ref_name.removeprefix(f"{STAGING_PREFIX}/").removesuffix("/tree")
        base = repo.refs.read_ref(_base_ref(key))
        areas.append(StagingArea(key=key, tree=tree_hash, base=base))
    return tuple(areas)


def open_staging(repo: "Repository", key: str, *, branch_ref: str) -> StagingArea:
    """Return ``key``'s staging area, creating it against ``branch_ref``'s
    current tip if it does not exist yet.

    A freshly opened area's tree starts identical to the branch tip's own
    tree — staging a memory state is not staging a *change*, it is staging a
    *copy that will be changed*, the same "``working`` always starts as the
    full ``before`` state" invariant slice 5's runtime already enforces for
    an agent turn.
    """
    existing = read_staging(repo, key)
    if existing is not None:
        return existing

    tip = repo.refs.read_ref(branch_ref)
    if tip is None:
        base_hash = None
        tree_hash = EMPTY_TREE_HASH
    else:
        base_hash = tip
        tree_hash = repo.read_commit(tip).tree

    repo.refs.write_ref(_tree_ref(key), tree_hash, expect=None, op="stage", reason="open")
    if base_hash is not None:
        repo.refs.write_ref(_base_ref(key), base_hash, expect=None, op="stage", reason="open")
    return StagingArea(key=key, tree=tree_hash, base=base_hash)


def stage(repo: "Repository", key: str, facts: list[Fact], *, based_on: str) -> StagingArea:
    """Replace ``key``'s staged fact set with ``facts`` (the whole set, not a
    delta — matching :meth:`~memgit.core.repository.Repository.commit`'s own
    contract).

    Args:
        based_on: The staged tree hash ``facts`` were folded against —
            typically a previous :func:`open_staging` or :func:`stage`
            call's ``.tree``. This is compared against, **never re-read**,
            for the compare-and-swap: the whole point of a CAS is to check
            the value the caller actually saw its fold against, not whatever
            happens to be live the instant this function runs. Re-reading
            "current" internally here would silently absorb a concurrent
            writer's update instead of detecting it — the caller's ``facts``
            were folded against stale data, but the swap would go through
            because it happens to match itself.

    Raises:
        StagingConflictError: another writer moved this staging area past
            ``based_on`` first — the caller re-reads (:func:`read_staging`),
            re-applies its fold on top of the new value, and retries once.
    """
    new_tree_hash = repo.write_tree(facts)
    try:
        repo.refs.write_ref(
            _tree_ref(key), new_tree_hash, expect=based_on, op="stage", reason="remember/forget"
        )
    except ValueError as exc:
        raise StagingConflictError(str(exc)) from exc
    base = repo.refs.read_ref(_base_ref(key))
    return StagingArea(key=key, tree=new_tree_hash, base=base)


def drop_staging(repo: "Repository", key: str) -> None:
    """Discard ``key``'s staging area. A no-op if nothing was staged there.

    Deletion is logged by the same :class:`~memgit.core.reflog.RefLogger`
    hook every ref delete gets, so the staging area's full history — every
    tree it passed through, including its last one — survives in
    ``.memgit/logs/refs/memgit/staging/<key>/tree`` even after the refs
    themselves are gone.
    """
    repo.refs.delete_ref(_tree_ref(key), op="stage", reason="drop")
    repo.refs.delete_ref(_base_ref(key), op="stage", reason="drop")


def touched_keys(base_tree: Tree, staged_tree: Tree) -> list[FactKey]:
    """Which ``(subject, predicate)`` keys differ between ``base_tree`` and
    ``staged_tree``.

    A structural comparison of hash tuples, not a call into the diff engine:
    sealing only needs to know *which* keys this session touched, not how to
    classify the change (``contradicted`` vs. ``value_added`` vs. ...) —
    that classification needs a :class:`~memgit.core.cardinality.CardinalityMap`
    and a fact read per changed key, neither of which sealing has any use
    for. Reusing ``diff_trees`` here would pay for answers this call site
    never asks.
    """
    before_by_key = base_tree.by_key()
    after_by_key = staged_tree.by_key()
    keys = set(before_by_key) | set(after_by_key)
    return sorted(key for key in keys if before_by_key.get(key) != after_by_key.get(key))
