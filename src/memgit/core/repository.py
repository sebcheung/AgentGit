r"""Repository — owns the on-disk ``.memgit`` layout end to end.

Everything below this module (``ObjectStore``, ``Tree``, ``Commit``,
``RefStore``, ``graph``) is deliberately ignorant of directory names and of
each other's lifecycle. ``Repository`` is the one place that knows what
``.memgit/`` looks like and wires the pieces together — ``init`` lays it out,
``commit`` drives the write path end to end, ``log`` drives the read path.

**Core raises, the CLI formats.** Every exception here is a plain Python
exception carrying a message meant for a human, with no dependency on typer.
That is what lets the same discovery and commit logic be reused later by the
FastAPI layer and the MCP server (components 8 and 10) without dragging a CLI
framework in along with it — ``cli.py`` is the only place that turns these
into exit codes.

``init`` creates::

    .memgit/objects/            empty; ObjectStore fills in fanout dirs lazily
    .memgit/refs/heads/         empty; the default branch is unborn
    .memgit/HEAD                "ref: refs/heads/main\\n"
    .memgit/config              {"format_version": 1, "default_branch": "main"}

``config`` is JSON rather than an INI file — everything else in this project
is JSON, and a second config dialect would be unearned. It exists for two
concrete reasons right now: ``format_version`` is the escape hatch that makes
the flat-tree decision in ``tree.py`` reversible without a future slice
having to guess what an old repository's objects mean, and it is the natural
home for a default ``author`` once slice 5 needs one. No ``refs/tags/`` is
created — nothing tags anything yet.

**Objects are written before the ref that points at them moves — always.**
If a crash happens between those two steps, the result is unreferenced
objects sitting in the store: garbage, collectable later, harmless. The
reverse order could leave a ref pointing at an object that was never written,
which is corruption a reader can't recover from. This ordering is the whole
safety argument for :meth:`Repository.commit`, and it is why the sequence
below is fixed rather than incidental.

**Checkout is only a HEAD move.** Git's checkout is dangerous because it
rewrites files that might hold uncommitted work; MemGit has no working tree
and no index (see PLAN.md's "decisions locked in"), so there is no
uncommitted state a checkout could ever clobber. :meth:`checkout` therefore
never needs ``--force`` and never merges anything — it validates a revision,
then moves HEAD, attached or detached.

**Rewind, not reset, is the primary way to go back.** :meth:`rewind`
re-commits a past state forward as a new commit — non-destructive, and
attributable in the log like anything else. :meth:`reset` is the destructive
alternative that moves a branch pointer and leaves the abandoned commits
unreferenced (recoverable via the reflog, otherwise harmless garbage, per the
same ordering argument above). A debugging tool should default to keeping
the evidence, which is why ``rewind`` exists at all rather than only
``reset`` — see each method's docstring for the full argument.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from memgit.core.cardinality import CardinalityMap
from memgit.core.commit import Commit
from memgit.core.decay import DecayPolicy
from memgit.core.diff import Diff, diff_trees
from memgit.core.fact import Fact, FactKey
from memgit.core.graph import merge_base, walk
from memgit.core.reflog import RefLog, RefLogEntry, RefLogger, ReflogNotFoundError
from memgit.core.refs import Head, RefStore
from memgit.core.revparse import AncestryError, RevSyntaxError, apply_steps, parse_revision
from memgit.core.staging import (
    StagingArea,
    drop_staging,
    list_staging,
    open_staging,
    read_staging,
    session_key,
    touched_keys,
)
from memgit.core.staging import (
    stage as _stage_facts,
)
from memgit.core.state import MemoryState
from memgit.core.store import (
    AmbiguousPrefixError,
    ObjectNotFoundError,
    ObjectStore,
    is_object_hash,
)
from memgit.core.tree import EMPTY_TREE_HASH, Tree
from memgit.retrieval.embed import Embedder, default_embedder
from memgit.retrieval.index import VectorIndex
from memgit.retrieval.rank import Retriever

__all__ = [
    "CheckoutResult",
    "EmptyCommitError",
    "NotARepositoryError",
    "Repository",
    "RepositoryExistsError",
    "RevisionNotFoundError",
]


@dataclass(frozen=True, slots=True)
class CheckoutResult:
    """What :meth:`Repository.checkout` did.

    Deliberately carries no diff of its own — computing one costs a fact read
    per hash, and checkout is the fast path slice 5 will call every turn. A
    caller that wants to show what changed asks for
    ``repo.diff(previous.commit, head.commit)`` itself.
    """

    previous: Head
    head: Head
    created_branch: str | None = None

    @property
    def detached(self) -> bool:
        """True if the checkout landed on a detached HEAD."""
        return self.head.is_detached

    @property
    def moved(self) -> bool:
        """True if HEAD actually points somewhere new."""
        return self.previous.commit != self.head.commit

_CONFIG_NAME = "config"
_HEAD_NAME = "HEAD"
_CARDINALITY_NAME = "cardinality.json"
_DECAY_NAME = "decay.json"


class NotARepositoryError(Exception):
    """Raised when no ``.memgit`` directory can be found."""


class RepositoryExistsError(Exception):
    """Raised by ``init`` when a ``.memgit`` directory already exists.

    Git's reinit-is-safe behavior is a courtesy that depends on git being
    careful about exactly what it overwrites. Here, silently reinitializing
    would just be a way to lose a HEAD no one meant to lose, so this refuses
    instead.
    """


class RevisionNotFoundError(KeyError):
    """Raised when a revision string does not resolve to a commit."""


class EmptyCommitError(Exception):
    """Raised when a commit's tree would be identical to its sole parent's.

    Matches git's default. Slice 5 (the agent runtime) may want to flip this
    with ``allow_empty=True`` — "this turn changed nothing" is itself signal
    worth recording for causal attribution — but that is a deliberate choice
    for that caller to make, not a silent default here.
    """


class Repository:
    """Owns one repository's ``.memgit`` directory.

    Construct via :meth:`init`, :meth:`open`, or :meth:`discover` rather than
    the bare constructor, which assumes ``memgit_dir`` already exists and is
    laid out correctly.
    """

    DIR_NAME = ".memgit"
    FORMAT_VERSION = 1

    def __init__(self, memgit_dir: Path) -> None:
        self.memgit_dir = Path(memgit_dir)
        self._store: ObjectStore | None = None
        self._refs: RefStore | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def init(cls, path: Path | str = ".", *, default_branch: str = "main") -> Repository:
        """Create a new repository at ``path/.memgit``.

        Raises:
            RepositoryExistsError: ``.memgit`` already exists at ``path``.
        """
        root = Path(path).resolve()
        memgit_dir = root / cls.DIR_NAME
        if memgit_dir.exists():
            raise RepositoryExistsError(f"a repository already exists at {memgit_dir}")

        (memgit_dir / "objects").mkdir(parents=True)
        (memgit_dir / "refs" / "heads").mkdir(parents=True)

        config = {"format_version": cls.FORMAT_VERSION, "default_branch": default_branch}
        (memgit_dir / _CONFIG_NAME).write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        repo = cls(memgit_dir)
        repo.refs.set_head(f"refs/heads/{default_branch}")
        return repo

    @classmethod
    def open(cls, path: Path | str) -> Repository:
        """Open the repository whose ``.memgit`` directory is exactly ``path``."""
        memgit_dir = Path(path).resolve()
        if not memgit_dir.is_dir() or not (memgit_dir / _HEAD_NAME).is_file():
            raise NotARepositoryError(f"not a memgit repository: {memgit_dir}")
        return cls(memgit_dir)

    @classmethod
    def discover(cls, start: Path | str | None = None) -> Repository:
        """Walk upward from ``start`` (default: cwd) looking for ``.memgit``.

        Mirrors how git commands work from any subdirectory of a checkout.
        """
        current = Path(start or Path.cwd()).resolve()
        for candidate in [current, *current.parents]:
            memgit_dir = candidate / cls.DIR_NAME
            if memgit_dir.is_dir():
                return cls(memgit_dir)
        raise NotARepositoryError(
            "not a memgit repository (no .memgit directory found in this "
            "directory or any parent)"
        )

    # -- layout ------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The working directory this repository lives in (parent of ``.memgit``)."""
        return self.memgit_dir.parent

    @property
    def store(self) -> ObjectStore:
        """This repository's object store, opened lazily and cached."""
        if self._store is None:
            self._store = ObjectStore(self.memgit_dir / "objects")
        return self._store

    @property
    def refs(self) -> RefStore:
        """This repository's ref store, opened lazily and cached."""
        if self._refs is None:
            author = self.config().get("author", "unknown")
            logger = RefLogger(self.memgit_dir, author=author)
            self._refs = RefStore(self.memgit_dir, logger=logger)
        return self._refs

    def config(self) -> dict[str, Any]:
        """This repository's ``.memgit/config``, as a plain dict."""
        path = self.memgit_dir / _CONFIG_NAME
        if not path.is_file():
            return {"format_version": self.FORMAT_VERSION}
        return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))

    # -- HEAD and branches -----------------------------------------------

    def head(self) -> Head:
        """The current :class:`Head` — branch/detached state and commit."""
        return self.refs.read_head()

    def head_commit(self) -> str | None:
        """The commit HEAD resolves to, or ``None`` on an unborn branch."""
        return self.head().commit

    def current_branch(self) -> str | None:
        """The branch HEAD points at, or ``None`` if detached."""
        return self.head().branch

    def resolve(self, revision: str = "HEAD") -> str:
        """Resolve ``revision`` to a commit hash.

        Beyond a plain ``HEAD`` / branch name / full hash (``RefStore``'s own
        job), this also understands ancestry suffixes (``HEAD~3``,
        ``main^2``), reflog indexing (``HEAD@{1}``), and abbreviated hash
        prefixes — see ``revparse.py`` for the grammar. Every failure mode
        collapses into :class:`RevisionNotFoundError` so no CLI command needs
        to know which of the underlying pieces produced it.

        Raises:
            RevisionNotFoundError: ``revision`` does not name anything.
        """
        try:
            expr = parse_revision(revision)
        except RevSyntaxError as exc:
            raise RevisionNotFoundError(revision) from exc

        if expr.reflog_index is not None:
            ref_name = self._reflog_ref_name(expr.base)
            try:
                base_hash = RefLog(self.memgit_dir, ref_name).at(expr.reflog_index)
            except ReflogNotFoundError as exc:
                raise RevisionNotFoundError(revision) from exc
        else:
            base_hash = self._resolve_base(expr.base)

        try:
            return apply_steps(base_hash, expr.steps, self.read_commit)
        except AncestryError as exc:
            raise RevisionNotFoundError(revision) from exc

    @staticmethod
    def _reflog_ref_name(base: str) -> str:
        """Map a revparse base to the ref name its reflog lives under."""
        if base == "HEAD" or base.startswith("refs/"):
            return base
        return f"refs/heads/{base}"

    def _resolve_base(self, base: str) -> str:
        """Resolve a bare base (no ancestry suffix, no reflog index).

        Tries ``RefStore``'s own resolution first (HEAD, a full hash, a full
        ref path, a bare branch name), then — only as a last resort, so an
        existing branch name is never shadowed — an abbreviated hash prefix.
        """
        resolved = self.refs.resolve(base)
        if resolved is not None:
            return resolved
        if is_object_hash(base):
            raise RevisionNotFoundError(base)
        try:
            return self.store.resolve_prefix(base)
        except AmbiguousPrefixError as exc:
            raise RevisionNotFoundError(f"{base!r} is ambiguous: {exc}") from exc
        except (ValueError, ObjectNotFoundError) as exc:
            raise RevisionNotFoundError(base) from exc

    def branches(self) -> dict[str, str]:
        """Every local branch, mapped to the commit hash it points at."""
        return {
            name.removeprefix("refs/heads/"): commit_hash
            for name, commit_hash in self.refs.list_refs("refs/heads/").items()
        }

    def create_branch(self, name: str, at: str = "HEAD") -> str:
        """Create branch ``name`` pointing at ``at``; return its commit hash."""
        target = self.resolve(at)
        self.refs.write_ref(
            f"refs/heads/{name}", target, expect=None, op="branch", reason=f"created from {at}"
        )
        return target

    def delete_branch(self, name: str, *, force: bool = False) -> None:
        """Delete branch ``name``.

        Raises:
            ValueError: ``name`` is the branch HEAD currently points at,
                unless ``force`` is set.
        """
        if not force and self.current_branch() == name:
            raise ValueError(f"cannot delete the current branch {name!r}")
        self.refs.delete_ref(f"refs/heads/{name}", op="branch", reason="branch deleted")

    # -- reading -----------------------------------------------------------

    def read_commit(self, rev: str) -> Commit:
        """Resolve ``rev`` and load the :class:`Commit` object it names."""
        return Commit.read(self.store, self.resolve(rev))

    def read_tree(self, rev: str) -> Tree:
        """Read the tree at ``rev`` — a commit-ish, or a tree hash directly.

        ``EMPTY_TREE_HASH`` is special-cased rather than looked up: nothing
        ever writes that object to the store (``init`` stays object-free), so
        a naive lookup would raise ``ObjectNotFoundError`` for the one tree
        hash that is guaranteed to exist conceptually — "believe nothing",
        the baseline :meth:`diff` uses for a root commit.
        """
        if rev == EMPTY_TREE_HASH:
            return Tree(())
        if is_object_hash(rev):
            payload = self.store.get(rev)
            if payload.get("type") == "tree":
                return Tree.from_dict(payload)
        return Tree.read(self.store, self.read_commit(rev).tree)

    def log(self, start: str = "HEAD", *, limit: int | None = None) -> Iterator[Commit]:
        """Walk history from ``start``, most-recent-first."""
        start_hash = self.resolve(start)
        for _commit_hash, commit in walk(start_hash, self.read_commit, limit=limit):
            yield commit

    def read_fact(self, fact_hash: str) -> Fact:
        """Load the fact stored at ``fact_hash``. The ``FactReader`` ``diff()`` uses."""
        return Fact.from_dict(self.store.get(fact_hash))

    # -- cardinality ---------------------------------------------------------

    def cardinality(self) -> CardinalityMap:
        """This repository's cardinality map — the diff engine's single/multi schema.

        Repo-local and uncommitted, unlike everything else in ``.memgit``: see
        ``cardinality.py``'s module docstring for why. A missing file is a
        normal state, matching :meth:`config`.
        """
        path = self.memgit_dir / _CARDINALITY_NAME
        if not path.is_file():
            return CardinalityMap.default_map()
        return CardinalityMap.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def set_cardinality(self, mapping: CardinalityMap) -> None:
        """Overwrite the cardinality map, atomically."""
        path = self.memgit_dir / _CARDINALITY_NAME
        lock = path.with_name(path.name + ".lock")
        lock.write_text(json.dumps(mapping.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lock.replace(path)

    # -- decay -----------------------------------------------------------

    def decay(self) -> DecayPolicy:
        """This repository's decay policy — how confidence fades between commits.

        Repo-local and uncommitted, for the same reason as :meth:`cardinality`:
        see ``decay.py``'s module docstring. A missing file is a normal
        state, matching :meth:`cardinality`.
        """
        path = self.memgit_dir / _DECAY_NAME
        if not path.is_file():
            return DecayPolicy.default_map()
        return DecayPolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def set_decay(self, policy: DecayPolicy) -> None:
        """Overwrite the decay policy, atomically."""
        path = self.memgit_dir / _DECAY_NAME
        lock = path.with_name(path.name + ".lock")
        lock.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lock.replace(path)

    # -- retrieval -------------------------------------------------------

    @property
    def embeddings_dir(self) -> Path:
        """Where vector caches live, namespaced per embedder.

        Not created by :meth:`init` — like the objects fanout directories,
        it comes into existence lazily on the first vector written, so a
        slice 0-6 repository that has never seen retrieval stays untouched.
        """
        return self.memgit_dir / "embeddings"

    def embedder(self) -> Embedder:
        """This repository's configured embedder — a zero-dependency default.

        Reads the ``"retrieval"`` section of :meth:`config`; see
        :func:`memgit.retrieval.embed.default_embedder` for the config
        shape and error behavior.
        """
        return default_embedder(self.config().get("retrieval"))

    def vector_index(self, embedder: Embedder | None = None) -> VectorIndex:
        """This repository's vector cache for ``embedder`` (default: :meth:`embedder`)."""
        resolved = embedder if embedder is not None else self.embedder()
        return VectorIndex.open(self.embeddings_dir, embedder_id=resolved.id, dim=resolved.dim)

    def retriever(self, *, confidence_weight: float | None = None) -> Retriever:
        """Assembles this repository's embedder, vector cache, and decay policy.

        The caller still supplies the :class:`~memgit.core.state.MemoryState`
        to :meth:`~memgit.retrieval.rank.Retriever.retrieve` — this factory
        only wires up the repo-local configuration, never the candidate set,
        which is exactly what keeps commit/branch scoping a property of the
        call site rather than of this method.
        """
        embedder = self.embedder()
        index = self.vector_index(embedder)
        kwargs = {} if confidence_weight is None else {"confidence_weight": confidence_weight}
        return Retriever(index, embedder, self.decay(), **kwargs)

    # -- diff ------------------------------------------------------------

    def diff(
        self,
        before: str | None = None,
        after: str = "HEAD",
        *,
        include_unchanged: bool = False,
        use_merge_base: bool = False,
    ) -> Diff:
        """Compare two memory states.

        Args:
            before: A revision, or ``None`` for the empty tree — "believe
                nothing" — which is what a root commit is diffed against.
                Defaults to ``after``'s first parent (empty tree for a root
                commit) rather than requiring the caller to know that.
            after: A revision. Defaults to ``HEAD``.
            use_merge_base: Diff ``after`` against ``merge_base(before, after)``
                instead of ``before`` directly — the honest comparison for two
                branches that have diverged, since a direct diff would also
                report facts ``before`` simply hasn't received yet.
        """
        after_hash = self.resolve(after)

        if before is None:
            after_commit = self.read_commit(after_hash)
            before_hash = after_commit.parents[0] if after_commit.parents else None
        else:
            before_hash = self.resolve(before)

        if use_merge_base:
            if before_hash is None:
                raise ValueError(
                    "use_merge_base requires two explicit revisions, not the implicit parent"
                )
            before_hash = merge_base(before_hash, after_hash, self.read_commit)

        before_tree = self.read_tree(before_hash) if before_hash is not None else Tree(())
        after_tree = self.read_tree(after_hash)

        return diff_trees(
            before_tree,
            after_tree,
            self.read_fact,
            cardinality=self.cardinality(),
            include_unchanged=include_unchanged,
        )

    # -- writing -------------------------------------------------------

    def write_tree(self, facts: Iterable[Fact]) -> str:
        """Write every fact in ``facts``, then the ``Tree`` they build; return its hash.

        The object-writing half of :meth:`commit`, pulled out on its own so a
        caller can produce a real, hashed, diffable tree without also
        producing a ``Commit`` — which is exactly what MCP write staging
        needs (see ``core/staging.py``): a session's in-progress memory state
        has to live *somewhere* between two process invocations, and "a
        content-addressed tree, referenced by a ref that isn't
        ``refs/heads/*``" is the answer the no-index decision in PLAN.md
        already points at, rather than any new, unhashed, git-index-shaped
        state. Facts are written before the tree, matching the module
        docstring's "objects before the ref moves" ordering one level down:
        a crash here leaves unreferenced fact objects, not a tree pointing at
        one that was never written.
        """
        fact_list = list(facts)
        for fact in fact_list:
            self.store.put(fact.to_dict())
        new_tree = Tree.from_facts(fact_list)
        return new_tree.write(self.store)

    def overlay(
        self, onto_tree: Tree, from_tree: Tree, keys: Sequence[FactKey]
    ) -> list[Fact]:
        """Apply just ``keys`` from ``from_tree`` onto ``onto_tree``; return the facts.

        For each key: present in ``from_tree`` means "set to that value"
        (:meth:`~memgit.core.tree.Tree.with_key`); absent means "remove it"
        (:meth:`~memgit.core.tree.Tree.without_key`). A key not listed in
        ``keys`` is left exactly as ``onto_tree`` has it, whether or not
        ``from_tree`` also has an opinion about it — "put back just this one
        belief" (or, for MCP sealing, "apply just the beliefs this session
        actually touched"), never a wholesale replacement.

        Extracted from :meth:`rewind`'s own ``keys=`` branch, which was
        first to need exactly this operation; sealing a staged MCP session
        onto a branch tip that has moved since staging began is the second
        caller (see ``mcp/session.py``), and the reason this earned a name of
        its own instead of staying inlined in one method.
        """
        by_key = from_tree.by_key()
        merged_tree = onto_tree
        for key in keys:
            if key in by_key:
                merged_tree = merged_tree.with_key(key, by_key[key])
            else:
                merged_tree = merged_tree.without_key(key)
        return merged_tree.load(self.store)

    def commit(
        self,
        facts: Iterable[Fact],
        message: str,
        *,
        author: str = "unknown",
        parents: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_empty: bool = False,
        onto: str | None = None,
    ) -> str:
        """Write ``facts`` as a new commit; return its hash.

        Objects are written before the ref moves — see the module docstring
        for why that ordering, not the reverse, is what keeps a crash
        mid-commit recoverable instead of corrupting.

        Args:
            facts: The complete memory state for this commit, not a delta.
            parents: Explicit parent hashes. ``None`` means "whatever
                ``onto`` (or HEAD) currently resolves to" (or no parent, on
                an unborn branch).
            allow_empty: Permit a commit whose tree is identical to its sole
                parent's. Off by default, matching git.
            onto: Advance this branch ref (``"refs/heads/<name>"`` or a bare
                branch name) instead of HEAD's current branch, and leave HEAD
                itself untouched. ``None`` (the default) reproduces every
                prior slice's behavior exactly. This is what lets an MCP
                write target a fixed, configured branch rather than
                whatever the human's HEAD happens to point at in another
                terminal — HEAD is the human's cursor, not a session's.

        Raises:
            EmptyCommitError: the tree would be identical to the sole
                parent's tree, and ``allow_empty`` is not set.
        """
        onto_ref = None if onto is None else (onto if onto.startswith("refs/") else f"refs/heads/{onto}")

        if parents is None:
            current = self.refs.read_ref(onto_ref) if onto_ref is not None else self.head_commit()
            parent_hashes: tuple[str, ...] = (current,) if current else ()
        else:
            parent_hashes = tuple(parents)

        new_tree_hash = self.write_tree(facts)

        if not allow_empty and len(parent_hashes) == 1:
            parent_commit = Commit.read(self.store, parent_hashes[0])
            if parent_commit.tree == new_tree_hash:
                raise EmptyCommitError(
                    f"commit would be empty: tree unchanged from parent {parent_hashes[0]}"
                )

        new_commit = Commit(
            tree=new_tree_hash,
            parents=parent_hashes,
            message=message,
            author=author,
            metadata=metadata,
        )
        new_hash = new_commit.write(self.store)

        if onto_ref is not None:
            self.refs.write_ref(
                onto_ref, new_hash, expect=parent_hashes[0] if parent_hashes else None,
                op="commit", reason=new_commit.summary,
            )
            return new_hash

        head = self.head()
        if head.is_detached:
            self.refs.detach_head(new_hash, op="commit", reason=new_commit.summary)
        else:
            assert head.ref is not None
            self.refs.write_ref(
                head.ref, new_hash, expect=head.commit, op="commit", reason=new_commit.summary
            )

        return new_hash

    # -- checkout, rewind and reset -----------------------------------------

    def _existing_branch_ref(self, base: str) -> str | None:
        """The full ref path for ``base`` if it names a branch, else ``None``.

        A branch "exists" here even if it is unborn (HEAD points at it, but
        it has no ref file yet, since a ref file is only written on its first
        commit) — checking out the branch you are already on must still
        attach, not detach.
        """
        ref_name = base if base.startswith("refs/") else f"refs/heads/{base}"
        if self.refs.read_ref(ref_name) is not None:
            return ref_name
        current = self.head()
        if not current.is_detached and current.ref == ref_name:
            return ref_name
        return None

    def checkout(
        self,
        revision: str = "HEAD",
        *,
        create: str | None = None,
        detach: bool = False,
    ) -> CheckoutResult:
        """Move HEAD to ``revision``.

        See the module docstring for why this is *only* a HEAD move: no
        working tree and no index means nothing is ever at risk of being
        overwritten, so there is no ``--force`` and no merge here.

        Args:
            create: Create this branch at ``revision`` first, then attach to
                it — mirrors ``git checkout -b``. If the repository is
                unborn and ``revision`` (default ``"HEAD"``) resolves to
                nothing, the new branch is left unborn too, matching
                ``git init && git checkout -b x``.
            detach: Force a detached checkout even when ``revision`` names a
                branch — mirrors ``git checkout --detach``.

        Raises:
            RevisionNotFoundError: ``revision`` does not resolve (unless
                ``create`` is given and the repository is simply unborn).
        """
        previous = self.head()

        if create is not None:
            try:
                self.create_branch(create, at=revision)
            except RevisionNotFoundError:
                pass  # unborn: nothing to point the new branch at yet
            self.refs.set_head(
                f"refs/heads/{create}", op="checkout", reason=f"checkout -b {create} from {revision}"
            )
            return CheckoutResult(previous=previous, head=self.head(), created_branch=create)

        if detach:
            self.refs.detach_head(
                self.resolve(revision), op="checkout", reason=f"checkout {revision} (detach)"
            )
            return CheckoutResult(previous=previous, head=self.head())

        expr = parse_revision(revision)
        branch_ref = self._existing_branch_ref(expr.base) if expr.is_bare else None
        if branch_ref is not None:
            self.refs.set_head(branch_ref, op="checkout", reason=f"checkout {revision}")
        else:
            # Not a bare branch name (e.g. "main~1"), or names no branch at
            # all: detach, even if the base happens to be a branch — is_bare
            # is exactly the distinction between "the branch" and "an
            # ancestor of it".
            self.refs.detach_head(self.resolve(revision), op="checkout", reason=f"checkout {revision}")

        return CheckoutResult(previous=previous, head=self.head())

    def rewind(
        self,
        revision: str,
        message: str | None = None,
        *,
        author: str = "unknown",
        keys: Sequence[FactKey] | None = None,
        allow_empty: bool = False,
    ) -> str:
        """Re-commit ``revision``'s memory state forward onto HEAD.

        The primary, non-destructive way to "go back": the rewind is itself
        a new commit, visible in ``log`` and diffable like any other change,
        rather than history quietly disappearing. See the module docstring
        for the full argument against making :meth:`reset` the default
        instead.

        Args:
            revision: The past memory state to bring forward.
            message: Commit message. Defaults to naming what was rewound.
            keys: If given, overlay only these ``(subject, predicate)`` keys
                from ``revision``'s tree onto HEAD's current tree, instead of
                replacing the whole state — "put back just this one belief",
                which slice 6's ablation engine needs first. A key from
                ``revision`` that HEAD doesn't have is added; a key omitted
                from ``revision`` (whose keys are given) is not itself
                removed unless explicitly listed.
            allow_empty: Passed through to :meth:`commit`.

        Raises:
            RevisionNotFoundError: ``revision`` does not resolve.
            EmptyCommitError: The rewind would not change anything.
        """
        target_hash = self.resolve(revision)
        target_tree = self.read_tree(target_hash)

        if keys is None:
            facts = target_tree.load(self.store)
        else:
            current_tree = self.read_tree("HEAD") if self.head_commit() is not None else Tree(())
            facts = self.overlay(current_tree, target_tree, keys)

        if message is None:
            message = f"rewind memory to {revision} ({target_hash[:8]})"

        return self.commit(
            facts,
            message,
            author=author,
            metadata={"rewind_of": target_hash, "rewind_from": revision},
            allow_empty=allow_empty,
        )

    def reset(self, revision: str, *, branch: str | None = None) -> str:
        """Move a branch to point at ``revision`` — destructive.

        With no working tree and no index, git's ``--soft``/``--mixed``/
        ``--hard`` split collapses into this one operation: there is no
        uncommitted state for the distinction to apply to. Commits solely
        reachable from the branch's old position become unreachable (not
        deleted — harmless garbage per the module docstring's ordering
        argument, and recoverable via the reflog while it still holds the
        old value). Prefer :meth:`rewind` unless the history itself needs to
        change, not just the memory state.

        Args:
            branch: Which branch to move. Defaults to HEAD's own branch, or
                moves HEAD directly if it is detached.

        Raises:
            RevisionNotFoundError: ``revision`` does not resolve.
            ValueError: HEAD is unborn and no ``branch`` was given.
        """
        target_hash = self.resolve(revision)
        head = self.head()

        if branch is not None:
            ref_name = f"refs/heads/{branch}"
            current = self.refs.read_ref(ref_name)
            self.refs.write_ref(
                ref_name, target_hash, expect=current, op="reset", reason=f"reset to {revision}"
            )
            return target_hash

        if head.is_detached:
            self.refs.detach_head(target_hash, op="reset", reason=f"reset to {revision}")
            return target_hash

        if head.commit is None:
            raise ValueError("cannot reset: HEAD is unborn and no branch was given")

        assert head.ref is not None
        self.refs.write_ref(
            head.ref, target_hash, expect=head.commit, op="reset", reason=f"reset to {revision}"
        )
        return target_hash

    def reflog(self, ref: str = "HEAD") -> tuple[RefLogEntry, ...]:
        """Every recorded movement of ``ref``, oldest first.

        Empty if the ref exists but has never moved since a logger was
        attached (or, on a freshly opened repository, if it has moved but
        the ``.memgit/logs`` directory predates this slice).
        """
        return RefLog(self.memgit_dir, self._reflog_ref_name(ref)).entries()

    # -- memory state --------------------------------------------------------

    def state(self, rev: str = "HEAD") -> MemoryState:
        """Materialize the full memory state at ``rev``.

        ``rev`` follows the same resolution rules as everywhere else: HEAD,
        a branch, an ancestry expression, or a commit/tree hash directly.
        """
        tree = self.read_tree(rev)
        return MemoryState.from_tree(tree, self.read_fact, commit=self.resolve(rev))

    # -- staging (slice 8: MCP write sessions) ------------------------------

    def _write_branch_ref(self, branch: str | None) -> str:
        """The full ref path a write session commits onto.

        Defaults to the repository's configured ``default_branch`` — **never
        HEAD** — because HEAD is the human's cursor at the CLI: an MCP
        session's memory should not relocate because someone ran
        ``memgit checkout`` in another terminal, and a detached HEAD would
        make the session's commits reachable from nothing at all.
        """
        name = branch if branch is not None else self.config().get("default_branch", "main")
        return name if name.startswith("refs/") else f"refs/heads/{name}"

    def staging_area(self, key: str) -> StagingArea | None:
        """The staging area at hashed key ``key``, or ``None`` if nothing is staged there.

        ``key`` is the hash :func:`~memgit.core.staging.session_key` produces.
        Read-only — never opens one; see :meth:`stage`.
        """
        return read_staging(self, key)

    def staging_areas(self) -> tuple[StagingArea, ...]:
        """Every open staging area, for ``memgit status`` / ``memgit staging list``."""
        return list_staging(self)

    def drop_staging(self, key: str) -> None:
        """Discard the staging area at hashed key ``key``. A no-op if absent."""
        drop_staging(self, key)

    def open_staging(self, session_id: str, *, branch: str | None = None) -> StagingArea:
        """Return ``session_id``'s staging area, opening one if it doesn't exist yet.

        A freshly opened area starts against ``branch``'s current tip.
        ``session_id`` is the caller's own identifier (an MCP transport
        session id, typically); it is hashed via
        :func:`~memgit.core.staging.session_key` before ever touching a ref
        path, and the resulting :class:`StagingArea` carries the hashed
        ``key`` a human can pass to ``memgit staging show``. Call this (or
        keep the ``StagingArea`` a previous :meth:`stage` call returned)
        before folding one more ``remember``/``forget`` onto its facts, so
        the fold has a ``.tree`` to pass as :meth:`stage`'s ``based_on``.
        """
        key = session_key(session_id)
        return open_staging(self, key, branch_ref=self._write_branch_ref(branch))

    def stage(
        self, session_id: str, facts: Iterable[Fact], *, based_on: str
    ) -> StagingArea:
        """Replace ``session_id``'s staged fact set with ``facts``.

        The whole set, not a delta — matching :meth:`commit`'s own contract.

        Args:
            based_on: The tree hash ``facts`` were folded against — a
                previous :meth:`open_staging` or :meth:`stage` call's
                ``.tree``. See :func:`memgit.core.staging.stage` for why this
                must be the value the caller actually saw, never re-read
                internally: that is what makes the compare-and-swap below
                capable of detecting a concurrent writer at all.

        Raises:
            ~memgit.core.staging.StagingConflictError: another writer moved
                this session's staging area past ``based_on`` first. The
                caller re-reads the area, re-applies its fold on top of the
                new value, and retries once — this method does not retry on
                its own, since only the caller knows how to redo the fold
                that produced the losing write.
        """
        key = session_key(session_id)
        return _stage_facts(self, key, list(facts), based_on=based_on)

    def seal_staging(
        self,
        session_id: str,
        message: str,
        *,
        branch: str | None = None,
        author: str = "unknown",
        metadata: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Commit ``session_id``'s staged facts onto ``branch``; drop the staging area either way.

        Sealing does **not** commit the staged tree wholesale: if the target
        branch has moved since staging began, doing so would silently
        discard every fact another writer added since. Instead it computes
        which ``(subject, predicate)`` keys *this session* touched (a
        structural comparison against its own ``base``, not a full diff) and
        :meth:`overlay`s just those onto the branch's current tip — the
        common case (``base`` still equals the tip) costs one structural
        comparison and produces a tree bit-identical to committing the
        staged tree wholesale; a diverged tip loses nothing untouched by this
        session. A key both sessions touched has no principled merge (no
        slice owns that), so it is recorded in the new commit's
        ``restaged_over`` metadata rather than silently resolved — the same
        posture :func:`~memgit.core.graph.merge_base` already takes for
        divergent branches.

        Returns:
            The new commit's hash, or ``None`` if nothing was staged, or
            this session touched no keys, or the resulting commit would have
            been empty (:class:`EmptyCommitError` is swallowed here, not
            raised, since a model calling ``commit`` after a no-op turn
            should be told nothing happened, not handed a tool error).
        """
        return self.seal_staging_key(
            session_key(session_id), message, branch=branch, author=author, metadata=metadata
        )

    def seal_staging_key(
        self,
        key: str,
        message: str,
        *,
        branch: str | None = None,
        author: str = "unknown",
        metadata: Mapping[str, Any] | None = None,
    ) -> str | None:
        """:meth:`seal_staging`, keyed directly by an already-hashed staging key.

        For ``memgit staging commit <key>`` — the CLI never has the MCP
        session's raw id, only the hashed key ``memgit staging`` already
        printed, and hashing that key a second time would look up the wrong
        ref entirely. :meth:`seal_staging` is just this method behind one
        extra hashing step.
        """
        area = read_staging(self, key)
        if area is None:
            return None

        branch_ref = self._write_branch_ref(branch)
        base_tree = self.read_tree(area.base) if area.base is not None else Tree(())
        staged_tree = self.read_tree(area.tree)
        touched = touched_keys(base_tree, staged_tree)
        if not touched:
            drop_staging(self, key)
            return None

        tip_hash = self.refs.read_ref(branch_ref)
        tip_tree = self.read_tree(tip_hash) if tip_hash is not None else Tree(())
        facts = self.overlay(tip_tree, staged_tree, touched)

        moved_since_base = touched_keys(base_tree, tip_tree)
        restaged_over = sorted(set(touched) & set(moved_since_base))

        full_metadata = dict(metadata) if metadata is not None else {}
        full_metadata["staged_on"] = area.base
        if restaged_over:
            full_metadata["restaged_over"] = restaged_over

        try:
            commit_hash = self.commit(
                facts, message, author=author, metadata=full_metadata, onto=branch_ref
            )
        except EmptyCommitError:
            drop_staging(self, key)
            return None

        drop_staging(self, key)
        return commit_hash

    def __repr__(self) -> str:
        return f"Repository({str(self.memgit_dir)!r})"
