"""Repository — owns the on-disk ``.memgit`` layout end to end.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from memgit.core.cardinality import CardinalityMap
from memgit.core.commit import Commit
from memgit.core.decay import DecayPolicy
from memgit.core.diff import Diff, diff_trees
from memgit.core.fact import Fact, FactKey
from memgit.core.graph import merge_base, walk
from memgit.core.reflog import RefLog, RefLogEntry, RefLogger, ReflogNotFoundError
from memgit.core.refs import Head, RefStore
from memgit.core.revparse import AncestryError, RevSyntaxError, apply_steps, parse_revision
from memgit.core.state import MemoryState
from memgit.core.store import (
    AmbiguousPrefixError,
    ObjectNotFoundError,
    ObjectStore,
    is_object_hash,
)
from memgit.core.tree import EMPTY_TREE_HASH, Tree

__all__ = [
    "Repository",
    "CheckoutResult",
    "NotARepositoryError",
    "RepositoryExistsError",
    "RevisionNotFoundError",
    "EmptyCommitError",
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
        return self.head.is_detached

    @property
    def moved(self) -> bool:
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
    def init(cls, path: Path | str = ".", *, default_branch: str = "main") -> "Repository":
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
    def open(cls, path: Path | str) -> "Repository":
        """Open the repository whose ``.memgit`` directory is exactly ``path``."""
        memgit_dir = Path(path).resolve()
        if not memgit_dir.is_dir() or not (memgit_dir / _HEAD_NAME).is_file():
            raise NotARepositoryError(f"not a memgit repository: {memgit_dir}")
        return cls(memgit_dir)

    @classmethod
    def discover(cls, start: Path | str | None = None) -> "Repository":
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
        if self._store is None:
            self._store = ObjectStore(self.memgit_dir / "objects")
        return self._store

    @property
    def refs(self) -> RefStore:
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
        return json.loads(path.read_text(encoding="utf-8"))

    # -- HEAD and branches -----------------------------------------------

    def head(self) -> Head:
        return self.refs.read_head()

    def head_commit(self) -> str | None:
        return self.head().commit

    def current_branch(self) -> str | None:
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

    def commit(
        self,
        facts: Iterable[Fact],
        message: str,
        *,
        author: str = "unknown",
        parents: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> str:
        """Write ``facts`` as a new commit; return its hash.

        Objects are written before the ref moves — see the module docstring
        for why that ordering, not the reverse, is what keeps a crash
        mid-commit recoverable instead of corrupting.

        Args:
            facts: The complete memory state for this commit, not a delta.
            parents: Explicit parent hashes. ``None`` means "whatever HEAD
                currently resolves to" (or no parent, on an unborn branch).
            allow_empty: Permit a commit whose tree is identical to its sole
                parent's. Off by default, matching git.

        Raises:
            EmptyCommitError: the tree would be identical to the sole
                parent's tree, and ``allow_empty`` is not set.
        """
        if parents is None:
            current = self.head_commit()
            parent_hashes: tuple[str, ...] = (current,) if current else ()
        else:
            parent_hashes = tuple(parents)

        fact_list = list(facts)
        for fact in fact_list:
            self.store.put(fact.to_dict())
        new_tree = Tree.from_facts(fact_list)
        new_tree_hash = new_tree.write(self.store)

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
    ) -> "CheckoutResult":
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
            by_key = target_tree.by_key()
            merged_tree = current_tree
            for key in keys:
                if key in by_key:
                    merged_tree = merged_tree.with_key(key, by_key[key])
                else:
                    merged_tree = merged_tree.without_key(key)
            facts = merged_tree.load(self.store)

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

    def __repr__(self) -> str:
        return f"Repository({str(self.memgit_dir)!r})"
