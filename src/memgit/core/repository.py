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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from memgit.core.commit import Commit
from memgit.core.fact import Fact
from memgit.core.graph import walk
from memgit.core.refs import Head, RefStore
from memgit.core.store import ObjectStore, is_object_hash
from memgit.core.tree import Tree

__all__ = [
    "Repository",
    "NotARepositoryError",
    "RepositoryExistsError",
    "RevisionNotFoundError",
    "EmptyCommitError",
]

_CONFIG_NAME = "config"
_HEAD_NAME = "HEAD"


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
            self._refs = RefStore(self.memgit_dir)
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

        Raises:
            RevisionNotFoundError: ``revision`` does not name anything.
        """
        resolved = self.refs.resolve(revision)
        if resolved is None:
            raise RevisionNotFoundError(revision)
        return resolved

    def branches(self) -> dict[str, str]:
        """Every local branch, mapped to the commit hash it points at."""
        return {
            name.removeprefix("refs/heads/"): commit_hash
            for name, commit_hash in self.refs.list_refs("refs/heads/").items()
        }

    def create_branch(self, name: str, at: str = "HEAD") -> str:
        """Create branch ``name`` pointing at ``at``; return its commit hash."""
        target = self.resolve(at)
        self.refs.write_ref(f"refs/heads/{name}", target, expect=None)
        return target

    def delete_branch(self, name: str, *, force: bool = False) -> None:
        """Delete branch ``name``.

        Raises:
            ValueError: ``name`` is the branch HEAD currently points at,
                unless ``force`` is set.
        """
        if not force and self.current_branch() == name:
            raise ValueError(f"cannot delete the current branch {name!r}")
        self.refs.delete_ref(f"refs/heads/{name}")

    # -- reading -----------------------------------------------------------

    def read_commit(self, rev: str) -> Commit:
        return Commit.read(self.store, self.resolve(rev))

    def read_tree(self, rev: str) -> Tree:
        """Read the tree at ``rev`` — a commit-ish, or a tree hash directly."""
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
            self.refs.detach_head(new_hash)
        else:
            assert head.ref is not None
            self.refs.write_ref(head.ref, new_hash, expect=head.commit)

        return new_hash

    def __repr__(self) -> str:
        return f"Repository({str(self.memgit_dir)!r})"
