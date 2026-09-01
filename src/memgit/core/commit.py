"""The Commit — one node in the memory DAG.

A commit ties a tree (a full snapshot, see ``tree.py``) to the history that
produced it: the commits that came before it, who or what wrote it, and why.
Branches (``refs.py``) are just named pointers at a commit; the DAG itself is
this module plus ``parents``.

A few choices here depart from git on purpose, and each is worth stating
rather than leaving as an implicit git-clone assumption:

**``parents`` is always emitted, even as ``[]``.** Every other optional field
in this codebase (``Fact.source``, this module's own ``metadata``) is omitted
from the serialized payload when unset, so that "never set" and "explicitly
empty" hash identically — see ``Fact.to_dict``'s docstring for that reasoning.
An empty parent list does not have that ambiguity: it is not the absence of a
value, it is the specific and meaningful assertion "history begins here." That
is worth a fact, not a null.

**One ``author`` field, not git's author/committer split.** Git needs both
because a patch can be written by one person and applied by another days
later. MemGit has no patch-mail workflow — whoever calls ``commit()`` is both
at once — so a second identity field would be structure copied from git
without a reader to serve.

**``committed_at`` is inside the hash.** Two commits with identical trees,
parents, and messages made a second apart are still two different objects.
That mirrors ``Fact.reaffirm``: re-asserting something unchanged is still a
new, addressable event, not something that collapses into the old one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from memgit.core.canonical import canonical_json, hash_payload, utcnow
from memgit.core.store import is_object_hash

if TYPE_CHECKING:
    from memgit.core.store import ObjectStore

__all__ = ["Commit"]

_FIELDS = frozenset(
    {"type", "tree", "parents", "message", "author", "committed_at", "metadata"}
)


@dataclass(frozen=True, slots=True)
class Commit:
    """A snapshot's place in history: a tree hash, its parent(s), and why.

    Frozen for the same reason ``Fact`` is: a commit's address is the hash of
    its own fields, so mutating one in place would leave it filed under a
    hash that no longer describes it. There is deliberately no stored
    ``hash`` field on this class — the hash is a pure function of the
    content, and storing it would create a second source of truth that could
    disagree with the first. ``.hash`` recomputes it every time, which
    doubles as a free integrity check on the way to the object store.

    Attributes:
        tree: Hash of this commit's :class:`~memgit.core.tree.Tree` — the full
            memory state at this point in history.
        parents: Hashes of the commit(s) this one follows. Empty for the root
            commit. A list (not a single value) so merge commits are
            representable later, even though nothing in this slice creates
            one.
        message: Why this commit exists, in the committer's own words.
        author: Who or what made this commit (``"agent:claude-opus-5"``,
            ``"cli:sebastian"``). Opaque to MemGit, like ``Fact.source``.
        committed_at: ISO-8601 UTC timestamp of when this was recorded.
        metadata: Free-form, structured tags for machinery built on top of the
            commit graph — for instance slice 6's ablation engine marking a
            commit as "arm B of experiment 17". Kept separate from ``message``
            (for humans) so a later feature can search on it without parsing
            prose. Omitted from the payload when unset, matching ``Fact``.
    """

    tree: str
    parents: tuple[str, ...] = ()
    message: str = ""
    author: str = "unknown"
    committed_at: str = field(default_factory=utcnow)
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not is_object_hash(self.tree):
            raise ValueError(f"Commit.tree must be a valid object hash, got {self.tree!r}")
        for parent in self.parents:
            if not is_object_hash(parent):
                raise ValueError(
                    f"Commit.parents must all be valid object hashes, got {parent!r}"
                )
        if len(set(self.parents)) != len(self.parents):
            raise ValueError(f"Commit.parents contains a duplicate: {self.parents!r}")
        if not isinstance(self.message, str):
            raise TypeError(f"Commit.message must be a str, got {type(self.message).__name__}")
        if not isinstance(self.author, str) or not self.author.strip():
            raise ValueError("Commit.author must be a non-empty string")
        if self.metadata is not None:
            # Fail now, at construction, rather than later at write() time far
            # from whatever code built an un-JSON-able metadata value.
            try:
                canonical_json(dict(self.metadata))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Commit.metadata is not JSON-serializable: {exc}") from exc

    # -- identity ----------------------------------------------------------

    @property
    def hash(self) -> str:
        """This commit's content address in the object store."""
        return hash_payload(self.to_dict())

    @property
    def is_root(self) -> bool:
        """Whether this commit has no parent — where history begins."""
        return len(self.parents) == 0

    @property
    def is_merge(self) -> bool:
        """Whether this commit has more than one parent."""
        return len(self.parents) > 1

    @property
    def summary(self) -> str:
        """The first line of ``message`` — what ``log`` prints per commit."""
        return self.message.splitlines()[0] if self.message else ""

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Unlike ``Fact``, ``parents`` is written even when empty — see the
        module docstring for why an empty list is not "unset" here. Only
        ``metadata`` follows the omit-when-unset convention.
        """
        payload: dict[str, Any] = {
            "type": "commit",
            "tree": self.tree,
            "parents": list(self.parents),
            "message": self.message,
            "author": self.author,
            "committed_at": self.committed_at,
        }
        if self.metadata is not None:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Commit:
        """Rebuild a commit from :meth:`to_dict` output.

        Strict about shape, matching ``Tree.from_dict``: an unknown key or a
        non-``"commit"`` type tag is rejected, so a commit's hash can never
        silently stop describing everything filed under it.
        """
        unknown = payload.keys() - _FIELDS
        if unknown:
            raise ValueError(f"commit payload has unknown keys: {sorted(unknown)}")

        kind = payload.get("type")
        if kind != "commit":
            raise ValueError(f"expected a commit object, got type={kind!r}")

        try:
            tree = payload["tree"]
            parents = payload["parents"]
        except KeyError as exc:
            raise ValueError(f"commit payload is missing {exc}") from exc

        return cls(
            tree=tree,
            parents=tuple(parents),
            message=payload.get("message", ""),
            author=payload.get("author", "unknown"),
            committed_at=payload.get("committed_at", utcnow()),
            metadata=payload.get("metadata"),
        )

    def write(self, store: "ObjectStore") -> str:
        """Store this commit; return its hash."""
        return store.put(self.to_dict())

    @classmethod
    def read(cls, store: "ObjectStore", commit_hash: str) -> Commit:
        """Load and validate the commit stored at ``commit_hash``."""
        return cls.from_dict(store.get(commit_hash))

    def __str__(self) -> str:
        return f"{self.hash[:8]} {self.summary}"
