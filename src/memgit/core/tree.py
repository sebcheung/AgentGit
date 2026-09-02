"""The Tree — a snapshot of an entire memory state.

Where a :class:`~memgit.core.fact.Fact` is one belief, a tree is *every* belief
an agent holds at a single point in history: the thing a commit points at, and
the thing slice 4's rewind will materialize wholesale. Trees are full snapshots,
not deltas against a parent — that costs storage (paid back by the object
store's deduplication: an unchanged fact's hash simply reappears in the next
tree, free), and buys back something a delta model would have to build
separately: any commit's full state is one object read away, with no chain of
patches to replay first.

Two shapes were considered for the entries, and this module took the plainer
one on purpose. Git's own model — a root tree of subject "directories", each
holding a per-predicate sub-tree of hashes — was the more obviously git-like
answer. It was rejected because it does not actually change the asymptotics:
the root tree is still rewritten on every commit and still grows with the
number of subjects, so the *sharded* design only lowers a constant factor, at
real cost — a recursive walk instead of a linear scan, right in the module
(the diff engine, slice 3) that is this project's core intellectual work. A
flat, single object is simpler to diff, simpler to test, and the entry
encoding is kept private behind this class's API — plus a ``format_version``
in the repository's config — so switching later is a contained change, not a
rewrite.

The other question a tree has to answer before the diff engine can be
written: what does one key hold? ``fact.py``'s docstring promises that
``(subject, predicate)`` is *the* diff key, but an agent can believe more than
one thing about a predicate at once (it can like both chess and go). Deciding
that here, in storage, would bake an ontology into the object format that only
the diff engine should own. So a tree entry holds a *list* of fact hashes per
key — cardinality ("is a second value here an addition or a contradiction?")
is left as a pure interpretation question for whoever reads the tree, not a
fact about how the tree is shaped. That question is answered by
:class:`memgit.core.cardinality.CardinalityMap`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from memgit.core.canonical import hash_payload
from memgit.core.fact import Fact, FactKey
from memgit.core.store import is_object_hash

if TYPE_CHECKING:
    from memgit.core.store import ObjectStore

__all__ = ["Tree", "TreeEntry", "EMPTY_TREE_HASH"]

# (subject, predicate, fact_hashes) — the hashes are already sorted and deduped
# by the time an entry exists on a Tree.
TreeEntry = tuple[str, str, tuple[str, ...]]

_RawHashes = str | Iterable[str]


@dataclass(frozen=True, slots=True)
class Tree:
    """An immutable, content-addressed snapshot of ``(subject, predicate)``
    keys to the fact hashes asserted for them.

    Construct via :meth:`from_facts` or :meth:`from_entries` — the bare
    constructor takes already-normalized entries and does not re-validate or
    re-sort them, which lets :meth:`with_key`/:meth:`without_key` stay cheap.
    """

    entries: tuple[TreeEntry, ...]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_facts(cls, facts: Iterable[Fact]) -> Tree:
        """Build a tree from a flat collection of facts, grouped by key."""
        grouped: dict[FactKey, set[str]] = {}
        for fact in facts:
            grouped.setdefault(fact.key, set()).add(fact.hash)
        return cls.from_entries(
            (subject, predicate, hashes)
            for (subject, predicate), hashes in grouped.items()
        )

    @classmethod
    def from_entries(
        cls, entries: Iterable[tuple[str, str, _RawHashes]]
    ) -> Tree:
        """Build a tree from ``(subject, predicate, hash_or_hashes)`` triples.

        Entries sharing a key are merged rather than rejected, so callers do
        not have to pre-group. The result is sorted by ``(subject, predicate)``
        with each key's hashes sorted and deduplicated — sorting here is not
        cosmetic, it is what makes the tree's hash a function of the memory
        state rather than of insertion order, and what lets the diff engine
        line up two trees with a linear merge-join instead of building dicts.
        """
        grouped: dict[FactKey, set[str]] = {}
        for subject, predicate, hash_or_hashes in entries:
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError(f"tree entry subject must be a non-empty str, got {subject!r}")
            if not isinstance(predicate, str) or not predicate.strip():
                raise ValueError(
                    f"tree entry predicate must be a non-empty str, got {predicate!r}"
                )

            hashes = [hash_or_hashes] if isinstance(hash_or_hashes, str) else list(hash_or_hashes)
            if not hashes:
                raise ValueError(f"tree entry for {(subject, predicate)!r} has no fact hashes")
            for h in hashes:
                if not is_object_hash(h):
                    raise ValueError(f"tree entry holds a malformed fact hash: {h!r}")

            grouped.setdefault((subject, predicate), set()).update(hashes)

        built = tuple(
            (subject, predicate, tuple(sorted(hashes)))
            for (subject, predicate), hashes in sorted(grouped.items())
        )
        return cls(built)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Tree:
        """Rebuild a tree from :meth:`to_dict` output.

        Strict about shape: an unknown top-level key or a non-``"tree"`` type
        tag is rejected outright, rather than ignored. A tree object silently
        tolerating unrecognized fields would mean its hash no longer describes
        everything it is filed under — the same round-trip property
        ``Fact.from_dict`` and ``Commit.from_dict`` enforce.
        """
        allowed = {"type", "entries"}
        unknown = payload.keys() - allowed
        if unknown:
            raise ValueError(f"tree payload has unknown keys: {sorted(unknown)}")

        kind = payload.get("type")
        if kind != "tree":
            raise ValueError(f"expected a tree object, got type={kind!r}")

        try:
            raw_entries = payload["entries"]
        except KeyError as exc:
            raise ValueError("tree payload is missing 'entries'") from exc

        entries: list[tuple[str, str, _RawHashes]] = []
        for entry in raw_entries:
            if (
                not isinstance(entry, list)
                or len(entry) != 3
                or not isinstance(entry[2], list)
            ):
                raise ValueError(f"malformed tree entry: {entry!r}")
            subject, predicate, hashes = entry
            entries.append((subject, predicate, hashes))

        return cls.from_entries(entries)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict, entries already in canonical order."""
        return {
            "type": "tree",
            "entries": [
                [subject, predicate, list(hashes)]
                for subject, predicate, hashes in self.entries
            ],
        }

    @property
    def hash(self) -> str:
        """This tree's content address in the object store."""
        return hash_payload(self.to_dict())

    def write(self, store: "ObjectStore") -> str:
        """Store this tree; return its hash."""
        return store.put(self.to_dict())

    @classmethod
    def read(cls, store: "ObjectStore", tree_hash: str) -> Tree:
        """Load and validate the tree stored at ``tree_hash``."""
        return cls.from_dict(store.get(tree_hash))

    def load(self, store: "ObjectStore") -> list[Fact]:
        """Materialize every fact this tree references.

        What ``show`` and ``ls-tree`` build on: reading a tree is cheap (one
        object), but reading the *facts* it names means one lookup per hash.
        """
        return [
            Fact.from_dict(store.get(h))
            for _subject, _predicate, hashes in self.entries
            for h in hashes
        ]

    # -- lookups -------------------------------------------------------------

    def __len__(self) -> int:
        """Number of distinct ``(subject, predicate)`` keys, not fact hashes."""
        return len(self.entries)

    def __iter__(self) -> Iterator[TreeEntry]:
        return iter(self.entries)

    def __contains__(self, key: FactKey) -> bool:
        subject, predicate = key
        return any(s == subject and p == predicate for s, p, _ in self.entries)

    def by_key(self) -> dict[FactKey, tuple[str, ...]]:
        """The multimap the diff engine aligns two trees by.

        One key can map to more than one hash — that is the tree's answer to
        cardinality: it makes no claim about whether a second hash means an
        addition or a contradiction. That call belongs to
        :class:`memgit.core.cardinality.CardinalityMap`, not to storage.
        """
        return {(subject, predicate): hashes for subject, predicate, hashes in self.entries}

    def fact_hashes(self) -> tuple[str, ...]:
        """Every fact hash referenced anywhere in this tree, sorted."""
        return tuple(sorted({h for _s, _p, hashes in self.entries for h in hashes}))

    def subjects(self) -> tuple[str, ...]:
        """Every distinct subject referenced in this tree, sorted."""
        return tuple(sorted({subject for subject, _p, _h in self.entries}))

    # -- single-key mutation ---------------------------------------------

    def with_key(self, key: FactKey, hashes: _RawHashes) -> Tree:
        """Return a copy of this tree with ``key`` set to ``hashes``.

        Slice 6's ablation engine needs exactly this: branch from a commit,
        replace or drop one belief, commit again. Building the new tree by
        filtering and re-adding a single entry keeps that a one-line
        operation rather than a full ``from_facts`` rebuild.
        """
        remaining = [e for e in self.entries if (e[0], e[1]) != key]
        subject, predicate = key
        return Tree.from_entries([*remaining, (subject, predicate, hashes)])

    def without_key(self, key: FactKey) -> Tree:
        """Return a copy of this tree with ``key`` removed entirely."""
        remaining = tuple(e for e in self.entries if (e[0], e[1]) != key)
        return Tree(remaining)

    def __repr__(self) -> str:
        return f"Tree({len(self)} key(s))"


# The tree of a memory state with no facts in it at all — "believe nothing".
# Used as the diff baseline for a root commit (Repository.diff) and will be
# slice 4's full-rewind baseline too — one constant rather than each site
# re-deriving Tree(()).hash. Note that no caller writes this object to the
# store; Repository.read_tree special-cases the hash instead.
EMPTY_TREE_HASH = Tree(()).hash
