"""MemoryState — every belief an agent holds at one point in time, materialized.

Where a :class:`~memgit.core.tree.Tree` is the *storable* shape of a memory
state (hashes, sorted, content-addressed), a ``MemoryState`` is the shape
something actually *uses* one in: real :class:`~memgit.core.fact.Fact`
objects, indexed for lookup, with the provenance of which commit (if any)
they came from. ``Tree.load(store)`` already returns the facts as a bare
``list``; this module exists because slice 5 (the agent runtime) and slice 6
(replay/ablation) both need more than that list, and each needs something
slightly different from it — a lookup by subject/predicate, an ablated copy
with one belief removed, and a rendered context block to hand an LLM. Putting
any of that on ``Tree`` would drag presentation and retrieval concerns into
the storage object, which ``tree.py``'s own docstring is explicit about
avoiding.

**A ``MemoryState`` is never written to the object store and has no hash of
its own.** The storable form of a memory state is a ``Tree``; :meth:`to_tree`
is the one-way door back into storage. Treating a ``MemoryState`` as
content-addressed would be a second, competing notion of identity for the
same bytes.

**Any derived state drops its ``commit`` provenance.** :meth:`filter`,
:meth:`without`, and :meth:`without_fact` all set ``commit=None`` on the
result, even though most of the original facts are unchanged. A filtered
state is not *the* state at that commit — pretending otherwise is exactly the
leak PLAN.md's retrieval-layer decision promises to avoid ("never leak facts
from a later commit or a different branch"): once a caller has started
slicing a state up, provenance can no longer vouch for what's left.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Collection, Iterable, Iterator

from memgit.core.fact import Fact, FactKey
from memgit.core.tree import Tree

if TYPE_CHECKING:
    from memgit.core.diff import FactReader

__all__ = ["MemoryState"]


@dataclass(frozen=True, slots=True)
class MemoryState:
    """A materialized, indexed memory state.

    Construct via :meth:`from_tree`, :meth:`from_facts`, or :meth:`empty` —
    the bare constructor takes already-built ``facts`` and does not
    deduplicate or validate them.

    Attributes:
        facts: Every fact in this state, in no particular guaranteed order.
        tree: The hash of the :class:`~memgit.core.tree.Tree` this state's
            facts would build (``EMPTY_TREE_HASH`` for no facts at all).
        commit: Which commit this came from, or ``None`` if this state was
            derived (filtered, ablated) and its facts no longer describe a
            real point in history.
    """

    facts: tuple[Fact, ...]
    tree: str
    commit: str | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_tree(
        cls, tree: Tree, read_fact: "FactReader", *, commit: str | None = None
    ) -> "MemoryState":
        """Materialize every fact ``tree`` references.

        Args:
            tree: The tree to load.
            read_fact: Loads a :class:`Fact` given its hash — the same
                ``FactReader`` shape ``diff.py`` uses.
            commit: The commit this tree came from, if any.
        """
        return cls(facts=tuple(read_fact(h) for h in tree.fact_hashes()), tree=tree.hash, commit=commit)

    @classmethod
    def from_facts(cls, facts: Iterable[Fact], *, commit: str | None = None) -> "MemoryState":
        """Build a state directly from facts, e.g. ones not yet committed."""
        fact_tuple = tuple(facts)
        return cls(facts=fact_tuple, tree=Tree.from_facts(fact_tuple).hash, commit=commit)

    @classmethod
    def empty(cls) -> "MemoryState":
        """The state of believing nothing — no facts, no commit."""
        return cls(facts=(), tree=Tree(()).hash, commit=None)

    # -- collection ------------------------------------------------------

    def __len__(self) -> int:
        """Number of facts. Differs from ``Tree.__len__``, which counts keys."""
        return len(self.facts)

    def __iter__(self) -> Iterator[Fact]:
        return iter(self.facts)

    def __contains__(self, key: FactKey) -> bool:
        return any(fact.key == key for fact in self.facts)

    def keys(self) -> tuple[FactKey, ...]:
        """Every distinct ``(subject, predicate)`` key, sorted."""
        return tuple(sorted({fact.key for fact in self.facts}))

    def subjects(self) -> tuple[str, ...]:
        """Every distinct subject, sorted."""
        return tuple(sorted({fact.subject for fact in self.facts}))

    # -- lookup ------------------------------------------------------------

    def get(self, subject: str, predicate: str) -> tuple[Fact, ...]:
        """Every fact at this key — plural, since a key can hold more than one
        belief (see ``tree.py``'s cardinality discussion)."""
        return tuple(fact for fact in self.facts if fact.subject == subject and fact.predicate == predicate)

    def one(self, subject: str, predicate: str) -> Fact | None:
        """The single most-trustworthy fact at this key, or ``None``.

        For a caller that wants one answer regardless of cardinality (the
        common case for rendering a prompt): highest ``confidence`` wins,
        ties broken by the most recent ``asserted_at``, then by hash — the
        last purely so the choice is deterministic rather than dependent on
        iteration order.
        """
        candidates = self.get(subject, predicate)
        if not candidates:
            return None
        return max(candidates, key=lambda f: (f.confidence, f.asserted_at, f.hash))

    def by_subject(self, subject: str) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.subject == subject)

    def by_predicate(self, predicate: str) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.predicate == predicate)

    # -- derivation --------------------------------------------------------

    def filter(
        self,
        *,
        min_confidence: float = 0.0,
        subjects: Collection[str] | None = None,
        predicates: Collection[str] | None = None,
    ) -> "MemoryState":
        """Return a new state holding only facts that pass every given test.

        Slice 7's confidence-decay ranking hangs off ``min_confidence``; the
        decay math itself is not this method's job, only the cutoff.
        """
        kept = tuple(
            fact
            for fact in self.facts
            if fact.confidence >= min_confidence
            and (subjects is None or fact.subject in subjects)
            and (predicates is None or fact.predicate in predicates)
        )
        return MemoryState.from_facts(kept)

    def without(self, key: FactKey) -> "MemoryState":
        """Return a new state with every fact at ``key`` removed.

        What slice 6's ablation engine calls to drop one belief entirely
        before replaying a query.
        """
        subject, predicate = key
        kept = tuple(fact for fact in self.facts if fact.key != (subject, predicate))
        return MemoryState.from_facts(kept)

    def without_fact(self, fact_hash: str) -> "MemoryState":
        """Return a new state with the single fact ``fact_hash`` removed.

        The finer-grained sibling of :meth:`without`: drops one value at a
        multi-valued key rather than the whole key.
        """
        kept = tuple(fact for fact in self.facts if fact.hash != fact_hash)
        return MemoryState.from_facts(kept)

    # -- boundaries ----------------------------------------------------------

    def to_tree(self) -> Tree:
        """Rebuild this state's storable form.

        Round-trips: ``MemoryState.from_tree(t, read).to_tree().hash == t.hash``
        for any tree ``t``, since both build the same grouping from the same
        facts.
        """
        return Tree.from_facts(self.facts)

    def render(self, *, max_facts: int | None = None) -> str:
        """A deterministic, human- and prompt-readable rendering of this state.

        One default shape — grouped by subject, one line per fact, confidence
        shown only when it is below 1.0 (a fact held with full confidence
        does not need its number repeated everywhere). This is the context
        block slice 5 injects verbatim; refining how it reads for a specific
        model is that slice's job, not this one's.
        """
        by_subject: dict[str, list[Fact]] = defaultdict(list)
        for fact in self.facts:
            by_subject[fact.subject].append(fact)

        lines: list[str] = []
        count = 0
        for subject in sorted(by_subject):
            lines.append(f"{subject}:")
            for fact in sorted(by_subject[subject], key=lambda f: (f.predicate, f.object)):
                if max_facts is not None and count >= max_facts:
                    return "\n".join(lines)
                suffix = "" if fact.confidence >= 1.0 else f" ({fact.confidence:.2f})"
                lines.append(f"  {fact.predicate} {fact.object}{suffix}")
                count += 1
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        No ``"type"`` tag: unlike ``Fact``/``Tree``/``Commit``, this is not a
        content-addressed object, so there is no round-trip-through-a-hash
        property to protect.
        """
        payload: dict[str, Any] = {
            "tree": self.tree,
            "facts": [fact.to_dict() for fact in self.facts],
        }
        if self.commit is not None:
            payload["commit"] = self.commit
        return payload

    def __repr__(self) -> str:
        return f"MemoryState({len(self.facts)} fact(s), tree={self.tree[:8]}, commit={self.commit!r})"
