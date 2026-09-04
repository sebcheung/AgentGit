"""The diff engine — comparing two memory states.

Everything here is written against plain :class:`~memgit.core.tree.Tree`
objects and a ``FactReader`` callable, exactly the pattern ``graph.py`` uses
for ``CommitReader``: the algorithm never touches a filesystem, so it stays
testable against synthetic dicts. ``Repository`` is the only thing that knows
a ``FactReader`` means "look this hash up in ``.memgit/objects``".

Two diffs live in this module, at different costs:

``diff_hashes``
    A cheap, hash-only comparison. It can say a key was added, removed, or
    "something changed" — but it cannot tell a contradiction from a
    reaffirmation, because that distinction depends on ``Fact.object``, not
    on the hash. No fact is ever read to produce it.

``diff_trees``
    The full semantic diff this project's tagline promises. It answers
    `PLAN.md`'s question — "what counts as changed vs. contradicted vs. an
    unrelated new fact?" — using the taxonomy below.

Both are built on one merge-join over the two trees' sorted entries
(``_merge_join``), so there is exactly one way keys get aligned, not two
implementations that can quietly drift apart.

## The change taxonomy

A key's hash list can change in more than one way at once — ``knows`` can
gain ``rust``, lose ``haskell``, and reaffirm ``python`` in the same commit.
Collapsing that to a single label loses information a reader would want; not
labelling it at all makes the CLI unprintable. So the authoritative content
of a key's change is a list of per-*value* changes (:class:`ValueChange`,
aligned by ``Fact.object``), and the key-level :class:`ChangeKind` is a
derived summary with a total, ordered collapse rule — see
:func:`_classify_key` for the exact rule and the reasoning behind each case.

## Duplicate objects at one key

``Tree.from_facts`` dedupes fact *hashes*, not objects: two facts can
legitimately share a key and an object, differing only in ``asserted_at`` or
``confidence`` (a plain reassertion). Aligning by object then hits a
collision on one side. The representative is the one with the latest
``asserted_at``, tie-broken by higher ``confidence``, tie-broken by hash —
deterministic, not directional. The loser is preserved on
``KeyDiff.shadowed_before``/``shadowed_after`` rather than dropped, and takes
no part in classification. This is not a cardinality violation: the same
object twice is a redundant reassertion, not two rival beliefs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from memgit.core.cardinality import Cardinality, CardinalityMap
from memgit.core.fact import Fact, FactKey
from memgit.core.tree import Tree

__all__ = [
    "CardinalityViolation",
    "ChangeKind",
    "Diff",
    "DiffStat",
    "FactReader",
    "HashDiff",
    "HashKeyChange",
    "HashKeyEntry",
    "KeyDiff",
    "ValueChange",
    "ValueKind",
    "diff_hashes",
    "diff_trees",
]

FactReader = Callable[[str], Fact]


# ---------------------------------------------------------------------------
# The merge-join
# ---------------------------------------------------------------------------


def _merge_join(
    before: Tree, after: Tree
) -> Iterator[tuple[FactKey, tuple[str, ...] | None, tuple[str, ...] | None]]:
    r"""Yield every key in either tree, exactly once, in sorted order.

    Linear in ``len(before) + len(after)``: one pass over both trees' already
    -sorted entries, no dict built, no per-key allocation beyond the yielded
    tuple. This is the merge-join ``tree.py`` promises the diff engine.

    The comparison ``ka < kb`` below must be the identical ordering
    ``Tree.from_entries`` used (``sorted(grouped.items())`` over
    ``(subject, predicate)`` tuples) — it is, since both are plain Python
    tuple ordering over the same pair. That equivalence is not cosmetic: a
    concatenated ``f"{subject}\\t{predicate}"`` comparison would order
    ``("a", "b/c")`` and ``("a/b", "c")`` differently and silently mis-align
    two trees at that key.
    """
    a, b = before.entries, after.entries
    i = j = 0
    while i < len(a) and j < len(b):
        ka = (a[i][0], a[i][1])
        kb = (b[j][0], b[j][1])
        if ka == kb:
            yield ka, a[i][2], b[j][2]
            i += 1
            j += 1
        elif ka < kb:
            yield ka, a[i][2], None
            i += 1
        else:
            yield kb, None, b[j][2]
            j += 1
    while i < len(a):
        yield (a[i][0], a[i][1]), a[i][2], None
        i += 1
    while j < len(b):
        yield (b[j][0], b[j][1]), None, b[j][2]
        j += 1


# ---------------------------------------------------------------------------
# Hash-only diff
# ---------------------------------------------------------------------------


class HashKeyChange(StrEnum):
    """What changed at a key, without reading any fact — see module docstring."""

    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


HashKeyEntry = tuple[FactKey, HashKeyChange, tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class HashDiff:
    """The cheap comparison: which keys differ, without saying why.

    Powers callers that don't need the semantic detail: ``--name-only``, a
    dashboard's "how many keys did this commit touch" over hundreds of
    commits, and slice 4's checkout deciding what to materialize.
    """

    entries: tuple[HashKeyEntry, ...]

    def changed_keys(self) -> tuple[FactKey, ...]:
        """Every key whose hash set is not identical on both sides."""
        return tuple(key for key, kind, _b, _a in self.entries if kind != HashKeyChange.UNCHANGED)

    def __iter__(self) -> Iterator[HashKeyEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def diff_hashes(before: Tree, after: Tree) -> HashDiff:
    """Compare two trees by hash alone. Reads no facts."""
    entries: list[HashKeyEntry] = []
    for key, before_hashes, after_hashes in _merge_join(before, after):
        b = before_hashes or ()
        a = after_hashes or ()
        if before_hashes is None:
            kind = HashKeyChange.ADDED
        elif after_hashes is None:
            kind = HashKeyChange.REMOVED
        elif b == a:
            kind = HashKeyChange.UNCHANGED
        else:
            kind = HashKeyChange.CHANGED
        entries.append((key, kind, b, a))
    return HashDiff(tuple(entries))


# ---------------------------------------------------------------------------
# Semantic diff
# ---------------------------------------------------------------------------


class ValueKind(StrEnum):
    """What happened to one distinct object at a key — see module docstring."""

    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    REAFFIRMED = "reaffirmed"


@dataclass(frozen=True, slots=True)
class ValueChange:
    """One object's fate at a key: present, gone, new, or re-asserted."""

    kind: ValueKind
    object: str
    before: Fact | None
    after: Fact | None

    @property
    def confidence_delta(self) -> float | None:
        """``after.confidence - before.confidence``, or ``None`` unless both sides exist."""
        if self.before is None or self.after is None:
            return None
        return self.after.confidence - self.before.confidence

    @property
    def is_strengthened(self) -> bool:
        """True if a reaffirmation raised confidence."""
        delta = self.confidence_delta
        return delta is not None and delta > 0

    @property
    def is_weakened(self) -> bool:
        """True if a reaffirmation lowered confidence."""
        delta = self.confidence_delta
        return delta is not None and delta < 0

    def __str__(self) -> str:
        if self.kind is ValueKind.ADDED:
            return f"+{self.object}"
        if self.kind is ValueKind.REMOVED:
            return f"-{self.object}"
        if self.kind is ValueKind.REAFFIRMED:
            assert self.before is not None and self.after is not None
            if self.before.confidence != self.after.confidence:
                return f"{self.object} ({self.before.confidence:.2f} -> {self.after.confidence:.2f})"
            return self.object
        return self.object


class ChangeKind(StrEnum):
    """The key-level summary of a :class:`KeyDiff`.

    See the module docstring for the ordered collapse rule that derives this
    from ``values``.
    """

    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    REAFFIRMED = "reaffirmed"
    CONTRADICTED = "contradicted"
    VALUE_ADDED = "value_added"
    VALUE_REMOVED = "value_removed"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class CardinalityViolation:
    """A ``single``-declared key held more than one distinct object on one side.

    Reported, never raised — storage is deliberately agnostic about
    cardinality (see ``tree.py``), so a tree carrying a violation is not
    malformed, and refusing to diff it would make real commits undiffable.
    """

    key: FactKey
    side: Literal["before", "after"]
    objects: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-ready shape the CLI and API report."""
        return {"key": list(self.key), "side": self.side, "objects": list(self.objects)}


@dataclass(frozen=True, slots=True)
class KeyDiff:
    """Everything that changed at one ``(subject, predicate)`` key."""

    key: FactKey
    kind: ChangeKind
    cardinality: Cardinality
    values: tuple[ValueChange, ...] = ()
    before_hashes: tuple[str, ...] = ()
    after_hashes: tuple[str, ...] = ()
    shadowed_before: tuple[Fact, ...] = ()
    shadowed_after: tuple[Fact, ...] = ()

    @property
    def subject(self) -> str:
        """The first half of :attr:`key`."""
        return self.key[0]

    @property
    def predicate(self) -> str:
        """The second half of :attr:`key`."""
        return self.key[1]

    @property
    def changed_values(self) -> tuple[ValueChange, ...]:
        """:attr:`values`, minus the ones that carried over unchanged."""
        return tuple(v for v in self.values if v.kind != ValueKind.UNCHANGED)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-ready shape the CLI and API report."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "kind": str(self.kind),
            "cardinality": self.cardinality,
            "before_hashes": list(self.before_hashes),
            "after_hashes": list(self.after_hashes),
            "values": [
                {
                    "kind": str(v.kind),
                    "object": v.object,
                    "before": v.before.to_dict() if v.before is not None else None,
                    "after": v.after.to_dict() if v.after is not None else None,
                }
                for v in self.values
            ],
        }

    def __str__(self) -> str:
        detail = ", ".join(str(v) for v in self.changed_values)
        return f"{self.subject} {self.predicate} {detail}"


@dataclass(frozen=True, slots=True)
class DiffStat:
    """Aggregate counts over a :class:`Diff` — what ``--stat`` prints."""

    keys_changed: int
    by_kind: Mapping[ChangeKind, int]
    facts_added: int
    facts_removed: int
    facts_reaffirmed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-ready shape the CLI and API report."""
        return {
            "keys_changed": self.keys_changed,
            "by_kind": {str(k): v for k, v in self.by_kind.items()},
            "facts_added": self.facts_added,
            "facts_removed": self.facts_removed,
            "facts_reaffirmed": self.facts_reaffirmed,
        }


@dataclass(frozen=True, slots=True)
class Diff:
    """The result of comparing two memory states.

    Sorted by key for free — the merge-join never needs a separate sort.
    Deliberately has no ``from_dict``: this is a report, not a stored object,
    and adding a parser would imply a round-trip guarantee ``to_dict`` was
    never designed to satisfy.
    """

    keys: tuple[KeyDiff, ...]
    violations: tuple[CardinalityViolation, ...]
    cardinality: CardinalityMap

    @property
    def is_empty(self) -> bool:
        """True if no key differs between the two states."""
        return len(self.keys) == 0

    def by_kind(self) -> dict[ChangeKind, tuple[KeyDiff, ...]]:
        """Group :attr:`keys` by their :class:`ChangeKind`."""
        result: dict[ChangeKind, list[KeyDiff]] = {}
        for kd in self.keys:
            result.setdefault(kd.kind, []).append(kd)
        return {kind: tuple(kds) for kind, kds in result.items()}

    def stat(self) -> DiffStat:
        """Aggregate this diff into the counts ``--stat`` prints."""
        by_kind_counts: dict[ChangeKind, int] = {}
        facts_added = facts_removed = facts_reaffirmed = 0
        for kd in self.keys:
            by_kind_counts[kd.kind] = by_kind_counts.get(kd.kind, 0) + 1
            for v in kd.values:
                if v.kind is ValueKind.ADDED:
                    facts_added += 1
                elif v.kind is ValueKind.REMOVED:
                    facts_removed += 1
                elif v.kind is ValueKind.REAFFIRMED:
                    facts_reaffirmed += 1
        return DiffStat(
            keys_changed=len(self.keys),
            by_kind=by_kind_counts,
            facts_added=facts_added,
            facts_removed=facts_removed,
            facts_reaffirmed=facts_reaffirmed,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-ready shape the CLI and API report."""
        return {
            "type": "diff",
            "cardinality": self.cardinality.to_dict(),
            "stat": self.stat().to_dict(),
            "keys": [kd.to_dict() for kd in self.keys],
            "violations": [v.to_dict() for v in self.violations],
        }

    def __iter__(self) -> Iterator[KeyDiff]:
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)


def _representative(hashes: tuple[str, ...], read_fact: FactReader) -> tuple[dict[str, Fact], dict[str, list[Fact]]]:
    """Group facts at a key by object; pick a representative per object.

    Returns ``(representatives, shadowed)`` where ``shadowed`` holds every
    non-representative fact sharing an object with one that won. The
    representative is the fact with the latest ``asserted_at``, tie-broken by
    higher ``confidence``, tie-broken by hash — see the module docstring.
    """
    by_object: dict[str, list[Fact]] = {}
    for h in hashes:
        fact = read_fact(h)
        by_object.setdefault(fact.object, []).append(fact)

    representatives: dict[str, Fact] = {}
    shadowed: dict[str, list[Fact]] = {}
    for obj, facts in by_object.items():
        ordered = sorted(facts, key=lambda f: (f.asserted_at, f.confidence, f.hash))
        representatives[obj] = ordered[-1]
        if len(ordered) > 1:
            shadowed[obj] = ordered[:-1]
    return representatives, shadowed


def _classify_key(
    key: FactKey,
    before_hashes: tuple[str, ...] | None,
    after_hashes: tuple[str, ...] | None,
    read_fact: FactReader,
    cardinality_map: CardinalityMap,
) -> tuple[KeyDiff, tuple[CardinalityViolation, ...]]:
    predicate = key[1]
    cardinality = cardinality_map[predicate]

    if before_hashes is None:
        assert after_hashes is not None
        after_reps, after_shadowed = _representative(after_hashes, read_fact)
        values = tuple(
            ValueChange(ValueKind.ADDED, obj, None, fact) for obj, fact in sorted(after_reps.items())
        )
        violations = _violations(key, None, after_reps)
        return (
            KeyDiff(
                key=key,
                kind=ChangeKind.ADDED,
                cardinality=cardinality,
                values=values,
                before_hashes=(),
                after_hashes=after_hashes,
                shadowed_after=tuple(f for facts in after_shadowed.values() for f in facts),
            ),
            violations,
        )

    if after_hashes is None:
        before_reps, before_shadowed = _representative(before_hashes, read_fact)
        values = tuple(
            ValueChange(ValueKind.REMOVED, obj, fact, None) for obj, fact in sorted(before_reps.items())
        )
        violations = _violations(key, before_reps, None)
        return (
            KeyDiff(
                key=key,
                kind=ChangeKind.REMOVED,
                cardinality=cardinality,
                values=values,
                before_hashes=before_hashes,
                after_hashes=(),
                shadowed_before=tuple(f for facts in before_shadowed.values() for f in facts),
            ),
            violations,
        )

    if before_hashes == after_hashes:
        return (
            KeyDiff(
                key=key,
                kind=ChangeKind.UNCHANGED,
                cardinality=cardinality,
                before_hashes=before_hashes,
                after_hashes=after_hashes,
            ),
            (),
        )

    before_reps, before_shadowed = _representative(before_hashes, read_fact)
    after_reps, after_shadowed = _representative(after_hashes, read_fact)

    objects = sorted(set(before_reps) | set(after_reps))
    values: list[ValueChange] = []
    n_add = n_rem = n_reaff = 0
    for obj in objects:
        b = before_reps.get(obj)
        a = after_reps.get(obj)
        if b is None:
            values.append(ValueChange(ValueKind.ADDED, obj, None, a))
            n_add += 1
        elif a is None:
            values.append(ValueChange(ValueKind.REMOVED, obj, b, None))
            n_rem += 1
        elif b.hash == a.hash:
            values.append(ValueChange(ValueKind.UNCHANGED, obj, b, a))
        else:
            values.append(ValueChange(ValueKind.REAFFIRMED, obj, b, a))
            n_reaff += 1

    if n_add == n_rem == n_reaff == 0:
        kind = ChangeKind.UNCHANGED
    elif cardinality == "single":
        if n_add >= 1:
            kind = ChangeKind.CONTRADICTED
        elif n_rem >= 1:
            kind = ChangeKind.VALUE_REMOVED
        else:
            kind = ChangeKind.REAFFIRMED
    else:
        if n_add >= 1 and n_rem >= 1:
            kind = ChangeKind.MIXED
        elif n_add >= 1:
            kind = ChangeKind.VALUE_ADDED
        elif n_rem >= 1:
            kind = ChangeKind.VALUE_REMOVED
        else:
            kind = ChangeKind.REAFFIRMED

    violations = _violations(key, before_reps, after_reps)
    return (
        KeyDiff(
            key=key,
            kind=kind,
            cardinality=cardinality,
            values=tuple(values),
            before_hashes=before_hashes,
            after_hashes=after_hashes,
            shadowed_before=tuple(f for facts in before_shadowed.values() for f in facts),
            shadowed_after=tuple(f for facts in after_shadowed.values() for f in facts),
        ),
        violations,
    )


def _violations(
    key: FactKey,
    before_reps: dict[str, Fact] | None,
    after_reps: dict[str, Fact] | None,
) -> tuple[CardinalityViolation, ...]:
    """Check both sides of one key for more than one distinct object.

    Only meaningful for ``single``-declared keys, but checked unconditionally
    here; the caller (``diff_trees``) only keeps these when the key is
    actually ``single``.
    """
    violations: list[CardinalityViolation] = []
    if before_reps is not None and len(before_reps) > 1:
        violations.append(CardinalityViolation(key, "before", tuple(sorted(before_reps))))
    if after_reps is not None and len(after_reps) > 1:
        violations.append(CardinalityViolation(key, "after", tuple(sorted(after_reps))))
    return tuple(violations)


def diff_trees(
    before: Tree,
    after: Tree,
    read_fact: FactReader,
    *,
    cardinality: CardinalityMap | None = None,
    include_unchanged: bool = False,
) -> Diff:
    """Compute the semantic diff between two memory states.

    ``read_fact`` is invoked only for keys whose hash tuples actually differ
    — an unchanged key costs zero fact reads, which is what makes diffing two
    large, mostly-identical trees cheap.
    """
    cardinality_map = cardinality if cardinality is not None else CardinalityMap.default_map()

    keys: list[KeyDiff] = []
    violations: list[CardinalityViolation] = []
    for key, before_hashes, after_hashes in _merge_join(before, after):
        key_diff, key_violations = _classify_key(key, before_hashes, after_hashes, read_fact, cardinality_map)
        if key_diff.cardinality == "single":
            violations.extend(key_violations)
        if include_unchanged or key_diff.kind != ChangeKind.UNCHANGED:
            keys.append(key_diff)

    return Diff(keys=tuple(keys), violations=tuple(violations), cardinality=cardinality_map)
