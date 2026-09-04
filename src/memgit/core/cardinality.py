"""The cardinality map — the diff engine's single/multi schema.

``tree.py`` is deliberately agnostic about what a second fact hash at one
``(subject, predicate)`` key means: an agent can believe exactly one thing
about ``favorite_editor`` but several things at once about ``likes``, and
storage should not have to guess which. This module is where that guess
finally gets made — a lens the diff engine (``diff.py``) applies when it
decides whether a second value at a key is an addition or a contradiction.

**This map is repo-local config, not a tracked object.** A diff is a question
you ask, not data you store: declaring ``likes`` ``multi`` and getting a
different classification for yesterday's commits means you asked a better
question, not that history changed underneath you. Committing the map instead
would force an unanswerable choice — whose declaration wins when diffing two
commits made under different schemas? There is no principled answer, so the
map lives at ``.memgit/cardinality.json``, outside the object store, applied
uniformly to both sides of every diff.

The honest cost of that choice: an archived diff's classifications are not
reproducible from the object store alone, since the map that produced them is
not itself an object. The mitigation is that every diff embeds the map it
used (see ``Diff.to_dict``), so nothing is silently lost — it just travels
with the result rather than living in history.

The default cardinality is ``single``, not ``multi``, because this is a
debugging tool and should default toward the loud classification: a second
value at an undeclared predicate reads as ``contradicted`` unless a human
says otherwise, rather than silently coexisting as an unremarkable addition.
False positives cost one declaration; false negatives are the exact bug class
this project exists to catch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

__all__ = ["Cardinality", "CardinalityMap"]

Cardinality = Literal["single", "multi"]

_VALID_CARDINALITIES = frozenset({"single", "multi"})
_FIELDS = frozenset({"version", "default", "predicates"})
_VERSION = 1


@dataclass(frozen=True, slots=True)
class CardinalityMap:
    """Predicate -> ``single``/``multi``, with a default for anything undeclared.

    Keyed on predicate alone, not ``(subject, predicate)`` — ``likes`` means
    the same thing whether the subject is ``user`` or ``project:memgit``.
    Subject-scoped cardinality is a schema feature with no caller yet; the
    ``version`` field is the escape hatch for adding it later without a
    silent format change.
    """

    predicates: Mapping[str, Cardinality] = field(default_factory=lambda: MappingProxyType({}))
    default: Cardinality = "single"

    def __post_init__(self) -> None:
        if self.default not in _VALID_CARDINALITIES:
            raise ValueError(f"CardinalityMap.default must be 'single' or 'multi', got {self.default!r}")
        for predicate, cardinality in self.predicates.items():
            if not isinstance(predicate, str) or not predicate.strip():
                raise ValueError(f"cardinality predicate must be a non-empty str, got {predicate!r}")
            if cardinality not in _VALID_CARDINALITIES:
                raise ValueError(
                    f"cardinality for {predicate!r} must be 'single' or 'multi', got {cardinality!r}"
                )
        object.__setattr__(self, "predicates", MappingProxyType(dict(self.predicates)))

    # -- lookups -------------------------------------------------------------

    def __getitem__(self, predicate: str) -> Cardinality:
        return self.predicates.get(predicate, self.default)

    def is_multi(self, predicate: str) -> bool:
        """True if ``predicate`` is declared (or defaulted to) ``"multi"``."""
        return self[predicate] == "multi"

    def declared(self) -> tuple[tuple[str, Cardinality], ...]:
        """Every explicitly declared predicate, sorted."""
        return tuple(sorted(self.predicates.items()))

    # -- derivation ------------------------------------------------------

    def with_predicate(self, predicate: str, cardinality: Cardinality) -> CardinalityMap:
        """Return a copy declaring ``predicate`` as ``cardinality``."""
        updated = dict(self.predicates)
        updated[predicate] = cardinality
        return CardinalityMap(updated, default=self.default)

    def without_predicate(self, predicate: str) -> CardinalityMap:
        """Return a copy with ``predicate``'s declaration removed, if any."""
        updated = dict(self.predicates)
        updated.pop(predicate, None)
        return CardinalityMap(updated, default=self.default)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-ready shape :meth:`from_dict` expects back."""
        return {
            "version": _VERSION,
            "default": self.default,
            "predicates": dict(self.declared()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CardinalityMap:
        """Rebuild a map from :meth:`to_dict` output.

        Strict about shape, matching the rest of the codebase: an unknown
        top-level key, an unrecognized ``version``, or a cardinality value
        that isn't ``single``/``multi`` is rejected rather than tolerated —
        this is config a human edits by hand, so a clear error at load time
        beats a confusing misclassification three commands later.
        """
        unknown = payload.keys() - _FIELDS
        if unknown:
            raise ValueError(f"cardinality map has unknown keys: {sorted(unknown)}")

        version = payload.get("version", _VERSION)
        if version != _VERSION:
            raise ValueError(f"unsupported cardinality map version: {version!r}")

        return cls(
            predicates=payload.get("predicates", {}),
            default=payload.get("default", "single"),
        )

    @classmethod
    def default_map(cls) -> CardinalityMap:
        """The map in effect when ``.memgit/cardinality.json`` does not exist."""
        return cls()

    def __repr__(self) -> str:
        return f"CardinalityMap(default={self.default!r}, {len(self.predicates)} declared)"
