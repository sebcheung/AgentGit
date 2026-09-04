"""The decay policy — how a fact's confidence fades between commits.

A belief recorded once and never revisited should not be trusted forever: a
fact's stored ``confidence`` is a snapshot of how sure the agent was *at
``asserted_at``*, and staleness itself is signal a debugging tool should
surface. This module computes that staleness; it never stores it.

**Decay is a read-time lens, exactly like** :class:`~memgit.core.cardinality.CardinalityMap`
**— repo-local config, not committed, not hashed.** Declaring ``timezone`` a
30-day half-life and re-running a query against last month's commit changes
the ranking, not history: you asked a better question, the facts underneath
did not move. Committing the policy would force the same unanswerable
"whose declaration wins" question ``cardinality.py`` already refuses to
answer for its own map.

**Decay is computed, never stored.** ``confidence`` sits inside a
:class:`~memgit.core.fact.Fact`'s hashed payload
(:meth:`~memgit.core.fact.Fact.to_dict`) by design — writing a decayed value
back would mint a new fact hash, which can only be recorded as a commit, and
that commit would fabricate a belief the agent never actually asserted. So
the only two places decayed confidence may appear are a retrieval ranker and
an explicit ``--as-of`` *display* flag. ``memgit diff`` and
:meth:`~memgit.core.repository.Repository.read_fact` never see it: the diff
engine's ``reaffirmed``/``unchanged`` classification is decided purely by
fact hash (see ``diff.py``), so decay cannot make that ambiguous, but a
confidence-strength comparison (``is_strengthened``/``is_weakened``) *would*
go wrong if fed a decayed number on only one side of a diff.

**Reaffirmation resets decay for free.** :meth:`~memgit.core.fact.Fact.reaffirm`
bumps ``asserted_at``, and the agent runtime's merge step already replaces a
same-triple fact with the newly-asserted one — decay reads ``asserted_at``,
so a fresh assertion is simply young again. No new machinery, no explicit
"reset" method: this is a consequence of facts being immutable values, not a
feature built on top of them.

``as_of`` is always an explicit argument, never read from the wall clock
inside this module. That is what keeps every decay and retrieval-ranking
test deterministic, and what keeps a replay reproducible on a later day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping

from memgit.core.fact import Fact

__all__ = ["DecayPolicy"]

_FIELDS = frozenset({"version", "default", "half_lives", "floor"})
_VERSION = 1
_DEFAULT_HALF_LIFE_DAYS = 180.0


@dataclass(frozen=True, slots=True)
class DecayPolicy:
    """Predicate -> half-life in days, with a default for anything undeclared.

    Keyed on predicate alone, matching :class:`~memgit.core.cardinality.CardinalityMap`'s
    reasoning: ``prefers_language`` decays the same way whichever subject
    holds it. ``None`` for a half-life (default or per-predicate) means the
    predicate never decays — appropriate for something like ``born_in`` that
    is true forever once asserted.

    Attributes:
        half_lives: Predicate -> half-life in days, or ``None`` for "never
            decays".
        default: The half-life used for any predicate not in ``half_lives``.
            Defaults to 180 days — decay on by default, following
            ``cardinality.json``'s "default toward the loud classification":
            a debugging tool should default to surfacing staleness. This is
            safe because decay only ever reweights ranking; it never filters
            or hides a fact.
        floor: The minimum decayed confidence a fact can reach. Zero by
            default — a fact can decay all the way to "essentially
            forgotten" rather than being propped up at some minimum trust.
    """

    half_lives: Mapping[str, float | None] = field(default_factory=lambda: MappingProxyType({}))
    default: float | None = _DEFAULT_HALF_LIFE_DAYS
    floor: float = 0.0

    def __post_init__(self) -> None:
        if self.default is not None and self.default <= 0:
            raise ValueError(f"DecayPolicy.default must be positive or None, got {self.default!r}")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError(f"DecayPolicy.floor must be within [0.0, 1.0], got {self.floor!r}")
        for predicate, half_life in self.half_lives.items():
            if not isinstance(predicate, str) or not predicate.strip():
                raise ValueError(f"decay predicate must be a non-empty str, got {predicate!r}")
            if half_life is not None and half_life <= 0:
                raise ValueError(
                    f"half-life for {predicate!r} must be positive or None, got {half_life!r}"
                )
        object.__setattr__(self, "half_lives", MappingProxyType(dict(self.half_lives)))

    # -- lookups -------------------------------------------------------------

    def half_life(self, predicate: str) -> float | None:
        """The half-life in days for ``predicate``, or ``None`` if it never decays."""
        return self.half_lives.get(predicate, self.default)

    def declared(self) -> tuple[tuple[str, float | None], ...]:
        """Every explicitly declared predicate, sorted."""
        return tuple(sorted(self.half_lives.items()))

    # -- decay math ------------------------------------------------------

    def age_days(self, fact: Fact, *, as_of: datetime) -> float:
        """How old ``fact`` is at ``as_of``, in days, clamped to zero.

        A negative age (clock skew, or a fact whose ``asserted_at`` is
        unparseable) clamps to zero rather than propagating — the same
        clock-skew posture ``graph.walk`` already takes, and it keeps
        :meth:`decayed` from ever needing to reject a malformed timestamp.
        """
        try:
            asserted = datetime.fromisoformat(fact.asserted_at)
        except ValueError:
            return 0.0
        delta = (as_of - asserted).total_seconds() / 86400.0
        return max(0.0, delta)

    def decayed(self, fact: Fact, *, as_of: datetime) -> float:
        """``fact``'s confidence, decayed to ``as_of``.

        Exponential half-life: memoryless, single-parameter, and never
        crosses zero — ``confidence * 0.5 ** (age / half_life)``. A fact
        whose ``asserted_at`` cannot be parsed, or whose predicate has no
        half-life, keeps its stored confidence unchanged.
        """
        half_life = self.half_life(fact.predicate)
        if half_life is None:
            return fact.confidence

        try:
            datetime.fromisoformat(fact.asserted_at)
        except ValueError:
            return fact.confidence

        age = self.age_days(fact, as_of=as_of)
        value = fact.confidence * 0.5 ** (age / half_life)
        return max(self.floor, value)

    # -- derivation ------------------------------------------------------

    def with_half_life(self, predicate: str, half_life: float | None) -> "DecayPolicy":
        """Return a copy declaring ``predicate``'s half-life."""
        updated = dict(self.half_lives)
        updated[predicate] = half_life
        return DecayPolicy(updated, default=self.default, floor=self.floor)

    def without_half_life(self, predicate: str) -> "DecayPolicy":
        """Return a copy with ``predicate``'s declaration removed, if any."""
        updated = dict(self.half_lives)
        updated.pop(predicate, None)
        return DecayPolicy(updated, default=self.default, floor=self.floor)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "default": self.default,
            "floor": self.floor,
            "half_lives": dict(self.declared()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecayPolicy":
        """Rebuild a policy from :meth:`to_dict` output.

        Strict about shape, matching ``CardinalityMap.from_dict``: an
        unknown top-level key or an unrecognized ``version`` is rejected
        rather than tolerated.
        """
        unknown = payload.keys() - _FIELDS
        if unknown:
            raise ValueError(f"decay policy has unknown keys: {sorted(unknown)}")

        version = payload.get("version", _VERSION)
        if version != _VERSION:
            raise ValueError(f"unsupported decay policy version: {version!r}")

        return cls(
            half_lives=payload.get("half_lives", {}),
            default=payload.get("default", _DEFAULT_HALF_LIFE_DAYS),
            floor=payload.get("floor", 0.0),
        )

    @classmethod
    def default_map(cls) -> "DecayPolicy":
        """The policy in effect when ``.memgit/decay.json`` does not exist."""
        return cls()

    def __repr__(self) -> str:
        return f"DecayPolicy(default={self.default!r}, floor={self.floor!r}, {len(self.half_lives)} declared)"
