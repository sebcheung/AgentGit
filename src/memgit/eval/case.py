"""Eval cases: declarative, closed-vocabulary assertions about a memory.

An eval case is a suite author's yardstick — "at this revision, the agent
should still believe X" — expressed as one or more :data:`Check`\\ s. Every
check kind reads only :class:`~memgit.core.state.MemoryState`,
:class:`~memgit.core.diff.Diff`, or a deterministic retrieval — never a model
reply — so a whole suite is evaluable offline, with no API key and no
network. See ``runner.py`` for how a check is actually evaluated, and the
package docstring (``__init__.py``) for why that offline property is the
point of this package.

The check vocabulary is closed and small on purpose, the same instinct as
``diff.py``'s eight-kind change taxonomy: an open, pluggable assertion
registry is a framework nobody asked for. Adding a ninth kind means editing
this module, not registering a plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Union

from memgit.core.diff import ChangeKind

__all__ = [
    "KeyExists",
    "KeyAbsent",
    "ValueIs",
    "ConfidenceAtLeast",
    "DiffKind",
    "NoViolations",
    "Recalls",
    "Check",
    "EvalFormatError",
]


class EvalFormatError(ValueError):
    """A suite or case file is malformed.

    Strict about shape, matching ``CardinalityMap.from_dict`` and
    ``DecayPolicy.from_dict``: an unknown key, a missing required field, or
    an unrecognized check kind is rejected at load time rather than silently
    skipped. A suite that quietly drops a case it can't parse is a suite that
    lies about its own coverage.
    """


# ---------------------------------------------------------------------------
# The closed check vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KeyExists:
    """At least one fact holds at ``(subject, predicate)``."""

    subject: str
    predicate: str


@dataclass(frozen=True, slots=True)
class KeyAbsent:
    """No fact holds at ``(subject, predicate)``."""

    subject: str
    predicate: str


@dataclass(frozen=True, slots=True)
class ValueIs:
    """Some fact at ``(subject, predicate)`` holds ``object``."""

    subject: str
    predicate: str
    object: str


@dataclass(frozen=True, slots=True)
class ConfidenceAtLeast:
    """The key's most-trustworthy fact (:meth:`MemoryState.one`) has stored
    confidence at least ``min``.

    Stored confidence, never decayed — decay is a read-time lens the
    retrieval ranker and ``--as-of`` display apply, and a regression check
    that moved every day without a memory change would be worse than useless.
    """

    subject: str
    predicate: str
    min: float


@dataclass(frozen=True, slots=True)
class DiffKind:
    """The revision's diff against its first parent classified this key as
    one of ``kinds``.

    Reuses ``diff.py``'s eight-kind taxonomy verbatim: "did this commit
    record a ``contradicted`` at ``(user, city)``" is a discrete, stable
    event, unlike anything asked of a model's reply.
    """

    subject: str
    predicate: str
    kinds: tuple[ChangeKind, ...]


@dataclass(frozen=True, slots=True)
class NoViolations:
    """The revision's diff reports no cardinality violations."""


@dataclass(frozen=True, slots=True)
class Recalls:
    """Retrieving ``query`` at this revision surfaces ``(subject, predicate)``
    within the top ``k``.

    ``as_of`` is required, not defaulted to "now": ``HashingEmbedder`` is
    deterministic but decayed confidence is not, so a case without a pinned
    ``as_of`` would pass today and fail on its own on a later day with no
    memory change at all — the exact non-determinism this suite exists to
    avoid.
    """

    query: str
    subject: str
    predicate: str
    as_of: datetime
    k: int = 8


Check = Union[KeyExists, KeyAbsent, ValueIs, ConfidenceAtLeast, DiffKind, NoViolations, Recalls]

# kind -> (class, required field names, optional field names)
_CHECK_FIELDS: dict[str, tuple[type, frozenset[str], frozenset[str]]] = {
    "key_exists": (KeyExists, frozenset({"subject", "predicate"}), frozenset()),
    "key_absent": (KeyAbsent, frozenset({"subject", "predicate"}), frozenset()),
    "value_is": (ValueIs, frozenset({"subject", "predicate", "object"}), frozenset()),
    "confidence_at_least": (ConfidenceAtLeast, frozenset({"subject", "predicate", "min"}), frozenset()),
    "diff_kind": (DiffKind, frozenset({"subject", "predicate", "kinds"}), frozenset()),
    "no_violations": (NoViolations, frozenset(), frozenset()),
    "recalls": (Recalls, frozenset({"query", "subject", "predicate", "as_of"}), frozenset({"k"})),
}


def _decode_check(payload: Mapping[str, Any]) -> Check:
    if "check" not in payload:
        raise EvalFormatError(f"check is missing its 'check' kind: {payload!r}")
    kind = payload["check"]
    spec = _CHECK_FIELDS.get(kind)
    if spec is None:
        raise EvalFormatError(f"unknown check kind: {kind!r}")
    cls, required, optional = spec

    unknown = payload.keys() - required - optional - {"check"}
    if unknown:
        raise EvalFormatError(f"check {kind!r} has unknown keys: {sorted(unknown)}")
    missing = required - payload.keys()
    if missing:
        raise EvalFormatError(f"check {kind!r} is missing required keys: {sorted(missing)}")

    kwargs = {key: value for key, value in payload.items() if key != "check"}

    if kind == "diff_kind":
        try:
            kwargs["kinds"] = tuple(ChangeKind(value) for value in kwargs["kinds"])
        except (TypeError, ValueError) as exc:
            raise EvalFormatError(f"check 'diff_kind' has an invalid 'kinds' entry: {exc}") from exc
        if not kwargs["kinds"]:
            raise EvalFormatError("check 'diff_kind' requires at least one kind in 'kinds'")

    if kind == "recalls":
        try:
            kwargs["as_of"] = datetime.fromisoformat(kwargs["as_of"])
        except (TypeError, ValueError) as exc:
            raise EvalFormatError(f"check 'recalls' has an invalid 'as_of': {exc}") from exc

    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise EvalFormatError(f"check {kind!r} is malformed: {exc}") from exc
