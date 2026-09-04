r"""Eval cases: declarative, closed-vocabulary assertions about a memory.

An eval case is a suite author's yardstick — "at this revision, the agent
should still believe X" — expressed as one or more :data:`Check`\\ s. Every
check kind reads only :class:`~memgit.core.state.MemoryState`,
:class:`~memgit.core.diff.Diff`, or a deterministic retrieval — never a model
reply — so a whole suite is evaluable offline, with no API key and no
network. See ``runner.py`` for how a check is actually evaluated, and the
package docstring (``__init__.py``) for why that offline property is the
point of this package.

**Suites are authored JSON, not a stored object.** They live at
``.memgit/eval/*.json`` — repo-local and never committed into memory
history, exactly like ``.memgit/cardinality.json`` and ``.memgit/decay.json``
(see those modules' docstrings). A suite is a question you ask, not data you
store: committing it into the history it grades would force the same
unanswerable "whose suite wins when diffing two commits" the cardinality map
already refuses to answer. JSON, not YAML — no YAML dialect exists anywhere
else in this project, and a second config format would be unearned.

The check vocabulary is closed and small on purpose, the same instinct as
``diff.py``'s eight-kind change taxonomy: an open, pluggable assertion
registry is a framework nobody asked for. Adding a ninth kind means editing
this module, not registering a plugin.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memgit.core.diff import ChangeKind

if TYPE_CHECKING:
    from memgit.core.repository import Repository

__all__ = [
    "SUITE_SUBDIR",
    "Check",
    "ConfidenceAtLeast",
    "DiffKind",
    "EvalCase",
    "EvalFormatError",
    "EvalSuite",
    "KeyAbsent",
    "KeyExists",
    "NoViolations",
    "Recalls",
    "ValueIs",
    "load_suites",
]

SUITE_SUBDIR = "eval"
_VERSION = 1


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
    """The key's most-trustworthy fact has stored confidence at least ``min``.

    The most-trustworthy fact is :meth:`MemoryState.one`'s pick. Stored
    confidence, never decayed — decay is a read-time lens the
    retrieval ranker and ``--as-of`` display apply, and a regression check
    that moved every day without a memory change would be worse than useless.
    """

    subject: str
    predicate: str
    min: float


@dataclass(frozen=True, slots=True)
class DiffKind:
    """The revision's diff against its first parent classified this key as one of ``kinds``.

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
    """Retrieving ``query`` at this revision surfaces ``(subject, predicate)`` within the top ``k``.

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


Check = KeyExists | KeyAbsent | ValueIs | ConfidenceAtLeast | DiffKind | NoViolations | Recalls

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


# ---------------------------------------------------------------------------
# Cases and suites
# ---------------------------------------------------------------------------

_CASE_FIELDS = frozenset({"id", "checks", "description"})
_SUITE_FIELDS = frozenset({"version", "cases"})


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One named yardstick: a case id and the checks it must satisfy."""

    id: str
    checks: tuple[Check, ...]
    description: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvalCase:
        """Parse one case from its JSON shape. Strict: unknown keys are an error."""
        if not isinstance(payload, Mapping):
            raise EvalFormatError(f"a case must be a JSON object, got {type(payload).__name__}")
        unknown = payload.keys() - _CASE_FIELDS
        if unknown:
            raise EvalFormatError(f"case has unknown keys: {sorted(unknown)}")

        case_id = payload.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvalFormatError(f"case 'id' must be a non-empty string, got {case_id!r}")

        checks_payload = payload.get("checks")
        if not isinstance(checks_payload, list) or not checks_payload:
            raise EvalFormatError(f"case {case_id!r} must declare a non-empty 'checks' list")
        checks = tuple(_decode_check(check) for check in checks_payload)

        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            raise EvalFormatError(f"case {case_id!r} has a non-string 'description'")

        return cls(id=case_id, checks=checks, description=description)


@dataclass(frozen=True, slots=True)
class EvalSuite:
    r"""A named group of :class:`EvalCase`\\ s, loaded from one JSON file."""

    name: str
    cases: tuple[EvalCase, ...]

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    @classmethod
    def from_dict(cls, name: str, payload: Mapping[str, Any]) -> EvalSuite:
        """Parse a whole suite from its JSON shape. Strict: unknown keys are an error."""
        if not isinstance(payload, Mapping):
            raise EvalFormatError(f"suite {name!r} must be a JSON object, got {type(payload).__name__}")
        unknown = payload.keys() - _SUITE_FIELDS
        if unknown:
            raise EvalFormatError(f"suite {name!r} has unknown keys: {sorted(unknown)}")

        version = payload.get("version", _VERSION)
        if version != _VERSION:
            raise EvalFormatError(f"suite {name!r}: unsupported version {version!r}")

        cases_payload = payload.get("cases")
        if not isinstance(cases_payload, list):
            raise EvalFormatError(f"suite {name!r} must declare a 'cases' list")

        cases: list[EvalCase] = []
        seen: set[str] = set()
        for case_payload in cases_payload:
            case = EvalCase.from_dict(case_payload)
            if case.id in seen:
                raise EvalFormatError(f"suite {name!r} has a duplicate case id: {case.id!r}")
            seen.add(case.id)
            cases.append(case)

        return cls(name=name, cases=tuple(cases))


def _load_suite_file(path: Path) -> EvalSuite:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalFormatError(f"{path}: could not be read: {exc}") from exc
    except ValueError as exc:
        raise EvalFormatError(f"{path}: not valid JSON: {exc}") from exc
    return EvalSuite.from_dict(path.stem, payload)


def load_suites(repo: Repository, path: Path | None = None) -> tuple[EvalSuite, ...]:
    """Load every declared suite for ``repo``.

    Args:
        repo: The repository whose ``.memgit/eval/`` directory is the
            default location — see the module docstring for why suites live
            outside the object store.
        path: Overrides the default. A single file loads as one suite; a
            directory loads every ``*.json`` file in it, sorted by name so
            output order is stable across runs.

    Returns:
        An empty tuple if the default directory does not exist — matching
        :meth:`~memgit.core.repository.Repository.cardinality`'s "a missing
        file is a normal state," not an error.
    """
    if path is not None:
        if path.is_file():
            return (_load_suite_file(path),)
        if not path.is_dir():
            raise EvalFormatError(f"{path}: no such suite file or directory")
        root = path
    else:
        root = repo.memgit_dir / SUITE_SUBDIR
        if not root.is_dir():
            return ()

    return tuple(_load_suite_file(f) for f in sorted(root.glob("*.json")))
