"""The eval runner: evaluate one case, or a whole suite, at one revision.

Every check reads exactly the surface :meth:`~memgit.core.repository.
Repository.diff` and ``memgit diff``/``memgit show`` already read: a
materialized :class:`~memgit.core.state.MemoryState` and a
:class:`~memgit.core.diff.Diff` against the revision's first parent (the
same bare-diff default ``memgit diff`` uses). Nothing here calls the agent
runtime or the replay engine, and nothing writes an object, a ref, or a
commit — a regression check that mutated the thing it measures would be a
defect, not a feature.

This is deliberately **not** "reuse the replay engine as a regression
check," despite PLAN.md's component 11 wording it that way.
``AblationResult.changed`` compares two independent, stochastically-sampled
API replies with no temperature or seed available — its false-positive rate
equals the model's own sampling variance, which is honest for a command a
human reads and disqualifying for an automated gate. The eval suite is
replay's deterministic sibling: it asks the same question ("did a memory
change break something?") without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memgit.core.diff import ChangeKind, Diff
from memgit.core.repository import Repository
from memgit.core.state import MemoryState
from memgit.eval.case import (
    Check,
    ConfidenceAtLeast,
    DiffKind,
    EvalCase,
    EvalSuite,
    KeyAbsent,
    KeyExists,
    NoViolations,
    ValueIs,
)

__all__ = ["CheckResult", "CaseResult", "SuiteResult", "run_case", "run_suite"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one :data:`~memgit.eval.case.Check`."""

    ok: bool
    kind: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The outcome of one :class:`~memgit.eval.case.EvalCase`: every check's result."""

    case_id: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.ok)

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "ok": self.ok, "checks": [c.to_dict() for c in self.checks]}


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """The outcome of running every case in an :class:`~memgit.eval.case.EvalSuite`
    at one resolved revision."""

    rev: str
    cases: tuple[CaseResult, ...]

    @property
    def ok(self) -> bool:
        return all(case.ok for case in self.cases)

    @property
    def failed(self) -> tuple[CaseResult, ...]:
        return tuple(case for case in self.cases if not case.ok)

    def to_dict(self) -> dict[str, Any]:
        return {"rev": self.rev, "ok": self.ok, "cases": [c.to_dict() for c in self.cases]}


# ---------------------------------------------------------------------------
# Per-check evaluation
# ---------------------------------------------------------------------------


def _run_key_exists(state: MemoryState, check: KeyExists) -> CheckResult:
    ok = len(state.get(check.subject, check.predicate)) > 0
    detail = (
        f"{check.subject} {check.predicate}: exists"
        if ok
        else f"expected a fact at {check.subject} {check.predicate}, found none"
    )
    return CheckResult(ok=ok, kind="key_exists", detail=detail)


def _run_key_absent(state: MemoryState, check: KeyAbsent) -> CheckResult:
    facts = state.get(check.subject, check.predicate)
    ok = len(facts) == 0
    detail = (
        f"{check.subject} {check.predicate}: absent"
        if ok
        else f"expected {check.subject} {check.predicate} absent, found {len(facts)} fact(s)"
    )
    return CheckResult(ok=ok, kind="key_absent", detail=detail)


def _run_value_is(state: MemoryState, check: ValueIs) -> CheckResult:
    objects = sorted({fact.object for fact in state.get(check.subject, check.predicate)})
    ok = check.object in objects
    detail = (
        f"{check.subject} {check.predicate} == {check.object!r}"
        if ok
        else f"expected {check.subject} {check.predicate} == {check.object!r}, got {objects!r}"
    )
    return CheckResult(ok=ok, kind="value_is", detail=detail)


def _run_confidence_at_least(state: MemoryState, check: ConfidenceAtLeast) -> CheckResult:
    fact = state.one(check.subject, check.predicate)
    ok = fact is not None and fact.confidence >= check.min
    got = "no fact" if fact is None else f"{fact.confidence:.2f}"
    detail = (
        f"{check.subject} {check.predicate} confidence {got} >= {check.min:.2f}"
        if ok
        else f"expected {check.subject} {check.predicate} confidence >= {check.min:.2f}, got {got}"
    )
    return CheckResult(ok=ok, kind="confidence_at_least", detail=detail)


def _run_diff_kind(diff: Diff | None, check: DiffKind) -> CheckResult:
    if diff is None:
        return CheckResult(ok=False, kind="diff_kind", detail="no diff available at this revision")
    matches = [key_diff for key_diff in diff.keys if key_diff.key == (check.subject, check.predicate)]
    kind = matches[0].kind if matches else ChangeKind.UNCHANGED
    ok = kind in check.kinds
    wanted = "/".join(k.value for k in check.kinds)
    detail = (
        f"{check.subject} {check.predicate}: {kind.value}"
        if ok
        else f"expected {check.subject} {check.predicate} in [{wanted}], got {kind.value}"
    )
    return CheckResult(ok=ok, kind="diff_kind", detail=detail)


def _run_no_violations(diff: Diff | None, check: NoViolations) -> CheckResult:
    if diff is None:
        return CheckResult(ok=False, kind="no_violations", detail="no diff available at this revision")
    ok = len(diff.violations) == 0
    detail = "no cardinality violations" if ok else f"{len(diff.violations)} cardinality violation(s)"
    return CheckResult(ok=ok, kind="no_violations", detail=detail)


def _run_check(repo: Repository, state: MemoryState, diff: Diff | None, check: Check) -> CheckResult:
    if isinstance(check, KeyExists):
        return _run_key_exists(state, check)
    if isinstance(check, KeyAbsent):
        return _run_key_absent(state, check)
    if isinstance(check, ValueIs):
        return _run_value_is(state, check)
    if isinstance(check, ConfidenceAtLeast):
        return _run_confidence_at_least(state, check)
    if isinstance(check, DiffKind):
        return _run_diff_kind(diff, check)
    if isinstance(check, NoViolations):
        return _run_no_violations(diff, check)
    raise TypeError(f"unhandled check type: {type(check).__name__}")  # pragma: no cover


_NEEDS_DIFF = (DiffKind, NoViolations)


def run_case(repo: Repository, case: EvalCase, rev: str = "HEAD") -> CaseResult:
    """Evaluate every check in ``case`` against ``rev``.

    The diff (``rev`` vs its first parent — the locked-in bare-diff default,
    see :meth:`Repository.diff`) is computed at most once, lazily, since a
    state-only case never needs it.
    """
    resolved = repo.resolve(rev)
    state = repo.state(resolved)
    diff: Diff | None = None
    if any(isinstance(check, _NEEDS_DIFF) for check in case.checks):
        diff = repo.diff(after=resolved)
    results = tuple(_run_check(repo, state, diff, check) for check in case.checks)
    return CaseResult(case_id=case.id, checks=results)


def run_suite(repo: Repository, suite: EvalSuite, rev: str = "HEAD") -> SuiteResult:
    """Evaluate every case in ``suite`` against the same resolved ``rev``.

    Resolving once up front means every case in one suite run compares
    against the exact same commit even if a concurrent process moves the
    branch a case's revision argument names.
    """
    resolved = repo.resolve(rev)
    cases = tuple(run_case(repo, case, resolved) for case in suite.cases)
    return SuiteResult(rev=resolved, cases=cases)
