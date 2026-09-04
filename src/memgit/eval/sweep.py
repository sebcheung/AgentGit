"""Sweep a suite across a commit range to find where each case first broke.

``memgit eval`` answers "is memory healthy right now"; this answers "which
commit made it unhealthy" — the sentence PLAN.md's component 11 actually
wants ("flag if a memory change broke something") pointed at a real
revision instead of just the working tip. A linear walk, not a binary
bisect: ``git bisect``'s halving only pays for itself when each probe is
expensive, and every check in this package is offline and deterministic, so
walking every commit is cheap and assumes nothing about monotonicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memgit.core.graph import walk
from memgit.core.repository import Repository, RevisionNotFoundError
from memgit.eval.case import EvalSuite
from memgit.eval.runner import run_suite

__all__ = ["FirstFailure", "sweep"]


@dataclass(frozen=True, slots=True)
class FirstFailure:
    """The earliest commit, within the walked range, where a case failed.

    Attributes:
        commit: The full hash of the first failing commit.
        message: The failing check detail(s) at that commit — the same
            ``CheckResult.detail`` text ``memgit eval`` itself prints.
    """

    case_id: str
    commit: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-ready shape the CLI reports."""
        return {"case_id": self.case_id, "commit": self.commit, "message": self.message}


def sweep(repo: Repository, suite: EvalSuite, *, since: str, until: str = "HEAD") -> tuple[FirstFailure, ...]:
    """Walk ``since..until`` oldest-first, reporting each case's first failure.

    A case not in the returned tuple never failed anywhere in the range. The
    result is "evidence, not proof" of a unique cause, in the same spirit as
    the replay engine's own ablation caveat: a case can fail because of an
    unrelated concurrent change at the same commit, not only the one a
    reader assumes. Inherits ``walk``'s own clock-skew caveat.

    Raises:
        RevisionNotFoundError: ``since`` does not resolve, or does not lie
            on ``until``'s first-parent-or-better ancestry as walked by
            :func:`~memgit.core.graph.walk`.
    """
    since_hash = repo.resolve(since)
    until_hash = repo.resolve(until)

    walked: list[tuple[str, Any]] = []
    for commit_hash, commit in walk(until_hash, repo.read_commit):
        walked.append((commit_hash, commit))
        if commit_hash == since_hash:
            break
    else:
        raise RevisionNotFoundError(f"{since!r} is not an ancestor of {until!r}")
    walked.reverse()  # oldest first, so a case's history reads chronologically

    failures: dict[str, FirstFailure] = {}
    for commit_hash, _commit in walked:
        result = run_suite(repo, suite, commit_hash)
        for case_result in result.cases:
            if case_result.ok or case_result.case_id in failures:
                continue
            detail = "; ".join(check.detail for check in case_result.failures)
            failures[case_result.case_id] = FirstFailure(
                case_id=case_result.case_id, commit=commit_hash, message=detail
            )

    return tuple(failures[case.id] for case in suite.cases if case.id in failures)
