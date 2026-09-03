"""Revision expressions — ``HEAD~3``, ``main^2``, ``HEAD@{1}``.

A commit hash names one specific point in history; a ref names whichever
commit is current. Neither can say "three turns before that", which is
exactly the question a rewind slice needs answered. This module owns that one
piece of syntax, and nothing else — it is deliberately ignorant of the
filesystem, the object store, and even of what a "ref" is, so it can be
tested against synthetic commit graphs the same way ``graph.py`` already is.

Two independent things happen here, split the way ``graph.py`` splits
traversal from ancestry: :func:`parse_revision` turns a string into structured
steps (no I/O at all — pure string handling), and :func:`apply_steps` walks
those steps against a :data:`~memgit.core.graph.CommitReader` (pure over that
callable, no I/O of its own either). ``Repository.resolve`` is the one place
that supplies a real reader and turns this module's two narrow exceptions
into the single ``RevisionNotFoundError`` every CLI command already expects.

Syntax, matching git exactly because there is no reason to diverge:

- ``^`` / ``^N`` — the Nth parent (1-indexed); ``^0`` is the commit itself.
- ``~`` / ``~N`` — N hops along first parents; ``~0`` is the commit itself.
- Steps apply strictly left to right: ``main~2^2~1``.
- ``@{N}`` selects the Nth-most-recent entry in *the base's own* reflog, and
  may only appear immediately after the base — ``HEAD@{1}~2`` is legal,
  ``HEAD~2@{1}`` is not, because ``@{N}`` names a starting point, not an
  ancestor of one.

Merge-parent steps (``^2``) parse and apply here for free, because
``graph.py``'s ``CommitReader`` already exposes every parent of a merge
commit — not because this slice creates merge commits. Nothing in MemGit
writes one yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from memgit.core.graph import CommitReader

__all__ = [
    "Step",
    "StepKind",
    "RevExpr",
    "parse_revision",
    "apply_steps",
    "RevSyntaxError",
    "AncestryError",
]


class RevSyntaxError(ValueError):
    """Raised when a revision string is not well-formed.

    A parsing failure, not a lookup failure — ``"HEAD~x"`` is wrong no matter
    what the commit graph looks like.
    """


class AncestryError(LookupError):
    """Raised when a revision parses fine but the graph has no such ancestor.

    ``"HEAD^3"`` is well-formed syntax that fails only if ``HEAD`` turns out
    to have fewer than three parents.
    """


class StepKind(str, Enum):
    """Which of the two ancestry operators a :class:`Step` came from."""

    PARENT = "parent"
    ANCESTOR = "ancestor"


@dataclass(frozen=True, slots=True)
class Step:
    """One ``^N`` or ``~N`` applied in sequence."""

    kind: StepKind
    n: int


@dataclass(frozen=True, slots=True)
class RevExpr:
    """A parsed revision: a base plus the steps to walk from it.

    Attributes:
        base: The unparsed base — ``"HEAD"``, a branch name, a full ref path,
            a commit hash, or a hash prefix. Left as a string because
            resolving it is ``Repository``'s job, not this module's.
        steps: Ancestry steps to apply, in order.
        reflog_index: The ``N`` from a trailing ``base@{N}``, or ``None`` if
            the revision did not use reflog syntax.
    """

    base: str
    steps: tuple[Step, ...] = ()
    reflog_index: int | None = None

    @property
    def is_bare(self) -> bool:
        """Whether this is exactly its base — no steps, no reflog selector.

        What lets a caller distinguish ``"main"`` (a branch name, worth
        staying attached to) from ``"main~1"`` (an ancestor of that branch,
        which is not the branch itself) without re-parsing the string.
        """
        return not self.steps and self.reflog_index is None


def parse_revision(revision: str) -> RevExpr:
    """Parse a revision string into a :class:`RevExpr`.

    Raises:
        RevSyntaxError: ``revision`` is not well-formed.
    """
    if not isinstance(revision, str) or revision == "":
        raise RevSyntaxError(f"revision must be a non-empty str, got {revision!r}")

    length = len(revision)
    i = 0

    start = i
    while i < length and revision[i] not in "~^@":
        i += 1
    base = revision[start:i]
    if not base:
        raise RevSyntaxError(f"revision has no base: {revision!r}")

    reflog_index: int | None = None
    if i < length and revision[i] == "@":
        i += 1
        if i >= length or revision[i] != "{":
            raise RevSyntaxError(f"expected '{{' after '@' in {revision!r}")
        i += 1
        digit_start = i
        while i < length and revision[i] != "}":
            i += 1
        if i >= length:
            raise RevSyntaxError(f"unterminated '@{{' in {revision!r}")
        digits = revision[digit_start:i]
        if not digits.isdigit():
            raise RevSyntaxError(f"'@{{...}}' must contain digits, got {digits!r}")
        reflog_index = int(digits)
        i += 1  # past the closing '}'

    steps: list[Step] = []
    while i < length:
        ch = revision[i]
        if ch == "@":
            raise RevSyntaxError(
                f"'@{{n}}' may only appear immediately after the base: {revision!r}"
            )
        if ch not in "~^":
            raise RevSyntaxError(f"unexpected character {ch!r} in revision {revision!r}")
        i += 1
        digit_start = i
        while i < length and revision[i].isdigit():
            i += 1
        digits = revision[digit_start:i]
        num = int(digits) if digits else 1
        kind = StepKind.ANCESTOR if ch == "~" else StepKind.PARENT
        steps.append(Step(kind=kind, n=num))

    return RevExpr(base=base, steps=tuple(steps), reflog_index=reflog_index)


def apply_steps(commit_hash: str, steps: Sequence[Step], read: "CommitReader") -> str:
    """Walk ``steps`` from ``commit_hash``; return the resulting commit hash.

    Pure over ``read``, exactly like ``graph.py``'s traversal functions — no
    filesystem, no ``Repository``, testable against a plain dict.

    Raises:
        AncestryError: A step asks for a parent that does not exist (a
            missing Nth parent, or a first-parent hop off a root commit).
    """
    current = commit_hash
    for step in steps:
        if step.n == 0:
            # ^0 and ~0 are both "the commit itself" — no read needed.
            continue
        if step.kind is StepKind.PARENT:
            commit = read(current)
            if step.n > len(commit.parents):
                raise AncestryError(
                    f"commit {current} has {len(commit.parents)} parent(s), "
                    f"no parent {step.n}"
                )
            current = commit.parents[step.n - 1]
        else:
            for _ in range(step.n):
                commit = read(current)
                if not commit.parents:
                    raise AncestryError(f"commit {current} has no parent (it is a root)")
                current = commit.parents[0]
    return current
