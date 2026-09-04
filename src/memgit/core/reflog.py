r"""The reflog — a journal of every place a ref (or HEAD) has pointed.

A branch or HEAD only ever remembers where it points *now*; the moment it
moves, where it used to point is gone unless something wrote it down first.
That is exactly the safety net a destructive operation like ``reset`` needs
before it is trustworthy: a ref move without one is a taken-away memory state
with no way back, which is the opposite of what this project is for. This
module owns the format and the on-disk journal; ``refs.py`` owns *when* to
call it.

Layout mirrors git on purpose, like everything else on disk here::

    .memgit/logs/HEAD                  every place HEAD has pointed
    .memgit/logs/refs/heads/<name>     every place that branch has pointed

Each line is one movement, oldest first, appended (never rewritten)::

    <old-hash> <new-hash> <author> <iso8601-utc> <op>\t<message>

Three deliberate departures from git's own reflog format, each worth stating
rather than leaving as an unexplained mismatch:

**Timestamps are ISO-8601 UTC**, via the same ``canonical.utcnow`` every other
timestamp in this project uses — not git's ``<unix-epoch> <tz-offset>``. Every
other clock in MemGit already speaks one dialect; a second one earns nothing,
and the "this looks like git" demo value is already carried by ``HEAD``,
``refs/`` and now ``logs/`` existing at all.

**``op`` is a short verb** (``commit``, ``checkout``, ``branch``, ``reset``,
``rewind``, ...) that a reader can filter on without parsing the free-text
``message`` that follows the tab. Git folds both into one string; splitting
them costs nothing and buys grep-ability.

**No file locking on append.** A ref write goes through a lock-file-and-rename
because a half-written ref is unreadable corruption. A half-written reflog
*line* is not: at worst a concurrent-write race truncates or interleaves one
entry, which is a lost breadcrumb, not a broken repository. A plain
``open(..., "a")`` is judged good enough for that risk, and the tradeoff is
named here rather than pretended away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from memgit.core.canonical import utcnow

__all__ = [
    "ZERO_HASH",
    "RefLog",
    "RefLogEntry",
    "RefLogger",
    "ReflogNotFoundError",
]

# "Nothing was here before" / "nothing is here now" — git's convention for a
# ref's reflog entry either side of a create or a delete.
ZERO_HASH = "0" * 64


class ReflogNotFoundError(KeyError):
    """Raised when a reflog is missing, or an index reaches past its start."""


@dataclass(frozen=True, slots=True)
class RefLogEntry:
    """One line of a reflog: where a ref was, where it went, and why.

    Attributes:
        old: The ref's value before this movement, or ``None`` for "did not
            exist yet" (an unborn branch, or a brand-new one).
        new: The ref's value after this movement, or ``None`` for "no longer
            exists" (a deleted branch, or HEAD pointing at an unborn one).
        author: Whoever's action caused this movement. Reuses the same
            free-form identity ``Commit.author`` and ``Fact.source`` carry.
        at: ISO-8601 UTC timestamp of the movement.
        op: A short verb naming the kind of movement (``"commit"``,
            ``"checkout"``, ``"branch"``, ``"reset"``, ``"rewind"``).
        message: A free-text reason, e.g. a commit's summary line.
    """

    old: str | None
    new: str | None
    author: str
    at: str
    op: str
    message: str = ""

    def format(self) -> str:
        """Render as one reflog line, without a trailing newline.

        Raises:
            ValueError: ``author`` or ``op`` contains whitespace, or
                ``message`` contains a newline — each would corrupt the
                one-line-per-entry format for the reader that comes after.
        """
        if any(ch.isspace() for ch in self.author):
            raise ValueError(f"reflog author must not contain whitespace: {self.author!r}")
        if any(ch.isspace() for ch in self.op):
            raise ValueError(f"reflog op must not contain whitespace: {self.op!r}")
        if "\n" in self.message or "\r" in self.message:
            raise ValueError("reflog message must not contain a newline")

        old_field = self.old if self.old is not None else ZERO_HASH
        new_field = self.new if self.new is not None else ZERO_HASH
        # Message is free text (e.g. a commit's message) and reaches a
        # line-oriented file; a tab in it would silently join two fields.
        message = self.message.replace("\t", " ")
        return f"{old_field} {new_field} {self.author} {self.at} {self.op}\t{message}"

    @classmethod
    def parse(cls, line: str) -> RefLogEntry:
        """Parse one line of :meth:`format` output."""
        line = line.rstrip("\n")
        header, _tab, message = line.partition("\t")
        parts = header.split(" ")
        if len(parts) != 5:
            raise ValueError(f"malformed reflog line: {line!r}")
        old_field, new_field, author, at, op = parts
        return cls(
            old=None if old_field == ZERO_HASH else old_field,
            new=None if new_field == ZERO_HASH else new_field,
            author=author,
            at=at,
            op=op,
            message=message,
        )


def _ref_log_path(memgit_dir: Path, ref: str) -> Path:
    if ref != "HEAD" and not ref.startswith("refs/"):
        raise ValueError(f"reflog ref must be 'HEAD' or start with 'refs/', got {ref!r}")
    return memgit_dir / "logs" / ref


class RefLog:
    """The append-only journal for one ref (or ``"HEAD"``).

    Args:
        memgit_dir: The repository's ``.memgit`` directory.
        ref: ``"HEAD"``, or a full ref path like ``"refs/heads/main"``.
    """

    def __init__(self, memgit_dir: Path, ref: str) -> None:
        self.memgit_dir = Path(memgit_dir)
        self.ref = ref

    @property
    def path(self) -> Path:
        """The on-disk file this ref's log is (or would be) stored at."""
        return _ref_log_path(self.memgit_dir, self.ref)

    def exists(self) -> bool:
        """True if this ref has ever been logged."""
        return self.path.is_file()

    def append(self, entry: RefLogEntry) -> None:
        """Append one entry. Creates ``logs/`` (and any subdirectory) lazily.

        Not locked — see the module docstring for why a plain append is the
        judged tradeoff here, unlike every ref write itself.
        """
        line = entry.format()
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def entries(self) -> tuple[RefLogEntry, ...]:
        """Every entry, oldest first. Empty if the log does not exist yet."""
        path = self.path
        if not path.is_file():
            return ()
        text = path.read_text(encoding="utf-8")
        return tuple(RefLogEntry.parse(line) for line in text.splitlines() if line)

    def at(self, index: int) -> str:
        """Resolve ``ref@{index}`` to a commit hash.

        ``@{0}`` is where the ref is now (its most recent entry's ``new``);
        ``@{n}`` for ``n >= 1`` is where it was *before* its nth-most-recent
        movement (that entry's ``old``) — git's own indexing.

        Raises:
            ReflogNotFoundError: No log exists, the index reaches past the
                start of it, or the entry at that index has no commit (the
                ref did not exist there, e.g. before it was first created).
        """
        if index < 0:
            raise ReflogNotFoundError(f"{self.ref}@{{{index}}}: index must not be negative")

        entries = self.entries()
        if not entries:
            raise ReflogNotFoundError(f"no reflog for {self.ref!r}")

        if index == 0:
            result = entries[-1].new
        else:
            if index > len(entries):
                raise ReflogNotFoundError(
                    f"{self.ref}@{{{index}}}: only {len(entries)} entries in the reflog"
                )
            result = entries[-index].old

        if result is None:
            raise ReflogNotFoundError(f"{self.ref}@{{{index}}} names no commit")
        return result


class RefLogger:
    """What ``RefStore`` calls on every ref movement, if attached.

    A thin convenience over :class:`RefLog`: it owns the author identity (one
    per repository open, not per call) and builds the :class:`RefLogEntry`'s
    timestamp, so ``refs.py``'s writers only have to say *what* moved and why.
    """

    def __init__(self, memgit_dir: Path, *, author: str = "unknown") -> None:
        self.memgit_dir = Path(memgit_dir)
        self.author = author

    def log(
        self,
        ref: str,
        old: str | None,
        new: str | None,
        *,
        op: str,
        message: str = "",
    ) -> None:
        """Record one movement of ``ref`` from ``old`` to ``new``."""
        entry = RefLogEntry(old=old, new=new, author=self.author, at=utcnow(), op=op, message=message)
        RefLog(self.memgit_dir, ref).append(entry)
