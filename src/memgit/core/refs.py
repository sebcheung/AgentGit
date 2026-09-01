"""Refs and HEAD — names for commits.

A commit hash is a fine address but a poor thing to type or to build a demo
around. Refs give commits names: ``refs/heads/main`` is a plain text file
holding a commit hash, and ``HEAD`` says which ref (or, detached, which raw
commit) is "current". This is git's model taken directly, not reinvented,
because the whole point of this project's storage layer is that its on-disk
shape can sit next to a real ``.git/`` and look recognizable.

Refs are plain UTF-8 text files with a trailing newline — not JSON, not rows
in a table — specifically so that ``cat .memgit/HEAD`` next to ``cat
.git/HEAD`` is the demo, not an approximation of one.

**HEAD's detached form ships in this slice even though nothing detaches it
yet.** Nothing before slice 4 (checkout/rewind) needs a raw-hash HEAD. Adding
the capability now rather than later means there is only ever one HEAD format
existing repositories have to speak — the alternative is a migration the day
detached HEAD is introduced.

**Every ref write goes through one function.** That is what makes slice 4's
reflog a hook added to :meth:`RefStore.write_ref` rather than a grep-and-patch
across every call site that ever moves a ref.

Ref names reach the filesystem, so they get the same scrutiny
``ObjectStore._path_for`` gives hashes: a ref name is untrusted input the
moment it can be typed on a command line, and the validation below assumes
that, rather than trusting a list of "obviously bad" characters to be
complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memgit.core.store import is_object_hash

__all__ = [
    "Head",
    "RefStore",
    "InvalidRefNameError",
    "RefNotFoundError",
]

_SYMREF_PREFIX = "ref: "
_MAX_SYMREF_DEPTH = 5
_MAX_NAME_LENGTH = 255
_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./"
)

# Sentinel distinguishing "no compare-and-swap requested" from "expect the ref
# to be absent" (which is spelled `expect=None`).
_UNSET = object()


class InvalidRefNameError(ValueError):
    """Raised when a ref name is not safe or not well-formed."""


class RefNotFoundError(KeyError):
    """Raised when an operation requires a ref that does not exist."""


def _validate_ref_name(memgit_dir: Path, name: str) -> None:
    if not isinstance(name, str) or not name:
        raise InvalidRefNameError(f"ref name must be a non-empty str, got {name!r}")
    if not name.startswith("refs/"):
        raise InvalidRefNameError(f"ref name must start with 'refs/', got {name!r}")
    if len(name) > _MAX_NAME_LENGTH:
        raise InvalidRefNameError(f"ref name is too long ({len(name)} characters): {name!r}")
    if any(ch not in _ALLOWED_CHARS for ch in name):
        raise InvalidRefNameError(f"ref name contains disallowed characters: {name!r}")
    if "//" in name or name.endswith("/"):
        raise InvalidRefNameError(f"ref name has an empty path component: {name!r}")

    components = name.split("/")
    for component in components:
        if component in ("", ".", ".."):
            raise InvalidRefNameError(f"ref name has an invalid path component: {name!r}")
        if component.startswith("."):
            raise InvalidRefNameError(f"ref name component may not start with '.': {name!r}")
        if component.endswith(".lock"):
            raise InvalidRefNameError(f"ref name may not end with '.lock': {name!r}")

    # Belt and braces: enumerating bad characters is a claim that ages badly,
    # so also require the resolved path to actually land under refs/.
    refs_root = (memgit_dir / "refs").resolve()
    resolved = (memgit_dir / name).resolve()
    if refs_root != resolved and refs_root not in resolved.parents:
        raise InvalidRefNameError(f"ref name escapes the refs directory: {name!r}")


@dataclass(frozen=True, slots=True)
class Head:
    """Where HEAD points: a branch name, or (detached) a raw commit.

    Attributes:
        ref: The symbolic ref HEAD points at (``"refs/heads/main"``), or
            ``None`` when HEAD is detached.
        commit: The commit hash HEAD currently resolves to, or ``None`` on an
            unborn branch — a repository that exists but has no commits yet.
            That is a normal state, not an error: every fresh ``init`` starts
            there.
    """

    ref: str | None
    commit: str | None

    @property
    def is_detached(self) -> bool:
        return self.ref is None

    @property
    def branch(self) -> str | None:
        """The branch name, if HEAD is symbolic (``"main"`` for
        ``refs/heads/main``); ``None`` when detached."""
        if self.ref is None:
            return None
        return self.ref.removeprefix("refs/heads/")


class RefStore:
    """Owns ``HEAD`` and everything under ``refs/`` for one repository.

    Args:
        memgit_dir: The repository's ``.memgit`` directory. Refs live at
            ``memgit_dir/refs/...``; HEAD at ``memgit_dir/HEAD``.
    """

    def __init__(self, memgit_dir: Path) -> None:
        self.memgit_dir = Path(memgit_dir)

    # -- plain refs ----------------------------------------------------------

    def _ref_path(self, name: str) -> Path:
        _validate_ref_name(self.memgit_dir, name)
        return self.memgit_dir / name

    def read_ref(self, name: str) -> str | None:
        """The commit hash ``name`` points at, or ``None`` if it doesn't exist.

        A missing ref is not an error — an unborn branch is a normal state
        (see :class:`Head`), and every caller that cares about "does this
        exist" can check for ``None`` without a ``try/except``.
        """
        path = self._ref_path(name)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8").strip()

    def write_ref(self, name: str, commit: str, *, expect: Any = _UNSET) -> None:
        """Point ``name`` at ``commit``, atomically.

        Args:
            expect: If given, the write only proceeds when the ref's current
                value equals ``expect`` (``None`` meaning "must not exist yet").
                This compare-and-swap is what lets a future fast-forward check
                be expressed without redesigning this method.

        Raises:
            ValueError: ``commit`` is not a valid object hash, or ``expect``
                was given and did not match the ref's current value.
        """
        if not is_object_hash(commit):
            raise ValueError(f"ref target must be a valid object hash, got {commit!r}")

        path = self._ref_path(name)
        if expect is not _UNSET:
            current = self.read_ref(name)
            if current != expect:
                raise ValueError(
                    f"ref {name!r} is at {current!r}, expected {expect!r} (concurrent write?)"
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_name(path.name + ".lock")
        lock.write_text(commit + "\n", encoding="utf-8")
        lock.replace(path)

    def delete_ref(self, name: str) -> None:
        """Remove ``name``. A no-op if it does not exist."""
        path = self._ref_path(name)
        path.unlink(missing_ok=True)

    def list_refs(self, prefix: str = "refs/") -> dict[str, str]:
        """Every ref under ``prefix``, mapped to its commit hash."""
        if not prefix.startswith("refs/") or any(part in ("", ".", "..") for part in prefix.split("/")[:-1]):
            raise InvalidRefNameError(f"invalid ref prefix: {prefix!r}")
        root = (self.memgit_dir / prefix).resolve()
        refs_root = (self.memgit_dir / "refs").resolve()
        if refs_root != root and refs_root not in root.parents:
            raise InvalidRefNameError(f"ref prefix escapes the refs directory: {prefix!r}")
        if not root.is_dir():
            return {}
        result: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.endswith(".lock"):
                name = prefix.rstrip("/") + "/" + str(path.relative_to(root)).replace("\\", "/")
                result[name] = path.read_text(encoding="utf-8").strip()
        return result

    # -- HEAD ------------------------------------------------------------

    def read_head(self) -> Head:
        """The current HEAD: symbolic (on a branch) or detached (on a commit)."""
        head_path = self.memgit_dir / "HEAD"
        content = head_path.read_text(encoding="utf-8").strip()

        if not content.startswith(_SYMREF_PREFIX):
            # A bare line in HEAD is a detached commit hash.
            if not is_object_hash(content):
                raise ValueError(f"HEAD is malformed: {content!r}")
            return Head(ref=None, commit=content)

        target = content[len(_SYMREF_PREFIX):].strip()
        seen: set[str] = set()
        depth = 0
        while True:
            if target in seen:
                raise ValueError(f"symref cycle detected at {target!r}")
            seen.add(target)
            depth += 1
            if depth > _MAX_SYMREF_DEPTH:
                raise ValueError("symref chain is too deep — possible cycle")

            path = self._ref_path(target)
            if not path.is_file():
                # Points at a branch that has no commits yet — unborn, not
                # an error.
                return Head(ref=target, commit=None)

            raw = path.read_text(encoding="utf-8").strip()
            if raw.startswith(_SYMREF_PREFIX):
                target = raw[len(_SYMREF_PREFIX):].strip()
                continue
            if not is_object_hash(raw):
                raise ValueError(f"ref {target!r} is malformed: {raw!r}")
            return Head(ref=target, commit=raw)

    def set_head(self, ref: str) -> None:
        """Point HEAD at branch ``ref`` symbolically."""
        _validate_ref_name(self.memgit_dir, ref)
        head_path = self.memgit_dir / "HEAD"
        lock = head_path.with_name(head_path.name + ".lock")
        lock.write_text(f"{_SYMREF_PREFIX}{ref}\n", encoding="utf-8")
        lock.replace(head_path)

    def detach_head(self, commit: str) -> None:
        """Point HEAD directly at ``commit``, leaving no branch current.

        Slice 4's rewind is built on this: checking out an arbitrary past
        commit does not — and must not — silently create or move a branch.
        """
        if not is_object_hash(commit):
            raise ValueError(f"HEAD target must be a valid object hash, got {commit!r}")
        head_path = self.memgit_dir / "HEAD"
        lock = head_path.with_name(head_path.name + ".lock")
        lock.write_text(commit + "\n", encoding="utf-8")
        lock.replace(head_path)

    # -- resolution --------------------------------------------------------

    def resolve(self, revision: str) -> str | None:
        """Resolve a revision string to a commit hash, or ``None`` if unresolvable.

        Handles, in order: the literal ``"HEAD"``, a full 64-hex commit hash,
        a full ref path (``"refs/heads/main"``), and a bare branch name
        (``"main"``, tried as ``refs/heads/main``).

        ``HEAD~3`` / ``main^2``-style ancestry suffixes are deliberately not
        handled here — rewind (slice 4) is the feature that motivates them,
        and this is the one place that suffix parsing would be added.
        """
        if revision == "HEAD":
            return self.read_head().commit
        if is_object_hash(revision):
            return revision
        if revision.startswith("refs/"):
            return self.read_ref(revision)
        return self.read_ref(f"refs/heads/{revision}")
