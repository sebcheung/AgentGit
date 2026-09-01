"""Content-addressable object store.

This is git's blob model, implemented directly rather than imported. An object
is stored at a path derived from the SHA-256 of its own contents:

    .memgit/objects/ab/cdef0123...      # hash = "abcdef0123..."

Three properties fall out of that, and they are the reason MemGit is built this
way rather than on a table with a hash column:

**Deduplication is automatic.** Writing a fact that already exists computes the
same address and finds the file already there, so it costs nothing. An agent
that reaffirms 500 unchanged facts across 50 commits stores them once.

**Objects are immutable.** Changing an object changes its address, so nothing
can be edited out from under a commit that references it. History is
append-only by construction, not by convention.

**Corruption is detectable.** The address is a checksum of the contents, so
re-hashing on read verifies the object end-to-end for free.

The two-character directory fanout (``ab/cdef...`` rather than ``abcdef...``)
exists because filesystems degrade badly with very large flat directories. 256
subdirectories keeps each one small enough to stay fast.

Objects are zlib-compressed. Facts are small, repetitive JSON — highly
compressible — and this is what git does, which keeps the parallel honest.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Iterator

from memgit.core.canonical import canonical_json, hash_payload

__all__ = [
    "ObjectStore",
    "CorruptObjectError",
    "ObjectNotFoundError",
    "AmbiguousPrefixError",
    "is_object_hash",
]

# Compression level 6 is zlib's default: a deliberate middle of the road on the
# size/speed curve. Facts are tiny, so this is unlikely to ever be a bottleneck.
_COMPRESSION_LEVEL = 6

# Length of the directory prefix in the fanout. Two hex chars = 256 buckets,
# matching git.
_FANOUT = 2


class ObjectNotFoundError(KeyError):
    """Raised when an object hash has no corresponding file in the store."""


class CorruptObjectError(Exception):
    """Raised when a stored object fails verification on read.

    Either zlib could not decompress it, or the bytes decompressed fine but
    re-hashed to a different address than the one they were filed under.
    """


class AmbiguousPrefixError(ValueError):
    """Raised when a hash prefix matches more than one object in the store."""


def is_object_hash(value: Any) -> bool:
    """Whether ``value`` has the shape of a valid object hash: 64 hex characters.

    Trees, commits, and refs will all hold hashes as opaque pointers into this
    store — a hash string reaching the filesystem unchecked is a path-traversal
    vector one hop removed (see ``ObjectStore._path_for``), so every module that
    carries one applies this exact rule rather than a slightly different
    reimplementation of it.
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


class ObjectStore:
    """A SHA-256 content-addressable store rooted at a directory.

    Args:
        objects_dir: Directory holding the fanned-out object files. Created on
            demand; the store is happy to point at a path that does not exist
            yet and will materialize it on first write.
    """

    def __init__(self, objects_dir: Path | str) -> None:
        self.objects_dir = Path(objects_dir)

    # -- addressing ------------------------------------------------------

    def _path_for(self, obj_hash: str) -> Path:
        """Map a hash to its on-disk path, validating the hash's shape.

        Validation matters here beyond tidiness: this value reaches the
        filesystem, so an unchecked hash string is a path-traversal vector.
        Rejecting anything that is not 64 hex characters means ``..`` and
        absolute paths can never get through.
        """
        if is_object_hash(obj_hash):
            return self.objects_dir / obj_hash[:_FANOUT] / obj_hash[_FANOUT:]

        # Fell through validation — figure out which rule was broken so the
        # error names it specifically, rather than reporting only "invalid".
        if not isinstance(obj_hash, str):
            raise TypeError(f"object hash must be a str, got {type(obj_hash).__name__}")
        if len(obj_hash) != 64:
            raise ValueError(
                f"object hash must be 64 hex characters, got {len(obj_hash)}"
            )
        raise ValueError(f"object hash is not hexadecimal: {obj_hash!r}")

    # -- writing ---------------------------------------------------------

    def put(self, payload: Any) -> str:
        """Store a JSON-serializable object; return its hash.

        Storing an object that is already present is a no-op — that is the
        deduplication property, and it is why an unchanged fact costs nothing
        to carry forward across commits.

        The write goes to a temporary file that is then atomically renamed into
        place. A crash mid-write therefore leaves either no object or a
        complete one, never a truncated file that would fail verification
        forever after.
        """
        raw = canonical_json(payload)
        obj_hash = hashlib.sha256(raw).hexdigest()
        path = self._path_for(obj_hash)

        if path.exists():
            return obj_hash

        path.parent.mkdir(parents=True, exist_ok=True)
        compressed = zlib.compress(raw, _COMPRESSION_LEVEL)

        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(compressed)
        # os.replace semantics: atomic on POSIX, and on Windows for same-volume
        # renames. Path.replace overwrites an existing destination, which is
        # what we want if a concurrent writer beat us to it — the contents are
        # identical by construction.
        tmp.replace(path)

        return obj_hash

    # -- reading ---------------------------------------------------------

    def get(self, obj_hash: str) -> Any:
        """Load and verify the object stored at ``obj_hash``.

        Raises:
            ObjectNotFoundError: No object is stored at that hash.
            CorruptObjectError: The object is unreadable, or its contents no
                longer hash to the address they are filed under.
        """
        path = self._path_for(obj_hash)
        if not path.exists():
            raise ObjectNotFoundError(obj_hash)

        try:
            raw = zlib.decompress(path.read_bytes())
        except zlib.error as exc:
            raise CorruptObjectError(
                f"object {obj_hash} could not be decompressed: {exc}"
            ) from exc

        # Verify before parsing. Content addressing gives us an integrity check
        # for free, and skipping it would waste the single best property of
        # this design.
        actual = hashlib.sha256(raw).hexdigest()
        if actual != obj_hash:
            raise CorruptObjectError(
                f"object {obj_hash} hashes to {actual}: contents do not match address"
            )

        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise CorruptObjectError(
                f"object {obj_hash} is not valid canonical JSON: {exc}"
            ) from exc

    def contains(self, obj_hash: str) -> bool:
        """Whether an object is stored at ``obj_hash``."""
        try:
            return self._path_for(obj_hash).exists()
        except (TypeError, ValueError):
            return False

    def __contains__(self, obj_hash: str) -> bool:
        return self.contains(obj_hash)

    # -- introspection ---------------------------------------------------

    def iter_hashes(self) -> Iterator[str]:
        """Yield every object hash in the store, in unspecified order."""
        if not self.objects_dir.exists():
            return
        for bucket in sorted(self.objects_dir.iterdir()):
            if not bucket.is_dir() or len(bucket.name) != _FANOUT:
                continue
            for obj in sorted(bucket.iterdir()):
                if obj.suffix == ".tmp":
                    continue
                yield bucket.name + obj.name

    def count(self) -> int:
        """Number of objects in the store."""
        return sum(1 for _ in self.iter_hashes())

    def size_on_disk(self) -> int:
        """Total compressed bytes of all stored objects.

        Paired with the uncompressed size of what was written, this is half of
        the storage-efficiency metric the README will quote — the other half
        being how many writes deduplicated away entirely.
        """
        total = 0
        for obj_hash in self.iter_hashes():
            total += self._path_for(obj_hash).stat().st_size
        return total

    def resolve_prefix(self, prefix: str, *, min_len: int = 4) -> str:
        """Expand a short hex prefix to the one full hash it identifies.

        The CLI's counterpart to ``git rev-parse`` accepting abbreviated SHAs:
        typing 64 hex characters in every demo command is the kind of friction
        that makes a tool feel unfinished, and the object store is exactly
        where the ambiguity check belongs, since it is the only thing that
        knows every hash that exists.

        Raises:
            ValueError: ``prefix`` is shorter than ``min_len`` or not hex.
            ObjectNotFoundError: no stored object's hash starts with ``prefix``.
            AmbiguousPrefixError: more than one does.
        """
        if len(prefix) < min_len:
            raise ValueError(
                f"prefix must be at least {min_len} characters, got {len(prefix)}"
            )
        try:
            int(prefix, 16)
        except ValueError as exc:
            raise ValueError(f"prefix is not hexadecimal: {prefix!r}") from exc

        if is_object_hash(prefix):
            if self.contains(prefix):
                return prefix
            raise ObjectNotFoundError(prefix)

        matches = [h for h in self.iter_hashes() if h.startswith(prefix)]
        if not matches:
            raise ObjectNotFoundError(prefix)
        if len(matches) > 1:
            raise AmbiguousPrefixError(
                f"prefix {prefix!r} matches {len(matches)} objects"
            )
        return matches[0]

    def verify(self) -> list[str]:
        """Read every object and return the hashes of any that fail to verify.

        The offline equivalent of ``git fsck`` for the object database. An
        empty list means the store is intact.
        """
        broken: list[str] = []
        for obj_hash in self.iter_hashes():
            try:
                self.get(obj_hash)
            except (CorruptObjectError, ObjectNotFoundError):
                broken.append(obj_hash)
        return broken

    def __repr__(self) -> str:
        return f"ObjectStore({str(self.objects_dir)!r})"


def hash_object(payload: Any) -> str:
    """Compute an object's hash without storing it.

    The counterpart to ``git hash-object`` without ``-w``: useful for asking
    "would this be a new object?" before deciding to write it.
    """
    return hash_payload(payload)
