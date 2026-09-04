"""``VectorIndex`` — a ``fact_hash -> vector`` cache, persisted on disk.

Laid out one directory per embedder, mirroring the on-disk-format instinct
``store.py`` already established for objects::

    .memgit/embeddings/<embedder-id-slug>/vectors.pack    raw float32, append-only
    .memgit/embeddings/<embedder-id-slug>/index.json      {"records": {"<fact_hash>": <record#>}}

Keyed by ``fact_hash`` because :class:`~memgit.core.fact.Fact`'s hash covers
every field — the same triple asserted with different confidence or
provenance is a different fact and gets its own vector, and a vector once
written is **never stale for content reasons**, only if the embedder itself
changes (hence the embedder-id namespace: swapping embedders means writing
into a fresh directory, not invalidating the old one).

**The pack is written before the index that points into it — always.** This
is the exact ordering argument ``repository.py``'s module docstring makes
for objects-before-refs, one level down: a crash between the two steps
leaves trailing, unreferenced bytes in the pack (harmless, collectable
garbage) rather than an index entry pointing past the end of the pack file
(corruption a reader cannot recover from).

This is a **cache**, not history: nothing here is committed or hashed into
``.memgit/objects``, and every byte is reconstructible from the object store
plus an embedder by re-running ``memgit embed``. That is a stronger claim
than ``cardinality.json``'s "repo-local config" status — this data isn't
even a question worth asking twice, it's a memoized answer.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path

__all__ = ["VectorIndex", "VectorIndexError"]

_INDEX_NAME = "index.json"
_PACK_NAME = "vectors.pack"
_VERSION = 1
_FLOAT_SIZE = 4  # struct format "f", little-endian


class VectorIndexError(Exception):
    """Raised when the on-disk vector index is missing, malformed, or corrupt."""


def _slug(embedder_id: str) -> str:
    """Turn an embedder id (``"hash-v1/256"``) into a filesystem-safe directory name."""
    return embedder_id.replace("/", "_")


class VectorIndex:
    """A ``fact_hash -> vector`` cache for one embedder, rooted at ``directory``.

    Construct with :meth:`open`, which resolves the embedder-specific
    subdirectory; the bare constructor assumes ``path`` already is that
    subdirectory (or does not exist yet — nothing is created until the
    first :meth:`put`).
    """

    def __init__(self, path: Path, *, embedder_id: str, dim: int) -> None:
        self.path = Path(path)
        self.embedder_id = embedder_id
        self.dim = dim
        self._offsets: dict[str, int] | None = None

    @classmethod
    def open(cls, embeddings_dir: Path, *, embedder_id: str, dim: int) -> VectorIndex:
        """Open (without creating) the index for ``embedder_id`` under ``embeddings_dir``."""
        return cls(Path(embeddings_dir) / _slug(embedder_id), embedder_id=embedder_id, dim=dim)

    # -- paths -------------------------------------------------------------

    @property
    def _pack_path(self) -> Path:
        return self.path / _PACK_NAME

    @property
    def _index_path(self) -> Path:
        return self.path / _INDEX_NAME

    # -- offsets -------------------------------------------------------------

    def _load_offsets(self) -> dict[str, int]:
        if self._offsets is not None:
            return self._offsets
        if not self._index_path.is_file():
            self._offsets = {}
            return self._offsets

        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorIndexError(f"malformed vector index at {self._index_path}: {exc}") from exc

        if payload.get("version") != _VERSION:
            raise VectorIndexError(f"unsupported vector index version: {payload.get('version')!r}")
        if payload.get("embedder") != self.embedder_id:
            raise VectorIndexError(
                f"vector index at {self._index_path} was built for embedder "
                f"{payload.get('embedder')!r}, not {self.embedder_id!r}"
            )
        if payload.get("dim") != self.dim:
            raise VectorIndexError(
                f"vector index at {self._index_path} has dim {payload.get('dim')!r}, expected {self.dim!r}"
            )

        self._offsets = dict(payload.get("records", {}))
        return self._offsets

    def _write_offsets(self, offsets: dict[str, int]) -> None:
        payload = {
            "version": _VERSION,
            "embedder": self.embedder_id,
            "dim": self.dim,
            "records": offsets,
        }
        lock = self._index_path.with_name(self._index_path.name + ".lock")
        lock.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lock.replace(self._index_path)
        self._offsets = offsets

    # -- lookup ------------------------------------------------------------

    def __contains__(self, fact_hash: str) -> bool:
        return fact_hash in self._load_offsets()

    def __len__(self) -> int:
        return len(self._load_offsets())

    def get(self, fact_hash: str) -> tuple[float, ...] | None:
        """The cached vector for ``fact_hash``, or ``None`` on a miss.

        Raises:
            VectorIndexError: the index names a record past the end of the
                pack file — a truncated pack from an interrupted write.
        """
        offsets = self._load_offsets()
        record = offsets.get(fact_hash)
        if record is None:
            return None

        record_size = self.dim * _FLOAT_SIZE
        byte_offset = record * record_size
        with self._pack_path.open("rb") as handle:
            handle.seek(byte_offset)
            raw = handle.read(record_size)
        if len(raw) != record_size:
            raise VectorIndexError(
                f"vector index entry for {fact_hash!r} points past the end of "
                f"{self._pack_path} — a truncated or corrupt pack file"
            )
        return struct.unpack(f"<{self.dim}f", raw)

    def put(self, fact_hash: str, vector: tuple[float, ...], *, force: bool = False) -> None:
        """Cache ``vector`` for ``fact_hash``, appending to the pack first.

        A no-op if ``fact_hash`` is already present and ``force`` is not
        set — a vector is ordinarily immutable once written, matching the
        fact it describes. ``force=True`` (``memgit embed --rebuild``)
        still only ever appends: the old record becomes unreferenced,
        harmless pack bytes, the same crash-safe "orphan, not corruption"
        posture the rest of this module keeps.

        Raises:
            VectorIndexError: ``len(vector) != self.dim``.
        """
        if len(vector) != self.dim:
            raise VectorIndexError(f"vector has {len(vector)} dim(s), expected {self.dim}")

        offsets = self._load_offsets()
        if fact_hash in offsets and not force:
            return

        self.path.mkdir(parents=True, exist_ok=True)
        record_size = self.dim * _FLOAT_SIZE
        with self._pack_path.open("ab") as handle:
            handle.write(struct.pack(f"<{self.dim}f", *vector))
            handle.flush()

        pack_size = self._pack_path.stat().st_size
        record_index = (pack_size - record_size) // record_size

        updated = dict(offsets)
        updated[fact_hash] = record_index
        self._write_offsets(updated)

    def keys(self) -> Iterator[str]:
        """Every ``fact_hash`` currently cached."""
        return iter(self._load_offsets())
