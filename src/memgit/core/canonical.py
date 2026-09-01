"""Canonical serialization and content hashing.

Every object type in MemGit — facts, and soon trees and commits — is addressed by
the SHA-256 of its own canonical JSON. That means the hashing rules can't live inside
any one object's module without becoming a lie the moment a second object type needs
them too. This module is the single place that defines what "the same logical value"
means in bytes.

Hashing has one hard requirement: the same logical value must always produce the same
bytes. A plain ``json.dumps`` does not guarantee that — key order follows insertion
order, and the default separators embed incidental whitespace. Both would let an
identical object hash two different ways depending on how it happened to be
constructed, which would silently break deduplication.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

__all__ = ["canonical_json", "hash_payload", "utcnow"]


def canonical_json(payload: Any) -> bytes:
    """Serialize ``payload`` to a byte string that is stable across runs.

    So we pin down every degree of freedom:

    - ``sort_keys=True`` makes key order a function of the value, not of
      construction order.
    - ``separators=(",", ":")`` removes whitespace entirely.
    - ``ensure_ascii=False`` + explicit UTF-8 keeps non-ASCII text as itself
      rather than as ``\\uXXXX`` escapes, so the encoding is one obvious thing.
    - ``allow_nan=False`` rejects ``NaN``/``Infinity``, which are not valid
      JSON and do not round-trip through other parsers.
    """
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def hash_payload(payload: Any) -> str:
    """Return the SHA-256 hex digest of ``payload``'s canonical JSON.

    SHA-256 rather than git's SHA-1 because there is no legacy to be compatible
    with, and choosing a broken hash on purpose in 2026 is hard to defend.
    """
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def utcnow() -> str:
    """Current UTC time as an ISO-8601 string with an explicit offset."""
    return datetime.now(timezone.utc).isoformat()
