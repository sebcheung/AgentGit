"""The Fact — MemGit's unit of memory.

A fact is a subject-predicate-object triple plus provenance metadata. The
triple is what makes memory *diffable*: `(subject, predicate)` acts as an
identity key, so two commits can be compared by asking, for each key, whether
the object changed. Raw text blobs cannot be compared that way.

Two distinct notions of identity live in this module, and keeping them apart is
the whole design:

``key``
    ``(subject, predicate)`` — *semantic* identity. "What claim is this about?"
    Used by the diff engine to line facts up across commits.

``fact_hash``
    SHA-256 over the canonical JSON of *every* field — *content* identity, the
    content-addressable-store address. Used for deduplication.

Because ``confidence``, ``asserted_at``, and ``source`` are inside the hashed
payload, re-asserting the same triple with higher confidence produces a
different hash — a genuinely new object at the same key. That is deliberate: it
is exactly what lets the diff engine see a fact being reaffirmed or
strengthened rather than silently collapsing it into the old one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from memgit.core.canonical import canonical_json, hash_payload, utcnow

__all__ = ["Fact", "FactKey", "canonical_json", "hash_payload"]

# The semantic identity of a fact: what claim it makes, ignoring how sure we
# are or where it came from.
FactKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Fact:
    """One thing an agent believes.

    Frozen because a fact is a *value*, not a record: the content-addressable
    store is keyed on the hash of its fields, so mutating one in place would
    leave it filed under an address that no longer describes it. Updating a
    belief means creating a new fact and committing it, which is what gives us
    a history to diff.

    Attributes:
        subject: What the claim is about (``"user"``, ``"project:memgit"``).
        predicate: The relation asserted (``"prefers_language"``).
        object: The value of the relation (``"Python"``). Always a string —
            typed values are a deliberate non-goal; see
            :class:`memgit.core.cardinality.CardinalityMap` for the small
            amount of schema we *do* keep.
        confidence: How sure the agent is, in ``[0.0, 1.0]``.
        asserted_at: ISO-8601 UTC timestamp of when this was recorded.
        source: Where the belief came from (``"session:42/turn:7"``). Opaque to
            MemGit; meaningful to whoever wrote it.
        source_text: The original natural-language claim, kept verbatim. This
            is the honesty field: the triple is a lossy interpretation, and
            keeping the raw text means a human reviewing a diff can always
            check whether the structure did the source justice.
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    asserted_at: str = field(default_factory=utcnow)
    source: str | None = None
    source_text: str | None = None

    def __post_init__(self) -> None:
        # Validate at construction so a malformed fact can never reach the
        # store. An unhashable or empty-keyed fact would corrupt the diff
        # engine's assumptions in ways that are painful to debug later.
        for name in ("subject", "predicate", "object"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"Fact.{name} must be a str, got {type(value).__name__}")
            if not value.strip():
                raise ValueError(f"Fact.{name} must be a non-empty string")

        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise TypeError("Fact.confidence must be a number")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Fact.confidence must be within [0.0, 1.0], got {self.confidence}"
            )

    @property
    def key(self) -> FactKey:
        """The ``(subject, predicate)`` pair the diff engine aligns facts by."""
        return (self.subject, self.predicate)

    @property
    def triple(self) -> tuple[str, str, str]:
        """The bare claim, ignoring confidence and provenance."""
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict.

        Optional fields are omitted when unset rather than written as ``null``,
        so a fact carrying no provenance hashes the same whether its optional
        fields were never passed or explicitly passed as ``None``. One value,
        one hash.
        """
        payload: dict[str, Any] = {
            "type": "fact",
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "asserted_at": self.asserted_at,
        }
        if self.source is not None:
            payload["source"] = self.source
        if self.source_text is not None:
            payload["source_text"] = self.source_text
        return payload

    _FIELDS = frozenset(
        {"type", "subject", "predicate", "object", "confidence", "asserted_at", "source", "source_text"}
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Fact:
        """Rebuild a fact from :meth:`to_dict` output.

        Strict about shape: an unknown key is rejected rather than silently
        dropped. Tolerating unrecognized fields would mean
        ``Fact.from_dict(store.get(h)).hash != h`` is reachable for a payload
        carrying one — a round-trip violation in a system whose entire premise
        is that a hash describes everything filed under it.
        """
        unknown = payload.keys() - cls._FIELDS
        if unknown:
            raise ValueError(f"fact payload has unknown keys: {sorted(unknown)}")

        kind = payload.get("type")
        if kind != "fact":
            raise ValueError(f"expected a fact object, got type={kind!r}")
        return cls(
            subject=payload["subject"],
            predicate=payload["predicate"],
            object=payload["object"],
            confidence=payload.get("confidence", 1.0),
            asserted_at=payload["asserted_at"],
            source=payload.get("source"),
            source_text=payload.get("source_text"),
        )

    @property
    def hash(self) -> str:
        """This fact's content address in the object store."""
        return hash_payload(self.to_dict())

    def reaffirm(
        self,
        *,
        confidence: float | None = None,
        source: str | None = None,
        source_text: str | None = None,
        asserted_at: str | None = None,
    ) -> Fact:
        """Return a copy of this fact re-asserted at the current time.

        The new fact shares this one's ``key`` and ``triple`` but hashes
        differently, so the store keeps both and the diff engine can report the
        belief as reaffirmed rather than unchanged.
        """
        return replace(
            self,
            confidence=self.confidence if confidence is None else confidence,
            source=self.source if source is None else source,
            source_text=self.source_text if source_text is None else source_text,
            asserted_at=asserted_at or utcnow(),
        )

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object} ({self.confidence:.2f})"
