"""Tests for the Commit — one node in the memory DAG.

Grouped around the properties that carry the design: identity (hash changes
with any field, including the ones that behave differently from Fact),
validation (a malformed commit can't reach the store), serialization
(parents always emitted, metadata omitted when unset, unknown keys rejected),
and the small helper properties log/show build on.
"""

from __future__ import annotations

import pytest

from memgit.core.commit import Commit
from memgit.core.store import ObjectStore

TREE_A = "a" * 64
TREE_B = "b" * 64
PARENT = "c" * 64


@pytest.fixture
def store(tmp_path) -> ObjectStore:
    return ObjectStore(tmp_path / "objects")


def make_commit(**overrides) -> Commit:
    defaults = dict(
        tree=TREE_A,
        parents=(PARENT,),
        message="learned the user's timezone",
        author="agent:claude-opus-5",
        committed_at="2026-09-01T14:03:22+00:00",
    )
    defaults.update(overrides)
    return Commit(**defaults)


class TestIdentity:
    def test_same_fields_hash_the_same(self):
        assert make_commit().hash == make_commit().hash

    def test_hash_changes_with_tree(self):
        assert make_commit().hash != make_commit(tree=TREE_B).hash

    def test_hash_changes_with_parents(self):
        assert make_commit().hash != make_commit(parents=()).hash

    def test_hash_changes_with_message(self):
        assert make_commit().hash != make_commit(message="different").hash

    def test_hash_changes_with_author(self):
        assert make_commit().hash != make_commit(author="cli:sebastian").hash

    def test_hash_changes_with_committed_at(self):
        other = make_commit(committed_at="2026-09-02T00:00:00+00:00")
        assert make_commit().hash != other.hash

    def test_hash_changes_with_metadata(self):
        assert make_commit().hash != make_commit(metadata={"session": "42"}).hash

    def test_hash_is_not_a_stored_field(self):
        """The hash must be a pure function of content, not a second source
        of truth that a stale value could disagree with."""
        assert "hash" not in make_commit().to_dict()


class TestValidation:
    def test_rejects_non_hex_tree(self):
        with pytest.raises(ValueError, match="tree"):
            Commit(tree="not-a-hash")

    def test_rejects_non_hex_parent(self):
        with pytest.raises(ValueError, match="parents"):
            Commit(tree=TREE_A, parents=("not-a-hash",))

    def test_rejects_duplicate_parents(self):
        with pytest.raises(ValueError, match="duplicate"):
            Commit(tree=TREE_A, parents=(PARENT, PARENT))

    def test_rejects_empty_author(self):
        with pytest.raises(ValueError, match="author"):
            Commit(tree=TREE_A, author="")

    def test_rejects_non_serializable_metadata(self):
        with pytest.raises(ValueError, match="metadata"):
            Commit(tree=TREE_A, metadata={"session": object()})


class TestProperties:
    def test_is_root_when_no_parents(self):
        assert make_commit(parents=()).is_root
        assert not make_commit(parents=(PARENT,)).is_root

    def test_is_merge_when_multiple_parents(self):
        other_parent = "d" * 64
        assert make_commit(parents=(PARENT, other_parent)).is_merge
        assert not make_commit(parents=(PARENT,)).is_merge

    def test_summary_is_first_line_of_message(self):
        commit = make_commit(message="first line\nsecond line")
        assert commit.summary == "first line"

    def test_summary_of_empty_message_is_empty(self):
        assert make_commit(message="").summary == ""


class TestSerialization:
    def test_parents_are_always_emitted_even_when_empty(self):
        """Unlike Fact's optional fields, an empty parent list is a
        meaningful assertion ('history begins here'), not an unset value."""
        payload = make_commit(parents=()).to_dict()
        assert payload["parents"] == []

    def test_metadata_omitted_when_unset(self):
        payload = make_commit().to_dict()
        assert "metadata" not in payload

    def test_metadata_present_when_set(self):
        payload = make_commit(metadata={"session": "42"}).to_dict()
        assert payload["metadata"] == {"session": "42"}

    def test_round_trips(self):
        commit = make_commit(metadata={"session": "42"})
        assert Commit.from_dict(commit.to_dict()) == commit

    def test_round_trip_preserves_hash(self):
        commit = make_commit()
        assert Commit.from_dict(commit.to_dict()).hash == commit.hash

    def test_from_dict_rejects_other_object_types(self):
        with pytest.raises(ValueError, match="expected a commit"):
            Commit.from_dict({"type": "tree", "tree": TREE_A, "parents": []})

    def test_from_dict_rejects_unknown_keys(self):
        payload = make_commit().to_dict()
        payload["extra"] = "surprise"
        with pytest.raises(ValueError, match="unknown keys"):
            Commit.from_dict(payload)

    def test_from_dict_rejects_missing_tree(self):
        with pytest.raises(ValueError, match="missing"):
            Commit.from_dict({"type": "commit", "parents": []})

    def test_write_then_read_preserves_hash(self, store):
        commit = make_commit()
        commit_hash = commit.write(store)
        assert Commit.read(store, commit_hash).hash == commit_hash

    def test_str_shows_short_hash_and_summary(self):
        commit = make_commit(message="seed")
        text = str(commit)
        assert commit.hash[:8] in text
        assert "seed" in text
