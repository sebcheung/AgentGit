"""Tests for staging areas — the MCP write path's cross-process buffer.

Grouped by the properties slice 8 leans on: the ref layout stays inspectable
with existing commands, a stage/seal round-trip behaves like a normal
commit when nobody else wrote in the meantime, a diverged branch tip loses
nothing untouched by the session, an abandoned staging area is neither lost
nor mistaken for commit history, and a concurrent writer is *detected*
rather than silently overwritten.
"""

from __future__ import annotations

import pytest

from memgit.core.fact import Fact
from memgit.core.repository import Repository
from memgit.core.staging import StagingConflictError, session_key
from memgit.core.tree import EMPTY_TREE_HASH


def make_fact(**overrides) -> Fact:
    defaults = {
        "subject": "user",
        "predicate": "prefers_language",
        "object": "Python",
        "confidence": 0.9,
        "asserted_at": "2026-08-11T12:00:00+00:00",
    }
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository.init(tmp_path)


def stage_fresh(repo: Repository, session_id: str, facts: list[Fact]):
    """Open + stage in one step, for tests that don't care about a fold."""
    opened = repo.open_staging(session_id)
    return repo.stage(session_id, facts, based_on=opened.tree)


class TestSessionKey:
    def test_deterministic(self):
        assert session_key("abc") == session_key("abc")

    def test_different_ids_different_keys(self):
        assert session_key("abc") != session_key("xyz")

    def test_key_is_ref_safe(self):
        # A raw id with a slash must never reach the filesystem unhashed.
        key = session_key("../../etc/passwd")
        assert "/" not in key and ".." not in key


class TestOpenStagingOnUnbornBranch:
    def test_opens_against_the_empty_tree(self, repo):
        area = repo.open_staging("s1")
        assert area.base is None
        assert area.tree == EMPTY_TREE_HASH

    def test_staging_area_is_inspectable_via_read_tree(self, repo):
        area = stage_fresh(repo, "s1", [make_fact()])
        tree = repo.read_tree(area.tree)
        assert [f.object for f in tree.load(repo.store)] == ["Python"]

    def test_staging_area_is_diffable_against_head(self, repo):
        # Explicit `before` (never a bare `None`, which asks `diff` to
        # resolve a commit's parent — meaningless for a raw tree hash):
        # the empty tree is the honest "believe nothing" baseline for a
        # staging area opened on an unborn branch.
        area = stage_fresh(repo, "s1", [make_fact()])
        diff = repo.diff(before=EMPTY_TREE_HASH, after=area.tree)
        assert len(diff.keys) == 1


class TestOpenStagingOnBornBranch:
    def test_opens_from_the_current_tip(self, repo):
        first = repo.commit([make_fact(object="Python")], "seed")
        area = repo.open_staging("s1")
        assert area.base == first
        assert area.tree == repo.read_commit(first).tree


class TestSealFastPath:
    def test_seal_matches_a_direct_commit_when_nothing_else_moved(self, repo):
        repo.commit([make_fact(object="Python")], "seed")
        area = stage_fresh(repo, "s1", [make_fact(object="Rust")])
        commit_hash = repo.seal_staging("s1", "renamed")
        assert commit_hash is not None
        assert repo.read_commit(commit_hash).tree == area.tree

    def test_seal_drops_the_staging_area(self, repo):
        repo.commit([make_fact()], "seed")
        stage_fresh(repo, "s1", [make_fact(object="Rust")])
        repo.seal_staging("s1", "renamed")
        assert repo.staging_area(session_key("s1")) is None

    def test_seal_advances_the_configured_branch_not_head(self, repo):
        repo.commit([make_fact()], "seed")
        repo.checkout("HEAD", detach=True)  # detach HEAD; the write path must not care
        stage_fresh(repo, "s1", [make_fact(object="Rust")])
        commit_hash = repo.seal_staging("s1", "renamed")
        assert repo.branches()["main"] == commit_hash

    def test_nothing_staged_returns_none(self, repo):
        assert repo.seal_staging("nonexistent", "message") is None

    def test_seal_with_no_touched_keys_returns_none_and_drops(self, repo):
        repo.commit([make_fact()], "seed")
        stage_fresh(repo, "s1", [make_fact()])  # identical to the tip: touches nothing
        assert repo.seal_staging("s1", "no-op") is None
        assert repo.staging_area(session_key("s1")) is None


class TestSealUnderADivergedTip:
    def test_untouched_keys_from_the_other_writer_survive(self, repo):
        repo.commit(
            [make_fact(predicate="prefers_language", object="Python"), make_fact(predicate="timezone", object="UTC")],
            "seed",
        )
        stage_fresh(
            repo,
            "s1",
            [make_fact(predicate="prefers_language", object="Rust"), make_fact(predicate="timezone", object="UTC")],
        )

        # A second writer changes a *different* key after s1 started staging.
        repo.commit(
            [make_fact(predicate="prefers_language", object="Python"), make_fact(predicate="timezone", object="PST")],
            "other writer",
        )

        commit_hash = repo.seal_staging("s1", "s1 seals")
        state = repo.state(commit_hash)
        assert state.get("user", "prefers_language")[0].object == "Rust"  # s1's change won
        assert state.get("user", "timezone")[0].object == "PST"  # other writer's change survived

    def test_same_key_collision_is_recorded_not_silently_resolved(self, repo):
        repo.commit([make_fact(object="Python")], "seed")
        stage_fresh(repo, "s1", [make_fact(object="Rust")])
        repo.commit([make_fact(object="Go")], "other writer")  # same key, different value

        commit_hash = repo.seal_staging("s1", "s1 seals")
        commit = repo.read_commit(commit_hash)
        assert commit.metadata["restaged_over"] == [["user", "prefers_language"]]
        # Last sealer wins the contested key.
        assert repo.state(commit_hash).get("user", "prefers_language")[0].object == "Rust"

    def test_seal_records_staged_on_metadata(self, repo):
        first = repo.commit([make_fact()], "seed")
        stage_fresh(repo, "s1", [make_fact(object="Rust")])
        commit_hash = repo.seal_staging("s1", "s1 seals")
        assert repo.read_commit(commit_hash).metadata["staged_on"] == first


class TestSealEmptyCommit:
    def test_a_diverged_seal_that_ends_up_empty_returns_none(self, repo):
        repo.commit([make_fact(object="Python")], "seed")
        stage_fresh(repo, "s1", [make_fact(object="Rust")])
        # Someone else makes the exact change s1 was going to make.
        repo.commit([make_fact(object="Rust")], "other writer")
        assert repo.seal_staging("s1", "s1 seals") is None
        assert repo.staging_area(session_key("s1")) is None


class TestAbandonedStaging:
    def test_survives_and_stays_readable_with_no_seal(self, repo):
        repo.commit([make_fact()], "seed")
        area = stage_fresh(repo, "s1", [make_fact(object="Rust")])
        # Simulate a fresh process: open a brand new Repository handle.
        reopened = Repository.open(repo.memgit_dir)
        assert reopened.staging_area(session_key("s1")) == area

    def test_never_confused_with_commit_history(self, repo):
        area = stage_fresh(repo, "s1", [make_fact()])
        with pytest.raises(ValueError):
            repo.read_commit(area.tree)

    def test_staging_areas_lists_every_open_session(self, repo):
        repo.open_staging("s1")
        repo.open_staging("s2")
        keys = {area.key for area in repo.staging_areas()}
        assert keys == {session_key("s1"), session_key("s2")}

    def test_drop_staging_discards_without_committing(self, repo):
        repo.open_staging("s1")
        repo.drop_staging(session_key("s1"))
        assert repo.staging_area(session_key("s1")) is None
        assert repo.head_commit() is None


class TestConcurrentStaging:
    def test_stale_based_on_raises_staging_conflict_error(self, repo):
        # Two "clients" both read the same starting point...
        client_a_view = repo.open_staging("s1")
        client_b_view = repo.open_staging("s1")

        # ...client A writes first and wins the compare-and-swap...
        repo.stage("s1", [make_fact(object="Rust")], based_on=client_a_view.tree)

        # ...client B's fold was computed against the now-stale snapshot, so
        # its write must be rejected rather than silently clobbering A's.
        with pytest.raises(StagingConflictError):
            repo.stage("s1", [make_fact(object="Go")], based_on=client_b_view.tree)

        # A's write is the one that stuck.
        assert repo.staging_area(session_key("s1")).tree == repo.write_tree([make_fact(object="Rust")])

    def test_retry_after_conflict_succeeds(self, repo):
        area = repo.open_staging("s1")
        repo.stage("s1", [make_fact(object="Rust")], based_on=area.tree)

        with pytest.raises(StagingConflictError):
            repo.stage("s1", [make_fact(object="Go")], based_on=area.tree)

        # Re-read and retry, exactly as a real caller would.
        current = repo.staging_area(session_key("s1"))
        retried = repo.stage("s1", [make_fact(object="Go")], based_on=current.tree)
        assert repo.read_tree(retried.tree).load(repo.store)[0].object == "Go"
