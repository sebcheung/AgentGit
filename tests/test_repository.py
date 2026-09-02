"""Tests for Repository — the .memgit layout and the commit protocol.

Grouped by the properties that carry the design: layout (init produces
exactly the expected shape, discover walks upward), the commit protocol
(parents chain correctly, empty commits are refused by default, detached
HEAD moves HEAD not a branch), the dedup metric the README quotes, and
crash-safety ordering (objects before refs, always).
"""

from __future__ import annotations

import pytest

from memgit.core.cardinality import CardinalityMap
from memgit.core.diff import ChangeKind
from memgit.core.fact import Fact
from memgit.core.repository import (
    EmptyCommitError,
    NotARepositoryError,
    Repository,
    RepositoryExistsError,
    RevisionNotFoundError,
)
from memgit.core.tree import EMPTY_TREE_HASH


def make_fact(**overrides) -> Fact:
    defaults = dict(
        subject="user",
        predicate="prefers_language",
        object="Python",
        confidence=0.9,
        asserted_at="2026-08-11T12:00:00+00:00",
    )
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository.init(tmp_path)


class TestInit:
    def test_creates_the_expected_layout(self, tmp_path):
        Repository.init(tmp_path)
        memgit_dir = tmp_path / ".memgit"
        assert (memgit_dir / "objects").is_dir()
        assert (memgit_dir / "refs" / "heads").is_dir()
        assert (memgit_dir / "HEAD").is_file()
        assert (memgit_dir / "config").is_file()

    def test_head_points_at_an_unborn_default_branch(self, tmp_path):
        repo = Repository.init(tmp_path)
        head = repo.head()
        assert head.ref == "refs/heads/main"
        assert head.commit is None

    def test_respects_a_custom_default_branch(self, tmp_path):
        repo = Repository.init(tmp_path, default_branch="trunk")
        assert repo.head().ref == "refs/heads/trunk"

    def test_reinit_raises(self, tmp_path):
        Repository.init(tmp_path)
        with pytest.raises(RepositoryExistsError):
            Repository.init(tmp_path)

    def test_config_records_format_version(self, tmp_path):
        repo = Repository.init(tmp_path)
        assert repo.config()["format_version"] == Repository.FORMAT_VERSION


class TestDiscover:
    def test_finds_the_repo_from_its_own_directory(self, tmp_path):
        Repository.init(tmp_path)
        found = Repository.discover(tmp_path)
        assert found.memgit_dir == tmp_path.resolve() / ".memgit"

    def test_finds_the_repo_from_a_nested_subdirectory(self, tmp_path):
        Repository.init(tmp_path)
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        found = Repository.discover(nested)
        assert found.memgit_dir == tmp_path.resolve() / ".memgit"

    def test_raises_outside_any_repository(self, tmp_path):
        with pytest.raises(NotARepositoryError):
            Repository.discover(tmp_path)

    def test_open_requires_the_exact_directory(self, tmp_path):
        Repository.init(tmp_path)
        with pytest.raises(NotARepositoryError):
            Repository.open(tmp_path / "not-there")
        assert Repository.open(tmp_path / ".memgit").memgit_dir == tmp_path.resolve() / ".memgit"


class TestFirstCommit:
    def test_has_no_parents(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        assert repo.read_commit(commit_hash).parents == ()

    def test_creates_the_branch_ref(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        assert repo.branches()["main"] == commit_hash

    def test_head_now_resolves_to_it(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        assert repo.head_commit() == commit_hash

    def test_tree_contains_the_committed_facts(self, repo):
        fact = make_fact()
        commit_hash = repo.commit([fact], "seed")
        tree = repo.read_tree(commit_hash)
        assert tree.by_key()[fact.key] == (fact.hash,)


class TestSubsequentCommits:
    def test_second_commit_parents_the_first(self, repo):
        first = repo.commit([make_fact()], "seed")
        second = repo.commit([make_fact(object="Rust")], "update")
        assert repo.read_commit(second).parents == (first,)

    def test_log_returns_both_newest_first(self, repo):
        first = repo.commit([make_fact()], "seed")
        second = repo.commit([make_fact(object="Rust")], "update")
        assert [c.hash for c in repo.log()] == [second, first]

    def test_log_respects_limit(self, repo):
        repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        assert len(list(repo.log(limit=1))) == 1


class TestEmptyCommits:
    def test_empty_commit_raises_by_default(self, repo):
        fact = make_fact()
        repo.commit([fact], "seed")
        with pytest.raises(EmptyCommitError):
            repo.commit([fact], "no-op")

    def test_allow_empty_succeeds_with_a_distinct_hash(self, repo):
        fact = make_fact()
        first = repo.commit([fact], "seed")
        second = repo.commit([fact], "no-op", allow_empty=True)
        assert first != second

    def test_first_commit_of_an_empty_state_is_never_empty(self, repo):
        # No parent to compare against, so this must succeed without allow_empty.
        commit_hash = repo.commit([], "empty state")
        assert repo.read_commit(commit_hash).is_root


class TestDetachedHead:
    def test_committing_while_detached_moves_head_not_the_branch(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.refs.detach_head(first)

        second = repo.commit([make_fact(object="Rust")], "detached update")

        assert repo.head_commit() == second
        assert repo.head().is_detached
        assert repo.branches()["main"] == first


class TestDedupMetric:
    def test_reaffirming_the_same_facts_costs_almost_nothing(self, repo):
        """The number the README quotes should be a test, not a spreadsheet."""
        facts = [make_fact(subject=f"user{i}", predicate="likes", object="chess") for i in range(20)]
        repo.commit(facts, "seed")
        before = repo.store.count()

        for i in range(9):
            repo.commit(facts, f"reaffirm {i}", allow_empty=True)

        after = repo.store.count()
        # None of the 20 facts is re-stored, and since nothing changed, the
        # tree object itself is identical too — only a new commit object is
        # added per reaffirmation, not a new tree and a new commit.
        assert after - before == 9


class TestCrashSafety:
    def test_ref_write_failure_leaves_objects_but_does_not_move_the_ref(self, repo, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("simulated crash")

        monkeypatch.setattr(repo.refs, "write_ref", boom)

        with pytest.raises(OSError):
            repo.commit([make_fact()], "seed")

        # The tree and commit objects were written before the failed ref
        # write — garbage now, but not corruption.
        assert repo.store.count() >= 2
        assert repo.head().commit is None


class TestResolve:
    def test_resolves_head(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        assert repo.resolve("HEAD") == commit_hash

    def test_resolves_a_branch_name(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        assert repo.resolve("main") == commit_hash

    def test_resolves_a_full_hash(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        assert repo.resolve(commit_hash) == commit_hash

    def test_unknown_revision_raises(self, repo):
        with pytest.raises(RevisionNotFoundError):
            repo.resolve("does-not-exist")


class TestBranches:
    def test_create_branch_at_head(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        assert repo.create_branch("experiment") == commit_hash
        assert repo.branches()["experiment"] == commit_hash

    def test_create_branch_at_explicit_revision(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        repo.create_branch("from-first", at=first)
        assert repo.branches()["from-first"] == first

    def test_delete_branch(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        repo.create_branch("experiment", at=commit_hash)
        repo.delete_branch("experiment")
        assert "experiment" not in repo.branches()

    def test_deleting_the_current_branch_is_refused(self, repo):
        repo.commit([make_fact()], "seed")
        with pytest.raises(ValueError, match="current branch"):
            repo.delete_branch("main")

    def test_force_deleting_the_current_branch_is_allowed(self, repo):
        repo.commit([make_fact()], "seed")
        repo.delete_branch("main", force=True)
        assert "main" not in repo.branches()


class TestEmptyTree:
    def test_read_tree_of_empty_hash_does_not_touch_the_store(self, repo):
        tree = repo.read_tree(EMPTY_TREE_HASH)
        assert len(tree) == 0
        assert EMPTY_TREE_HASH not in repo.store


class TestCardinality:
    def test_defaults_when_the_file_is_absent(self, repo):
        assert repo.cardinality() == CardinalityMap.default_map()

    def test_set_then_get_round_trips(self, repo):
        repo.set_cardinality(CardinalityMap({"likes": "multi"}))
        assert repo.cardinality() == CardinalityMap({"likes": "multi"})

    def test_set_writes_atomically_no_lock_left_behind(self, repo):
        repo.set_cardinality(CardinalityMap({"likes": "multi"}))
        assert not (repo.memgit_dir / "cardinality.json.lock").exists()
        assert (repo.memgit_dir / "cardinality.json").is_file()

    def test_malformed_file_raises_clearly(self, repo):
        (repo.memgit_dir / "cardinality.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            repo.cardinality()


class TestDiff:
    def test_root_commit_diffs_against_the_empty_tree(self, repo):
        fact = make_fact()
        commit_hash = repo.commit([fact], "seed")
        result = repo.diff(after=commit_hash)
        assert len(result.keys) == 1
        assert result.keys[0].kind == ChangeKind.ADDED

    def test_diff_resolves_branch_names(self, repo):
        repo.commit([make_fact()], "seed")
        repo.create_branch("experiment")
        result = repo.diff("experiment", "main")
        assert result.is_empty  # both branches point at the same commit

    def test_use_merge_base_differs_from_direct_diff_on_a_fork(self, repo):
        repo.commit([make_fact()], "seed")
        repo.create_branch("experiment")
        repo.commit([make_fact(predicate="current_project", object="memgit")], "main-only")
        repo.refs.detach_head(repo.resolve("experiment"))
        repo.commit([make_fact(predicate="timezone", object="UTC")], "experiment-only")
        repo.create_branch("experiment-tip", at="HEAD")

        direct = repo.diff("main", "experiment-tip")
        via_base = repo.diff("main", "experiment-tip", use_merge_base=True)
        assert direct.keys != via_base.keys

    def test_use_merge_base_requires_an_explicit_before_revision(self, repo):
        repo.commit([make_fact()], "seed")
        with pytest.raises(ValueError):
            repo.diff(use_merge_base=True)
