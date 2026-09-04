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
from memgit.core.decay import DecayPolicy
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

    def test_resolves_head_tilde_n(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        assert repo.resolve("HEAD~1") == first

    def test_resolves_a_branch_name_with_ancestry_suffix(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        assert repo.resolve("main~1") == first

    def test_malformed_ancestry_suffix_raises_revision_not_found(self, repo):
        repo.commit([make_fact()], "seed")
        with pytest.raises(RevisionNotFoundError):
            repo.resolve("HEAD~x")

    def test_ancestry_suffix_past_a_root_raises_revision_not_found(self, repo):
        repo.commit([make_fact()], "seed")
        with pytest.raises(RevisionNotFoundError):
            repo.resolve("HEAD~5")

    def test_resolves_an_abbreviated_hash(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        assert repo.resolve(commit_hash[:8]) == commit_hash

    def test_a_branch_name_shadows_a_valid_hash_prefix(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        # A branch literally named after a hash prefix must win over prefix
        # resolution — bare-name resolution is tried first.
        repo.create_branch(commit_hash[:8])
        assert repo.resolve(commit_hash[:8]) == commit_hash

    def test_resolves_head_at_reflog_index(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        assert repo.resolve("HEAD@{1}") == first

    def test_reflog_index_past_the_start_raises(self, repo):
        repo.commit([make_fact()], "seed")
        with pytest.raises(RevisionNotFoundError):
            repo.resolve("HEAD@{99}")


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


class TestDecay:
    def test_defaults_when_the_file_is_absent(self, repo):
        assert repo.decay() == DecayPolicy.default_map()

    def test_set_then_get_round_trips(self, repo):
        repo.set_decay(DecayPolicy({"likes": 30.0}))
        assert repo.decay() == DecayPolicy({"likes": 30.0})

    def test_set_writes_atomically_no_lock_left_behind(self, repo):
        repo.set_decay(DecayPolicy({"likes": 30.0}))
        assert not (repo.memgit_dir / "decay.json.lock").exists()
        assert (repo.memgit_dir / "decay.json").is_file()

    def test_malformed_file_raises_clearly(self, repo):
        (repo.memgit_dir / "decay.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            repo.decay()

    def test_init_does_not_create_a_decay_file(self, repo):
        assert not (repo.memgit_dir / "decay.json").exists()


class TestRetrieval:
    def test_init_does_not_create_the_embeddings_dir(self, repo):
        assert not repo.embeddings_dir.exists()

    def test_embedder_defaults_to_hashing(self, repo):
        embedder = repo.embedder()
        assert embedder.id.startswith("hash-v1/")

    def test_vector_index_is_scoped_to_the_embedder(self, repo):
        index = repo.vector_index()
        assert index.embedder_id == repo.embedder().id
        assert index.dim == repo.embedder().dim

    def test_vector_index_put_creates_the_embeddings_dir(self, repo):
        embedder = repo.embedder()
        index = repo.vector_index(embedder)
        index.put("abc123", embedder.embed("hello"))
        assert repo.embeddings_dir.is_dir()


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


class TestCheckout:
    def test_checkout_a_branch_attaches(self, repo):
        repo.commit([make_fact()], "seed")
        repo.create_branch("experiment")
        result = repo.checkout("experiment")
        assert not result.detached
        assert repo.current_branch() == "experiment"

    def test_checkout_a_commit_detaches(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        result = repo.checkout(first)
        assert result.detached
        assert repo.head_commit() == first
        assert repo.branches()["main"] != first  # the branch itself did not move

    def test_checkout_an_ancestry_expression_detaches_even_though_the_base_is_a_branch(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        result = repo.checkout("main~1")
        assert result.detached
        assert repo.head_commit() == first

    def test_checkout_detach_flag_forces_detachment_of_a_branch(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        result = repo.checkout("main", detach=True)
        assert result.detached
        assert repo.head_commit() == commit_hash

    def test_checkout_unknown_revision_raises(self, repo):
        repo.commit([make_fact()], "seed")
        with pytest.raises(RevisionNotFoundError):
            repo.checkout("does-not-exist")

    def test_checkout_create_and_switch(self, repo):
        repo.commit([make_fact()], "seed")
        result = repo.checkout("HEAD", create="topic")
        assert result.created_branch == "topic"
        assert repo.current_branch() == "topic"
        assert repo.branches()["topic"] == repo.head_commit()

    def test_checkout_create_on_an_unborn_repository_leaves_it_unborn(self, repo):
        repo.checkout("HEAD", create="topic")
        assert repo.current_branch() == "topic"
        assert repo.head_commit() is None
        assert "topic" not in repo.branches()  # no ref file yet — still unborn

    def test_checkout_result_previous_and_moved(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        result = repo.checkout(first)
        assert result.previous.commit != result.head.commit
        assert result.moved

    def test_checkout_the_current_branch_is_a_no_op(self, repo):
        repo.commit([make_fact()], "seed")
        result = repo.checkout("main")
        assert not result.moved
        assert repo.current_branch() == "main"


class TestRewind:
    def test_rewind_produces_a_commit_with_the_targets_tree(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        rewound = repo.rewind(first)
        assert repo.read_commit(rewound).tree == repo.read_commit(first).tree

    def test_rewind_is_non_destructive_history_stays(self, repo):
        first = repo.commit([make_fact()], "seed")
        second = repo.commit([make_fact(object="Rust")], "update")
        rewound = repo.rewind(first)
        assert repo.read_commit(rewound).parents == (second,)
        # Both prior commits remain reachable.
        assert {c.hash for c in repo.log()} >= {first, second, rewound}

    def test_rewind_of_the_current_state_raises_empty_commit(self, repo):
        repo.commit([make_fact()], "seed")
        with pytest.raises(EmptyCommitError):
            repo.rewind("HEAD")

    def test_rewind_records_provenance_metadata(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        rewound = repo.rewind(first)
        commit = repo.read_commit(rewound)
        assert commit.metadata["rewind_of"] == first
        assert commit.metadata["rewind_from"] == first

    def test_rewind_default_message_names_the_target(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        rewound = repo.rewind(first)
        assert first[:8] in repo.read_commit(rewound).message

    def test_partial_rewind_touches_only_the_given_keys(self, repo):
        first = repo.commit(
            [make_fact(predicate="prefers_language", object="Python"), make_fact(predicate="timezone", object="UTC")],
            "seed",
        )
        repo.commit(
            [make_fact(predicate="prefers_language", object="Rust"), make_fact(predicate="timezone", object="PST")],
            "update",
        )
        current_tree_before_rewind = repo.read_tree(repo.resolve("HEAD"))

        rewound = repo.rewind(first, keys=[("user", "prefers_language")])
        by_key = repo.read_tree(rewound).by_key()

        # The rewound key matches the old state...
        old_tree = repo.read_tree(first)
        assert by_key[("user", "prefers_language")] == old_tree.by_key()[("user", "prefers_language")]
        # ...but the untouched key keeps HEAD's value from before the rewind.
        assert by_key[("user", "timezone")] == current_tree_before_rewind.by_key()[("user", "timezone")]

    def test_rewind_allow_empty(self, repo):
        first = repo.commit([make_fact()], "seed")
        rewound = repo.rewind(first, allow_empty=True)
        assert repo.read_commit(rewound).tree == repo.read_commit(first).tree


class TestWriteTree:
    def test_write_tree_matches_tree_from_facts_hash(self, repo):
        facts = [make_fact(), make_fact(predicate="timezone", object="UTC")]
        tree_hash = repo.write_tree(facts)
        from memgit.core.tree import Tree

        assert tree_hash == Tree.from_facts(facts).hash

    def test_write_tree_stores_every_fact_object(self, repo):
        fact = make_fact()
        tree_hash = repo.write_tree([fact])
        assert repo.store.contains(fact.hash)
        assert repo.store.contains(tree_hash)

    def test_write_tree_does_not_write_a_commit_or_move_any_ref(self, repo):
        repo.write_tree([make_fact()])
        assert repo.head_commit() is None


class TestOverlay:
    def test_overlay_sets_a_key_present_in_the_source(self, repo):
        onto = repo.read_tree(repo.commit([make_fact(object="Python")], "seed"))
        source = repo.read_tree(
            repo.commit([make_fact(object="Rust")], "update")
        )
        facts = repo.overlay(onto, source, [("user", "prefers_language")])
        assert [f.object for f in facts] == ["Rust"]

    def test_overlay_removes_a_key_absent_from_the_source(self, repo):
        onto = repo.read_tree(
            repo.commit(
                [make_fact(predicate="prefers_language"), make_fact(predicate="timezone", object="UTC")],
                "seed",
            )
        )
        empty_source = repo.read_tree(EMPTY_TREE_HASH)
        facts = repo.overlay(onto, empty_source, [("user", "timezone")])
        assert [f.predicate for f in facts] == ["prefers_language"]

    def test_overlay_leaves_unlisted_keys_untouched(self, repo):
        onto = repo.read_tree(
            repo.commit(
                [make_fact(predicate="prefers_language", object="Python"), make_fact(predicate="timezone", object="UTC")],
                "seed",
            )
        )
        source = repo.read_tree(EMPTY_TREE_HASH)
        facts = repo.overlay(onto, source, [("user", "prefers_language")])
        # "prefers_language" was overlaid from an empty source (removed);
        # "timezone" was never listed, so it must survive untouched.
        assert [f.predicate for f in facts] == ["timezone"]


class TestCommitOnto:
    def test_onto_advances_the_named_branch_and_leaves_head_alone(self, repo):
        repo.commit([make_fact()], "seed")
        repo.create_branch("other")
        head_before = repo.head_commit()

        new_hash = repo.commit([make_fact(object="Rust")], "update", onto="other")

        assert repo.branches()["other"] == new_hash
        assert repo.head_commit() == head_before

    def test_onto_accepts_a_full_ref_path(self, repo):
        repo.commit([make_fact()], "seed")
        repo.create_branch("other")
        new_hash = repo.commit([make_fact(object="Rust")], "update", onto="refs/heads/other")
        assert repo.branches()["other"] == new_hash

    def test_onto_an_unborn_branch_creates_a_root_commit(self, repo):
        new_hash = repo.commit([make_fact()], "seed", onto="feature")
        assert repo.branches()["feature"] == new_hash
        assert repo.read_commit(new_hash).is_root

    def test_onto_still_respects_empty_commit_detection(self, repo):
        repo.commit([make_fact()], "seed")
        repo.create_branch("other")
        with pytest.raises(EmptyCommitError):
            repo.commit([make_fact()], "no-op", onto="other")


class TestReset:
    def test_reset_moves_the_current_branch(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        repo.reset(first)
        assert repo.head_commit() == first
        assert repo.branches()["main"] == first

    def test_reset_on_detached_head_moves_head_not_a_branch(self, repo):
        first = repo.commit([make_fact()], "seed")
        second = repo.commit([make_fact(object="Rust")], "update")
        repo.checkout(first)
        repo.reset(second)
        assert repo.head_commit() == second
        assert repo.branches()["main"] == second  # main never moved from second in the first place
        assert repo.head().is_detached

    def test_reset_an_explicit_branch_not_the_current_one(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.create_branch("other", at=first)
        repo.commit([make_fact(object="Rust")], "update")
        repo.reset(first, branch="other")
        assert repo.branches()["other"] == first
        assert repo.current_branch() == "main"  # unaffected

    def test_reset_on_unborn_head_without_a_branch_raises(self, repo):
        commit_hash = repo.commit([make_fact()], "seed")
        # "bogus" fails to resolve, so checkout(create=...) leaves "topic"
        # genuinely unborn even though the repository itself has history.
        repo.checkout("bogus", create="topic")
        with pytest.raises(ValueError):
            repo.reset(commit_hash)

    def test_reset_unknown_revision_raises(self, repo):
        repo.commit([make_fact()], "seed")
        with pytest.raises(RevisionNotFoundError):
            repo.reset("does-not-exist")


class TestState:
    def test_state_at_head(self, repo):
        fact = make_fact()
        commit_hash = repo.commit([fact], "seed")
        state = repo.state()
        assert state.commit == commit_hash
        assert fact in state.facts

    def test_state_at_an_ancestor(self, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(object="Rust")], "update")
        state = repo.state("HEAD~1")
        assert state.commit == first
        assert state.one("user", "prefers_language").object == "Python"

    def test_state_at_the_empty_tree(self, repo):
        state = repo.state(EMPTY_TREE_HASH)
        assert len(state) == 0
