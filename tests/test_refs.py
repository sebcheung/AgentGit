"""Tests for refs and HEAD.

Grouped by the properties that carry the design: round-tripping (a ref reads
back what was written, including through git-identical text formatting),
HEAD state (unborn branch, detached commit, symref chains and cycles),
safety (a hostile ref name cannot escape refs/), and atomicity (writes never
leave a half-written ref or a stray lock file behind).
"""

from __future__ import annotations

import pytest

from memgit.core.refs import Head, InvalidRefNameError, RefStore

COMMIT_A = "a" * 64
COMMIT_B = "b" * 64


@pytest.fixture
def refs(tmp_path) -> RefStore:
    memgit_dir = tmp_path / ".memgit"
    (memgit_dir / "refs" / "heads").mkdir(parents=True)
    return RefStore(memgit_dir)


class TestRoundTrip:
    def test_writes_and_reads_a_branch_ref(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        assert refs.read_ref("refs/heads/main") == COMMIT_A

    def test_write_appends_a_trailing_newline(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        path = refs.memgit_dir / "refs" / "heads" / "main"
        assert path.read_text(encoding="utf-8") == COMMIT_A + "\n"

    def test_read_tolerates_a_missing_trailing_newline(self, refs):
        path = refs.memgit_dir / "refs" / "heads" / "main"
        path.write_text(COMMIT_A, encoding="utf-8")
        assert refs.read_ref("refs/heads/main") == COMMIT_A

    def test_read_of_missing_ref_is_none_not_an_exception(self, refs):
        assert refs.read_ref("refs/heads/nope") is None

    def test_write_rejects_a_non_hash_target(self, refs):
        with pytest.raises(ValueError, match="object hash"):
            refs.write_ref("refs/heads/main", "not-a-hash")

    def test_delete_ref_is_a_no_op_when_absent(self, refs):
        refs.delete_ref("refs/heads/nope")  # must not raise

    def test_delete_ref_removes_it(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        refs.delete_ref("refs/heads/main")
        assert refs.read_ref("refs/heads/main") is None


class TestHead:
    def test_unborn_branch_has_no_commit(self, refs):
        refs.set_head("refs/heads/main")
        head = refs.read_head()
        assert head == Head(ref="refs/heads/main", commit=None)
        assert not head.is_detached
        assert head.branch == "main"

    def test_head_resolves_through_to_the_commit(self, refs):
        refs.set_head("refs/heads/main")
        refs.write_ref("refs/heads/main", COMMIT_A)
        assert refs.read_head().commit == COMMIT_A

    def test_detached_head_round_trips(self, refs):
        refs.detach_head(COMMIT_A)
        head = refs.read_head()
        assert head.is_detached
        assert head.commit == COMMIT_A
        assert head.branch is None

    def test_detach_head_rejects_a_non_hash(self, refs):
        with pytest.raises(ValueError, match="object hash"):
            refs.detach_head("not-a-hash")

    def test_symref_chain_is_followed(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        (refs.memgit_dir / "refs" / "heads" / "alias").write_text(
            "ref: refs/heads/main\n", encoding="utf-8"
        )
        (refs.memgit_dir / "HEAD").write_text("ref: refs/heads/alias\n", encoding="utf-8")
        assert refs.read_head().commit == COMMIT_A

    def test_symref_cycle_raises_rather_than_hangs(self, refs):
        (refs.memgit_dir / "refs" / "heads" / "a").write_text(
            "ref: refs/heads/b\n", encoding="utf-8"
        )
        (refs.memgit_dir / "refs" / "heads" / "b").write_text(
            "ref: refs/heads/a\n", encoding="utf-8"
        )
        (refs.memgit_dir / "HEAD").write_text("ref: refs/heads/a\n", encoding="utf-8")
        with pytest.raises(ValueError, match="cycle"):
            refs.read_head()

    def test_malformed_head_raises(self, refs):
        (refs.memgit_dir / "HEAD").write_text("garbage\n", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            refs.read_head()


class TestRefNameSafety:
    """A ref name reaches the filesystem, so it is untrusted input."""

    @pytest.mark.parametrize(
        "name",
        [
            "../../evil",
            "refs/../../x",
            "/absolute/path",
            "refs/heads/.hidden",
            "refs/heads/x.lock",
            "heads/main",  # missing refs/ prefix
            "refs/",
            "refs//heads/main",
            "refs/heads/",
            "refs/heads/\x00null",
            "",
        ],
    )
    def test_rejects_unsafe_names(self, refs, name):
        with pytest.raises(InvalidRefNameError):
            refs.write_ref(name, COMMIT_A)

    def test_nothing_is_written_outside_refs(self, refs, tmp_path):
        with pytest.raises(InvalidRefNameError):
            refs.write_ref("../../escape", COMMIT_A)
        assert not (tmp_path.parent / "escape").exists()
        assert not (tmp_path / "escape").exists()


class TestCompareAndSwap:
    def test_expect_none_requires_absence(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A, expect=None)
        assert refs.read_ref("refs/heads/main") == COMMIT_A

    def test_expect_none_rejects_when_already_present(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        with pytest.raises(ValueError, match="concurrent write"):
            refs.write_ref("refs/heads/main", COMMIT_B, expect=None)

    def test_expect_matching_value_succeeds(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        refs.write_ref("refs/heads/main", COMMIT_B, expect=COMMIT_A)
        assert refs.read_ref("refs/heads/main") == COMMIT_B

    def test_expect_stale_value_is_rejected(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        with pytest.raises(ValueError, match="concurrent write"):
            refs.write_ref("refs/heads/main", COMMIT_B, expect=COMMIT_B)


class TestAtomicity:
    def test_no_lock_file_left_behind_on_success(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        assert list(refs.memgit_dir.rglob("*.lock")) == []

    def test_no_lock_file_left_behind_by_set_head(self, refs):
        refs.set_head("refs/heads/main")
        assert list(refs.memgit_dir.rglob("*.lock")) == []

    def test_no_lock_file_left_behind_by_detach_head(self, refs):
        refs.detach_head(COMMIT_A)
        assert list(refs.memgit_dir.rglob("*.lock")) == []


class TestListRefs:
    def test_lists_every_ref_under_the_default_prefix(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        refs.write_ref("refs/heads/experiment", COMMIT_B)
        assert refs.list_refs() == {
            "refs/heads/main": COMMIT_A,
            "refs/heads/experiment": COMMIT_B,
        }

    def test_respects_a_narrower_prefix(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        (refs.memgit_dir / "refs" / "stage").parent.mkdir(exist_ok=True)
        refs.write_ref("refs/heads/main", COMMIT_A)
        assert refs.list_refs("refs/heads/") == {"refs/heads/main": COMMIT_A}

    def test_empty_when_prefix_directory_does_not_exist(self, refs):
        assert refs.list_refs("refs/tags/") == {}

    def test_excludes_lock_files(self, refs):
        (refs.memgit_dir / "refs" / "heads" / "main.lock").write_text(COMMIT_A, encoding="utf-8")
        assert refs.list_refs() == {}


class TestResolve:
    def test_resolves_head(self, refs):
        refs.set_head("refs/heads/main")
        refs.write_ref("refs/heads/main", COMMIT_A)
        assert refs.resolve("HEAD") == COMMIT_A

    def test_resolves_a_full_hash(self, refs):
        assert refs.resolve(COMMIT_A) == COMMIT_A

    def test_resolves_a_full_ref_path(self, refs):
        refs.write_ref("refs/heads/main", COMMIT_A)
        assert refs.resolve("refs/heads/main") == COMMIT_A

    def test_resolves_a_bare_branch_name(self, refs):
        refs.write_ref("refs/heads/experiment", COMMIT_B)
        assert refs.resolve("experiment") == COMMIT_B

    def test_unresolvable_name_returns_none(self, refs):
        assert refs.resolve("does-not-exist") is None
