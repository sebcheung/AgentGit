"""Tests for the reflog format and journal, independent of RefStore.

Grouped by the properties that carry the design: line round-tripping
(including the characters that would break a line-oriented format if left
unescaped), the zero-hash convention for "did not exist", and ``@{n}``
indexing.
"""

from __future__ import annotations

import pytest

from memgit.core.reflog import ZERO_HASH, RefLog, RefLogEntry, RefLogger, ReflogNotFoundError

COMMIT_A = "a" * 64
COMMIT_B = "b" * 64
COMMIT_C = "c" * 64


class TestEntryRoundTrip:
    def test_formats_and_parses_back_identically(self):
        entry = RefLogEntry(
            old=COMMIT_A,
            new=COMMIT_B,
            author="cli:sebastian",
            at="2026-01-01T00:00:00+00:00",
            op="commit",
            message="seed the agent's beliefs",
        )
        assert RefLogEntry.parse(entry.format()) == entry

    def test_old_none_renders_as_zero_hash(self):
        entry = RefLogEntry(old=None, new=COMMIT_A, author="a", at="t", op="branch")
        line = entry.format()
        assert line.startswith(ZERO_HASH + " ")

    def test_new_none_renders_as_zero_hash(self):
        entry = RefLogEntry(old=COMMIT_A, new=None, author="a", at="t", op="delete")
        line = entry.format()
        assert f" {ZERO_HASH} " in line

    def test_zero_hash_parses_back_to_none(self):
        entry = RefLogEntry(old=None, new=None, author="a", at="t", op="delete")
        assert RefLogEntry.parse(entry.format()) == entry

    def test_empty_message_round_trips(self):
        entry = RefLogEntry(old=COMMIT_A, new=COMMIT_B, author="a", at="t", op="checkout")
        assert RefLogEntry.parse(entry.format()).message == ""

    def test_tab_in_message_is_replaced_not_left_to_corrupt_the_line(self):
        entry = RefLogEntry(old=COMMIT_A, new=COMMIT_B, author="a", at="t", op="commit", message="a\tb")
        parsed = RefLogEntry.parse(entry.format())
        assert "\t" not in parsed.message

    def test_newline_in_message_is_rejected(self):
        entry = RefLogEntry(old=COMMIT_A, new=COMMIT_B, author="a", at="t", op="commit", message="a\nb")
        with pytest.raises(ValueError, match="newline"):
            entry.format()

    def test_whitespace_in_author_is_rejected(self):
        entry = RefLogEntry(old=COMMIT_A, new=COMMIT_B, author="a b", at="t", op="commit")
        with pytest.raises(ValueError, match="author"):
            entry.format()

    def test_whitespace_in_op_is_rejected(self):
        entry = RefLogEntry(old=COMMIT_A, new=COMMIT_B, author="a", at="t", op="a b")
        with pytest.raises(ValueError, match="op"):
            entry.format()

    def test_malformed_line_raises_on_parse(self):
        with pytest.raises(ValueError):
            RefLogEntry.parse("not enough fields")


class TestRefLog:
    def test_missing_log_has_no_entries(self, tmp_path):
        log = RefLog(tmp_path, "HEAD")
        assert log.entries() == ()
        assert not log.exists()

    def test_append_creates_the_file_lazily(self, tmp_path):
        log = RefLog(tmp_path, "HEAD")
        log.append(RefLogEntry(old=None, new=COMMIT_A, author="a", at="t", op="commit"))
        assert log.exists()
        assert log.path == tmp_path / "logs" / "HEAD"

    def test_branch_log_path_matches_refs_layout(self, tmp_path):
        log = RefLog(tmp_path, "refs/heads/main")
        log.append(RefLogEntry(old=None, new=COMMIT_A, author="a", at="t", op="commit"))
        assert log.path == tmp_path / "logs" / "refs" / "heads" / "main"

    def test_entries_are_returned_oldest_first(self, tmp_path):
        log = RefLog(tmp_path, "HEAD")
        log.append(RefLogEntry(old=None, new=COMMIT_A, author="a", at="t0", op="commit"))
        log.append(RefLogEntry(old=COMMIT_A, new=COMMIT_B, author="a", at="t1", op="commit"))
        log.append(RefLogEntry(old=COMMIT_B, new=COMMIT_C, author="a", at="t2", op="commit"))
        entries = log.entries()
        assert [e.new for e in entries] == [COMMIT_A, COMMIT_B, COMMIT_C]

    def test_rejects_a_ref_that_is_neither_head_nor_under_refs(self, tmp_path):
        with pytest.raises(ValueError):
            _ = RefLog(tmp_path, "heads/main").path


class TestRefLogAtIndex:
    @pytest.fixture
    def log(self, tmp_path) -> RefLog:
        log = RefLog(tmp_path, "refs/heads/main")
        log.append(RefLogEntry(old=None, new=COMMIT_A, author="a", at="t0", op="branch"))
        log.append(RefLogEntry(old=COMMIT_A, new=COMMIT_B, author="a", at="t1", op="commit"))
        log.append(RefLogEntry(old=COMMIT_B, new=COMMIT_C, author="a", at="t2", op="commit"))
        return log

    def test_at_zero_is_the_current_value(self, log):
        assert log.at(0) == COMMIT_C

    def test_at_one_is_one_move_ago(self, log):
        assert log.at(1) == COMMIT_B

    def test_at_past_the_start_raises(self, log):
        with pytest.raises(ReflogNotFoundError):
            log.at(4)

    def test_at_the_boundary_entry_before_creation_has_no_commit(self, log):
        # @{3}: the value *before* the branch's very first movement (its
        # creation) is "did not exist" — not a resolvable revision.
        with pytest.raises(ReflogNotFoundError):
            log.at(3)

    def test_negative_index_raises(self, log):
        with pytest.raises(ReflogNotFoundError):
            log.at(-1)

    def test_missing_log_raises(self, tmp_path):
        with pytest.raises(ReflogNotFoundError):
            RefLog(tmp_path, "refs/heads/gone").at(0)


class TestRefLogger:
    def test_log_appends_a_well_formed_entry(self, tmp_path):
        logger = RefLogger(tmp_path, author="cli:sebastian")
        logger.log("refs/heads/main", None, COMMIT_A, op="branch", message="create main")
        entries = RefLog(tmp_path, "refs/heads/main").entries()
        assert len(entries) == 1
        assert entries[0].old is None
        assert entries[0].new == COMMIT_A
        assert entries[0].author == "cli:sebastian"
        assert entries[0].op == "branch"
        assert entries[0].message == "create main"
