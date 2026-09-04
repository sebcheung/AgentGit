"""Tests for the CLI's translation of core exceptions into exit codes.

Deliberately the one test module that doesn't map 1:1 onto a single core
module: ``cli.py`` is the only place a Repository/Tree/Commit exception
becomes an exit code and a stderr message, and that translation was
previously untested. Covers the happy path end to end, then one case per
error a user is likely to hit.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memgit.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


FACTS = json.dumps(
    [
        {
            "type": "fact",
            "subject": "user",
            "predicate": "prefers_language",
            "object": "Python",
            "confidence": 0.9,
            "asserted_at": "2026-09-01T00:00:00+00:00",
        }
    ]
)


def _init_and_commit(tmp_path, message="seed"):
    (tmp_path / "facts.json").write_text(FACTS, encoding="utf-8")
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["commit", "-m", message, "--file", "facts.json"])
    assert result.exit_code == 0, result.output
    return result.output.strip()


class TestHappyPath:
    def test_init_creates_a_repository(self, tmp_path):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (tmp_path / ".memgit").is_dir()

    def test_commit_prints_a_hash(self, tmp_path):
        runner.invoke(app, ["init"])
        (tmp_path / "facts.json").write_text(FACTS, encoding="utf-8")
        result = runner.invoke(app, ["commit", "-m", "seed", "--file", "facts.json"])
        assert result.exit_code == 0
        assert len(result.output.strip()) == 64

    def test_log_shows_the_commit(self, tmp_path):
        commit_hash = _init_and_commit(tmp_path)
        result = runner.invoke(app, ["log", "--oneline"])
        assert result.exit_code == 0
        assert commit_hash[:8] in result.output
        assert "seed" in result.output

    def test_show_lists_the_committed_fact(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["show", "HEAD"])
        assert result.exit_code == 0
        assert "prefers_language" in result.output

    def test_ls_tree_prints_tab_separated_entries(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["ls-tree", "HEAD"])
        assert result.exit_code == 0
        assert "user\tprefers_language\t" in result.output

    def test_branch_creates_and_lists(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["branch", "experiment"])
        assert result.exit_code == 0

        result = runner.invoke(app, ["branch"])
        assert result.exit_code == 0
        assert "* main" in result.output
        assert "experiment" in result.output

    def test_rev_parse_resolves_head(self, tmp_path):
        commit_hash = _init_and_commit(tmp_path)
        result = runner.invoke(app, ["rev-parse", "HEAD"])
        assert result.exit_code == 0
        assert result.output.strip() == commit_hash

    def test_status_reports_branch_and_fact_count(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "On branch main" in result.output
        assert "1 fact(s)" in result.output

    def test_fsck_reports_all_objects_verified(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["fsck"])
        assert result.exit_code == 0
        assert "all objects verified" in result.output


class TestErrors:
    def test_commands_fail_outside_a_repository(self, tmp_path):
        result = runner.invoke(app, ["log"])
        assert result.exit_code == 1
        assert "not a memgit repository" in result.output

    def test_reinit_fails(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1

    def test_unknown_revision_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["log", "does-not-exist"])
        assert result.exit_code == 1
        assert "unknown revision" in result.output

    def test_unknown_object_fails(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["cat-file", "a" * 64])
        assert result.exit_code == 1
        assert "object not found" in result.output

    def test_bad_json_file_fails(self, tmp_path):
        runner.invoke(app, ["init"])
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        result = runner.invoke(app, ["commit", "-m", "x", "--file", "bad.json"])
        assert result.exit_code == 1
        assert "not valid JSON" in result.output

    def test_empty_commit_fails_without_allow_empty(self, tmp_path):
        _init_and_commit(tmp_path)
        (tmp_path / "facts.json").write_text(FACTS, encoding="utf-8")
        result = runner.invoke(app, ["commit", "-m", "no-op", "--file", "facts.json"])
        assert result.exit_code == 1

    def test_deleting_current_branch_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["branch", "-d", "main"])
        assert result.exit_code == 1


def _second_commit(tmp_path, predicate="favorite_editor", obj="vim", message="second"):
    payload = json.dumps(
        [
            {
                "type": "fact",
                "subject": "user",
                "predicate": predicate,
                "object": obj,
                "confidence": 0.9,
                "asserted_at": "2026-09-02T00:00:00+00:00",
            }
        ]
    )
    (tmp_path / "facts2.json").write_text(payload, encoding="utf-8")
    result = runner.invoke(app, ["commit", "-m", message, "--file", "facts2.json"])
    assert result.exit_code == 0, result.output
    return result.output.strip()


class TestDiff:
    def test_zero_args_shows_root_commit_as_all_added(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["diff"])
        assert result.exit_code == 0
        assert result.output.startswith("+")

    def test_unborn_branch_fails_with_no_commits_yet(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["diff"])
        assert result.exit_code == 1
        assert "no commits yet" in result.output

    def test_two_explicit_revs(self, tmp_path):
        first = _init_and_commit(tmp_path)
        second = _second_commit(tmp_path)
        result = runner.invoke(app, ["diff", first, second])
        assert result.exit_code == 0
        assert "favorite_editor" in result.output

    def test_two_dot_range(self, tmp_path):
        first = _init_and_commit(tmp_path)
        second = _second_commit(tmp_path)
        result = runner.invoke(app, ["diff", f"{first}..{second}"])
        assert result.exit_code == 0
        assert "favorite_editor" in result.output

    def test_three_dot_range_parses_and_uses_merge_base(self, tmp_path):
        # A trivial (already-equal) pair of branches is enough to prove the
        # "..." syntax parses and takes the --merge-base path rather than
        # erroring; a genuinely forked history is exercised in
        # TestDiff.test_use_merge_base_differs_from_direct_diff_on_a_fork
        # over in test_repository.py.
        _init_and_commit(tmp_path)
        runner.invoke(app, ["branch", "experiment"])
        three_dot = runner.invoke(app, ["diff", "main...experiment"])
        assert three_dot.exit_code == 0
        assert three_dot.output == ""

    def test_unknown_revision_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["diff", "does-not-exist"])
        assert result.exit_code == 1
        assert "unknown revision" in result.output

    def test_json_output_parses(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["diff", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["type"] == "diff"
        assert payload["keys"][0]["kind"] == "added"

    def test_stat_prints_counts_not_per_key_lines(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["diff", "--stat"])
        assert result.exit_code == 0
        assert "key(s) changed" in result.output
        assert "prefers_language" not in result.output

    def test_name_only_prints_tab_separated_keys(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["diff", "--name-only"])
        assert result.exit_code == 0
        assert "user\tprefers_language" in result.output

    def test_contradicted_key_line_starts_with_bang(self, tmp_path):
        _init_and_commit(tmp_path)
        _second_commit(tmp_path, predicate="prefers_language", obj="Rust")
        result = runner.invoke(app, ["diff"])
        assert result.exit_code == 0
        assert any(line.startswith("!") for line in result.output.splitlines())

    def test_strict_cardinality_exits_nonzero_on_violation(self, tmp_path):
        _init_and_commit(tmp_path)
        # Two facts sharing a "single" predicate in the same commit: a
        # genuine cardinality violation, not just a contradiction.
        payload = json.dumps(
            [
                {
                    "type": "fact",
                    "subject": "user",
                    "predicate": "favorite_editor",
                    "object": "vim",
                    "confidence": 0.9,
                    "asserted_at": "2026-09-02T00:00:00+00:00",
                },
                {
                    "type": "fact",
                    "subject": "user",
                    "predicate": "favorite_editor",
                    "object": "neovim",
                    "confidence": 0.9,
                    "asserted_at": "2026-09-02T00:00:01+00:00",
                },
            ]
        )
        (tmp_path / "rivals.json").write_text(payload, encoding="utf-8")
        runner.invoke(app, ["commit", "-m", "rival beliefs", "--file", "rivals.json"])

        clean = runner.invoke(app, ["diff"])
        assert clean.exit_code == 0

        strict = runner.invoke(app, ["diff", "--strict-cardinality"])
        assert strict.exit_code == 1


class TestShow:
    def test_shows_a_diff_by_default(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["show", "HEAD"])
        assert result.exit_code == 0
        assert "prefers_language" in result.output

    def test_facts_flag_restores_the_full_listing(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["show", "HEAD", "--facts"])
        assert result.exit_code == 0
        assert "0.90" in result.output


class TestCheckout:
    def test_checkout_a_commit_detaches(self, tmp_path):
        first = _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        result = runner.invoke(app, ["checkout", first])
        assert result.exit_code == 0
        assert "detached" in result.output
        assert first[:8] in result.output

    def test_checkout_reports_memory_change(self, tmp_path):
        first = _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        result = runner.invoke(app, ["checkout", first])
        assert "key(s) changed" in result.output

    def test_checkout_a_branch_reattaches(self, tmp_path):
        _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        runner.invoke(app, ["checkout", "HEAD~1"])
        result = runner.invoke(app, ["checkout", "main"])
        assert result.exit_code == 0
        assert "Switched to branch 'main'" in result.output
        status = runner.invoke(app, ["status"])
        assert "On branch main" in status.output

    def test_checkout_create_branch(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["checkout", "HEAD", "-b", "topic"])
        assert result.exit_code == 0
        assert "new branch 'topic'" in result.output
        status = runner.invoke(app, ["status"])
        assert "On branch topic" in status.output

    def test_checkout_unknown_revision_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["checkout", "does-not-exist"])
        assert result.exit_code == 1
        assert "unknown revision" in result.output

    def test_status_hints_reattachment_when_detached(self, tmp_path):
        first = _init_and_commit(tmp_path)
        runner.invoke(app, ["checkout", first])
        result = runner.invoke(app, ["status"])
        assert "memgit checkout" in result.output


class TestRewind:
    def test_rewind_prints_a_commit_hash(self, tmp_path):
        first = _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        result = runner.invoke(app, ["rewind", first])
        assert result.exit_code == 0
        assert len(result.output.strip()) == 64

    def test_rewound_state_matches_the_target(self, tmp_path):
        first = _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        runner.invoke(app, ["rewind", first, "-m", "roll back"])
        show = runner.invoke(app, ["show", "HEAD", "--facts"])
        assert "prefers_language" in show.output

    def test_rewind_a_no_op_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["rewind", "HEAD"])
        assert result.exit_code == 1

    def test_rewind_unknown_revision_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["rewind", "does-not-exist"])
        assert result.exit_code == 1
        assert "unknown revision" in result.output

    def test_rewind_bad_key_flag_fails(self, tmp_path):
        first = _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        result = runner.invoke(app, ["rewind", first, "--key", "no-colon-here"])
        assert result.exit_code == 1


class TestReset:
    def test_reset_moves_the_branch(self, tmp_path):
        first = _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        result = runner.invoke(app, ["reset", first])
        assert result.exit_code == 0
        assert first[:8] in result.output

    def test_reset_warns_about_unreachable_commits(self, tmp_path):
        first = _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        result = runner.invoke(app, ["reset", first])
        assert "no longer reachable" in result.output

    def test_reset_unknown_revision_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["reset", "does-not-exist"])
        assert result.exit_code == 1


class TestReflog:
    def test_reflog_lists_movements_newest_first(self, tmp_path):
        first = _init_and_commit(tmp_path)
        second = _second_commit(tmp_path)
        result = runner.invoke(app, ["reflog"])
        assert result.exit_code == 0
        lines = result.output.strip().splitlines()
        assert lines[0].startswith(second[:8])
        assert any(line.startswith(first[:8]) for line in lines)

    def test_reflog_limit(self, tmp_path):
        _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        result = runner.invoke(app, ["reflog", "-n", "1"])
        assert len(result.output.strip().splitlines()) == 1


class TestState:
    def test_state_lists_facts_at_head(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["state"])
        assert result.exit_code == 0
        assert "prefers_language" in result.output

    def test_state_at_an_ancestor(self, tmp_path):
        _init_and_commit(tmp_path)
        _second_commit(tmp_path)
        result = runner.invoke(app, ["state", "HEAD~1"])
        assert result.exit_code == 0
        assert "prefers_language" in result.output
        assert "favorite_editor" not in result.output

    def test_state_render_flag(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["state", "--render"])
        assert result.exit_code == 0
        assert "user:" in result.output

    def test_state_json_flag(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["state", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "facts" in payload

    def test_state_filters_by_subject(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["state", "--subject", "nobody"])
        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_state_unknown_revision_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["state", "does-not-exist"])
        assert result.exit_code == 1


class TestEmbed:
    def test_embeds_every_fact_reachable_from_head(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["embed", "--stats"])
        assert result.exit_code == 0
        assert "embedded 1 new fact(s), 0 already present" in result.output

    def test_second_run_is_idempotent(self, tmp_path):
        _init_and_commit(tmp_path)
        runner.invoke(app, ["embed"])
        result = runner.invoke(app, ["embed", "--stats"])
        assert result.exit_code == 0
        assert "embedded 0 new fact(s), 1 already present" in result.output

    def test_rebuild_reembeds_everything(self, tmp_path):
        _init_and_commit(tmp_path)
        runner.invoke(app, ["embed"])
        result = runner.invoke(app, ["embed", "--rebuild", "--stats"])
        assert result.exit_code == 0
        assert "embedded 1 new fact(s), 0 already present" in result.output

    def test_unknown_revision_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["embed", "does-not-exist"])
        assert result.exit_code == 1


class TestCardinality:
    def test_lists_default_with_no_declarations(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["cardinality"])
        assert result.exit_code == 0
        assert "default: single" in result.output

    def test_set_then_diff_reclassifies_a_key(self, tmp_path):
        _init_and_commit(tmp_path)
        # A second commit that keeps the original value *and* adds a new one
        # at the same predicate: a rival belief under "single", a coexisting
        # one under "multi". Repository.commit takes the whole state, so
        # both facts must be listed together to keep the old one.
        payload = json.dumps(
            [
                {
                    "type": "fact",
                    "subject": "user",
                    "predicate": "prefers_language",
                    "object": "Python",
                    "confidence": 0.9,
                    "asserted_at": "2026-09-01T00:00:00+00:00",
                },
                {
                    "type": "fact",
                    "subject": "user",
                    "predicate": "prefers_language",
                    "object": "Rust",
                    "confidence": 0.9,
                    "asserted_at": "2026-09-02T00:00:00+00:00",
                },
            ]
        )
        (tmp_path / "both.json").write_text(payload, encoding="utf-8")
        commit_result = runner.invoke(app, ["commit", "-m", "also likes rust", "--file", "both.json"])
        assert commit_result.exit_code == 0, commit_result.output

        before = runner.invoke(app, ["diff"])
        assert any(line.startswith("!") for line in before.output.splitlines())

        set_result = runner.invoke(app, ["cardinality", "set", "prefers_language", "multi"])
        assert set_result.exit_code == 0

        after = runner.invoke(app, ["diff"])
        assert any(line.startswith(">") for line in after.output.splitlines())

    def test_unset_reverts(self, tmp_path):
        runner.invoke(app, ["init"])
        runner.invoke(app, ["cardinality", "set", "likes", "multi"])
        result = runner.invoke(app, ["cardinality", "unset", "likes"])
        assert result.exit_code == 0
        listing = runner.invoke(app, ["cardinality"])
        assert "likes" not in listing.output

    def test_set_bad_value_fails(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["cardinality", "set", "likes", "several"])
        assert result.exit_code == 1


class TestDecay:
    def test_lists_default_with_no_declarations(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["decay"])
        assert result.exit_code == 0
        assert "default: 180.0 day(s)" in result.output

    def test_set_then_get_round_trips(self, tmp_path):
        runner.invoke(app, ["init"])
        set_result = runner.invoke(app, ["decay", "set", "prefers_language", "30"])
        assert set_result.exit_code == 0
        listing = runner.invoke(app, ["decay"])
        assert "prefers_language\t30.0" in listing.output

    def test_set_never_exempts_a_predicate(self, tmp_path):
        runner.invoke(app, ["init"])
        runner.invoke(app, ["decay", "set", "born_in", "never"])
        listing = runner.invoke(app, ["decay"])
        assert "born_in\tnever" in listing.output

    def test_unset_reverts(self, tmp_path):
        runner.invoke(app, ["init"])
        runner.invoke(app, ["decay", "set", "likes", "30"])
        result = runner.invoke(app, ["decay", "unset", "likes"])
        assert result.exit_code == 0
        listing = runner.invoke(app, ["decay"])
        assert "likes" not in listing.output

    def test_set_bad_value_fails(self, tmp_path):
        runner.invoke(app, ["init"])
        result = runner.invoke(app, ["decay", "set", "likes", "not-a-number"])
        assert result.exit_code == 1

    def test_state_as_of_shows_decayed_confidence(self, tmp_path):
        _init_and_commit(tmp_path)
        runner.invoke(app, ["decay", "set", "prefers_language", "1"])
        result = runner.invoke(app, ["state", "--as-of", "2027-09-01T00:00:00+00:00"])
        assert result.exit_code == 0
        assert "as of" in result.output

    def test_state_json_as_of_includes_decayed_confidence(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["state", "--as-of", "2026-09-01T00:00:00+00:00", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "decayed_confidence" in payload["facts"][0]

    def test_state_as_of_bad_timestamp_fails(self, tmp_path):
        _init_and_commit(tmp_path)
        result = runner.invoke(app, ["state", "--as-of", "not-a-timestamp"])
        assert result.exit_code == 1
