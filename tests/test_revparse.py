"""Tests for revision expression parsing and application.

Split the way the module is split: parsing is pure string handling (no
fixtures needed beyond the strings themselves), application is pure over a
``CommitReader`` and tested against synthetic dict graphs in ``test_graph.py``'s
style.
"""

from __future__ import annotations

import pytest

from memgit.core.commit import Commit
from memgit.core.revparse import (
    AncestryError,
    RevExpr,
    RevSyntaxError,
    Step,
    StepKind,
    apply_steps,
    parse_revision,
)

EMPTY_TREE = "0" * 64


class TestParseBare:
    def test_bare_head(self):
        assert parse_revision("HEAD") == RevExpr(base="HEAD")

    def test_bare_branch_name(self):
        assert parse_revision("main") == RevExpr(base="main")

    def test_bare_ref_path(self):
        assert parse_revision("refs/heads/main") == RevExpr(base="refs/heads/main")

    def test_is_bare_true_with_no_steps_or_reflog(self):
        assert parse_revision("main").is_bare

    def test_empty_string_is_a_syntax_error(self):
        with pytest.raises(RevSyntaxError):
            parse_revision("")


class TestParseAncestorSteps:
    def test_tilde_defaults_to_one(self):
        assert parse_revision("HEAD~").steps == (Step(StepKind.ANCESTOR, 1),)

    def test_tilde_n(self):
        assert parse_revision("HEAD~3").steps == (Step(StepKind.ANCESTOR, 3),)

    def test_tilde_zero_is_identity_step(self):
        assert parse_revision("HEAD~0").steps == (Step(StepKind.ANCESTOR, 0),)

    def test_not_bare_once_a_step_is_present(self):
        assert not parse_revision("HEAD~1").is_bare


class TestParseParentSteps:
    def test_caret_defaults_to_one(self):
        assert parse_revision("HEAD^").steps == (Step(StepKind.PARENT, 1),)

    def test_caret_n(self):
        assert parse_revision("HEAD^2").steps == (Step(StepKind.PARENT, 2),)

    def test_caret_zero_is_identity_step(self):
        assert parse_revision("HEAD^0").steps == (Step(StepKind.PARENT, 0),)


class TestParseChains:
    def test_mixed_chain_applies_left_to_right(self):
        expr = parse_revision("main~2^2~1")
        assert expr.base == "main"
        assert expr.steps == (
            Step(StepKind.ANCESTOR, 2),
            Step(StepKind.PARENT, 2),
            Step(StepKind.ANCESTOR, 1),
        )


class TestParseReflog:
    def test_head_at_n(self):
        expr = parse_revision("HEAD@{2}")
        assert expr.base == "HEAD"
        assert expr.reflog_index == 2
        assert expr.steps == ()

    def test_not_bare_with_reflog_index(self):
        assert not parse_revision("HEAD@{0}").is_bare

    def test_reflog_then_steps_is_legal(self):
        expr = parse_revision("HEAD@{1}~2")
        assert expr.reflog_index == 1
        assert expr.steps == (Step(StepKind.ANCESTOR, 2),)

    def test_steps_then_reflog_is_a_syntax_error(self):
        # @{n} names a starting point, not an ancestor of one.
        with pytest.raises(RevSyntaxError):
            parse_revision("HEAD~2@{1}")

    def test_branch_name_reflog(self):
        expr = parse_revision("main@{3}")
        assert expr.base == "main"
        assert expr.reflog_index == 3


class TestParseRejects:
    @pytest.mark.parametrize(
        "revision",
        [
            "HEAD~x",
            "HEAD^-1",
            "HEAD@{",
            "HEAD@{}",
            "HEAD@{x}",
            "HEAD@main",
            "~HEAD",
            "^HEAD",
        ],
    )
    def test_malformed_revisions_raise_syntax_error(self, revision):
        with pytest.raises(RevSyntaxError):
            parse_revision(revision)


# -- apply_steps -------------------------------------------------------------


def _commit(tree=EMPTY_TREE, parents=(), message="c", at="2026-01-01T00:00:00+00:00", **kw):
    return Commit(tree=tree, parents=parents, message=message, committed_at=at, **kw)


def build_chain(n: int) -> tuple[dict[str, Commit], list[str]]:
    store: dict[str, Commit] = {}
    hashes: list[str] = []
    parent: tuple[str, ...] = ()
    for i in range(n):
        c = _commit(parents=parent, message=f"c{i}", at=f"2026-01-01T00:{i:02d}:00+00:00")
        store[c.hash] = c
        hashes.append(c.hash)
        parent = (c.hash,)
    return store, hashes


class _Reader:
    def __init__(self, store: dict[str, Commit]) -> None:
        self._store = store

    def __call__(self, commit_hash: str) -> Commit:
        return self._store[commit_hash]


class TestApplyAncestorSteps:
    def test_zero_steps_is_identity(self):
        store, hashes = build_chain(3)
        assert apply_steps(hashes[-1], (), _Reader(store)) == hashes[-1]

    def test_tilde_zero_is_identity(self):
        store, hashes = build_chain(3)
        result = apply_steps(hashes[-1], (Step(StepKind.ANCESTOR, 0),), _Reader(store))
        assert result == hashes[-1]

    def test_tilde_one_hop(self):
        store, hashes = build_chain(3)
        result = apply_steps(hashes[-1], (Step(StepKind.ANCESTOR, 1),), _Reader(store))
        assert result == hashes[-2]

    def test_tilde_multiple_hops(self):
        store, hashes = build_chain(5)
        result = apply_steps(hashes[-1], (Step(StepKind.ANCESTOR, 3),), _Reader(store))
        assert result == hashes[-4]

    def test_tilde_off_a_root_raises_ancestry_error(self):
        store, hashes = build_chain(2)
        with pytest.raises(AncestryError):
            apply_steps(hashes[-1], (Step(StepKind.ANCESTOR, 5),), _Reader(store))


class TestApplyParentSteps:
    def test_caret_zero_is_identity(self):
        store, hashes = build_chain(2)
        result = apply_steps(hashes[-1], (Step(StepKind.PARENT, 0),), _Reader(store))
        assert result == hashes[-1]

    def test_caret_one_is_first_parent(self):
        store, hashes = build_chain(3)
        result = apply_steps(hashes[-1], (Step(StepKind.PARENT, 1),), _Reader(store))
        assert result == hashes[-2]

    def test_caret_selects_a_merge_parent(self):
        root = _commit(message="root", at="2026-01-01T00:00:00+00:00")
        left = _commit(parents=(root.hash,), message="left", at="2026-01-01T00:01:00+00:00")
        right = _commit(parents=(root.hash,), message="right", at="2026-01-01T00:01:00+00:00")
        merge = _commit(
            parents=(left.hash, right.hash), message="merge", at="2026-01-01T00:02:00+00:00"
        )
        store = {c.hash: c for c in (root, left, right, merge)}
        assert apply_steps(merge.hash, (Step(StepKind.PARENT, 1),), _Reader(store)) == left.hash
        assert apply_steps(merge.hash, (Step(StepKind.PARENT, 2),), _Reader(store)) == right.hash

    def test_caret_past_the_last_parent_raises_ancestry_error(self):
        store, hashes = build_chain(2)
        with pytest.raises(AncestryError):
            apply_steps(hashes[-1], (Step(StepKind.PARENT, 2),), _Reader(store))

    def test_caret_off_a_root_raises_ancestry_error(self):
        store, hashes = build_chain(1)
        with pytest.raises(AncestryError):
            apply_steps(hashes[0], (Step(StepKind.PARENT, 1),), _Reader(store))


class TestApplyChains:
    def test_chain_applies_left_to_right(self):
        # root -> a -> b -> {left, right}; merge = (left, right)
        root = _commit(message="root", at="2026-01-01T00:00:00+00:00")
        a = _commit(parents=(root.hash,), message="a", at="2026-01-01T00:01:00+00:00")
        b = _commit(parents=(a.hash,), message="b", at="2026-01-01T00:02:00+00:00")
        left = _commit(parents=(b.hash,), message="left", at="2026-01-01T00:03:00+00:00")
        right = _commit(parents=(b.hash,), message="right", at="2026-01-01T00:03:00+00:00")
        merge = _commit(
            parents=(left.hash, right.hash), message="merge", at="2026-01-01T00:04:00+00:00"
        )
        store = {c.hash: c for c in (root, a, b, left, right, merge)}
        read = _Reader(store)

        # merge^2~1 -> right's parent -> b
        result = apply_steps(
            merge.hash, (Step(StepKind.PARENT, 2), Step(StepKind.ANCESTOR, 1)), read
        )
        assert result == b.hash
