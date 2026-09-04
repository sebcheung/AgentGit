"""MemGit command line interface."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from memgit import __version__
from memgit.core.commit import Commit
from memgit.core.decay import DecayPolicy
from memgit.core.diff import ChangeKind, Diff
from memgit.core.fact import Fact
from memgit.core.graph import ancestors, is_ancestor, walk
from memgit.core.refs import InvalidRefNameError
from memgit.core.repository import (
    EmptyCommitError,
    NotARepositoryError,
    Repository,
    RepositoryExistsError,
    RevisionNotFoundError,
)
from memgit.core.store import CorruptObjectError, ObjectNotFoundError, hash_object
from memgit.retrieval.embed import embed_text

app = typer.Typer(
    name="memgit",
    help="Version control and time-travel debugging for AI agent memory.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"memgit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the MemGit version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """MemGit — git for what your agent believes."""


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _parse_as_of(value: str | None) -> datetime:
    """Parse an ``--as-of`` flag, defaulting to now.

    The CLI is the one place allowed to read the wall clock for decay — every
    call into ``core.decay`` or ``retrieval`` takes ``as_of`` explicitly, so
    a decay-aware command stays reproducible everywhere except at the exact
    point a human asked "what does it look like right now."
    """
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _fail(f"--as-of must be ISO-8601, got {value!r}")
        raise  # unreachable; _fail always raises
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _repo() -> Repository:
    """Discover the current repository, or exit with an error.

    All discovery and validation logic lives on ``Repository`` itself and
    raises plain exceptions — this function is the one place that translates
    those into the CLI's exit-code convention, so the same core code stays
    reusable from the FastAPI layer and the MCP server later without either
    one depending on typer.
    """
    try:
        return Repository.discover()
    except NotARepositoryError as exc:
        _fail(str(exc))
        raise  # unreachable; satisfies type checkers


def _load_facts(file: str | None) -> list[Fact]:
    """Read a JSON array of fact dicts from ``file``, or stdin if ``file`` is ``-``."""
    if file is None or file == "-":
        if file is None and sys.stdin.isatty():
            _fail("provide --file PATH, or pipe a JSON array of facts on stdin")
        raw = sys.stdin.read()
    else:
        try:
            raw = Path(file).read_text(encoding="utf-8")
        except OSError as exc:
            _fail(f"could not read {file}: {exc}")

    try:
        payloads = json.loads(raw)
    except ValueError as exc:
        _fail(f"not valid JSON: {exc}")

    if not isinstance(payloads, list):
        _fail("expected a JSON array of facts")

    try:
        return [Fact.from_dict(p) for p in payloads]
    except (TypeError, ValueError) as exc:
        _fail(str(exc))
        raise  # unreachable


def _format_commit_oneline(commit_hash: str, commit: Commit) -> str:
    return f"{commit_hash[:8]} {commit.summary}"


@app.command("init")
def init_cmd(
    path: str = typer.Argument(".", help="Where to create the repository."),
) -> None:
    """Create a new, empty repository. Mirrors ``git init``."""
    try:
        repo = Repository.init(path)
    except RepositoryExistsError as exc:
        _fail(str(exc))
        return
    typer.echo(f"Initialized empty MemGit repository in {repo.memgit_dir}")


@app.command("commit")
def commit_cmd(
    message: str = typer.Option(..., "-m", "--message", help="Commit message."),
    file: str = typer.Option(
        None, "--file", "-f", help="JSON file of facts, or '-' for stdin."
    ),
    author: str = typer.Option("unknown", "--author", help="Who or what made this commit."),
    allow_empty: bool = typer.Option(
        False, "--allow-empty", help="Permit a commit identical to its parent."
    ),
) -> None:
    """Record the given facts as a new commit. Mirrors ``git commit``."""
    repo = _repo()
    facts = _load_facts(file)
    try:
        commit_hash = repo.commit(facts, message, author=author, allow_empty=allow_empty)
    except EmptyCommitError as exc:
        _fail(str(exc))
        return
    typer.echo(commit_hash)


@app.command("log")
def log_cmd(
    rev: str = typer.Argument("HEAD", help="Revision to start from."),
    limit: int = typer.Option(None, "-n", "--limit", help="Show at most this many commits."),
    oneline: bool = typer.Option(False, "--oneline", help="One line per commit."),
) -> None:
    """Walk history from ``rev``, most-recent-first. Mirrors ``git log``."""
    repo = _repo()
    try:
        commits = list(repo.log(rev, limit=limit))
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return

    for commit in commits:
        if oneline:
            typer.echo(_format_commit_oneline(commit.hash, commit))
        else:
            typer.echo(f"commit {commit.hash}")
            if commit.parents:
                typer.echo(f"parents: {' '.join(commit.parents)}")
            typer.echo(f"author:  {commit.author}")
            typer.echo(f"date:    {commit.committed_at}")
            typer.echo(f"\n    {commit.message}\n")


_DIFF_PREFIXES: dict[ChangeKind, tuple[str, str | None]] = {
    ChangeKind.ADDED: ("+", typer.colors.GREEN),
    ChangeKind.REMOVED: ("-", typer.colors.RED),
    ChangeKind.REAFFIRMED: ("~", typer.colors.BLUE),
    ChangeKind.CONTRADICTED: ("!", typer.colors.YELLOW),
    ChangeKind.VALUE_ADDED: (">", typer.colors.GREEN),
    ChangeKind.VALUE_REMOVED: ("<", typer.colors.RED),
    ChangeKind.MIXED: ("*", typer.colors.YELLOW),
    ChangeKind.UNCHANGED: (" ", None),
}


def _print_diff_human(diff: Diff, *, name_only: bool) -> None:
    for kd in diff.keys:
        if name_only:
            typer.echo(f"{kd.subject}\t{kd.predicate}")
            continue
        prefix, color = _DIFF_PREFIXES[kd.kind]
        detail = ", ".join(str(v) for v in kd.changed_values)
        line = f"{prefix} {kd.subject}\t{kd.predicate}\t{detail}"
        if color is not None:
            typer.secho(line, fg=color)
        else:
            typer.echo(line)


def _print_diff_stat(diff: Diff) -> None:
    stat = diff.stat()
    by_kind = ", ".join(
        f"{count} {str(kind).replace('_', ' ')}" for kind, count in sorted(stat.by_kind.items())
    )
    summary = f"{stat.keys_changed} key(s) changed"
    if by_kind:
        summary += f": {by_kind}"
    typer.echo(summary)
    total_facts = stat.facts_added + stat.facts_removed + stat.facts_reaffirmed
    typer.echo(f"{total_facts} fact(s): +{stat.facts_added} -{stat.facts_removed} ~{stat.facts_reaffirmed}")


def _print_violations(diff: Diff) -> None:
    if not diff.violations:
        return
    typer.secho("cardinality:", fg=typer.colors.YELLOW, err=True)
    for v in diff.violations:
        typer.secho(
            f"  {v.key[0]} {v.key[1]} ({v.side}): {', '.join(v.objects)}",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _parse_diff_range(
    rev_a: str | None, rev_b: str | None, merge_base_flag: bool
) -> tuple[str | None, str, bool]:
    """Resolve the CLI's revision arguments to ``(before, after, use_merge_base)``.

    No working tree and no index means there is no uncommitted state to
    diff, so zero arguments falls back to "what did the newest commit
    change" rather than git's "what's staged". A lone ``A...B`` or ``A..B``
    packed into ``rev_a`` is split here since it is user-input syntax, not a
    revision-resolution concern ``Repository``/``RefStore`` should own.
    """
    if rev_a is None:
        return None, "HEAD", merge_base_flag
    if rev_b is None:
        if "..." in rev_a:
            left, right = rev_a.split("...", 1)
            return left, right, True
        if ".." in rev_a:
            left, right = rev_a.split("..", 1)
            return left, right, merge_base_flag
        return None, rev_a, merge_base_flag
    return rev_a, rev_b, merge_base_flag


@app.command("diff")
def diff_cmd(
    rev_a: str = typer.Argument(None, help="Revision, or a range: A..B / A...B."),
    rev_b: str = typer.Argument(None, help="Second revision, if not given as a range."),
    stat: bool = typer.Option(False, "--stat", help="Summary counts only."),
    name_only: bool = typer.Option(False, "--name-only", help="List changed keys only."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    include_unchanged: bool = typer.Option(False, "--unchanged", help="Include unchanged keys."),
    merge_base_flag: bool = typer.Option(
        False, "--merge-base", help="Compare against merge_base(A, B) instead of A directly."
    ),
    strict_cardinality: bool = typer.Option(
        False, "--strict-cardinality", help="Exit 1 if any cardinality violation was found."
    ),
) -> None:
    """Compare two memory states. Mirrors ``git diff``.

    MemGit has no staging area, so there is no uncommitted state to diff:
    with no arguments this shows what the newest commit changed, i.e. HEAD
    against its first parent.
    """
    repo = _repo()
    if repo.head().commit is None:
        _fail("no commits yet")
        return

    before_rev, after_rev, use_merge_base = _parse_diff_range(rev_a, rev_b, merge_base_flag)

    try:
        result = repo.diff(
            before_rev, after_rev, include_unchanged=include_unchanged, use_merge_base=use_merge_base
        )
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return
    except ValueError as exc:
        _fail(str(exc))
        return

    if not use_merge_base and before_rev is not None:
        try:
            before_hash = repo.resolve(before_rev)
            after_hash = repo.resolve(after_rev)
            diverged = not is_ancestor(before_hash, after_hash, repo.read_commit) and not is_ancestor(
                after_hash, before_hash, repo.read_commit
            )
        except RevisionNotFoundError:
            diverged = False
        if diverged:
            typer.secho(
                f"note: {before_rev} and {after_rev} have diverged; "
                f"try `memgit diff {before_rev}...{after_rev}`",
                fg=typer.colors.BLUE,
                err=True,
            )

    if as_json:
        payload = result.to_dict()
        payload["before"] = before_rev
        payload["after"] = after_rev
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    elif stat:
        _print_diff_stat(result)
    else:
        _print_diff_human(result, name_only=name_only)

    _print_violations(result)

    if strict_cardinality and result.violations:
        raise typer.Exit(code=1)


@app.command("show")
def show_cmd(
    rev: str = typer.Argument("HEAD", help="Revision to show."),
    facts: bool = typer.Option(
        False, "--facts", help="List every fact in the tree instead of the diff."
    ),
    stat: bool = typer.Option(False, "--stat", help="Summary counts only (diff mode)."),
) -> None:
    """Show a commit's metadata and what it changed. Mirrors ``git show``."""
    repo = _repo()
    try:
        commit_hash = repo.resolve(rev)
        commit = repo.read_commit(commit_hash)
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return
    except (ObjectNotFoundError, CorruptObjectError) as exc:
        _fail(str(exc))
        return

    typer.echo(f"commit {commit_hash}")
    typer.echo(f"author: {commit.author}")
    typer.echo(f"date:   {commit.committed_at}")
    typer.echo(f"\n    {commit.message}\n")

    if facts:
        tree = repo.read_tree(commit_hash)
        for fact in tree.load(repo.store):
            typer.echo(f"  {fact}")
        return

    result = repo.diff(after=commit_hash)
    if stat:
        _print_diff_stat(result)
    else:
        _print_diff_human(result, name_only=False)
    _print_violations(result)


@app.command("ls-tree")
def ls_tree_cmd(rev: str = typer.Argument("HEAD", help="Revision whose tree to list.")) -> None:
    """List a tree's entries. Mirrors ``git ls-tree``."""
    repo = _repo()
    try:
        tree = repo.read_tree(rev)
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return
    except (ObjectNotFoundError, CorruptObjectError) as exc:
        _fail(str(exc))
        return

    for subject, predicate, hashes in tree:
        for h in hashes:
            typer.echo(f"{subject}\t{predicate}\t{h}")


@app.command("branch")
def branch_cmd(
    name: str = typer.Argument(None, help="Branch to create."),
    at: str = typer.Option("HEAD", "--at", help="Revision the new branch should point at."),
    delete: str = typer.Option(None, "-d", "--delete", help="Branch to delete."),
) -> None:
    """List, create, or delete branches. Mirrors ``git branch``."""
    repo = _repo()

    if delete is not None:
        try:
            repo.delete_branch(delete)
        except ValueError as exc:
            _fail(str(exc))
        return

    if name is not None:
        try:
            repo.create_branch(name, at=at)
        except (RevisionNotFoundError, ValueError) as exc:
            _fail(str(exc))
        return

    current = repo.current_branch()
    for branch_name in sorted(repo.branches()):
        marker = "*" if branch_name == current else " "
        typer.echo(f"{marker} {branch_name}")


@app.command("rev-parse")
def rev_parse_cmd(rev: str = typer.Argument(..., help="Revision to resolve.")) -> None:
    """Resolve a revision string to a commit hash. Mirrors ``git rev-parse``."""
    repo = _repo()
    try:
        typer.echo(repo.resolve(rev))
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")


@app.command("checkout")
def checkout_cmd(
    rev: str = typer.Argument("HEAD", help="Revision, branch, or ancestry expression to check out."),
    create: str = typer.Option(None, "-b", "--branch", help="Create and switch to a new branch at REV."),
    detach: bool = typer.Option(False, "--detach", help="Detach HEAD even if REV names a branch."),
    stat: bool = typer.Option(False, "--stat", help="Show the full diff summary of what changed."),
) -> None:
    """Move HEAD. Mirrors ``git checkout``, minus everything about a working tree.

    MemGit has no working tree and no index, so this never needs --force:
    nothing is ever at risk of being overwritten.
    """
    repo = _repo()
    try:
        result = repo.checkout(rev, create=create, detach=detach)
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return
    except (ValueError, InvalidRefNameError) as exc:
        _fail(str(exc))
        return

    if result.created_branch is not None:
        typer.echo(f"Switched to a new branch '{result.created_branch}'")
    elif result.detached:
        commit_display = result.head.commit[:8] if result.head.commit else "(unborn)"
        typer.echo(f"HEAD is now detached at {commit_display}")
    else:
        typer.echo(f"Switched to branch '{result.head.branch}'")

    if result.moved and result.previous.commit is not None and result.head.commit is not None:
        diff_result = repo.diff(result.previous.commit, result.head.commit)
        if stat:
            _print_diff_stat(diff_result)
        else:
            typer.echo(f"Memory: {diff_result.stat().keys_changed} key(s) changed")


@app.command("rewind")
def rewind_cmd(
    rev: str = typer.Argument(..., help="Past revision to bring forward onto HEAD."),
    message: str = typer.Option(None, "-m", "--message", help="Commit message."),
    author: str = typer.Option("unknown", "--author", help="Who or what performed the rewind."),
    key: list[str] = typer.Option(
        None,
        "--key",
        help="Rewind only this SUBJECT:PREDICATE key (repeatable); split on the last ':'.",
    ),
    allow_empty: bool = typer.Option(False, "--allow-empty", help="Permit a rewind that changes nothing."),
) -> None:
    """Re-commit a past memory state forward onto HEAD. Non-destructive.

    Unlike ``memgit reset``, nothing becomes unreachable: the rewind is
    itself a new commit, visible in ``memgit log`` like any other change.
    """
    repo = _repo()

    keys = None
    if key:
        keys = []
        for raw in key:
            if ":" not in raw:
                _fail(f"--key must be SUBJECT:PREDICATE, got {raw!r}")
                return
            subject, _, predicate = raw.rpartition(":")
            keys.append((subject, predicate))

    try:
        commit_hash = repo.rewind(rev, message, author=author, keys=keys, allow_empty=allow_empty)
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return
    except EmptyCommitError as exc:
        _fail(str(exc))
        return

    typer.echo(commit_hash)


@app.command("reset")
def reset_cmd(
    rev: str = typer.Argument(..., help="Revision to reset to."),
    branch: str = typer.Option(
        None, "--branch", help="Branch to move. Defaults to HEAD's own branch, or HEAD itself if detached."
    ),
) -> None:
    """Move a branch pointer to REV — destructive. Prefer ``memgit rewind``.

    With no working tree and no index, git's --soft/--mixed/--hard split
    collapses into this one operation. Commits solely reachable from the old
    position become unreachable, but not deleted — recoverable via
    ``memgit reflog`` while it still holds the old value.
    """
    repo = _repo()

    if branch is not None:
        label = branch
        previous_hash = repo.branches().get(branch)
    elif repo.head().is_detached:
        label = "HEAD"
        previous_hash = repo.head_commit()
    else:
        label = repo.current_branch()
        previous_hash = repo.head_commit()

    try:
        new_hash = repo.reset(rev, branch=branch)
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return
    except ValueError as exc:
        _fail(str(exc))
        return

    typer.echo(f"{label} is now at {new_hash[:8]}")

    if previous_hash is not None and previous_hash != new_hash:
        orphaned = ancestors(previous_hash, repo.read_commit) - ancestors(new_hash, repo.read_commit)
        if orphaned:
            typer.secho(
                f"warning: {len(orphaned)} commit(s) are no longer reachable from {label}; "
                f"recover with 'memgit reflog'",
                fg=typer.colors.YELLOW,
                err=True,
            )


@app.command("reflog")
def reflog_cmd(
    ref: str = typer.Argument("HEAD", help="Ref whose reflog to show."),
    limit: int = typer.Option(None, "-n", "--limit", help="Show at most this many entries."),
) -> None:
    """Show every place REF has pointed. Mirrors ``git reflog``."""
    repo = _repo()
    entries = repo.reflog(ref)
    if limit is not None:
        entries = entries[-limit:]
    for offset, entry in enumerate(reversed(entries)):
        short = (entry.new or "0" * 64)[:8]
        typer.echo(f"{short} {ref}@{{{offset}}}: {entry.op}: {entry.message}")


@app.command("embed")
def embed_cmd(
    rev: str = typer.Argument(
        None, help="Only embed facts reachable from this revision (default: every branch)."
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Recompute and overwrite vectors already cached."
    ),
    stats: bool = typer.Option(False, "--stats", help="Print how many facts were embedded."),
) -> None:
    """Warm the vector cache — the backfill for repositories predating retrieval.

    Retrieval also embeds lazily on a cache miss, so this command is never
    required for correctness; it exists to make ``memgit recall`` fast the
    first time it runs against an existing repository, and to make the
    cache's cost visible with ``--stats``.
    """
    repo = _repo()
    embedder = repo.embedder()
    index = repo.vector_index(embedder)

    if rev is not None:
        try:
            starts = [repo.resolve(rev)]
        except RevisionNotFoundError as exc:
            _fail(f"unknown revision: {exc}")
            return
    else:
        starts = list(repo.branches().values())

    fact_hashes: set[str] = set()
    for start in starts:
        for _commit_hash, commit in walk(start, repo.read_commit):
            fact_hashes.update(repo.read_tree(commit.tree).fact_hashes())

    embedded = 0
    skipped = 0
    for fact_hash in sorted(fact_hashes):
        if not rebuild and fact_hash in index:
            skipped += 1
            continue
        fact = repo.read_fact(fact_hash)
        index.put(fact_hash, embedder.embed(embed_text(fact)), force=rebuild)
        embedded += 1

    if stats:
        typer.echo(f"embedded {embedded} new fact(s), {skipped} already present ({embedder.id})")


@app.command("recall")
def recall_cmd(
    query: str = typer.Argument(..., help="What to search memory for."),
    rev: str = typer.Argument("HEAD", help="Revision to search — the scope guarantee."),
    limit: int = typer.Option(8, "-k", "--limit", help="How many facts to return."),
    subject: str = typer.Option(None, "--subject", help="Restrict to this subject."),
    min_score: float = typer.Option(0.0, "--min-score", help="Drop results below this score."),
    as_of: str = typer.Option(None, "--as-of", help="Rank as of this ISO-8601 timestamp (default: now)."),
    no_decay: bool = typer.Option(False, "--no-decay", help="Ignore confidence decay; rank on similarity alone."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Search memory at REV for facts relevant to QUERY — the demo command.

    Shows exactly what the ``recall`` agent tool and a retrieval-mode prompt
    would see: the same :class:`~memgit.retrieval.rank.Retriever` code path,
    scoped to REV by construction (see ``retrieval/rank.py``'s module
    docstring) rather than by a filter that could leak a later commit or
    another branch.
    """
    repo = _repo()
    try:
        resolved = repo.resolve(rev)
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return

    state = repo.state(resolved)
    weight = 0.0 if no_decay else None
    retriever = repo.retriever(confidence_weight=weight)
    moment = _parse_as_of(as_of)

    result = retriever.retrieve(
        state,
        query,
        k=limit,
        as_of=moment,
        subjects={subject} if subject is not None else None,
        min_score=min_score,
    )

    if as_json:
        payload = {
            "query": result.query,
            "commit": result.commit,
            "candidates": result.candidates,
            "embedder": result.embedder,
            "as_of": result.as_of,
            "facts": [
                {
                    **r.fact.to_dict(),
                    "score": r.score,
                    "similarity": r.similarity,
                    "confidence": r.confidence,
                }
                for r in result.facts
            ],
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    label = f"{rev} ({resolved[:8]})"
    typer.echo(f"{len(result.facts)} of {result.candidates} fact(s) at {label}")
    typer.echo(result.render())


@app.command("state")
def state_cmd(
    rev: str = typer.Argument("HEAD", help="Revision whose memory state to show."),
    subject: str = typer.Option(None, "--subject", help="Only facts about this subject."),
    predicate: str = typer.Option(None, "--predicate", help="Only facts at this predicate."),
    min_confidence: float = typer.Option(0.0, "--min-confidence", help="Drop facts below this confidence."),
    render: bool = typer.Option(
        False, "--render", help="Print the context-block rendering instead of a listing."
    ),
    as_of: str = typer.Option(
        None, "--as-of", help="Show decayed confidence as of this ISO-8601 timestamp (default: now)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show what the agent believed at REV — the time-travel command.

    Distinct from ``memgit show --facts``: that is commit-centric and
    unfiltered, this is state-centric and filterable. ``--as-of`` adds a
    decayed-confidence column without ever touching stored confidence —
    decay is purely a display lens here, the same as it is for retrieval
    ranking (see ``core/decay.py``).
    """
    repo = _repo()
    try:
        memory = repo.state(rev)
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return
    except (ObjectNotFoundError, CorruptObjectError) as exc:
        _fail(str(exc))
        return

    if subject is not None or predicate is not None or min_confidence > 0.0:
        memory = memory.filter(
            min_confidence=min_confidence,
            subjects={subject} if subject is not None else None,
            predicates={predicate} if predicate is not None else None,
        )

    decay_moment = _parse_as_of(as_of) if as_of is not None else None

    if as_json:
        payload = memory.to_dict()
        if decay_moment is not None:
            policy = repo.decay()
            payload["as_of"] = decay_moment.isoformat()
            for fact_payload, fact in zip(payload["facts"], memory.facts):
                fact_payload["decayed_confidence"] = policy.decayed(fact, as_of=decay_moment)
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    elif render:
        typer.echo(memory.render())
    else:
        policy = repo.decay() if decay_moment is not None else None
        for fact in sorted(memory.facts, key=lambda f: (f.subject, f.predicate, f.object)):
            if policy is not None:
                decayed = policy.decayed(fact, as_of=decay_moment)
                typer.echo(f"  {fact}  [as of {as_of or 'now'}: {decayed:.2f}]")
            else:
                typer.echo(f"  {fact}")


@app.command("status")
def status_cmd() -> None:
    """Show the current branch, HEAD, and fact count."""
    repo = _repo()
    head = repo.head()
    if head.is_detached:
        typer.echo(f"HEAD detached at {head.commit}")
        typer.echo("  (use 'memgit checkout <branch>' to reattach)")
    else:
        typer.echo(f"On branch {head.branch}")
        if head.commit is None:
            typer.echo("No commits yet")

    if head.commit is not None:
        tree = repo.read_tree(head.commit)
        typer.echo(f"{len(tree)} key(s), {len(tree.fact_hashes())} fact(s)")


@app.command("hash-object")
def hash_object_cmd(
    write: bool = typer.Option(
        False, "-w", "--write", help="Write the object into the store."
    ),
    subject: str = typer.Option(None, "--subject", "-s", help="Fact subject."),
    predicate: str = typer.Option(None, "--predicate", "-p", help="Fact predicate."),
    obj: str = typer.Option(None, "--object", "-o", help="Fact object."),
    confidence: float = typer.Option(1.0, "--confidence", "-c", help="0.0 to 1.0."),
    source: str = typer.Option(None, "--source", help="Where this belief came from."),
    source_text: str = typer.Option(
        None, "--source-text", help="The original natural-language claim."
    ),
) -> None:
    """Compute a fact's object hash, optionally storing it.

    Mirrors ``git hash-object``. With no fact flags, reads a JSON object from
    stdin instead — handy for inspecting trees and commits too.
    """
    if subject is not None or predicate is not None or obj is not None:
        if not (subject and predicate and obj):
            typer.secho(
                "--subject, --predicate and --object must be given together",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            payload = Fact(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                source=source,
                source_text=source_text,
            ).to_dict()
        except (TypeError, ValueError) as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
    else:
        if sys.stdin.isatty():
            typer.secho(
                "provide a fact with --subject/--predicate/--object, "
                "or pipe a JSON object on stdin",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            payload = json.loads(sys.stdin.read())
        except ValueError as exc:
            typer.secho(f"stdin is not valid JSON: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    typer.echo(_repo().store.put(payload) if write else hash_object(payload))


@app.command("cat-file")
def cat_file_cmd(
    obj_hash: str = typer.Argument(..., help="Object hash to read."),
    pretty: bool = typer.Option(
        True, "--pretty/--raw", help="Pretty-print the JSON payload."
    ),
) -> None:
    """Read an object out of the store, verifying it on the way.

    Mirrors ``git cat-file -p``. Verification is free here: the address *is* a
    checksum of the contents, so a mismatch means the object was tampered with
    or damaged.
    """
    try:
        payload = _repo().store.get(obj_hash)
    except ObjectNotFoundError:
        typer.secho(f"object not found: {obj_hash}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except CorruptObjectError as exc:
        typer.secho(f"corrupt object: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except (TypeError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.echo(json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False))


@app.command("fsck")
def fsck_cmd() -> None:
    """Verify every object in the store.

    The offline equivalent of ``git fsck`` for the object database. Also prints
    the storage-efficiency numbers the README quotes.
    """
    store = _repo().store
    broken = store.verify()

    count = store.count()
    size = store.size_on_disk()
    typer.echo(f"{count} object(s), {size} bytes on disk")

    if broken:
        for obj_hash in broken:
            typer.secho(f"corrupt: {obj_hash}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("all objects verified", fg=typer.colors.GREEN)


def _agent_client(model: str) -> Any:
    """Build the real Anthropic-backed client. The one seam tests monkeypatch.

    Kept as its own function, rather than inlined into :func:`_make_agent`,
    so a test can replace it with a :class:`~memgit.agent.client.LLMClient`
    fake without ever importing ``anthropic`` or touching the network.
    """
    from memgit.agent.client import default_client

    return default_client(model=model)


def _make_agent(repo: Repository, *, model: str, record_empty: bool) -> "MemoryAgent":
    from memgit.agent.client import AgentError
    from memgit.agent.runtime import MemoryAgent

    try:
        client = _agent_client(model)
    except AgentError as exc:
        _fail(str(exc))
        raise  # unreachable
    return MemoryAgent(repo, client=client, model=model, record_empty=record_empty)


def _print_turn_result(repo: Repository, result: "TurnResult") -> None:
    typer.echo(result.reply)
    if result.commit is None:
        typer.secho("(no memory change)", fg=typer.colors.YELLOW, err=True)
        return
    commit = repo.read_commit(result.commit)
    typer.secho(_format_commit_oneline(result.commit, commit), fg=typer.colors.CYAN, err=True)
    _print_diff_stat(repo.diff(after=result.commit))


@app.command("replay")
def replay_cmd(
    revision: str = typer.Argument(..., help="Memory state to replay from."),
    subject: str = typer.Argument(..., help="Subject of the belief to ablate."),
    predicate: str = typer.Argument(..., help="Predicate of the belief to ablate."),
    query: str = typer.Argument(..., help="Query to replay against both states."),
    model: str = typer.Option("claude-opus-5", "--model", help="Model to replay with."),
    fact: str = typer.Option(
        None, "--fact", help="Ablate only this one fact hash at (subject, predicate), not every value there."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Replay QUERY against REVISION with and without one belief — never commits.

    Runs the same query twice: once against the memory at REVISION, once
    against that same state with (SUBJECT, PREDICATE) removed. Neither run
    changes the repository — this is an experiment, not a turn. See PLAN.md's
    "Honest caveats": a changed reply is evidence, not proof, that the
    ablated belief caused it.
    """
    from memgit.agent.client import AgentError
    from memgit.replay.engine import ablate_and_replay

    repo = _repo()
    try:
        client = _agent_client(model)
    except AgentError as exc:
        _fail(str(exc))
        return

    try:
        result = ablate_and_replay(
            repo, revision, (subject, predicate), query, client=client, model=model, fact_hash=fact
        )
    except RevisionNotFoundError as exc:
        _fail(f"unknown revision: {exc}")
        return
    except AgentError as exc:
        _fail(str(exc))
        return

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "query": result.query,
                    "key": list(result.key),
                    "fact_hash": result.fact_hash,
                    "baseline": {"reply": result.baseline.reply, "stop_reason": result.baseline.stop_reason},
                    "ablated": {"reply": result.ablated.reply, "stop_reason": result.ablated.stop_reason},
                    "changed": result.changed,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    typer.echo(f"baseline> {result.baseline.reply}")
    typer.echo(f"ablated>  {result.ablated.reply}")
    color = typer.colors.YELLOW if result.changed else typer.colors.CYAN
    typer.secho(f"changed: {'yes' if result.changed else 'no'}", fg=color, err=True)


@app.command("ask")
def ask_cmd(
    message: str = typer.Argument(..., help="What to say to the agent."),
    model: str = typer.Option("claude-opus-5", "--model", help="Model to run the turn with."),
    record_empty: bool = typer.Option(
        False, "--record-empty", help="Commit even when the turn remembers nothing."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Run one turn against the memory at HEAD, and commit whatever it remembers.

    One call to :class:`~memgit.agent.runtime.MemoryAgent`: the whole memory
    state at HEAD is shown to the model, ``remember``/``forget`` tool calls
    are applied, and the result lands as a single commit — unless nothing
    changed, in which case nothing is committed (see ``--record-empty``).
    """
    from memgit.agent.client import AgentError

    repo = _repo()
    agent = _make_agent(repo, model=model, record_empty=record_empty)
    try:
        result = agent.turn(message)
    except AgentError as exc:
        _fail(str(exc))
        return

    if as_json:
        typer.echo(
            json.dumps(
                {"reply": result.reply, "commit": result.commit, "stop_reason": result.stop_reason},
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    _print_turn_result(repo, result)


@app.command("chat")
def chat_cmd(
    model: str = typer.Option("claude-opus-5", "--model", help="Model to run turns with."),
    record_empty: bool = typer.Option(
        False, "--record-empty", help="Commit even when a turn remembers nothing."
    ),
) -> None:
    """An interactive REPL against the memory at HEAD.

    Each line is one turn — see :meth:`MemoryAgent.turn`. The chat transcript
    lives only for this process; what an agent remembers survives only
    through the commits each turn makes, which is what ``/state`` after a
    restart demonstrates. ``/state``, ``/log``, ``/diff`` inspect memory
    without spending a turn; ``/quit`` (or EOF) exits.
    """
    repo = _repo()
    agent = _make_agent(repo, model=model, record_empty=record_empty)
    typer.echo("memgit chat — /state, /log, /diff, /quit")

    from memgit.agent.client import AgentError

    while True:
        try:
            line = input("you> ")
        except EOFError:
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in ("/quit", "/exit"):
            break
        if stripped == "/state":
            if repo.head_commit() is None:
                typer.echo("No memories recorded yet.")
            else:
                typer.echo(repo.state().render())
            continue
        if stripped == "/log":
            if repo.head_commit() is None:
                typer.echo("No commits yet.")
            else:
                for commit_hash, commit in walk(repo.resolve("HEAD"), repo.read_commit, limit=10):
                    typer.echo(_format_commit_oneline(commit_hash, commit))
            continue
        if stripped == "/diff":
            if repo.head_commit() is None:
                typer.echo("No commits yet.")
            else:
                _print_diff_stat(repo.diff())
            continue

        try:
            result = agent.turn(line)
        except AgentError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            continue
        _print_turn_result(repo, result)


cardinality_app = typer.Typer(help="Inspect and edit the diff engine's single/multi schema.")
app.add_typer(cardinality_app, name="cardinality")


@cardinality_app.callback(invoke_without_command=True)
def cardinality_main(ctx: typer.Context) -> None:
    """List declared predicates and the effective default. Mirrors bare ``memgit cardinality``."""
    if ctx.invoked_subcommand is not None:
        return
    mapping = _repo().cardinality()
    typer.echo(f"default: {mapping.default}")
    for predicate, cardinality in mapping.declared():
        typer.echo(f"{predicate}\t{cardinality}")


@cardinality_app.command("set")
def cardinality_set_cmd(
    predicate: str = typer.Argument(..., help="Predicate to declare."),
    cardinality: str = typer.Argument(..., help="'single' or 'multi'."),
) -> None:
    """Declare PREDICATE as single- or multi-valued for the diff engine."""
    if cardinality not in ("single", "multi"):
        _fail(f"cardinality must be 'single' or 'multi', got {cardinality!r}")
        return
    repo = _repo()
    repo.set_cardinality(repo.cardinality().with_predicate(predicate, cardinality))


@cardinality_app.command("unset")
def cardinality_unset_cmd(
    predicate: str = typer.Argument(..., help="Predicate to remove the declaration for."),
) -> None:
    """Remove PREDICATE's declaration, reverting it to the map's default."""
    repo = _repo()
    repo.set_cardinality(repo.cardinality().without_predicate(predicate))


decay_app = typer.Typer(help="Inspect and edit temporal confidence decay half-lives.")
app.add_typer(decay_app, name="decay")


@decay_app.callback(invoke_without_command=True)
def decay_main(ctx: typer.Context) -> None:
    """List declared half-lives and the effective default. Mirrors bare ``memgit decay``."""
    if ctx.invoked_subcommand is not None:
        return
    policy = _repo().decay()
    typer.echo(f"default: {policy.default if policy.default is not None else 'never'} day(s)")
    for predicate, half_life in policy.declared():
        typer.echo(f"{predicate}\t{half_life if half_life is not None else 'never'}")


@decay_app.command("set")
def decay_set_cmd(
    predicate: str = typer.Argument(..., help="Predicate to declare a half-life for."),
    half_life: str = typer.Argument(..., help="Half-life in days, or 'never'."),
) -> None:
    """Declare PREDICATE's half-life, in days, or 'never' to exempt it from decay."""
    repo = _repo()
    if half_life == "never":
        days: float | None = None
    else:
        try:
            days = float(half_life)
        except ValueError:
            _fail(f"half-life must be a number of days or 'never', got {half_life!r}")
            return
    repo.set_decay(repo.decay().with_half_life(predicate, days))


@decay_app.command("unset")
def decay_unset_cmd(
    predicate: str = typer.Argument(..., help="Predicate to remove the declaration for."),
) -> None:
    """Remove PREDICATE's declaration, reverting it to the policy's default."""
    repo = _repo()
    repo.set_decay(repo.decay().without_half_life(predicate))


if __name__ == "__main__":
    app()
