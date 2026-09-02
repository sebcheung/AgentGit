"""MemGit command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from memgit import __version__
from memgit.core.commit import Commit
from memgit.core.diff import ChangeKind, Diff
from memgit.core.fact import Fact
from memgit.core.graph import is_ancestor
from memgit.core.repository import (
    EmptyCommitError,
    NotARepositoryError,
    Repository,
    RepositoryExistsError,
    RevisionNotFoundError,
)
from memgit.core.store import CorruptObjectError, ObjectNotFoundError, hash_object

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
def show_cmd(rev: str = typer.Argument("HEAD", help="Revision to show.")) -> None:
    """Show a commit's metadata and the facts in its tree. Mirrors ``git show``."""
    repo = _repo()
    try:
        commit_hash = repo.resolve(rev)
        commit = repo.read_commit(commit_hash)
        tree = repo.read_tree(commit_hash)
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
    for fact in tree.load(repo.store):
        typer.echo(f"  {fact}")


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


@app.command("status")
def status_cmd() -> None:
    """Show the current branch, HEAD, and fact count."""
    repo = _repo()
    head = repo.head()
    if head.is_detached:
        typer.echo(f"HEAD detached at {head.commit}")
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


if __name__ == "__main__":
    app()
