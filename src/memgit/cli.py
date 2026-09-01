"""MemGit command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from memgit import __version__
from memgit.core.fact import Fact
from memgit.core.store import (
    CorruptObjectError,
    ObjectNotFoundError,
    ObjectStore,
    hash_object,
)

app = typer.Typer(
    name="memgit",
    help="Version control and time-travel debugging for AI agent memory.",
    no_args_is_help=True,
    add_completion=False,
)

MEMGIT_DIR = ".memgit"


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


def _find_repo(start: Path | None = None) -> Path:
    """Walk upward looking for a ``.memgit`` directory, like git does.

    Searching ancestors rather than requiring the exact directory means
    commands work from anywhere inside a project, which is the behaviour
    anyone who has used git already expects.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / MEMGIT_DIR).is_dir():
            return candidate / MEMGIT_DIR
    typer.secho(
        "not a memgit repository (no .memgit directory found in this "
        "directory or any parent)",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)


def _store() -> ObjectStore:
    return ObjectStore(_find_repo() / "objects")


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

    typer.echo(_store().put(payload) if write else hash_object(payload))


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
        payload = _store().get(obj_hash)
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
    store = _store()
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
