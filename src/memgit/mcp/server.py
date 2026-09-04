"""The MCP tool surface: write + read, never a HEAD move.

**Every write is staged, never immediate.** ``remember``/``forget`` fold
onto this MCP session's own :class:`~memgit.core.staging.StagingArea`
(``mcp/session.py``); only ``commit`` makes anything durable. This is the
"declared, not observed, turn boundary" PLAN.md's turn-granularity row now
accounts for — see ``core/staging.py``'s module docstring for the full
argument against committing per tool call or on connection teardown.

**Every read takes an explicit, optional ``rev`` — never HEAD.** HEAD is the
human's cursor at the CLI; a read tool that silently followed it would be
unreproducible between two identical calls seconds apart (a `memgit
checkout` run in another terminal would move it) and could surface beliefs
from a branch this session never wrote to. ``rev`` defaults to ``"HEAD"``
only as the CLI's own default already means — the last commit on whatever
branch the repository's config names as its write target — not to a
per-session cursor of its own.

**No ``checkout``/``reset``/``rewind``/branch-delete/config tools.** Each
mutates state shared by every connected client and the human at the CLI
(HEAD, an existing branch, repo-local config), and nothing this server does
needs any of them — reads take an explicit ``rev``; writes go through
staging onto a fixed, configured branch. ``create_branch`` is the one
branch-mutating exception: it is purely additive, creates a new name, and
orphans nothing.

Tool functions raise :class:`~mcp.server.mcpserver.exceptions.ToolError` for
anything the model can react to and fix (a malformed fact, an unresolvable
revision) — never a bare ``return`` of an error string, which the SDK's own
docs call out as silently reading as success (``is_error`` stays ``False``).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError

from memgit.agent.tools import ToolCallError, decode_forget, decode_recall, decode_remember
from memgit.core.diff import ChangeKind, Diff
from memgit.core.repository import Repository, RevisionNotFoundError
from memgit.mcp.session import (
    forget as fold_forget,
)
from memgit.mcp.session import (
    remember as fold_remember,
)
from memgit.mcp.session import (
    resolve_session_id,
    seal,
)

__all__ = ["build_server", "run"]

_DIFF_PREFIXES: dict[ChangeKind, str] = {
    ChangeKind.ADDED: "+",
    ChangeKind.REMOVED: "-",
    ChangeKind.REAFFIRMED: "~",
    ChangeKind.CONTRADICTED: "!",
    ChangeKind.VALUE_ADDED: ">",
    ChangeKind.VALUE_REMOVED: "<",
    ChangeKind.MIXED: "*",
    ChangeKind.UNCHANGED: " ",
}


def _render_diff(diff: Diff) -> str:
    """A plain-text rendering of ``diff``, independent of typer.

    Mirrors ``cli.py``'s ``_print_diff_human`` (the same prefixes, the same
    taxonomy), but returns a string rather than printing — this module has
    no CLI dependency, matching the rest of the ``mcp`` package.
    """
    if diff.is_empty:
        return "(no changes)"
    lines = []
    for kd in diff.keys:
        detail = ", ".join(str(v) for v in kd.changed_values)
        lines.append(f"{_DIFF_PREFIXES[kd.kind]} {kd.subject}\t{kd.predicate}\t{detail}")
    return "\n".join(lines)


def build_server(repo: Repository, *, name: str = "memgit") -> MCPServer:
    """Build (but do not run) an MCP server wrapping ``repo``.

    Kept separate from :func:`run` so tests can drive it with an in-memory
    :class:`mcp.Client` — no transport, no subprocess, no port.
    """
    mcp = MCPServer(
        name,
        instructions=(
            "MemGit: versioned, diffable agent memory. `remember`/`forget` "
            "stage changes onto this session's own staging area; nothing is "
            "durable until `commit` is called. Every read tool (`recall`, "
            "`read_state`, `diff`, `log`) shows the last sealed commit on "
            "the repository's configured branch, never this session's own "
            "unsealed staging and never another session's."
        ),
    )
    # One id per server process, used only when the transport has no session
    # concept of its own (stdio, the in-memory test transport) — see
    # `session.resolve_session_id`.
    process_session_id = uuid.uuid4().hex

    def _sid(ctx: Context) -> str:
        return resolve_session_id(ctx, process_session_id=process_session_id)

    @mcp.tool()
    def remember(
        ctx: Context,
        subject: str,
        predicate: str,
        object: str,
        confidence: float,
        source_text: str,
    ) -> str:
        """Record one durable (subject, predicate, object) fact.

        Staged onto this session's own staging area — call `commit` to make
        it durable. Reuse an existing subject/predicate exactly when one
        already applies; see `read_state` for the current key inventory.
        """
        try:
            call = decode_remember(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "confidence": confidence,
                    "source_text": source_text,
                },
                source=f"mcp:{_sid(ctx)}",
            )
        except ToolCallError as exc:
            raise ToolError(str(exc)) from exc
        state = fold_remember(repo, _sid(ctx), call)
        return f"staged: {subject} {predicate}={object} ({len(state.facts)} fact(s) staged; call commit to seal)"

    @mcp.tool()
    def forget(
        ctx: Context, subject: str, predicate: str, object: str | None, reason: str
    ) -> str:
        """Retract a belief that is no longer true.

        `object=None` retracts every value at (subject, predicate). Use only
        for retraction, not revision — call `remember` with the new value
        for a revision instead.
        """
        try:
            call = decode_forget(
                {"subject": subject, "predicate": predicate, "object": object, "reason": reason}
            )
        except ToolCallError as exc:
            raise ToolError(str(exc)) from exc
        fold_forget(repo, _sid(ctx), call)
        target = f"{subject} {predicate}"
        return f"staged retraction: {target}" + (f"={object}" if object else " (all values)")

    @mcp.tool()
    def commit(ctx: Context, message: str) -> str:
        """Seal this session's staged `remember`/`forget` calls into a durable commit.

        A no-op turn (nothing staged, or nothing this session touched)
        reports that plainly rather than erroring.
        """
        commit_hash = seal(repo, _sid(ctx), message, author=f"mcp:{name}")
        if commit_hash is None:
            return "nothing to commit"
        return f"committed {commit_hash[:8]}: {message}"

    @mcp.tool()
    def recall(
        ctx: Context,
        query: str,
        subject: str | None,
        limit: int,
        rev: str = "HEAD",
    ) -> str:
        """Search memory for facts relevant to `query`, scoped to `rev` (default: the last commit)."""
        try:
            call = decode_recall({"query": query, "subject": subject, "limit": limit})
        except ToolCallError as exc:
            raise ToolError(str(exc)) from exc
        try:
            state = repo.state(rev)
        except RevisionNotFoundError as exc:
            raise ToolError(str(exc)) from exc
        retriever = repo.retriever()
        result = retriever.retrieve(
            state,
            call.query,
            k=call.limit,
            as_of=datetime.now(UTC),
            subjects={call.subject} if call.subject is not None else None,
        )
        return result.render()

    @mcp.tool()
    def read_state(ctx: Context, rev: str = "HEAD") -> str:
        """Render the full memory state at `rev` (default: the last commit)."""
        try:
            state = repo.state(rev)
        except RevisionNotFoundError as exc:
            raise ToolError(str(exc)) from exc
        return state.render()

    @mcp.tool()
    def diff(ctx: Context, before: str | None, after: str = "HEAD") -> str:
        """Show what changed between two commits. `before=None` means `after`'s first parent."""
        try:
            result = repo.diff(before=before, after=after)
        except RevisionNotFoundError as exc:
            raise ToolError(str(exc)) from exc
        return _render_diff(result)

    @mcp.tool()
    def log(ctx: Context, rev: str = "HEAD", limit: int = 10) -> str:
        """List recent commits on `rev`, most-recent-first."""
        try:
            commits = list(repo.log(rev, limit=limit))
        except RevisionNotFoundError as exc:
            raise ToolError(str(exc)) from exc
        if not commits:
            return "(no commits)"
        return "\n".join(str(c) for c in commits)

    @mcp.tool()
    def create_branch(ctx: Context, name: str, at: str = "HEAD") -> str:
        """Create a new branch at `at` (default: HEAD). Never moves HEAD or any existing branch."""
        try:
            target = repo.create_branch(name, at=at)
        except RevisionNotFoundError as exc:
            raise ToolError(str(exc)) from exc
        return f"created branch {name} at {target[:8]}"

    return mcp


def run(
    repo: Repository,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
) -> None:
    """Build and run a server over ``repo``. Blocks for the life of the server.

    ``transport`` is ``"stdio"`` (the default — a host launches this as a
    subprocess) or ``"streamable-http"``. SSE is deliberately not offered:
    the MCP spec superseded it with Streamable HTTP in the 2025-03-26
    protocol revision, and the SDK's own docs say plainly "don't build
    anything new on it."

    Args:
        api_key: Required on every streamable-HTTP request when set.
            Ignored (with a warning) on stdio — the host process already
            owns the pipe, so a key on a transport it already controls is
            theater, not a real access boundary.
    """
    server = build_server(repo)
    if transport == "stdio":
        if api_key is not None:
            import sys

            print(
                "warning: --api-key has no effect on stdio -- the host process already "
                "owns this pipe",
                file=sys.stderr,
            )
        server.run(transport="stdio")
        return

    # `MCPServer.run(transport="streamable-http")` has no hook for
    # middleware, so the key check is applied by hand: build the Starlette
    # app `run_streamable_http_async` would otherwise build internally, wrap
    # it in `ApiKeyMiddleware`, and run it the same way that method does.
    import uvicorn

    from memgit.mcp.transport import ApiKeyMiddleware

    starlette_app = server.streamable_http_app(host=host)
    gated_app = ApiKeyMiddleware(starlette_app, key=api_key)
    uvicorn.run(gated_app, host=host, port=port)
