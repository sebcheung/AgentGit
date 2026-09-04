# MemGit

[![test](https://github.com/sebcheung/AgentGit/actions/workflows/test.yml/badge.svg)](https://github.com/sebcheung/AgentGit/actions/workflows/test.yml)

Version control and time-travel debugging for AI agent memory — snapshot,
branch, diff, checkout, and rewind what an agent "believed" over time, so you
can causally explain *why* it behaved the way it did. See [PLAN.md](PLAN.md)
for the full design and build order.

## Diffing memory

`memgit diff` is the headline feature: a *semantic* diff over structured
facts, not a text diff. Every changed `(subject, predicate)` gets one line,
prefixed by what happened to it:

```sh
$ memgit diff
! user  favorite_editor    vim -> neovim (0.90)
+ user  timezone           Europe/Berlin (0.80)
> user  likes              +go (0.80)
```

| prefix | meaning |
|---|---|
| `+` / `-` | a belief the agent didn't have before / doesn't have anymore |
| `~` | reaffirmed — same claim, new confidence, timestamp, or source |
| `!` | contradicted — a `single`-valued predicate's value was replaced or rivalled |
| `>` / `<` | a `multi`-valued predicate gained / lost one coexisting value |
| `*` | mixed — a `multi`-valued predicate both gained and lost a value |

Whether a second value at a key is a contradiction or just another belief is
controlled by `memgit cardinality` (`.memgit/cardinality.json`, undeclared
predicates default to `single`):

```sh
$ memgit cardinality set likes multi
$ memgit diff        # the `likes` line above turns from `!` into `>`
```

`memgit diff a...b` compares against the two branches' merge base rather than
diffing them directly — the honest question for "what did *this* branch
actually change", since a direct diff would also report facts the other
branch simply hasn't received yet. See [PLAN.md](PLAN.md)'s "Decisions locked
in" for the full change taxonomy and the reasoning behind the cardinality
map's defaults.

## Checkout and rewind

MemGit has no working tree and no index — the memory state *is* the tree — so
`memgit checkout` only ever moves HEAD, attached to a branch or detached at a
commit. There's no `--force`, because there's nothing uncommitted a checkout
could ever overwrite:

```sh
$ memgit checkout HEAD~3     # detach at an ancestor
HEAD is now detached at 8d258c6e
Memory: 2 key(s) changed
$ memgit state               # what the agent believed there
user:
  prefers_language Python
$ memgit checkout main       # reattach
```

Going back to a past belief state has two flavors. `memgit rewind` is
non-destructive and primary: it re-commits the past state forward as a new
commit, so the rewind itself is attributable history, not a hole where one
used to be. `memgit reset` is the destructive alternative — it moves a branch
pointer and leaves the abandoned commits unreferenced, recoverable via
`memgit reflog` (`.memgit/logs/HEAD`, `.memgit/logs/refs/heads/*`) while it
still holds their hash:

```sh
$ memgit rewind HEAD~3 -m "roll back to before the bad turn"
$ memgit reflog
48aa20e9 HEAD@{0}: commit: roll back to before the bad turn
...
```

Revisions understand git's ancestry suffixes (`HEAD~3`, `main^2`), reflog
indexing (`HEAD@{1}`), and abbreviated hashes, on top of `HEAD`/branch names/
full hashes. See [PLAN.md](PLAN.md)'s "Decisions locked in" for why `rewind`
rather than `reset` is the default.

## On-disk layout

MemGit's object store, refs, and HEAD are deliberately laid out like git's,
built by hand rather than imported — the comparison below is real output
from `memgit init && memgit commit`, next to a real `git init && git commit`:

```
.git/                              .memgit/
├── HEAD                           ├── HEAD
├── config                         ├── config
├── objects/                       ├── objects/
│   ├── 45/…                       │   ├── 84/84b755be…
│   ├── 4a/…                       │   ├── ac/a7785f3d…
│   └── ba/…                       │   └── b1/1d51e54d…
├── refs/                          ├── refs/
│   └── heads/                     │   └── heads/
│       └── main                   │       └── main
└── logs/                          └── logs/
    ├── HEAD                           ├── HEAD
    └── refs/                          └── refs/
        └── heads/                         └── heads/
            └── main                           └── main
```

```sh
$ cat .git/HEAD              $ cat .memgit/HEAD
ref: refs/heads/main         ref: refs/heads/main
```

Same idea, same shape: a two-character fanout under `objects/` (256 buckets,
so no directory holds thousands of files), a plain-text `HEAD` that is either
a symbolic ref or a raw hash, and branches that are nothing more than a file
holding a commit hash under `refs/heads/`. Where MemGit differs — one flat
JSON config instead of git's INI, zlib-compressed JSON blobs instead of git's
packed objects, no `index`/staging area (see [PLAN.md](PLAN.md)'s "Decisions
locked in" for why) — is called out explicitly there rather than left as a
silent gap.

## Talking to it

`memgit ask` and `memgit chat` run a real `claude-opus-5` turn against the
memory at `HEAD`. Below 64 facts, the whole state is shown to the model, as
in slice 5; above it, a turn retrieves only the facts most relevant to the
query and the model can call `recall` to search the rest — see
[Retrieval and decay](#retrieval-and-decay) below. `remember` and `forget`
are the only tools that change memory; whatever the model decides lands as
one commit — unless it decides nothing, in which case nothing is committed:

```sh
$ memgit ask "I mostly write Python, and my timezone is Europe/Berlin."
Got it — noted both.
a1b2c3d4 turn: I mostly write Python, and my timezone is Europe/Berlin.
2 key(s) changed: 2 added
$ memgit ask "What language do I prefer?"
You prefer Python.
(no memory change)
```

The second `ask` answers correctly without creating a commit — memory
crossed the process boundary through the commit graph, not through a chat
transcript, which is the property this whole project is built to make
inspectable. `memgit chat` is the same turn loop as a REPL, with `/state`,
`/log`, and `/diff` to look without spending a turn. See `PLAN.md`'s
"Decisions locked in" for why the tool surface is exactly `remember` and
`forget`, and why commit messages are derived rather than model-authored.

## Explaining it: replay and ablation

`memgit replay` is the causal-attribution tool: it runs the same query twice
against a memory state — once as-is, once with one belief removed — and
reports whether the model's answer changed. Neither run commits anything;
an ablated state is explicitly not a real point in history.

```sh
$ memgit replay HEAD user prefers_language "what language do I prefer?"
baseline> You prefer Python.
ablated>  I don't know your language preference.
changed: yes
```

That's the "gave wrong answer, removed a belief, replayed, answer changed"
case PLAN.md's metrics section asks for — a single command instead of a
manual before/after. A changed reply is evidence the ablated fact mattered,
not proof: see PLAN.md's "Honest caveats" on the limits of ablation as a
methodology — including a retrieval-specific one, since `replay` pins one
retrieved set for both sides rather than re-retrieving after ablation, to
keep the experiment down to one variable.

## Retrieval and decay

Once a repository crosses 64 facts, `memgit ask`/`chat` stop injecting the
whole memory and instead retrieve the top-k facts most relevant to the
query — scoped to exactly the commit or branch being read, by construction:
retrieval only ever ranks the facts already materialized by
`Repository.state(rev)`, never a separate index, so a fact from a later
commit or another branch cannot leak in even if its vector is already
cached. A full `(subject, predicate)` key inventory is injected alongside
the retrieved subset, so the model can still recognize an existing belief it
wasn't shown in full rather than inventing a near-duplicate predicate for
it — and the model can call `recall` to search the rest directly:

```sh
$ memgit recall "what editor does the user prefer" HEAD
3 of 214 fact(s) at HEAD (a1b2c3d4)
  user favorite_editor neovim (score=0.81, sim=0.74, conf=0.95)
  ...
```

Ranking blends similarity with **temporal confidence decay** — a belief
loses confidence over time (an exponential half-life over `asserted_at`,
180 days by default) unless reaffirmed by a later commit, which resets it
for free: a reaffirmation is just a new fact at the same key with a fresh
timestamp. Decay is a read lens, never stored — `memgit diff` and `memgit
log` always show the confidence that was actually asserted:

```sh
$ memgit decay set favorite_editor 30
$ memgit state --as-of 2027-01-01T00:00:00Z
  user favorite_editor neovim (0.95)  [as of 2027-01-01T00:00:00+00:00: 0.12]
```

The default embedder is a deterministic, zero-dependency lexical hash — no
torch, no API key, fully offline — behind an `Embedder` seam shaped like the
LLM client seam, so a real semantic model can drop in later without
touching the retrieval or agent code around it. See PLAN.md's "Decisions
locked in" for the full reasoning, including why this isn't backed by a
vector database.

## MCP server

`memgit serve` exposes this same memory store to any MCP-compatible host —
Claude Desktop, or a live Claude conversation over a tunnel — as
`remember`/`forget`/`commit`/`recall`/`read_state`/`diff`/`log`/
`create_branch` tools. Two things carry over from the CLI's own design:

- **Writes are staged, not immediate.** `remember`/`forget` calls fold onto
  the connecting session's own staging area (a real, hashed, diffable tree
  under `refs/memgit/staging/<key>/`, inspectable with `memgit staging`);
  nothing is durable until the model calls `commit`. There is no MCP
  tool-calling loop for the server to observe ending, unlike
  `memgit ask`/`chat`, so the turn boundary is *declared* instead of
  *observed* — see PLAN.md's "Decisions locked in" for the full argument.
- **Reads never follow HEAD.** Every read tool takes an explicit, optional
  `rev` (default: the last commit on the repository's configured branch) —
  HEAD is the human's cursor at the CLI, not a session's.

```sh
py -3 -m uv sync --extra mcp --system-certs
py -3 -m uv run memgit serve                              # stdio, for Claude Desktop
py -3 -m uv run memgit serve --transport streamable-http   # a real HTTP server
```

For Claude Desktop, add to its config:

```json
{
  "mcpServers": {
    "memgit": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/AgentGit", "memgit", "serve"]
    }
  }
}
```

If a session never calls `commit` — a crashed host, a closed connection —
nothing is lost: `memgit staging` lists every open staging area,
`memgit staging show <key>` / `diff <key>` inspect it, and
`memgit staging commit <key> -m "..."` seals it from the CLI.

`memgit serve --transport streamable-http` requires an API key
(`--api-key`, or `$MEMGIT_API_KEY`) the moment `--host` isn't loopback —
see [Auth](#auth) below. stdio is exempt: the host process already owns
the pipe it runs over, so a key there would check nothing a malicious
host couldn't already do directly.

## CI for agent memory

`memgit eval` asserts on what an agent's memory *is*, not on what it says —
which fact holds at a key, what a commit's diff classified a key as, whether
a query still retrieves the right belief. Every check reads
`Repository.state`/`diff`/`retriever`, the same offline surface `memgit
diff` already reads, so a suite needs no API key, no network, and produces
the same result every run — which is what makes it a real CI gate rather
than a demo:

```sh
$ memgit eval                 # 1 passed
$ memgit ask "actually I moved to Berlin"
$ memgit eval                 # 1 failed
FAIL  knows-where-user-lives
        expected user city == 'Boston', got ['Berlin']
0/1 passed
$ memgit eval --since HEAD~5  # which commit broke it
FAIL  knows-where-user-lives  first broke at a1b2c3d4
        expected user city == 'Boston', got ['Berlin']
$ memgit diff a1b2c3d4
! user  city    Boston -> Berlin
```

Cases live in `.memgit/eval/*.json` — repo-local and never committed into
memory history, same reasoning as `cardinality.json`/`decay.json`: a suite is
a question you ask, not data you store. This is deliberately *not* "reuse
the replay engine as a regression check": `AblationResult.changed` compares
two independent, sampled API replies, so its false-positive rate is the
model's own sampling variance — honest for a command a human reads, useless
as an automated gate. `memgit eval` asserts on memory instead, which is
fully deterministic. See [PLAN.md](PLAN.md)'s "Decisions locked in" for the
full reasoning, and its "Honest caveats" for what a green suite does *not*
prove: an agent's memory not regressing is not the same claim as an agent
still answering correctly.

Drop the exit code into any CI:

```yaml
- run: uv run memgit eval
```

## Dashboard and REST API

`memgit serve-web` runs a read-only REST API and a small hand-written
dashboard over this same repository — no React, no build step, plain
HTML/CSS and ES-module JS served straight out of the package:

```sh
$ memgit serve-web
memgit dashboard: http://127.0.0.1:8001
```

Open it and you get a commit graph (an inline SVG, one lane per branch,
laid out in a single forward pass — no charting library), a diff viewer
over whatever two commits you click, a memory-state browser, and the
replay comparison from the section above as an actual button: click a
fact in the diff or state view to populate `(subject, predicate)`, type a
query, and get baseline vs. ablated side by side with a `changed` badge.

The API underneath is exactly what the CLI already computes —
`GET /api/log`, `/api/commits/{rev}`, `/api/state`, `/api/diff`,
`/api/recall`, and `POST /api/replay` — with `Diff.to_dict()` and
`MemoryState.to_dict()` passed straight through, byte-identical to
`memgit diff --json` / `memgit state --json`. `/docs` gets you the full
OpenAPI schema for free.

**Read-only, plus replay.** Every route but one is a `GET`; `POST
/api/replay` is the exception, and it's the one call in this project that
structurally cannot write memory (`ablate_and_replay` never calls
`Repository.commit`). Requires the `web` extra (`pip install memgit[web]`,
or `uv sync --extra web`). `memgit serve-web` requires an API key the
moment `--host` isn't loopback, same as `memgit serve` above — every route
but `/api/health` and `/api/ready` enforces it; see [Auth](#auth).

## Auth

Off by default on loopback — every route above works exactly as shown,
unauthenticated, the moment you run `memgit serve`/`serve-web` with no
`--api-key`. The moment `--host` isn't loopback, both commands refuse to
bind without one:

```sh
$ memgit serve-web --host 0.0.0.0
refusing to bind '0.0.0.0' with no API key configured -- set --api-key,
$MEMGIT_API_KEY, or pass --insecure to bind anyway
$ memgit serve-web --host 0.0.0.0 --api-key "$(py -3 -m uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
memgit dashboard: http://0.0.0.0:8001
```

Present the key as `X-API-Key: <key>` or `Authorization: Bearer <key>`; the
dashboard's own JS prompts for it once, on the first 401, and remembers it
for the tab (`sessionStorage`, never a cookie or a URL). `--insecure`
overrides the refusal with a loud warning — the explicit escape hatch for a
tunneled demo (`ngrok`/`cloudflared`), not a way to silence the warning.

This is a single shared key, not per-caller identity: every request that
presents it looks the same to the server, so there's no rotation and no
revocation list. That's the honest, minimal answer for a debugging tool one
person tunnels to themselves — see PLAN.md's "Honest caveats" for the full
statement of what this does and doesn't buy you.

## Blame and storage stats

Everything above reads the filesystem object store directly. A separate,
optional Postgres projection (`memgit.pg`, the `pg` extra) mirrors the
commit graph into relational tables for the one question the CAS answers
slowly — "which commits touched this key, and what did each one do to it":

```sh
$ export DATABASE_URL=postgresql+psycopg://memgit:...@localhost/memgit
$ memgit project
12 commit(s), 41 fact(s) added
$ memgit blame user favorite_editor
2026-01-03T09:00:00+00:00  a1b2c3d4  +5f9ee29a
2026-01-15T14:22:00+00:00  8d258c6e  -5f9ee29a
2026-01-15T14:22:00+00:00  8d258c6e  +6a225e44
$ memgit stats
41 distinct fact(s), 63 tree entries
dedup ratio: 1.54x
```

The projection is derived, never authoritative: `memgit.core` never
imports `memgit.pg`, and every command above still works with no database
configured at all. `memgit project` is idempotent — safe to re-run after
every commit, or on a schedule — and every query it answers is checked
against a full walk of the object store in `tests/test_pg_queries.py`. See
PLAN.md's "Decisions locked in" for why Postgres is a projection rather
than a second source of truth.

## Running it

```sh
cp .env.example .env   # set POSTGRES_PASSWORD and MEMGIT_API_KEY
docker compose up -d --build
curl http://localhost:8001/api/ready
```

One command, fully local, `$0`: `db` (Postgres), `migrate` (`alembic
upgrade head`, once), `init` (`memgit init` on a shared volume, once), then
`api` (the dashboard, port 8001) and `mcp` (streamable HTTP, port 8000).
Both `api` and `mcp` bind `0.0.0.0` inside the compose network, which is
exactly the case [Auth](#auth) requires a key for — `docker-compose.yml`
refuses to start either service without `MEMGIT_API_KEY` set. See
[docs/LOADTEST.md](docs/LOADTEST.md) for what this stack's read endpoints
look like under load.

## Quickstart

```sh
py -3 -m uv sync --all-extras --system-certs
py -3 -m uv run memgit init
py -3 -m uv run memgit commit -m "seed" --file facts.json
py -3 -m uv run memgit log --oneline
py -3 -m uv run memgit show HEAD
```

See [PLAN.md](PLAN.md) for setup details, the full CLI surface, and what's
built vs. planned.
