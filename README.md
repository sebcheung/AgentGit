# MemGit

[![test](https://github.com/sebcheung/AgentGit/actions/workflows/test.yml/badge.svg)](https://github.com/sebcheung/AgentGit/actions/workflows/test.yml)

MemGit is version control for an AI agent's memory. It borrows the commit,
branch, diff, and checkout model from git and applies it to the facts an
agent remembers about you, so you can snapshot what an agent believed at any
point in time, compare two points, roll back to an earlier one, and get a
real answer to "why did it just do that."

See [PLAN.md](PLAN.md) for the full design notes and the order things were
built in.

## Why this exists

Agents with persistent memory are everywhere now, but that memory is usually
a blob: a JSON file, a vector store, a row in a database that just gets
overwritten. When an agent starts behaving differently, there's no way to
ask "what changed in what it remembers, and when." You can look at the
current state, but not the history that produced it.

Git solved this exact problem for code decades ago. A commit is a snapshot,
a diff tells you what changed between two snapshots, and you can always
check out an old one. None of that is specific to source files. It works
just as well for a set of facts, as long as the facts are structured enough
to compare. So MemGit treats each memory as a fact (something like `user
favorite_editor neovim`, with a timestamp, a source, and a confidence
score) and reuses git's model on top of it: content-addressed storage,
a commit graph, branches, and a diff engine, all built from scratch rather
than imported from a library, since understanding how they work is most of
the point of this project.

## The core idea

A **fact** is a `(subject, predicate, object)` triple plus metadata:
when it was asserted, where it came from, and how confident the agent is
in it. Facts get hashed and stored the way git stores blobs. A **commit**
points at a tree of these facts plus a parent commit and a message, exactly
like a git commit points at a tree of files. A **branch** is just a name
pointing at a commit hash. None of this is a metaphor bolted on after the
fact: `.memgit/` really does look like `.git/` on disk (see
[On-disk layout](#on-disk-layout) below), and the CLI verbs (`commit`,
`branch`, `diff`, `checkout`, `log`, `reflog`) mean the same thing they mean
in git.

The one place MemGit genuinely diverges from git is the diff. Git diffs
text line by line. MemGit diffs facts, so `memgit diff` can tell you not
just that something changed, but what kind of change it was: a new belief,
a contradiction, a reaffirmed belief with a fresh confidence score, and so
on. That's covered in detail below.

## Feature tour

### Diffing memory

`memgit diff` is the feature the rest of the project is built around. It's
a semantic diff over facts, not a text diff, so every changed
`(subject, predicate)` gets exactly one line, prefixed by what kind of
change happened to it:

```sh
$ memgit diff
! user  favorite_editor    vim -> neovim (0.90)
+ user  timezone           Europe/Berlin (0.80)
> user  likes              +go (0.80)
```

| prefix | meaning |
|---|---|
| `+` / `-` | a belief the agent didn't have before, or one it no longer has |
| `~` | reaffirmed: the same claim, just a new confidence, timestamp, or source |
| `!` | contradicted: a single-valued fact's value got replaced |
| `>` / `<` | a multi-valued fact gained or lost one of several coexisting values |
| `*` | mixed: a multi-valued fact both gained and lost a value in the same diff |

Whether a second value at the same key counts as a contradiction or just
another belief depends on that key's declared **cardinality**. A predicate
like `favorite_editor` only makes sense with one value at a time, so a
second value replaces the first. A predicate like `likes` can hold several
values at once. This is configurable per predicate
(`.memgit/cardinality.json`; anything undeclared defaults to single-valued):

```sh
$ memgit cardinality set likes multi
$ memgit diff        # the `likes` line above turns from `!` into `>`
```

`memgit diff a...b` (three dots) compares each branch against their common
ancestor instead of comparing them to each other directly. That matters
because a direct diff between two branches would also report facts that one
branch simply hasn't received yet, which isn't really "what did this branch
change." See PLAN.md's "Decisions locked in" for the full reasoning behind
the change taxonomy and the cardinality defaults.

### Checkout and rewind

MemGit has no working tree and no staging area at the CLI level (the MCP
server has its own staging concept, covered later). The memory state at a
commit *is* the tree, so `memgit checkout` only ever moves `HEAD`, either
attached to a branch or detached at a specific commit. There's no
`--force` flag, because there's nothing uncommitted for a checkout to ever
overwrite:

```sh
$ memgit checkout HEAD~3     # detach at an ancestor
HEAD is now detached at 8d258c6e
Memory: 2 key(s) changed
$ memgit state               # what the agent believed there
user:
  prefers_language Python
$ memgit checkout main       # reattach
```

Going back to a past belief state can happen two ways. `memgit rewind` is
the one you should reach for by default: it re-commits the past state
forward as a brand new commit, so the rewind itself shows up in history
instead of leaving a gap where the old commits used to be. `memgit reset`
is the other option, and it's destructive: it moves a branch pointer and
leaves the old commits unreferenced. They aren't gone though, you can still
recover them through `memgit reflog` (backed by `.memgit/logs/HEAD` and
`.memgit/logs/refs/heads/*`) as long as you still have their hash:

```sh
$ memgit rewind HEAD~3 -m "roll back to before the bad turn"
$ memgit reflog
48aa20e9 HEAD@{0}: commit: roll back to before the bad turn
...
```

Anywhere MemGit expects a commit, you can use the same shorthand git
supports: ancestry suffixes (`HEAD~3`, `main^2`), reflog indexing
(`HEAD@{1}`), abbreviated hashes, branch names, or `HEAD` itself.

### On-disk layout

`.memgit/` is laid out like `.git/` on purpose, and it was built by hand
rather than by wrapping git or a database. Here's real output side by side,
from `memgit init && memgit commit` next to `git init && git commit`:

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

Same idea, same shape: objects live under a two-character fanout directory
so no single folder ends up holding thousands of files, `HEAD` is a plain
text file that's either a symbolic ref or a raw hash, and a branch is
nothing more than a file under `refs/heads/` holding a commit hash. The
real differences are small and intentional: MemGit uses one flat JSON
config file instead of git's INI format, and it stores objects as
zlib-compressed JSON instead of git's packed binary format. There's also no
index or staging area at the CLI level, since a single command like `commit
-m "..." --file facts.json` doesn't need one. PLAN.md's "Decisions locked
in" spells out the reasoning for each of these.

### Talking to it

`memgit ask` and `memgit chat` send a real message to Claude, with the
memory at `HEAD` as context. Below 64 facts, the model just sees the whole
memory state. Past that, a turn retrieves only the facts most relevant to
the question and the model can ask for more with a `recall` tool call (see
[Retrieval and decay](#retrieval-and-decay) below). The model has exactly
two tools that can change memory: `remember` and `forget`. Whatever it
decides to remember or forget lands as one commit; if it decides there's
nothing worth remembering, nothing gets committed at all:

```sh
$ memgit ask "I mostly write Python, and my timezone is Europe/Berlin."
Got it, noted both.
a1b2c3d4 turn: I mostly write Python, and my timezone is Europe/Berlin.
2 key(s) changed: 2 added
$ memgit ask "What language do I prefer?"
You prefer Python.
(no memory change)
```

That second `ask` answers correctly without creating a commit, which is
the point: the memory carried over through the commit graph, not through a
saved chat transcript. That's what makes it something you can actually
inspect later, instead of just trusting the model to remember consistently.
`memgit chat` runs the same loop as an interactive REPL, with `/state`,
`/log`, and `/diff` commands to look around without spending a turn on the
model.

### Explaining it: replay and ablation

`memgit replay` is the tool for answering "did this specific fact actually
matter?" It runs the same question twice against a memory state: once as
is, and once with one fact removed. Then it tells you whether the answer
changed. Neither run writes anything, an "ablated" state is a hypothetical
used for one experiment, not a real point in history:

```sh
$ memgit replay HEAD user prefers_language "what language do I prefer?"
baseline> You prefer Python.
ablated>  I don't know your language preference.
changed: yes
```

That's the whole "the agent gave a wrong answer, I removed a belief, and
the answer changed" workflow as a single command instead of a manual
before-and-after test. Worth being honest about what this proves though: a
changed answer is evidence that the removed fact mattered, not a
mathematical proof, since removing one fact can have knock-on effects on
others that reference it. PLAN.md's "Honest caveats" section covers this in
more depth, including a retrieval-specific wrinkle: `replay` reuses the
same retrieved set of facts for both runs instead of re-retrieving after
the ablation, specifically so the experiment only ever changes one thing at
a time.

### Retrieval and decay

Once a memory has more than 64 facts, `memgit ask`/`chat` stop showing the
model everything and instead retrieve the top few facts that are actually
relevant to the question. This retrieval is scoped to exactly the commit
or branch being read: it only ever ranks facts that already exist at that
point in history, so nothing from a later commit or a different branch can
leak in, even if it's already been embedded and cached. Alongside the
retrieved facts, the model also gets a full list of every existing
`(subject, predicate)` key, so it can recognize a belief it already has
even if it wasn't shown the value, instead of accidentally creating a
near-duplicate. The model can also call `recall` directly to search
everything:

```sh
$ memgit recall "what editor does the user prefer" HEAD
3 of 214 fact(s) at HEAD (a1b2c3d4)
  user favorite_editor neovim (score=0.81, sim=0.74, conf=0.95)
  ...
```

Ranking combines similarity with **temporal confidence decay**: a belief
slowly loses confidence over time (an exponential half-life measured from
when it was asserted, 180 days by default) unless a later commit
reaffirms it, which resets the clock for free, since a reaffirmation is
just a new fact at the same key with a newer timestamp. Decay only affects
how facts are read and ranked, it's never written to storage, so
`memgit diff` and `memgit log` always show the confidence that was
actually recorded at the time:

```sh
$ memgit decay set favorite_editor 30
$ memgit state --as-of 2027-01-01T00:00:00Z
  user favorite_editor neovim (0.95)  [as of 2027-01-01T00:00:00+00:00: 0.12]
```

The default way facts get turned into vectors for similarity search is a
deterministic hash of words and characters. No machine learning model, no
API key, works fully offline. An optional real sentence-embedding model
(via `fastembed`, the `semantic` extra) can be swapped in instead, since
both sit behind the same small interface. That interface is what makes the
swap possible without touching any of the retrieval or agent code around
it.

### MCP server

`memgit serve` exposes this same memory store to any MCP-compatible
client, like Claude Desktop or a live Claude conversation over a tunnel, as
a set of tools: `remember`, `forget`, `commit`, `recall`, `read_state`,
`diff`, `log`, and `create_branch`. Two things about it are worth calling
out:

- **Writes are staged, not immediate.** Calls to `remember`/`forget` build
  up in the connecting session's own staging area (a real, hashed,
  diffable tree under `refs/memgit/staging/<key>/`, which you can inspect
  with `memgit staging`). Nothing is saved for good until the model calls
  `commit`. Unlike `memgit ask`/`chat`, there's no natural point where an
  MCP server can tell a conversation is "over," so the model has to
  explicitly say when a turn is done rather than the server inferring it.
- **Reads never follow `HEAD`.** Every read tool takes an explicit,
  optional `rev` argument (defaulting to the latest commit on the
  repository's configured branch). `HEAD` is the human's cursor at the
  CLI, not something a remote session should be quietly following.

```sh
py -3 -m uv sync --extra mcp --system-certs
py -3 -m uv run memgit serve                              # stdio, for Claude Desktop
py -3 -m uv run memgit serve --transport streamable-http   # a real HTTP server
```

For Claude Desktop, add this to its config:

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

If a session never calls `commit`, whether the host crashed or the
connection just closed, nothing is lost. `memgit staging` lists every open
staging area, `memgit staging show <key>` and `diff <key>` let you look
inside one, and `memgit staging commit <key> -m "..."` seals it from the
CLI directly.

`memgit serve --transport streamable-http` requires an API key
(`--api-key`, or the `MEMGIT_API_KEY` environment variable) the moment
`--host` isn't loopback. See [Auth](#auth) below. Plain stdio doesn't need
one: whatever process is running it already has full access to the pipe it
talks over, so a key there wouldn't stop anything a malicious host process
couldn't already do directly.

### CI for agent memory

`memgit eval` checks what an agent's memory *is*, not what it says. Does a
given fact still hold the value it should? Did a commit's diff get
classified the way it should have? Does a query still retrieve the right
belief? Every check reads straight from `Repository.state`, `diff`, and the
retriever, the same offline data `memgit diff` already reads, so a whole
suite runs with no API key, no network call, and the exact same result
every time. That's what makes it usable as a real CI gate instead of just
a demo:

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

Test cases live in `.memgit/eval/*.json`, next to the repository but never
committed into memory history, for the same reason `cardinality.json` and
`decay.json` aren't either: a test case is a question you're asking, not a
fact the agent believes. This is a deliberately different tool from
`memgit replay`: `AblationResult.changed` compares two separate, sampled
replies from the model, so it can be noisy just from normal sampling
variance. That's fine for a command a person reads and interprets, but bad
for an automated gate. `memgit eval` checks memory state instead, which is
fully deterministic and won't give a false alarm from randomness.

Drop the exit code into any CI pipeline:

```yaml
- run: uv run memgit eval
```

### Dashboard and REST API

`memgit serve-web` runs a small read-only REST API and dashboard over the
same repository. No React, no build step, just plain HTML, CSS, and
ES-module JavaScript served directly from the package:

```sh
$ memgit serve-web
memgit dashboard: http://127.0.0.1:8001
```

Open it up and you get a commit graph (an inline SVG, one lane per branch,
laid out with no charting library), a diff viewer for any two commits you
click, a browser for memory state at a given commit, and the replay
comparison from above turned into an actual button: click a fact in the
diff or state view to fill in the subject and predicate, type a question,
and see baseline versus ablated answers side by side with a changed/
unchanged badge.

The API underneath is the same logic the CLI already runs:
`GET /api/log`, `/api/commits/{rev}`, `/api/state`, `/api/diff`,
`/api/recall`, and `POST /api/replay`. The JSON these return is exactly
what `memgit diff --json` and `memgit state --json` print. `/docs` gives
you the full OpenAPI schema for free, courtesy of FastAPI.

**Read-only, except for one route.** Every endpoint is a `GET` except
`POST /api/replay`, and that one is the exception specifically because it
structurally cannot write memory: the function behind it never calls
`Repository.commit`. This needs the `web` extra (`pip install memgit[web]`,
or `uv sync --extra web`), and like `memgit serve`, it requires an API key
the moment `--host` isn't loopback. Every route except `/api/health` and
`/api/ready` enforces it; see [Auth](#auth).

### Auth

Off by default on loopback. Everything above works exactly as shown, with
no login and no key, as long as you run `memgit serve` or `serve-web` with
no `--api-key` and the default host. The moment `--host` isn't loopback,
both commands refuse to start without a key:

```sh
$ memgit serve-web --host 0.0.0.0
refusing to bind '0.0.0.0' with no API key configured -- set --api-key,
$MEMGIT_API_KEY, or pass --insecure to bind anyway
$ memgit serve-web --host 0.0.0.0 --api-key "$(py -3 -m uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
memgit dashboard: http://0.0.0.0:8001
```

Send the key as either `X-API-Key: <key>` or
`Authorization: Bearer <key>`. The dashboard's own JavaScript asks for it
once, the first time it hits a 401, and keeps it for the tab in
`sessionStorage` (never a cookie, never a URL). `--insecure` skips the
check with a loud warning printed to the console. It exists for a quick
tunneled demo (`ngrok`, `cloudflared`), not as a way to make the warning go
away permanently.

This is one shared key, not per-user login. Anyone who has the key looks
identical to the server, so there's no way to revoke access from just one
caller or roll the key without affecting everyone using it. That's a fair
tradeoff for a debugging tool one person tunnels to themselves, but it's
worth being upfront about rather than implying this is a real
multi-user auth system. See PLAN.md's "Honest caveats" for the full
statement of what this does and doesn't protect against.

### Blame and storage stats

Everything described so far reads the object store on disk directly. On
top of that, there's an optional Postgres projection (`memgit.pg`, the
`pg` extra) that mirrors the commit graph into relational tables, built
for the one question the object store is slow at answering: "which commits
touched this fact, and what did each one do to it?"

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

The projection is a derived copy, never the source of truth: `memgit.core`
never imports anything from `memgit.pg`, and every command above still
works with no database configured at all. `memgit project` is safe to
re-run any time, whether after every commit or on a schedule, since it's
idempotent. Every query it answers is also checked against a full,
independent walk of the object store in `tests/test_pg_queries.py`, so the
projection can't silently drift from what the commits actually say. See
PLAN.md's "Decisions locked in" for why Postgres is a projection here
instead of a second source of truth.

## How this was built

This started from a single design question: could a memory system for AI
agents borrow git's model wholesale, not just as an analogy but as an
actual architecture? Everything below followed from answering that one
question honestly, then building it in order.

The object store, commit graph, diff engine, and refs are all hand-written,
not wrapped around `git` itself or a database. That was a deliberate
constraint: the interesting part of a project like this is in the
mechanics of content-addressed storage and a commit DAG, not in whichever
library already implements them, so importing one would have skipped the
part worth building.

The project was built in a sequence of slices, each one a small, working,
testable increment rather than one big push at the end:

1. **Fact schema and object store.** Decide what a "memory item" looks
   like, then hash and store it the way git stores a blob.
2. **Commit graph.** A commit is a tree of fact hashes plus a parent and a
   message. A branch is a name pointing at a commit.
3. **Diff engine.** Given two commits, work out what changed between them,
   and what kind of change it was.
4. **Checkout and rewind.** Materialize the full memory state at any past
   commit, and move between them safely.
5. **Agent runtime.** Wire up a real Claude call that reads memory at
   `HEAD` and writes new facts back as a new commit.
6. **Replay and ablation.** Remove one fact, re-ask the same question, and
   see if the answer changes.
7. **Retrieval and confidence decay.** Once memory gets big, retrieve only
   the relevant facts instead of showing everything, and let old,
   unconfirmed beliefs fade instead of staying at full confidence forever.
8. **MCP server.** Expose the same read and write operations to any
   MCP-compatible client, including a live Claude conversation.
9. **Eval suite.** Turn a handful of expected facts into a repeatable,
   offline check that can run in CI.
10. **Dashboard and REST API.** Make the commit graph, diffs, and replay
    tool visible and clickable instead of terminal-only.
11. **Production hardening.** Linting, type checking, structured logging,
    retries on the Anthropic API calls, API-key auth, a readiness check.
12. **Data and packaging.** A Postgres projection for blame and stats,
    Docker Compose so the whole thing runs with one command, GitHub Actions
    CI, and a local load test to see how the read endpoints hold up as
    memory grows.

Checkout and rewind (slice 4) came after the diff engine (slice 3) on
purpose. Restoring a past state is much easier to get right once the diff
engine has already forced a real decision about what a fact at a given key
means at any point in the tree.

Every slice landed with its own tests before the next one started. The
project has just over 50 test files, one for nearly every module, plus a
CLI test suite covering the part where a core error actually needs to
become the right exit code and error message for a person at a terminal.
CI runs the full test suite, ruff, mypy, a separate Postgres-backed test
job, and a real `docker compose up` against a health check, on every push.

### Choices worth explaining

A few decisions came up more than once while building this, so they're
worth stating plainly instead of leaving them implicit in the code:

- **The diff is semantic, not textual.** Facts are structured triples, so
  a diff can say "this belief was contradicted" instead of just "this line
  changed." That distinction is the actual point of the project: a text
  diff over a JSON memory dump would tell you bytes moved, not what the
  agent now believes differently.
- **`rewind` is the default way to go back, `reset` is the escape hatch.**
  Reverting a mistake should itself be visible in history, not a hole
  where old commits used to be. `reset` still exists for when you
  genuinely want to discard something, and the reflog keeps it recoverable
  either way.
- **Retrieval is scoped strictly to one commit or branch.** A fact from a
  later commit, or a different branch, must never leak into an answer
  just because it happened to already be embedded and cached. Getting this
  wrong would quietly break the entire point of having versioned memory in
  the first place.
- **The eval suite checks memory, not model output.** Asking a model the
  same question twice and comparing the replies is inherently a little
  noisy, since the model can sample differently each time. Memory state is
  exact, so that's what CI checks against.
- **The Postgres projection is derived, never authoritative.** `memgit.pg`
  can be deleted and rebuilt from the object store at any time with
  `memgit project`. Nothing in the core write path depends on Postgres
  being present or up to date.

### What's simplified, on purpose

A few things were left out deliberately, to keep this a finishable project
instead of an open-ended platform: multi-agent shared memory with conflict
resolution, a full observability stack beyond the dashboard, training a
custom embedding model, and a permanently hosted deployment. Docker Compose
plus a local load test tells the same "this actually runs" story as a
hosted deployment would, without an ongoing bill or an idle server nobody
is using.

A few limitations are also worth stating outright rather than leaving
someone to discover them the hard way:

- Ablation (`memgit replay`) shows that removing a fact changed the
  answer. It's evidence the fact mattered, not mathematical proof, since
  facts can have effects on each other that a single removal doesn't fully
  isolate.
- The default embedder is lexical (hashed words and character sequences),
  not a real language model. It won't recognize that "favorite editor" and
  "preferred text editor" mean the same thing unless the optional
  `fastembed`-based semantic embedder is enabled.
- Auth is a single shared API key, not per-user accounts. It's the right
  amount of security for a personal debugging tool tunneled to yourself,
  not a substitute for real identity and access management.
- The Postgres projection can go stale between runs of `memgit project`.
  That's intentional: keeping it always in sync would mean putting a
  database in the middle of the core write path, which is exactly what the
  projection is designed to avoid.
- This is a debugging and research tool for understanding agent memory,
  not a production memory backend for a live product serving real traffic.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python | Structured data and an ecosystem full of agent tooling |
| Object store | Hand-written, SHA-256 content-addressed, like git's blobs | The mechanics are the point of the project |
| Commit graph | A small DAG of commits, trees, and refs | Mirrors git closely enough to reuse its vocabulary directly |
| CLI | `typer` | A real `memgit commit` / `memgit diff` feel from the terminal |
| Agent | Anthropic API (Claude) | Instrumenting how an agent reads and writes memory, not building a new model |
| API | FastAPI, with Pydantic request/response models | Typed contracts instead of raw dicts passed around |
| Dashboard | Plain HTML/CSS/JS, no build step | The interesting part (laying out a commit graph) is a couple hundred lines a chart library wouldn't save |
| Tests | `pytest` | Especially important for the diff and replay logic, which is where the subtle bugs live |
| Relational storage | Postgres, via SQLAlchemy and Alembic migrations | A real second data model for blame/stats queries, with schema evolution handled properly |
| Retrieval | A deterministic lexical hash by default, `fastembed` optional | Zero-dependency and offline by default, upgradable behind one interface |
| Tool exposure | An MCP server over stdio and Streamable HTTP | Lets any MCP-compatible client, including Claude itself, use MemGit as memory |
| Packaging | Docker Compose | One command, fully local, no hosting bill |
| CI | GitHub Actions | Tests, lint, type checking, a Postgres job, and a real Docker build on every push |

## Getting started

```sh
py -3 -m uv sync --all-extras --system-certs
py -3 -m uv run memgit init
py -3 -m uv run memgit commit -m "seed" --file facts.json
py -3 -m uv run memgit log --oneline
py -3 -m uv run memgit show HEAD
```

To run the whole stack (dashboard, MCP server over HTTP, and Postgres) in
one shot:

```sh
cp .env.example .env   # set POSTGRES_PASSWORD and MEMGIT_API_KEY
docker compose up -d --build
curl http://localhost:8001/api/ready
```

This single `docker compose up` brings up Postgres, runs migrations once,
initializes a shared MemGit repository once, then starts the dashboard
(port 8001) and the MCP server over Streamable HTTP (port 8000). Both of
those bind `0.0.0.0` inside the Docker network, which is exactly the case
[Auth](#auth) requires a key for, so `docker-compose.yml` won't start
either service without `MEMGIT_API_KEY` set. See
[docs/LOADTEST.md](docs/LOADTEST.md) for what the read endpoints look like
under load.

See [PLAN.md](PLAN.md) for full setup details, the complete CLI reference,
and the current state of what's built versus planned.
