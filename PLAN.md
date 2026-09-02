# MemGit — Version Control for AI Agent Memory

## The Goal (one-liner)

A git-inspired version control and time-travel debugging system for AI agent memory — snapshot, branch, diff, rewind, and replay what an agent "believed" over time, so you can causally explain *why* an agent behaved the way it did.

**The problem it solves:** Modern agents carry persistent memory across sessions, but that memory is an opaque, unversioned blob. When an agent misbehaves, there's currently no way to answer "why did it do that" — was it the prompt, the model, or something it remembered three sessions ago? MemGit makes agent memory inspectable, diffable, and reversible, the same way git made code state inspectable.

**Elevator pitch:** "I built a memory versioning system for AI agents because nobody could explain why an agent behaved differently once it had persistent memory. It borrows git's commit/branch/diff model, but the diff is semantic — because memory is structured facts, not text."

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Core language | Python | Matches your strongest skills; also the dominant language for agent tooling |
| Memory representation | Structured facts (subject–predicate–object + metadata: timestamp, source, confidence) stored as JSON-serializable objects | Needs to be structured enough to diff meaningfully — raw text blobs can't be diffed semantically |
| Object store | Custom content-addressable store (SHA-256 hashed objects, like git blobs/trees) — implement yourself, don't import a VCS library | This is the part interviewers will ask about; building it yourself is the point |
| Commit graph | Simple DAG stored as JSON/SQLite — commits reference a tree hash + parent(s) + message | Mirrors git's model, small enough to hand-roll |
| Agent runtime | Anthropic API (or OpenAI) for the actual agent's reasoning/responses | You're not rebuilding an LLM, just instrumenting how it reads/writes memory |
| CLI | `typer` or `click` | Gives you a real `memgit commit`, `memgit branch`, `memgit diff` command-line feel |
| API layer | FastAPI | Thin REST layer so the dashboard (and eventually other users) can hit the memory store |
| Dashboard | React (or plain HTML/JS if you want to move faster) | Visualizes commit graph, renders diffs, drives replay comparisons |
| Testing | `pytest` | Needed especially for the diff/causal-attribution logic — this is where bugs hide |
| Storage backend | Postgres for the commit graph / fact metadata (relational, real foreign keys); object store blobs can stay as flat hashed files or a blob column | Gives you genuine relational data modeling instead of just a key-value blob store — a real gap-filler for backend depth |
| Migrations | Alembic | Shows you understand schema evolution isn't a one-time thing |
| API validation | Pydantic models for all request/response schemas (comes with FastAPI) | Explicit, typed contracts instead of raw dicts — a deliberate design choice worth naming in interviews |
| Auth | Simple API-key or token-based auth on the deployed API | Real answer to "how would you secure this," without building a full identity system |
| Retrieval / RAG | `sentence-transformers` or an embedding API + plain `numpy` cosine similarity or Chroma for the vector index | Needed once memory grows past what fits in context; scoped to correct commit/branch (see below) |
| Tool exposure | MCP server wrapping MemGit's read/write/diff/branch operations, served remotely over HTTP/SSE (not just local stdio) | Lets any MCP-compatible agent — including a live Claude conversation — use MemGit as a real, reachable memory backend |
| Deployment | Fly.io, Render, or a VPS (DigitalOcean droplet) | Proves you can ship something reachable, not just run it on localhost |
| CI/CD | GitHub Actions — run tests + deploy on push to `main` | Small setup, real "I do CI/CD" claim |
| Logging / error handling | Python `logging` with levels, retries on LLM calls, explicit error responses from the API | Production code is mostly about the failure paths, not the happy path |
| Monitoring | A `/health` endpoint + free-tier uptime monitor (e.g. UptimeRobot) | Small addition, genuinely senior-sounding sentence in an interview |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Dashboard (React)                   │
│   commit graph view · diff viewer · replay comparison    │
└───────────────────────────┬───────────────────────────────┘
                             │ REST
┌───────────────────────────▼───────────────────────────────┐
│                    API Layer (FastAPI)                    │
└───────────────────────────┬───────────────────────────────┘
                             │
       ┌─────────────────────┼─────────────────────┐
       ▼                     ▼                     ▼
┌─────────────┐      ┌───────────────┐     ┌─────────────────┐
│ Object Store │      │ Commit Graph  │     │  Diff Engine    │
│ (CAS, hashed │◄────►│ (DAG: commits,│◄───►│ (semantic fact  │
│  fact blobs) │      │  branches)    │     │  comparison)    │
└─────────────┘      └───────────────┘     └─────────────────┘
                             ▲
                             │
                    ┌────────┴─────────┐
                    │  Agent Runtime    │
                    │ (reads current    │
                    │  memory state,    │
                    │  calls LLM, writes│
                    │  new facts →      │
                    │  new commit)      │
                    └───────────────────┘
                             │
                    ┌────────┴─────────┐
                    │ Replay / Ablation │
                    │  Engine (rerun    │
                    │  task on 2 memory │
                    │  states, diff     │
                    │  outputs, attempt │
                    │  causal           │
                    │  attribution)     │
                    └───────────────────┘

┌─────────────────────────┐      ┌──────────────────────────┐
│   Retrieval Layer (RAG)  │      │   MCP Server (deployed,   │
│  embed facts at write    │      │   HTTP/SSE, not just      │
│  time, retrieve top-k    │◄────►│   local stdio)             │
│  scoped to commit/branch,│      │  exposes commit/branch/   │
│  weighted by confidence  │      │  diff/read/write as MCP   │
│  decay                   │      │  tools — any MCP-compatible│
└─────────────────────────┘      │  agent (incl. live Claude) │
                                  │  can use MemGit as memory  │
                                  └──────────────────────────┘

        Deployment: Fly.io/Render/VPS · GitHub Actions CI/CD ·
        Postgres · /health endpoint + uptime monitoring
```

### Core components, in build order

1. **Fact schema** — decide the structured format for a "memory item" (subject/predicate/object + confidence + timestamp + source). This decision shapes everything downstream — spend real time here.
2. **Object store** — content-addressable storage: hash each fact, store by hash, dedupe automatically (a fact that hasn't changed doesn't get re-stored). This is literally git's blob model.
3. **Commit graph** — a commit = a tree of fact-object hashes + parent commit + message. Branches are just pointers to a commit. This is the DAG git also uses.
4. **Diff engine** — given two commits, compute added/removed/changed facts. The interesting design question — what counts as "changed" vs. "contradicted" vs. "unrelated new fact"? — is answered in "Decisions locked in" below (change taxonomy, cardinality map).
5. **Checkout / rewind** — materialize the full memory state at any commit, so an agent can be "restored" to a past belief state.
6. **Agent runtime hookup** — wire a real agent (via the Anthropic API) to read from a memory state and write new facts back as a new commit after each interaction.
7. **Replay / ablation engine** — the causal-attribution piece: take a specific memory diff, remove just that one fact, replay the same query, and measure whether/how the output changes. This is what turns "a fancy database" into "a debugging tool" — treat it like a real experiment (control for everything except the one variable).
8. **Retrieval layer (RAG)** — embed each fact at write time; on each agent turn, embed the query and retrieve top-k relevant facts *scoped to the current commit/branch* (never leak facts from a later commit or a different branch) instead of injecting the full memory into context.
9. **Temporal confidence decay** — facts lose confidence over time unless reaffirmed by a later commit; retrieval ranking factors in decayed confidence alongside semantic similarity.
10. **MCP server wrapper** — expose commit/branch/diff/read/write as MCP tools, so any MCP-compatible agent (including Claude itself) can use MemGit as a real memory backend in a live conversation.
11. **Eval suite ("CI for agent memory")** — reuse the replay engine as a regression check: define a small set of test queries with expected behaviors, run them automatically against every new commit, and flag if a memory change broke something.
12. **Dashboard** — commit graph visualization, a diff viewer, and a "run replay comparison" button. This is what makes the live demo land.
13. **Production hardening** — migrate metadata storage to Postgres with Alembic migrations, add Pydantic request/response validation, add API-key auth, add structured logging + retry logic on LLM calls, add a `/health` endpoint.
14. **Deploy** — ship the API and MCP server to Fly.io/Render/a VPS, wire up GitHub Actions for test-and-deploy, and hook up basic uptime monitoring.

### Build order, as vertical slices

The 14 components above are grouped into slices — each one a working, demoable
increment, tested before moving on:

| Slice | Component(s) | Status |
|---|---|---|
| 0. Skeleton | — | ✅ |
| 1. Fact + CAS | 1, 2 | ✅ |
| 2. Commit graph | 3 | ✅ |
| 3. Diff engine | 4 | ✅ |
| 4. Checkout / rewind | 5 | ← next |
| 5. Agent runtime | 6 | |
| 6. Replay / ablation | 7 | |
| 7. Retrieval + confidence decay | 8 + 9 (merged) | |
| 8. MCP server | 10 (moved up — the tool-call fact-write decision makes this nearly the same code as slice 4) | |
| 9. Eval suite | 11 | |
| 10. FastAPI + dashboard | 12 | |
| 11. Production hardening + deploy | 13 + 14 (merged) | |

Checkout/rewind is deliberately its own slice (4), after the diff engine (3) rather
than bundled into the commit graph (2): materializing a past state is easiest to get
right once the diff engine has already forced a decision about what a tree entry
means at a given key.

---

## Decisions locked in

Design choices already made and built on, not up for re-litigation without a reason:

| Decision | Choice | Why |
|---|---|---|
| Fact schema | Strict `(subject, predicate, object)` + metadata + `source_text` | `(subject, predicate)` is the diff key; SHA-256 of canonical JSON is the CAS key. Free-form NL would make the diff fuzzy and untestable, which kills the core selling point. |
| Object store | Filesystem CAS, git-identical layout | `.memgit/objects/ab/cdef…`, `refs/heads/*`, `HEAD`. Max learning value; the on-disk layout can sit next to a real `.git`. SQLite/Postgres is added later *only* for the commit-graph metadata and eval results, not for slices 1-3. |
| Fact creation | Agent calls `remember(...)` deliberately as a tool | Structured at birth — no extraction step, no parsing noise. Same shape the MCP server needs, so the agent runtime and MCP server share code. Honest caveat: the agent only remembers what it decides to remember. |
| Tree shape | One flat, sorted `Tree` object per commit, not a subject-sharded two-level tree | Sharding (subject as git's "directory") only lowers a constant factor — the root tree is still rewritten every commit, so the asymptotics don't change — and it costs a recursive walk in the diff engine, the project's core intellectual work. Reversible: the entry encoding stays private behind `Tree`'s API, and `.memgit/config` carries a `format_version`. |
| Tree entry cardinality | One `(subject, predicate)` key maps to a *list* of fact hashes, not one | Keeps `(subject, predicate)` the single diff key exactly as `fact.py` promises, and leaves "is a second value an addition or a contradiction" as a pure diff-engine interpretation rule (slice 3's cardinality map) instead of baking an ontology into storage. |
| Staging / index | No `.memgit/index`; `Repository.commit()` takes the whole fact set | Git's index solves selective staging and a stat cache, neither of which applies to an agent turn (all-or-nothing, facts already in memory). An index is also mutable, uncommitted, unhashed state — exactly the one thing this project's premise says should always be diffable and reversible. When slice 5 needs staging across two process invocations, the answer is a ref holding a tree hash, not git's binary index. |
| Checkout / rewind | Its own slice (4), after the diff engine (3), not bundled into the commit graph | Materializing a past state is easiest to get right once the diff engine has already forced a decision about what a tree entry means at a given key. |
| Cardinality map | Repo-local `.memgit/cardinality.json`, keyed on predicate, undeclared predicates default to `single`; not committed, not hashed | A diff is a question you ask, not data you store — committing the schema forces an unanswerable "whose schema wins" when diffing two commits made under different declarations. `single` by default because a debugging tool should default toward the loud classification: a rival value at a functional predicate gets reported as `contradicted`, and declaring it `multi` is a one-time cost. Every machine-readable diff embeds the map it used, so results stay auditable without being versioned. |
| Change taxonomy | Eight key-level kinds (`unchanged`/`added`/`removed`/`reaffirmed`/`contradicted`/`value_added`/`value_removed`/`mixed`), with the authoritative detail carried per *value* | A key can change several ways at once, so one label can't be authoritative on its own — it's a summary over a per-value list, with a total, ordered collapse rule. `added` is the unrelated new fact, `value_added` the coexisting belief, `contradicted` the revision, `reaffirmed` the same claim at new confidence. |
| `memgit diff` with no arguments | `HEAD` vs its first parent | There is no working tree and no index (see the staging row), so there is no uncommitted state to diff. "What did the newest turn change?" is the honest analogue and the more useful default. |
| `merge_base` | Two-source painting + a reduce pass; `merge_bases` returns all candidates, `merge_base` picks one deterministically by date | Needed by `diff a...b`, the honest question for branch attribution: a direct two-dot diff would also report facts one branch simply hasn't received yet. Inherits `walk`'s clock-skew caveat; a synthesized virtual merge base for criss-cross histories would need a merge algorithm MemGit has no slice for, so multi-candidate ambiguity is surfaced rather than resolved. |
| LLM | Anthropic API, `claude-opus-5`, Python SDK | |

---

## Metrics worth tracking (for your README and interview talking points)

- % of behavior changes correctly attributed to a specific memory diff (your causal attribution accuracy)
- Storage efficiency from deduplication (bytes saved vs. naive full-copy snapshotting)
- Diff computation time as memory size scales
- A concrete before/after case study: "Agent gave wrong answer X, we branched memory, removed fact Y, replayed, answer changed to correct — here's the diff"

---

## Suggested build order (8-10 weeks)

- **Weeks 1-2:** Fact schema + object store + commit graph, backed by Postgres with Alembic migrations from the start (no LLM yet — pure data structures, test with synthetic data)
- **Weeks 3-4:** Diff engine + checkout/rewind, with a thorough test suite; API layer with Pydantic-validated schemas
- **Weeks 5-6:** Hook up a real agent via the Anthropic API; get memory reads/writes flowing into commits; add the retrieval/RAG layer once memory volume makes full-context injection impractical
- **Weeks 7-8:** Replay/ablation engine + temporal confidence decay + eval suite ("CI for agent memory")
- **Weeks 9-10:** MCP server wrapper (served remotely over HTTP/SSE) + dashboard
- **Weeks 11-12:** Production hardening (auth, logging/retries, health endpoint) + deploy to Fly.io/Render/VPS + GitHub Actions CI/CD + uptime monitoring
- **Buffer:** polish the demo case study, write the README with real numbers, publish to GitHub

Note: weeks 11-12 push this past the original 8-10 week estimate — if time is tight, the production-hardening pass is the right place to compress (e.g., skip Alembic and hand-write one clean schema, or skip uptime monitoring) rather than cutting the core versioning/diff/replay engine, which is the actual point of the project.

## Scope discipline — deliberately excluded

To keep this a focused, finishable summer project rather than an open-ended systems platform, the following were considered and intentionally left out:

- **Multi-agent shared memory** (conflict resolution and access control across multiple agents sharing a memory space) — real complexity, marginal added resume value over what's already here
- **Full distributed tracing / observability stack** (OpenTelemetry-style) — the dashboard and commit graph already provide equivalent debugging value at a fraction of the complexity
- **Training or fine-tuning a custom embedding model** — an off-the-shelf embedding model is the correct, unremarkable choice here; the project's value is the versioning/attribution system, not the embeddings themselves

---

## Honest caveats to be ready to discuss in an interview

- Semantic diffing of natural-language-derived facts is genuinely hard — you'll need to decide how facts get *extracted* from raw agent output (LLM-based extraction is easiest to start, but introduces its own noise/errors you should be upfront about).
- Causal attribution via ablation is a real methodology, but it's not bulletproof — removing one fact can have downstream effects on other facts that reference it, so be honest about the limits of "removed X, output changed, therefore X caused it."
- This is a debugging/observability tool, not a production memory system for a live product — scope it as such rather than overclaiming.

---

## Repo conventions

- Package lives in `src/memgit/`. Core is import-clean: `memgit.core.*` never
  touches the network, so slices 1-3 test fast and offline.
- Tests in `tests/`, one module per core module — with one deliberate
  exception: `tests/test_cli.py` covers `cli.py`, since the CLI is the only
  place a core exception becomes an exit code, and that translation needs its
  own coverage rather than being assumed from the core modules' tests.

## Development environment

- No `python` on PATH — use the `py` launcher, or `py -m uv run ...`.
- `uv` is installed as a pip package, so it's `py -m uv`, not bare `uv`.
- **The network does TLS interception.** Every `uv` command that touches the
  index needs `--system-certs` or it fails with `invalid peer certificate:
  UnknownIssuer`. e.g. `py -m uv sync --all-extras --system-certs`.

## Commands

```sh
py -m uv sync --all-extras --system-certs   # install/refresh deps
py -m uv run pytest                          # test
py -m uv run memgit --help                   # CLI
```
