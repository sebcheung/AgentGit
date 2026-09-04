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
| Auth | Simple API-key or token-based auth on the API | Real answer to "how would you secure this," without building a full identity system |
| Retrieval / RAG | `sentence-transformers` or an embedding API + plain `numpy` cosine similarity or Chroma for the vector index | Needed once memory grows past what fits in context; scoped to correct commit/branch (see below) |
| Tool exposure | MCP server wrapping MemGit's read/write/diff/branch operations, served over Streamable HTTP (not just local stdio) | Lets any MCP-compatible agent — including a live Claude conversation — use MemGit as a real memory backend; run locally, or tunneled temporarily (e.g. `ngrok`/`cloudflared`, both free) for a live demo, rather than permanently hosted. Originally scoped as "HTTP/SSE"; SSE was superseded by Streamable HTTP in the MCP spec's 2025-03-26 revision, and the SDK's own docs say plainly not to build anything new on it — see the slice 8 decision rows below |
| Packaging | Docker + `docker compose up` — no hosted deployment | Zero cost, zero ongoing maintenance surface; "clone it, run one command" is a complete, honest answer to "how would someone run this" for a project that isn't serving real traffic. See "Scope discipline" for why a paid or usage-billed host isn't worth the risk here |
| CI | GitHub Actions — run tests and build the Docker image on every push to `main` | Real "I do CI" claim without needing anywhere to deploy to |
| Logging / error handling | Python `logging` with levels, retries on LLM calls, explicit error responses from the API | Production code is mostly about the failure paths, not the happy path |
| Health check | A `/health` endpoint, exercised by Docker's own `HEALTHCHECK` and by CI, not by an external uptime monitor | Same signal ("this thing knows how to report its own health") without needing a live host for an uptime service to poll |

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
│   Retrieval Layer (RAG)  │      │ MCP Server (Streamable    │
│  embed facts at write    │      │   HTTP + stdio; local, or │
│  time, retrieve top-k    │◄────►│   tunneled for a live      │
│  scoped to commit/branch,│      │   demo)                    │
│  weighted by confidence  │      │  exposes remember/forget/ │
│  decay                   │      │  commit/recall/read/diff/ │
└─────────────────────────┘      │  log/create_branch as MCP │
                                  │  tools — any MCP-compatible│
                                  │  agent (incl. live Claude) │
                                  │  can use MemGit as memory  │
                                  └──────────────────────────┘

        Packaging: Docker Compose (local, one command) ·
        GitHub Actions CI · Postgres · /health endpoint
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
14. **Package, don't deploy** — containerize the API and MCP server with Docker Compose so the whole stack runs with one command, and wire up GitHub Actions to run tests (and build the image) on every push. No hosted deployment — see "Scope discipline" for why.

### Build order, as vertical slices

The 14 components above are grouped into slices — each one a working, demoable
increment, tested before moving on:

| Slice | Component(s) | Status |
|---|---|---|
| 0. Skeleton | — | ✅ |
| 1. Fact + CAS | 1, 2 | ✅ |
| 2. Commit graph | 3 | ✅ |
| 3. Diff engine | 4 | ✅ |
| 4. Checkout / rewind | 5 | ✅ |
| 5. Agent runtime | 6 | ✅ |
| 6. Replay / ablation | 7 | ✅ |
| 7. Retrieval + confidence decay | 8 + 9 (merged) | ✅ |
| 8. MCP server | 10 (moved up — the tool-call fact-write decision makes this nearly the same code as slice 4) | ✅ |
| 9. Eval suite | 11 | ✅ |
| 10. FastAPI + dashboard | 12 | |
| 11. Production hardening + packaging | 13 + 14 (merged) | |

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
| Rewind vs. reset | `memgit rewind` (re-commit a past tree forward, non-destructive) is primary; `memgit reset` (move a branch pointer, destructive) is secondary, made survivable by the reflog | The no-index decision above argues against ever creating "mutable, uncommitted, unhashed state" — a branch move that orphans commits is that same sin one level up. A rewind is itself an attributable commit; a reset's abandoned commits are recoverable only via `.memgit/logs/...` while it still holds their hash, otherwise harmless collectable garbage per the object-write-before-ref-move ordering. With no working tree and no index, git's `--soft`/`--mixed`/`--hard` split collapses into one operation, since there is no uncommitted state for the distinction to apply to. |
| Reflog timestamps | ISO-8601 UTC (`canonical.utcnow`), not git's `<epoch> <tz>` | Every other clock in the project already speaks one dialect; a second one for one file earns nothing, and the git-lookalike demo value is already carried by `HEAD`, `refs/`, and now `logs/` existing at all. |
| Cardinality map | Repo-local `.memgit/cardinality.json`, keyed on predicate, undeclared predicates default to `single`; not committed, not hashed | A diff is a question you ask, not data you store — committing the schema forces an unanswerable "whose schema wins" when diffing two commits made under different declarations. `single` by default because a debugging tool should default toward the loud classification: a rival value at a functional predicate gets reported as `contradicted`, and declaring it `multi` is a one-time cost. Every machine-readable diff embeds the map it used, so results stay auditable without being versioned. |
| Change taxonomy | Eight key-level kinds (`unchanged`/`added`/`removed`/`reaffirmed`/`contradicted`/`value_added`/`value_removed`/`mixed`), with the authoritative detail carried per *value* | A key can change several ways at once, so one label can't be authoritative on its own — it's a summary over a per-value list, with a total, ordered collapse rule. `added` is the unrelated new fact, `value_added` the coexisting belief, `contradicted` the revision, `reaffirmed` the same claim at new confidence. |
| `memgit diff` with no arguments | `HEAD` vs its first parent | There is no working tree and no index (see the staging row), so there is no uncommitted state to diff. "What did the newest turn change?" is the honest analogue and the more useful default. |
| `merge_base` | Two-source painting + a reduce pass; `merge_bases` returns all candidates, `merge_base` picks one deterministically by date | Needed by `diff a...b`, the honest question for branch attribution: a direct two-dot diff would also report facts one branch simply hasn't received yet. Inherits `walk`'s clock-skew caveat; a synthesized virtual merge base for criss-cross histories would need a merge algorithm MemGit has no slice for, so multi-candidate ambiguity is surfaced rather than resolved. |
| LLM | Anthropic API, `claude-opus-5`, Python SDK | |
| Agent tool surface | `remember` and `forget` only — no `reaffirm`, no `recall` | `remember` is the locked-in write path; a second `remember` of the same triple at a new confidence already reads as a reaffirmation, so a separate `reaffirm` verb would just be a second way to say the same thing and a chance to pick the wrong one. `recall` earns its place once slice 7 replaces full-context injection with scoped retrieval — until then the whole state is already in the prompt, and a second read path would let the model act on facts it was never shown. `forget` stays because retraction (`removed`/`value_removed`) is not expressible as any `remember`. |
| `remember` at an existing key | The repo's cardinality map decides: `single` replaces, `multi` adds alongside | `Tree` maps one key to a *list* of hashes; without this rule every `single` predicate would accumulate rivals and every diff would report a cardinality violation. Consulting the map at write time does not re-litigate "a diff is a question you ask, not data you store" — the runtime is a client of that repo-local lens exactly as `memgit diff` is, and the write itself is neither hashed nor committed. |
| Agent commit messages | Derived from the applied ops and the query, never model-authored | A model-authored message can disagree with what the diff actually shows. In a tool whose premise is trustworthy history, a terse derived message beats a plausible but possibly-wrong one. |
| Agent turn granularity | One turn = one commit, written once after the tool-calling loop ends, `allow_empty` off by default. **The boundary is observed in-process (the loop ending) or declared out-of-process (an MCP client calling `commit`)** — slice 8 adds this clause rather than special-casing itself out of the rule, since the subject was always what a commit means, not how the boundary is found | A turn is the unit a human attributes; per-tool-call commits would record beliefs the model revises within the same turn, and an API failure mid-loop would leave memory half-written. A turn that remembers nothing produces no commit unless `--record-empty` is passed — `EmptyCommitError`'s own docstring calls recording no-ops "a deliberate choice for that caller to make," not a default, since it would fill the log with identical trees from every chit-chat turn. Over MCP there is no tool-calling loop for the server to observe ending, so the boundary has to be declared instead — see the staging rows below. |
| Transcript vs. memory | The chat transcript lives only in-process, for one `MemoryAgent`; it is never committed | Persisting it would be exactly the "mutable, uncommitted, unhashed state" the no-index decision above already refuses. Memory crosses a process boundary through commits, not through the transcript — each turn re-reads `HEAD` and rebuilds the system prompt from it, so a fact remembered in one `memgit chat` invocation reaches the next. |
| Replay / ablation | `memgit replay` never calls `Repository.commit`; it runs two fresh, isolated one-turn conversations (baseline and ablated) sharing nothing but the `LLMClient` | An ablated `MemoryState` already drops its `commit` provenance (see `state.py`) — persisting one back into history would fabricate a belief the agent never held. Each replay starts its own `messages` list rather than reusing `MemoryAgent`'s transcript, so the only variable between the two runs is the one ablated belief, not conversational drift. |
| `AblationResult.changed` | Literal string inequality between the two replies, not a semantic diff | Cheap and honest about what it is: evidence a belief mattered, not proof — see "Honest caveats" below. Semantic-equivalence checking is a retrieval-grade problem (slice 7), not this slice's; slice 9 discharges its own half of this deferral by not needing one at all (see that slice's rows). |
| Embedding storage | Sidecar `fact_hash -> vector`, namespaced by embedder id (`.memgit/embeddings/<id>/`); never inside the `Fact` payload | A vector field would rehash every fact and every tree it appears in, invalidating all of slices 1-6's history and destroying the dedup metric below. `Fact.hash` already covers every field, so `fact_hash -> vector` is an immutable, content-keyed cache — a vector is never stale for content reasons, only if the embedder itself changes, which is exactly what the id namespace isolates. |
| Commit/branch scoping | The candidate set for retrieval is `Repository.state(rev).facts`, never a separate commit-aware index | `Repository.state(rev)` already materializes exactly the right fact set and is already tested; a second, commit-partitioned vector index would mean re-implementing that guarantee with a real chance of getting it wrong. A fact absent from the state passed to `Retriever.retrieve` cannot appear in its output, even if that fact's vector is already cached from a later commit or a different branch — the leak PLAN.md's retrieval-layer decision (component 8) warns against is structurally impossible, not merely filtered out. |
| Vector index shape | Brute-force cosine over a content-keyed cache (an append-only float32 pack plus a JSON offset index), not a vector database (Chroma) | Chroma or any collection-based index needs per-commit or per-branch partitioning to preserve the scoping guarantee above — i.e., building the guarantee instead of inheriting it. Brute-force cosine at 256 dimensions and realistic corpus sizes (~10^3-10^4 facts) costs single-digit milliseconds in pure Python; ~10^5 facts is the measured, documented crossover where this would need revisiting, not a surprise. |
| Vector index status | Repo-local, uncommitted, like `cardinality.json` — but for a different, stronger reason | `cardinality.json` is uncommitted because it is **config**, a question with no principled single answer across two commits. The vector index is uncommitted because it is **cache**: a pure function of content already in the object store plus an embedder id, fully reconstructible by `memgit embed`. Committing it would mean storing the answer to a question the store can already re-answer. |
| Default embedder | Deterministic, zero-dependency lexical feature hashing (`HashingEmbedder`), behind an `Embedder` protocol seam shaped exactly like `LLMClient` | `sentence-transformers` pulls in torch and a multi-gigabyte dependency footprint for a project whose only other runtime dependency is `typer`, and its model download breaks the "core stays offline" convention on first use in this network environment. A hosted embedding API means a second vendor and a second API key — Anthropic has no first-party embeddings endpoint, confirmed against the installed SDK — with a network call sitting inside the fact-write path. The seam lets a real backend (`sentence-transformers`, or Voyage AI, Anthropic's own documented recommendation) drop in later without touching anything else; `HashingEmbedder` is documented as lexical, not sold as a stand-in for semantic embeddings. |
| Decay | Computed at read time, never stored; exponential half-life over `asserted_at` (`confidence * 0.5 ** (age / half_life)`); `as_of` always an explicit argument, never read from the wall clock inside the math | `confidence` is inside a fact's hashed payload by design (see the fact schema row above), so writing a decayed value back would mint a new fact hash — recordable only as a commit that fabricates a belief the agent never actually reasserted. Exponential is memoryless and single-parameter; explicit `as_of` is what keeps every decay, ranking, and replay call deterministic and reproducible on a later day. |
| Decay's blast radius | Applied only by the retrieval ranker and an explicit `--as-of` display flag; never inside `Repository.read_fact` | `read_fact` is the one `FactReader` the diff engine uses, and it decides `reaffirmed` vs. `unchanged` purely by fact hash — decay there wouldn't break that classification, but it would corrupt every printed confidence-strength comparison (`is_strengthened`/`is_weakened`), making a changed key look universally weakened. `memgit diff` and `memgit log` always show stored confidence. |
| Decay map | `.memgit/decay.json`, keyed on predicate, per-predicate half-life in days or `null` for "never decays", repo-local and uncommitted, decay on by default (180-day default half-life) | Same reasoning as the cardinality map: declaring a half-life and re-ranking last month's commit changes the ranking, not history. Decay defaults on, following cardinality's "default toward the loud classification" — safe because decay only ever reweights ranking, it never filters or hides a fact from a listing. |
| Decay reset on reaffirmation | No new machinery — `Fact.reaffirm()` already bumps `asserted_at`, and the agent runtime's merge step already replaces a same-triple fact with the newly-asserted one | Decay reads `asserted_at`, so a freshly reaffirmed belief is simply young again. This is a consequence of facts being immutable values and reaffirmation already being modeled as "a new fact at the same key," not a feature that needed building. |
| Ranking | `(1 - w) * max(0, cosine similarity) + w * decayed confidence`, linear rather than multiplicative, `w = 0.25` by default | A product (`similarity * confidence`) lets a fact that has decayed toward zero vanish from results even when it is the only relevant belief in scope — hiding exactly the stale belief a debugging tool exists to surface. Linear degrades gracefully at both extremes of `w` and keeps relevance dominant while confidence breaks ties and demotes staleness rather than erasing it. |
| Injection mode threshold | Full-state injection below `config["retrieval"]["full_below"]` facts (default 64); top-k retrieval at or above it | Matches this file's own suggested build order ("add the retrieval/RAG layer once memory volume makes full-context injection impractical") literally — below the threshold, retrieval can only lose information while adding a failure mode, so full injection stays strictly better. |
| Key inventory | A full `(subject, predicate)` key listing is always injected alongside a retrieved subset, never top-k facts alone | Top-k alone lets the model lose sight of which keys already exist, so it invents a near-duplicate predicate for a belief it already holds but wasn't shown — and that drift lands in the diff engine as `added`, the taxonomy's most reassuring label, not the silent corruption of the diff key it actually is. The inventory is what lets the model recognize "I have a belief here, I just wasn't shown it" instead of manufacturing a new one. |
| `recall` | The one new agent tool this slice adds — offered only when a turn is in retrieval mode (the same flag that decides injection), searching the turn's committed `before` state, never the mutating `working` copy | `recall` earns its place exactly when PLAN.md said it would: "once slice 7 replaces full-context injection with scoped retrieval." Under full injection there is nothing left to recall, so tying its availability to retrieval mode is a one-flag answer rather than a second rule to keep in sync. Searching `before` rather than `working` keeps recall scoped to a real commit, so a call can never surface a belief the model only just invented earlier in the same turn. |
| Retrieval and the working state | A turn's mutable `working` copy always starts as the full `before` state, in retrieval mode or not; retrieval only narrows what gets rendered into the prompt | `Repository.commit` takes the whole fact set, never a delta. Seeding `working` from a retrieved subset instead of the full state would mean a turn that only saw 8 of 200 facts silently commits a memory with 192 of them deleted — the single most dangerous class of bug this slice could introduce. |
| Ablation + retrieval | `ablate_and_replay` retrieves exactly once, against the baseline state; the ablated side reuses that same retrieved set minus the ablated fact, with no backfill | Re-retrieving independently per side breaks the one-variable property the replay engine's whole design leans on: removing the ablated fact frees a slot in the top-k, a different fact backfills it, and the two prompts then differ in two facts instead of one — the control group changed, which is a defect in the experiment, not the "downstream effects" caveat already on record. The residual bias (the pinned set was ranked *with* the ablated fact present) is the honest trade for holding the stimulus constant, and is called out in "Honest caveats" below rather than hidden. |
| MCP write path | `remember`/`forget` stage onto a per-session ref holding a tree hash (`refs/memgit/staging/<key>/{tree,base}`); an explicit `commit` tool seals it. No auto-commit on connection teardown | Cashes in the no-index row's own escape hatch: "When slice 5 needs staging across two process invocations, the answer is a ref holding a tree hash, not git's binary index." A staging ref is mutable and uncommitted, but never *unhashed* — every intermediate tree is a real, content-addressed object with every fact already written, so it stays diffable (`memgit diff HEAD refs/memgit/staging/<key>/tree` works with zero new core code) and reflogged like any other ref. A commit needs an attributable author and message; a dropped connection has neither, so teardown nudges the model in tool-result text ("staged (N pending) — call commit to seal") rather than firing an unattributed commit. |
| Staging session identity | The ref key is `sha256(session_id).hexdigest()[:16]`, never the raw MCP session id | A ref name only forbids `../` and a few other shapes, not `/` — a raw, potentially attacker-influenced id (plausible for the tunneled HTTP transport this slice's demo story leans on) could otherwise steer a ref write outside `refs/memgit/staging/`. Hashing is deterministic (the same id always maps to the same key, so a tool result's key can be echoed straight into `memgit staging show`) and every output already satisfies the ref-name character set by construction. |
| Sealing under a moved branch tip | Key-level overlay: diff the staged tree against its own `base` to find which keys this session touched, then overlay just those onto the branch's *current* tip, never the staged tree wholesale | Committing the staged tree wholesale would silently delete every fact another writer added since `base` — the same catastrophe the "retrieval and the working state" row already warns against for a different reason. The overlay reuses `rewind`'s own `keys=` logic (extracted into `Repository.overlay`), costs one structural tree comparison when nothing else moved (the common case, bit-identical to a wholesale commit), and never deletes a belief this session didn't touch. A same-key collision between two sessions has no principled merge — no slice owns one — so it is recorded in the new commit's `restaged_over` metadata rather than silently resolved, the same posture `merge_base` already takes for divergent branches. |
| MCP write target | An explicitly configured branch ref (`config["mcp"]["branch"]`, default `default_branch`) via `Repository.commit`'s new `onto=` parameter — never wherever HEAD points | HEAD is the human's cursor at the CLI; a write that silently followed it would relocate an agent's memory because someone ran `memgit checkout` in another terminal, or land on nothing if HEAD is detached. `onto=None` preserves every prior slice's commit behavior exactly. |
| MCP tool surface | Write + read, no HEAD moves: `remember`/`forget`/`commit`/`recall`/`read_state`/`diff`/`log`/`create_branch`. No `checkout`/`reset`/`rewind`/branch-delete/`set_cardinality`/`set_decay` | Each excluded operation mutates state shared by every connected client and the human at the CLI, and nothing the server does needs any of them — reads take an explicit `rev`, writes go through staging onto a fixed branch. `create_branch` stays because it is purely additive and orphans nothing. |
| MCP read scoping | Every read tool takes an explicit, optional `rev` — never HEAD, and never a session's own unsealed staging | HEAD is the human's CLI cursor, not a session's; a read that silently followed it is unreproducible between two identical calls seconds apart and can leak beliefs from a branch this session never wrote to — the same cross-branch leak the commit/branch-scoping row already calls structurally impossible, but only because the candidate set is deliberate. Reading a session's own `working` staging area (rather than the last sealed commit) would let the model act on a belief it only just invented, unsealed — the same reasoning that already keeps `recall` scoped to a turn's committed `before` state in the agent runtime. |
| MCP transport | Streamable HTTP (default, loopback) + stdio. SSE dropped | The MCP spec superseded SSE with Streamable HTTP in the 2025-03-26 revision, and the Python SDK's own docs say plainly: "Don't build anything new on it." PLAN.md originally scoped this as "HTTP/SSE"; that wording predates the deprecation. |
| Eval assertions | Over *memory* — state, a commit's diff, and retrieval — never over a model's reply. Deterministic, offline, no API key | Component 11 says "reuse the replay engine as a regression check"; this slice deliberately does not. `AblationResult.changed` compares two independent, stochastically-sampled API replies with no temperature or seed available, so its false-positive rate equals the model's own sampling variance — honest for a command a human reads, disqualifying for an automated gate. The eval suite is replay's deterministic sibling: it asks the same question ("did a memory change break something?") without a model. An eval suite that needs an API key and a paid, non-reproducible call per case is not CI, it's a smoke test with a bill. |
| Semantic equivalence, slice 9's half | Not built | The `AblationResult.changed` row above defers semantic-equivalence checking to slice 7 (discharged there by `HashingEmbedder`'s own honest "lexical, not semantic" caveat). Slice 9 never needed a judge in the first place: an eval case asserts over memory, where `(subject, predicate)` is an exact key and `Fact.hash` is exact content identity — there is nothing fuzzy left to judge. An LLM judge would add a second stochastic process to adjudicate the first, leaving neither the signal nor the verdict reproducible. |
| Eval vs. pytest | Both, at different units, deliberately not merged | pytest tests MemGit's *code* at a fixed working tree, parametrized at import time — the existing 700+ tests stay exactly that. `memgit eval` tests *a memory repository that did not exist when the cases were written*, parametrized over that repository's commit DAG, and reports the first commit at which a case broke. That sweep-and-attribute is what pytest cannot express, and it is the only reason `memgit.eval` exists rather than being a pytest plugin. If an eval case ever asserts something about MemGit's own behavior rather than a memory's contents, the boundary has collapsed and the case belongs in `tests/` instead. |
| Eval suite storage | `.memgit/eval/*.json`, repo-local and uncommitted; results are not stored anywhere | Same reasoning as `cardinality.json`/`decay.json`: a suite is a question you ask, not data you store, so committing it into the history it grades would force the unanswerable "whose suite wins when diffing two commits" one level up. This also resolves the object-store row's forward reference to "SQLite/Postgres… for the commit-graph metadata and eval results" — a result is a pure function of `(suite, repo, rev)`, so persisting one stores an answer the store can already re-answer, the same argument the vector-index cache row makes. |
| `memgit.eval` import graph | Imports `memgit.core` and `memgit.retrieval`; never `memgit.agent` | The import graph *is* the design claim that this suite is CI-grade — every check reads the same offline surface `memgit diff` already reads. A dedicated test asserts this holds (in a fresh subprocess, since other tests in the same session legitimately import `memgit.agent` first). |
| Sweep, not bisect | `memgit eval --since` walks every commit between two revisions, oldest to newest, rather than halving the range | `git bisect`'s halving pays for itself only when each probe is expensive; every check here is offline and deterministic, so a linear sweep is cheap and — unlike bisect — assumes nothing about monotonicity. Inherits `walk`'s existing clock-skew caveat, and reports the first failure *along the walked ancestry*, not proof of a unique cause — the same "evidence, not proof" posture the replay engine already takes. |
| CI pulled forward for this slice | Only `.github/workflows/test.yml` (pytest on push/PR) | Shipping a slice headlined "CI for agent memory" into a repository with no CI at all was the ordering bug worth fixing; adopting ruff, mypy, and Docker in the same slice was not. A first lint/type-check pass over ~8,000 unannotated lines would swamp this slice's actual content — that stays in slice 11 with the rest of the hardening pass. |

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
- **Weeks 9-10:** MCP server wrapper (HTTP/SSE, run locally or tunneled for a demo) + dashboard
- **Weeks 11-12:** Production hardening (auth, logging/retries, health endpoint) + Docker Compose packaging + GitHub Actions CI + a local load-test script and write-up
- **Buffer:** polish the demo case study, write the README with real numbers, publish to GitHub

Note: weeks 11-12 push this past the original 8-10 week estimate — if time is tight, the production-hardening pass is the right place to compress (e.g., skip Alembic and hand-write one clean schema, or trim the load-test write-up to one paragraph) rather than cutting the core versioning/diff/replay engine, which is the actual point of the project.

## Scope discipline — deliberately excluded

To keep this a focused, finishable summer project rather than an open-ended systems platform, the following were considered and intentionally left out:

- **Multi-agent shared memory** (conflict resolution and access control across multiple agents sharing a memory space) — real complexity, marginal added resume value over what's already here
- **Full distributed tracing / observability stack** (OpenTelemetry-style) — the dashboard and commit graph already provide equivalent debugging value at a fraction of the complexity
- **Training or fine-tuning a custom embedding model** — an off-the-shelf embedding model is the correct, unremarkable choice here; the project's value is the versioning/attribution system, not the embeddings themselves
- **A permanently hosted deployment** (Fly.io, Render, a VPS, or anything else with a bill attached) — the engineering interviewers actually probe here is the CAS/diff/replay engine, not infra uptime, and a hobby project has no real traffic to justify the ongoing cost or maintenance. Docker Compose (one command, fully local) plus a local load-test run tells the same "this is production-shaped" story for $0
- **An LLM-judge / semantic-equivalence checker for the eval suite** (slice 9) — no slice owns this. It would make the eval suite paid, keyed, and non-reproducible, the three properties that disqualify something from being CI; an eval case asserts over memory, where equality is already exact.
- **Replay-backed eval cases** (slice 9) — asserting an automated pass/fail on `AblationResult.changed` would be asserting on sampling noise, not on a memory regression.
- **A stored eval-results database** (slice 9) — a result is a pure function of `(suite, repo, rev)` and fully recomputable; storing one anyway just duplicates the object store's own re-derivability guarantee for no benefit.
- **A `memgit-eval` pytest plugin** (slice 9) — would collapse the one distinction (parametrizing over a memory's commit DAG, not over code at a fixed working tree) that justifies `memgit.eval` existing as its own package.
- **Case severities, scoring, flakiness budgets, retry policies, an assertion DSL, `memgit eval init` scaffolding, watch mode** (slice 9) — each is a plausible extension with no caller asking for it yet, the same instinct that keeps the change taxonomy and the check vocabulary closed rather than pluggable.

---

## Honest caveats to be ready to discuss in an interview

- Semantic diffing of natural-language-derived facts is genuinely hard — you'll need to decide how facts get *extracted* from raw agent output (LLM-based extraction is easiest to start, but introduces its own noise/errors you should be upfront about).
- Causal attribution via ablation is a real methodology, but it's not bulletproof — removing one fact can have downstream effects on other facts that reference it, so be honest about the limits of "removed X, output changed, therefore X caused it."
- Under retrieval, ablation has a second, distinct confound on top of the one above: the baseline's retrieved set is ranked *with* the ablated fact present, so it may have displaced some other fact from the top-k. The ablated run reuses that same pinned set minus the ablated fact rather than re-retrieving (see "Decisions locked in"), which holds the stimulus constant but means the ablated run may see a slightly smaller, differently-composed context than a live turn would — a changed reply is evidence about the ablated fact specifically, not about what a from-scratch retrieval against the ablated state would have surfaced.
- The default embedder is lexical (hashed word and character-n-gram features), not semantic — it will not recognize that "favourite editor" and "preferred text editor" are the same claim. It is honest about being a zero-dependency baseline behind a seam, not a stand-in for a real sentence-embedding model.
- This is a debugging/observability tool, not a production memory system for a live product — scope it as such rather than overclaiming.
- The MCP server has no auth in this slice (PLAN.md defers that to slice 11's API-key work) — it binds to loopback by default, and the tunneled-demo path (`ngrok`/`cloudflared`) is unauthenticated write access to memory for as long as the tunnel is up. Be upfront that this is a deliberately deferred gap, not an oversight, and close the tunnel the moment the demo ends.
- The eval suite (slice 9) asserts over *memory*, never over agent behavior. A green suite means no belief regressed — it does not mean the agent still answers correctly. That stronger claim needs a non-deterministic, paid check, which is deliberately out of scope; see "Scope discipline" above.
- `memgit eval --since`'s sweep reports the first failing commit along the walked ancestry, not a proof of unique cause — the same "evidence, not proof" posture the replay engine already takes for ablation, and it inherits `walk`'s existing clock-skew caveat.

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
