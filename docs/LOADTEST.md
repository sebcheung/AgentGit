# Load test

`scripts/loadtest.py` seeds a synthetic repository, runs `memgit serve-web`
against it as a real subprocess, and hits the read endpoints with a thread
pool — one cold request, then many warm ones, reported separately since
`/api/recall`'s first call is the one that pays to build the vector index.

```
py -m uv run --extra web python scripts/loadtest.py --synthesize 5000
```

`/api/replay` is deliberately never touched: it spends two paid Anthropic
API calls per request, so a load test that hits it is a bill, not a
benchmark.

## What this measures

This is **not** a database benchmark. There is no database in the request
path for any of these four endpoints — every one of them reads the
filesystem content-addressable store directly, zlib-inflating each object
it touches. What these numbers characterize is that read path: how it
scales as a repository accumulates facts, and where the concurrency-facing
edges (the shared `retrieval_lock` — see PLAN.md's "Vector-cache writes
under concurrency" decision row) actually cost something.

## Results

One machine (Intel Core i9-10850K, Windows 11), `memgit serve-web`'s
default single-worker dev server, 8-way client concurrency, 30 warm
requests per endpoint per run. Absolute numbers are this machine's; the
*shape* — which endpoints scale with repository size and which don't — is
the actual finding.

| facts | `/api/log` (warm p50) | `/api/state` (warm p50) | `/api/diff` (warm p50) | `/api/recall` (warm p50) |
|---|---|---|---|---|
| 200   | 199.97 ms | 326.84 ms  | 58.01 ms  | 686.92 ms   |
| 1,000 | 206.84 ms | 1,504.89 ms | 133.61 ms | 3,292.09 ms |
| 5,000 | 200.56 ms | 7,538.07 ms | 556.85 ms | 16,760.53 ms |

## The honest read

**`/api/log` doesn't scale with fact count at all**, because it never
materializes a fact — `core.graph.walk` reads only commit objects, and
`--limit 200` bounds how many of those get read regardless of how large
any one tree is. Its ~200ms is fixed per-request overhead (Python's GIL
plus a fresh TCP connection per request in the load-test client itself, not
anything `walk` does).

**`/api/state` scales close to linearly with fact count**, and it is the
cleanest demonstration of the cost `Repository.state()` pays: materializing
a memory state means one `read_fact` call — one file read plus one zlib
inflate — *per fact in the tree*, with no batching and no cache between
requests. 5,000 facts is roughly 25x 200 facts; the p50 latency is roughly
23x. That ratio holding is exactly the signal that per-fact CAS reads, not
some fixed overhead, dominate this endpoint's cost. This is the "diff time
vs. memory size" curve PLAN.md's metrics section asks about, for the
state-materialization half of it.

**`/api/diff` scales too, but sub-linearly** relative to `/api/state` at
the same fact counts, because `diff_trees` only calls `read_fact` for keys
whose hash tuples actually changed between the two trees being compared —
an unchanged key costs zero fact reads. Since each synthetic commit in this
harness only ever adds new keys (nothing is ever reaffirmed or removed),
the "changed" set for the last commit's diff against its parent is always
much smaller than the full tree, so the endpoint pays for that one commit's
worth of facts, not the whole history's.

**`/api/recall` is the outlier, and the shared lock is why.** Its cold
call is the most expensive single request measured at every scale (26.3s
at 5,000 facts) because a full cache miss means embedding and indexing
every fact once. What makes the *warm* numbers worse than `/api/state`'s,
despite recall only ever pinning `k=8` facts per query, is
`deps.retrieval_lock`: every concurrent `/api/recall` request serializes
behind it (see PLAN.md's "Vector-cache writes under concurrency" row), so
8-way concurrency here does not mean 8 requests actually running at once —
it means up to 8 requests queued behind whichever one currently holds the
lock. That is the correct tradeoff for a single-user dashboard talking to
one process, and this load test is exactly what makes the cost of that
tradeoff visible instead of theoretical.

## What this doesn't tell you

- **Not a production capacity number.** `memgit serve-web` runs uvicorn
  with its default single worker; a real deployment would tune
  `--workers`, and every number above would change.
- **Not representative of a real fact distribution.** The synthetic
  repository is 100 commits of unique, never-reaffirmed, never-retracted
  facts — the cheapest possible shape for `/api/diff` and the shape most
  favorable to `/api/state`'s linear-not-worse scaling. A history with
  heavy reaffirmation would push more work into `diff_trees`'s
  `_representative` grouping step, untested here.
- **A cold start's cost is a one-time tax, not a steady-state one** for
  `/api/recall` specifically: `memgit embed` (warming the vector cache
  ahead of time, outside the request path) is the documented answer to the
  first number in that row — see PLAN.md's embedder decision rows.
