# MemGit

Version control and time-travel debugging for AI agent memory — snapshot,
branch, diff, and (soon) rewind what an agent "believed" over time, so you can
causally explain *why* it behaved the way it did. See [PLAN.md](PLAN.md) for
the full design and build order.

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

## Quickstart

```sh
py -m uv sync --all-extras --system-certs
py -m uv run memgit init
py -m uv run memgit commit -m "seed" --file facts.json
py -m uv run memgit log --oneline
py -m uv run memgit show HEAD
```

See [PLAN.md](PLAN.md) for setup details, the full CLI surface, and what's
built vs. planned.
