#!/usr/bin/env bash
set -euo pipefail

# Runs as the non-root `memgit` user, in /data (see the Dockerfile) -- a
# fresh named volume mounted there is empty on first boot, so a repository
# needs to exist before any of `serve`/`serve-web`/`project` can open one.
# Idempotent: an existing `.memgit` is left untouched on every later boot.
if [ ! -d ".memgit" ]; then
    memgit init
fi

exec memgit "$@"
