# syntax=docker/dockerfile:1

# Multi-stage: the builder resolves and installs every extra with uv's own
# cache mount, so a rebuild after a source-only change never re-downloads a
# dependency; the runtime stage copies only the finished .venv and source,
# never uv itself, keeping the image that actually runs small.
FROM python:3.12-slim AS base

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --all-extras

COPY . .
RUN chmod +x docker/entrypoint.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras

FROM base
# A non-root user, and a data directory it owns before the image ever
# changes user -- a named volume mounted over an empty-but-owned directory
# inherits that ownership on first creation, which is what lets `memgit
# init` (run as this user) write into a bind-mounted volume without a
# separate chown step at container start.
RUN useradd --create-home --uid 1000 memgit \
    && mkdir -p /data \
    && chown memgit:memgit /data

WORKDIR /app
COPY --from=builder --chown=memgit:memgit /app /app
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /data
USER memgit

# Plain stdlib urllib, not curl: the slim base has neither curl nor wget,
# and installing one just for this would cost more than it's worth on an
# image whose only other runtime dependency is the interpreter itself.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/ready', timeout=3)"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["serve-web", "--host", "0.0.0.0"]
