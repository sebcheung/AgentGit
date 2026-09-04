"""A local load-test harness for memgit's read API. Stdlib only, no /api/replay.

Seeds a synthetic repository, launches ``memgit serve-web`` against it as a
real subprocess (a separate process, so client-side thread contention never
shows up as server-side latency), then hits ``/api/log``, ``/api/state``,
``/api/diff``, and ``/api/recall`` with a thread pool -- one cold request
(the uncached path) followed by many warm ones, reported separately, since
``/api/recall``'s first call is the one that pays to build the vector
index.

``/api/replay`` is deliberately never touched here: it spends two paid
model calls per request, so a load test that hits it is a bill, not a
benchmark.

Usage::

    py -m uv run --extra web python scripts/loadtest.py --synthesize 5000
"""

from __future__ import annotations

import argparse
import http.client
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


def _synthesize_repo(root: Path, n: int) -> None:
    """Build a repository with ``n`` facts across roughly a hundred commits."""
    from memgit.core.fact import Fact
    from memgit.core.repository import Repository

    repo = Repository.init(root)
    per_commit = max(1, n // 100)
    facts: list[Fact] = []
    written = 0
    commit_no = 0
    while written < n:
        batch_size = min(per_commit, n - written)
        facts.extend(
            Fact(
                subject=f"entity-{written + i}",
                predicate="has_property",
                object=f"value-{written + i}",
                confidence=0.9,
            )
            for i in range(batch_size)
        )
        written += batch_size
        commit_no += 1
        repo.commit(facts, f"synthetic batch {commit_no}")
    print(f"seeded {written} fact(s) across {commit_no} commit(s) at {root}")


def _wait_for_health(host: str, port: int, *, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://{host}:{port}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.2)
    raise TimeoutError(f"server never became healthy at {url}")


def _one_request(host: str, port: int, path: str, *, timeout: float) -> float:
    """One GET, on its own fresh connection -- so latency includes real connection setup, not just the handler."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        start = time.perf_counter()
        conn.request("GET", path)
        response = conn.getresponse()
        response.read()
        if response.status >= 400:
            raise RuntimeError(f"{path} -> HTTP {response.status}")
        return (time.perf_counter() - start) * 1000
    finally:
        conn.close()


@dataclass(frozen=True, slots=True)
class EndpointResult:
    """One endpoint's cold latency plus warm p50/p95/p99 and throughput."""

    path: str
    cold_ms: float
    warm_ms: list[float]
    wall_s: float

    def report(self) -> str:
        """Render one printable summary line."""
        warm = sorted(self.warm_ms)
        n = len(warm)
        p50 = statistics.median(warm) if warm else 0.0
        p95 = warm[max(0, int(n * 0.95) - 1)] if warm else 0.0
        p99 = warm[max(0, int(n * 0.99) - 1)] if warm else 0.0
        rps = n / self.wall_s if self.wall_s > 0 else 0.0
        return (
            f"{self.path:<40} cold={self.cold_ms:7.2f}ms  "
            f"warm(n={n:<4}) p50={p50:7.2f}ms p95={p95:7.2f}ms p99={p99:7.2f}ms  "
            f"{rps:8.1f} req/s"
        )


def _run_endpoint(
    host: str, port: int, path: str, *, requests: int, concurrency: int, timeout: float
) -> EndpointResult:
    cold_ms = _one_request(host, port, path, timeout=timeout)
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        warm_ms = list(pool.map(lambda _: _one_request(host, port, path, timeout=timeout), range(requests)))
    wall_s = time.perf_counter() - wall_start
    return EndpointResult(path=path, cold_ms=cold_ms, warm_ms=warm_ms, wall_s=wall_s)


def main() -> int:
    """Parse arguments, seed a repo, run the server, print one report line per endpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesize", type=int, default=1000, help="Number of facts to seed.")
    parser.add_argument("--requests", type=int, default=200, help="Warm requests per endpoint.")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent client threads.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-request socket timeout, in seconds."
    )
    args = parser.parse_args()

    with TemporaryDirectory(prefix="memgit-loadtest-") as tmp:
        root = Path(tmp)
        _synthesize_repo(root, args.synthesize)

        server = subprocess.Popen(
            [sys.executable, "-m", "memgit.cli", "serve-web", "--host", args.host, "--port", str(args.port)],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_health(args.host, args.port)

            paths = [
                "/api/log?all=true&limit=200",
                "/api/state",
                "/api/diff",
                "/api/recall?query=value-1",
            ]
            print(f"\n{args.requests} warm request(s) per endpoint, {args.concurrency}-way concurrency\n")
            for path in paths:
                result = _run_endpoint(
                    args.host,
                    args.port,
                    path,
                    requests=args.requests,
                    concurrency=args.concurrency,
                    timeout=args.timeout,
                )
                print(result.report())
        finally:
            server.terminate()
            server.wait(timeout=10)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
