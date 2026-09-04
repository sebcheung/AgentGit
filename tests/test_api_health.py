"""Tests for /api/health (liveness) and /api/ready (readiness), both auth-exempt.

/api/health's own contract is already covered by test_api_app.py (moved
there from read.py's tests when health got its own router); this module
covers what's new here -- /api/ready's check registry, its 503 on a broken
dependency, and that both endpoints stay reachable with a key configured.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgit.api.app import create_app
from memgit.core.fact import Fact
from memgit.core.repository import Repository


@pytest.fixture
def repo(tmp_path) -> Repository:
    r = Repository.init(tmp_path)
    r.commit([Fact(subject="user", predicate="prefers_language", object="Python", confidence=0.9)], "seed")
    return r


@pytest.fixture
def client(repo) -> TestClient:
    return TestClient(create_app(repo, static=False))


class TestReady:
    def test_ready_on_a_healthy_repo(self, client: TestClient) -> None:
        response = client.get("/api/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["repository"]["ok"] is True
        assert body["checks"]["object_store"]["ok"] is True

    def test_degraded_when_the_repo_directory_is_gone(self, repo: Repository, client: TestClient) -> None:
        import shutil

        shutil.rmtree(repo.memgit_dir)

        response = client.get("/api/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["checks"]["repository"]["ok"] is False
        assert "does not exist" in body["checks"]["repository"]["detail"]

    def test_every_check_reports_a_duration(self, client: TestClient) -> None:
        body = client.get("/api/ready").json()
        for check in body["checks"].values():
            assert check["ms"] >= 0


class TestAuthExemption:
    def test_health_reachable_with_a_key_configured(self, repo: Repository) -> None:
        client = TestClient(create_app(repo, static=False, api_key="secret"))
        assert client.get("/api/health").status_code == 200

    def test_ready_reachable_with_a_key_configured(self, repo: Repository) -> None:
        client = TestClient(create_app(repo, static=False, api_key="secret"))
        assert client.get("/api/ready").status_code == 200
