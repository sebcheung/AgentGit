"""Tests for API-key auth: default-off, then 401 without a valid key once configured.

The most important test here is the first one -- confirming every existing
route still answers 200 with no key configured, which is what proves this
slice didn't change behavior for the 829 tests that came before it. Every
other test in this module builds its own app with a key set, since that
path is otherwise untested by anything else in the suite.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgit.api.app import create_app
from memgit.core.fact import Fact
from memgit.core.repository import Repository

_GET_ROUTES = ["/api/repo", "/api/log", "/api/state", "/api/diff", "/api/recall?query=x"]


@pytest.fixture
def repo(tmp_path) -> Repository:
    r = Repository.init(tmp_path)
    r.commit([Fact(subject="user", predicate="prefers_language", object="Python", confidence=0.9)], "seed")
    return r


class TestNoKeyConfigured:
    """The default: every route answers 200, exactly as before this slice."""

    @pytest.mark.parametrize("path", _GET_ROUTES)
    def test_every_read_route_is_unauthenticated_by_default(self, repo: Repository, path: str) -> None:
        client = TestClient(create_app(repo, static=False))
        assert client.get(path).status_code == 200

    def test_health_is_unauthenticated_by_default(self, repo: Repository) -> None:
        client = TestClient(create_app(repo, static=False))
        assert client.get("/api/health").status_code == 200


class TestKeyConfigured:
    @pytest.fixture
    def client(self, repo: Repository) -> TestClient:
        return TestClient(create_app(repo, static=False, api_key="secret"))

    @pytest.mark.parametrize("path", _GET_ROUTES)
    def test_missing_key_is_401(self, client: TestClient, path: str) -> None:
        response = client.get(path)
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"

    @pytest.mark.parametrize("path", _GET_ROUTES)
    def test_wrong_key_is_401(self, client: TestClient, path: str) -> None:
        response = client.get(path, headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_correct_key_via_x_api_key_header(self, client: TestClient) -> None:
        response = client.get("/api/repo", headers={"X-API-Key": "secret"})
        assert response.status_code == 200

    def test_correct_key_via_bearer_token(self, client: TestClient) -> None:
        response = client.get("/api/repo", headers={"Authorization": "Bearer secret"})
        assert response.status_code == 200

    def test_health_is_exempt_even_with_a_key_configured(self, client: TestClient) -> None:
        assert client.get("/api/health").status_code == 200

    def test_replay_requires_the_key_too(self, client: TestClient) -> None:
        response = client.post(
            "/api/replay",
            json={"subject": "user", "predicate": "prefers_language", "query": "what do I prefer?"},
        )
        assert response.status_code == 401
