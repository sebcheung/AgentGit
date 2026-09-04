"""Tests for the FastAPI app factory: wiring, static mount, error envelope.

Route-level behavior (``/api/log``, ``/api/diff``, ...) is covered once
those routes exist, in ``test_api_read.py`` and ``test_api_replay.py``; this
module only exercises what ``create_app`` itself wires up, independent of
any one route's logic — including the exception-handler mapping, tested
here against throwaway routes registered directly on the built app rather
than against real endpoints.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgit.api.app import create_app
from memgit.api.errors import LLMUnavailableError
from memgit.core.fact import Fact
from memgit.core.repository import (
    NotARepositoryError,
    Repository,
    RevisionNotFoundError,
)
from memgit.core.store import CorruptObjectError, ObjectNotFoundError


@pytest.fixture
def repo(tmp_path) -> Repository:
    r = Repository.init(tmp_path)
    r.commit([Fact(subject="user", predicate="prefers_language", object="Python", confidence=0.9)], "seed")
    return r


@pytest.fixture
def client(repo) -> TestClient:
    return TestClient(create_app(repo))


class TestWiring:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]

    def test_health_never_touches_the_repo(self) -> None:
        # No repository was ever opened -- health must still answer, since
        # it is liveness only (see PLAN.md's "/health" decision row).
        class _NeverOpened(Repository):
            def __init__(self) -> None:  # type: ignore[no-untyped-def]
                pass

        app = create_app(_NeverOpened())
        response = TestClient(app).get("/api/health")
        assert response.status_code == 200

    def test_static_index_served_at_root(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_static_mount_does_not_shadow_api_routes(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_static_mount_does_not_shadow_docs(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200

    def test_static_can_be_disabled(self, repo: Repository) -> None:
        app = create_app(repo, static=False)
        response = TestClient(app).get("/")
        assert response.status_code == 404


class TestExceptionMapping:
    """Each mapped exception, in isolation, against a throwaway test route."""

    def _app_raising(self, repo: Repository, exc: Exception):
        app = create_app(repo, static=False)

        @app.get("/api/_test/boom")
        def _boom() -> None:
            raise exc

        return TestClient(app)

    def test_revision_not_found_is_404(self, repo: Repository) -> None:
        client = self._app_raising(repo, RevisionNotFoundError("nope"))
        response = client.get("/api/_test/boom")
        assert response.status_code == 404
        assert response.json() == {"error": "revision_not_found", "detail": "'nope'"}

    def test_not_a_repository_is_503(self, repo: Repository) -> None:
        client = self._app_raising(repo, NotARepositoryError("no .memgit here"))
        response = client.get("/api/_test/boom")
        assert response.status_code == 503
        assert response.json()["error"] == "repository_unavailable"

    def test_object_not_found_is_500(self, repo: Repository) -> None:
        client = self._app_raising(repo, ObjectNotFoundError("deadbeef"))
        response = client.get("/api/_test/boom")
        assert response.status_code == 500
        assert response.json()["error"] == "object_store_corrupt"

    def test_corrupt_object_is_500(self, repo: Repository) -> None:
        client = self._app_raising(repo, CorruptObjectError("bad zlib"))
        response = client.get("/api/_test/boom")
        assert response.status_code == 500
        assert response.json()["error"] == "object_store_corrupt"

    def test_llm_unavailable_is_503(self, repo: Repository) -> None:
        client = self._app_raising(repo, LLMUnavailableError("install the agent extra"))
        response = client.get("/api/_test/boom")
        assert response.status_code == 503
        assert response.json()["error"] == "llm_unavailable"

    def test_agent_error_is_502(self, repo: Repository) -> None:
        from memgit.agent.client import AgentError

        client = self._app_raising(repo, AgentError("the model call failed"))
        response = client.get("/api/_test/boom")
        assert response.status_code == 502
        assert response.json()["error"] == "agent_error"


class TestUnbornRepository:
    def test_health_works_before_any_commit(self, tmp_path) -> None:
        empty_repo = Repository.init(tmp_path)
        client = TestClient(create_app(empty_repo))
        assert client.get("/api/health").status_code == 200
