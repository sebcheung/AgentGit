"""Tests for POST /api/replay, driven entirely with ScriptedClient — zero network.

``get_llm_client`` is overridden per test via ``app.dependency_overrides`` —
the ASGI equivalent of ``cli.py``'s ``_agent_client`` monkeypatch seam (see
``tests/test_cli_replay.py``, which patches the same seam at the CLI layer).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from conftest import ScriptedClient, text_message
from fastapi.testclient import TestClient

from memgit.api.app import create_app
from memgit.api.deps import get_llm_client
from memgit.core.fact import Fact
from memgit.core.repository import Repository


def make_fact(**overrides) -> Fact:
    defaults = {
        "subject": "user",
        "predicate": "prefers_language",
        "object": "Python",
        "confidence": 0.9,
        "source": "test",
        "source_text": "I like Python.",
    }
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.fixture
def repo(tmp_path) -> Repository:
    r = Repository.init(tmp_path)
    r.commit([make_fact()], "seed", author="test")
    return r


def _client_with(app_repo: Repository, llm_client) -> TestClient:
    app = create_app(app_repo, static=False)
    app.dependency_overrides[get_llm_client] = lambda: llm_client
    return TestClient(app)


REPLAY_BODY = {
    "rev": "HEAD",
    "subject": "user",
    "predicate": "prefers_language",
    "query": "what do I prefer?",
}


class TestReplay:
    def test_distinct_replies_report_changed(self, repo: Repository) -> None:
        client = ScriptedClient(
            responses=[text_message("You prefer Python."), text_message("I don't know.")]
        )
        http = _client_with(repo, client)
        response = http.post("/api/replay", json=REPLAY_BODY)

        assert response.status_code == 200
        body = response.json()
        assert body["changed"] is True
        assert body["baseline"]["reply"] == "You prefer Python."
        assert body["ablated"]["reply"] == "I don't know."
        assert body["key"] == ["user", "prefers_language"]

    def test_identical_replies_report_unchanged(self, repo: Repository) -> None:
        client = ScriptedClient(responses=[text_message("no idea"), text_message("no idea")])
        http = _client_with(repo, client)
        response = http.post("/api/replay", json=REPLAY_BODY)

        assert response.status_code == 200
        assert response.json()["changed"] is False

    def test_never_commits(self, repo: Repository) -> None:
        before_head = repo.head_commit()
        before_log_len = sum(1 for _ in repo.log())

        client = ScriptedClient(responses=[text_message("a"), text_message("b")])
        http = _client_with(repo, client)
        http.post("/api/replay", json=REPLAY_BODY)

        assert repo.head_commit() == before_head
        assert sum(1 for _ in repo.log()) == before_log_len

    def test_unknown_rev_is_404(self, repo: Repository) -> None:
        client = ScriptedClient(responses=[])
        http = _client_with(repo, client)
        response = http.post("/api/replay", json={**REPLAY_BODY, "rev": "does-not-exist"})
        assert response.status_code == 404

    def test_no_read_route_touches_the_llm_client(self, repo: Repository) -> None:
        class _ExplodingClient:
            def create_message(self, **kwargs):
                raise AssertionError("a read route must never build an LLM client")

        http = _client_with(repo, _ExplodingClient())
        for path in ("/api/health", "/api/repo", "/api/log", "/api/state", "/api/diff"):
            response = http.get(path)
            assert response.status_code in (200, 404), (path, response.text)


class TestReplayUnavailable:
    def test_missing_extra_or_key_is_503(self, repo: Repository, monkeypatch: pytest.MonkeyPatch) -> None:
        from memgit.agent.client import AgentError

        def _raise(**kwargs):
            raise AgentError("the agent runtime needs the anthropic SDK: install memgit[agent]")

        monkeypatch.setattr("memgit.agent.client.default_client", _raise)
        app = create_app(repo, static=False)
        response = TestClient(app).post("/api/replay", json=REPLAY_BODY)

        assert response.status_code == 503
        assert response.json()["error"] == "llm_unavailable"
        assert "agent" in response.json()["detail"].lower()

    def test_engine_agent_error_is_502(self, repo: Repository, monkeypatch: pytest.MonkeyPatch) -> None:
        from memgit.agent.client import AgentError

        class _FailingClient:
            def create_message(self, **kwargs):
                raise AgentError("upstream call failed")

        http = _client_with(repo, _FailingClient())
        response = http.post("/api/replay", json=REPLAY_BODY)

        assert response.status_code == 502
        assert response.json()["error"] == "agent_error"


class TestReplayBusy:
    def test_second_concurrent_replay_is_429(self, repo: Repository) -> None:
        from memgit.api import deps

        client = ScriptedClient(responses=[text_message("a"), text_message("b")])
        http = _client_with(repo, client)

        acquired = deps.replay_semaphore.acquire(blocking=False)
        assert acquired  # sanity: nothing else holds it
        try:
            response = http.post("/api/replay", json=REPLAY_BODY)
            assert response.status_code == 429
        finally:
            deps.replay_semaphore.release()
