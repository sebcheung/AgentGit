"""Tests for the per-request logging middleware: header echo, log line, contextvar lifetime.

Exercised through a real app via ``TestClient`` rather than by calling
``request_id_middleware`` directly -- what matters is the observable
contract (a response header, one log line, no leaked contextvar state
between requests), not the coroutine's internals.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgit.api.app import create_app
from memgit.core.fact import Fact
from memgit.core.repository import Repository
from memgit.logging_config import request_id_var


@pytest.fixture
def repo(tmp_path) -> Repository:
    r = Repository.init(tmp_path)
    r.commit([Fact(subject="user", predicate="prefers_language", object="Python", confidence=0.9)], "seed")
    return r


@pytest.fixture
def client(repo) -> TestClient:
    return TestClient(create_app(repo, static=False))


class TestRequestId:
    def test_response_carries_a_generated_request_id(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.headers["x-request-id"]

    def test_incoming_request_id_is_echoed_back(self, client: TestClient) -> None:
        response = client.get("/api/health", headers={"X-Request-ID": "req-abc"})
        assert response.headers["x-request-id"] == "req-abc"

    def test_two_requests_get_different_generated_ids(self, client: TestClient) -> None:
        first = client.get("/api/health").headers["x-request-id"]
        second = client.get("/api/health").headers["x-request-id"]
        assert first != second

    def test_contextvar_does_not_leak_past_the_request(self, client: TestClient) -> None:
        client.get("/api/health", headers={"X-Request-ID": "req-leak-check"})
        assert request_id_var.get() is None


class TestRequestLogging:
    def test_logs_one_http_request_line(self, client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="memgit.api"):
            client.get("/api/health")
        records = [r for r in caplog.records if r.message == "http.request"]
        assert len(records) == 1
        assert records[0].method == "GET"
        assert records[0].path == "/api/health"
        assert records[0].status == 200
        assert records[0].duration_ms >= 0
