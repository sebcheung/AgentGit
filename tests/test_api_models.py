"""Guards the Pydantic-vs-``to_dict`` boundary this API draws.

``Diff`` and ``MemoryState`` pass through as the dicts core already emits
rather than being mirrored field-by-field in Pydantic (see ``models.py``'s
module docstring). The trade for not mirroring them is that nothing type
checks their shape at declaration time, so this module is the mechanical
replacement: it asserts the passthrough's key set matches what
``Diff.to_dict()`` / ``MemoryState.to_dict()`` actually produce, against a
real repository, so a drift between the two surfaces here instead of as a
500 on a working diff.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from memgit.api.app import create_app
from memgit.core.fact import Fact
from memgit.core.repository import Repository


def make_fact(**overrides) -> Fact:
    defaults = {"subject": "user", "predicate": "prefers_language", "object": "Python", "confidence": 0.9}
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.fixture
def repo(tmp_path) -> Repository:
    r = Repository.init(tmp_path)
    r.commit([make_fact()], "first")
    r.commit([make_fact(object="Rust")], "second")
    return r


@pytest.fixture
def client(repo: Repository) -> TestClient:
    return TestClient(create_app(repo, static=False))


def test_diff_response_key_set_matches_diff_to_dict(repo: Repository, client: TestClient) -> None:
    expected = repo.diff().to_dict()
    response = client.get("/api/diff")
    assert response.json()["diff"].keys() == expected.keys()
    assert response.json()["diff"]["cardinality"] == expected["cardinality"]


def test_state_response_key_set_matches_memory_state_to_dict(
    repo: Repository, client: TestClient
) -> None:
    expected = repo.state().to_dict()
    response = client.get("/api/state")
    assert response.json()["state"].keys() == expected.keys()


def test_every_declared_response_model_validates_real_output(
    repo: Repository, client: TestClient
) -> None:
    from memgit.api.models import (
        CommitDetail,
        CommitNode,
        DiffResponse,
        HealthResponse,
        LogResponse,
        RecallResponse,
        RepoResponse,
        StateResponse,
    )

    head = repo.head_commit()

    HealthResponse.model_validate(client.get("/api/health").json())
    RepoResponse.model_validate(client.get("/api/repo").json())
    log_body = client.get("/api/log").json()
    LogResponse.model_validate(log_body)
    for node in log_body["commits"]:
        CommitNode.model_validate(node)
    CommitDetail.model_validate(client.get(f"/api/commits/{head}").json())
    StateResponse.model_validate(client.get("/api/state").json())
    DiffResponse.model_validate(client.get("/api/diff").json())
    RecallResponse.model_validate(client.get("/api/recall", params={"query": "language"}).json())
