"""Tests for the read-only routes: /repo, /log, /commits, /state, /diff, /recall."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from memgit.api.app import create_app  # noqa: E402
from memgit.core.commit import Commit  # noqa: E402
from memgit.core.fact import Fact  # noqa: E402
from memgit.core.repository import Repository  # noqa: E402


def make_fact(**overrides) -> Fact:
    defaults = dict(subject="user", predicate="prefers_language", object="Python", confidence=0.9)
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository.init(tmp_path)


@pytest.fixture
def client(repo: Repository) -> TestClient:
    return TestClient(create_app(repo, static=False))


class TestRepo:
    def test_unborn_repo(self, client: TestClient) -> None:
        response = client.get("/api/repo")
        assert response.status_code == 200
        body = response.json()
        assert body["unborn"] is True
        assert body["head"]["commit"] is None
        assert body["branches"] == {}

    def test_after_a_commit(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/repo")
        body = response.json()
        assert body["unborn"] is False
        assert body["head"]["branch"] == "main"
        assert body["head"]["detached"] is False
        assert "main" in body["branches"]


class TestLog:
    def test_unborn_repo_returns_empty_list(self, client: TestClient) -> None:
        response = client.get("/api/log")
        assert response.status_code == 200
        body = response.json()
        assert body["resolved"] is None
        assert body["commits"] == []

    def test_every_node_carries_its_own_hash(self, repo: Repository, client: TestClient) -> None:
        h1 = repo.commit([make_fact()], "first")
        h2 = repo.commit([make_fact(object="Rust")], "second")

        response = client.get("/api/log")
        assert response.status_code == 200
        commits = response.json()["commits"]
        hashes = {c["hash"] for c in commits}
        assert hashes == {h1, h2}

        # Every node's hash must equal a recomputed Commit.hash, and every
        # parent reference must resolve within the returned set -- the
        # commit-payload-over-the-wire property this route exists to
        # guarantee (Commit.to_dict() itself omits the hash).
        by_hash = {c["hash"]: c for c in commits}
        for node in commits:
            recomputed = Commit(
                tree=node["tree"],
                parents=tuple(node["parents"]),
                message=node["message"],
                author=node["author"],
                committed_at=node["committed_at"],
                metadata=node["metadata"],
            ).hash
            assert recomputed == node["hash"]
            for parent in node["parents"]:
                assert parent in by_hash

    def test_limit(self, repo: Repository, client: TestClient) -> None:
        for i in range(5):
            repo.commit([make_fact(object=str(i))], f"commit {i}")
        response = client.get("/api/log", params={"limit": 2})
        assert len(response.json()["commits"]) == 2

    def test_all_branches_deduplicates(self, repo: Repository, client: TestClient) -> None:
        base = repo.commit([make_fact()], "base")
        repo.create_branch("feature", at=base)
        tip = repo.commit([make_fact(object="Rust")], "on main")

        response = client.get("/api/log", params={"all": "true"})
        commits = response.json()["commits"]
        hashes = {c["hash"] for c in commits}
        assert hashes == {base, tip}
        assert response.json()["all"] is True

    def test_unknown_rev_is_404(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/log", params={"rev": "does-not-exist"})
        assert response.status_code == 404
        assert response.json()["error"] == "revision_not_found"


class TestCommitDetail:
    def test_head_and_ancestor(self, repo: Repository, client: TestClient) -> None:
        h1 = repo.commit([make_fact()], "first")
        h2 = repo.commit([make_fact(object="Rust")], "second")

        response = client.get(f"/api/commits/{h2}")
        assert response.status_code == 200
        assert response.json()["hash"] == h2

        response = client.get("/api/commits/HEAD~1")
        assert response.status_code == 200
        assert response.json()["hash"] == h1

    def test_abbreviated_hash(self, repo: Repository, client: TestClient) -> None:
        h1 = repo.commit([make_fact()], "first")
        response = client.get(f"/api/commits/{h1[:10]}")
        assert response.status_code == 200
        assert response.json()["hash"] == h1

    def test_stat_present(self, repo: Repository, client: TestClient) -> None:
        h1 = repo.commit([make_fact()], "first")
        response = client.get(f"/api/commits/{h1}")
        stat = response.json()["stat"]
        assert stat["facts_added"] == 1


class TestState:
    def test_basic(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/state")
        assert response.status_code == 200
        body = response.json()
        assert len(body["state"]["facts"]) == 1
        assert body["resolved"]

    def test_filters(self, repo: Repository, client: TestClient) -> None:
        repo.commit(
            [make_fact(), make_fact(subject="user", predicate="timezone", object="UTC")], "seed"
        )
        response = client.get("/api/state", params={"predicate": "timezone"})
        facts = response.json()["state"]["facts"]
        assert len(facts) == 1
        assert facts[0]["predicate"] == "timezone"

    def test_as_of_adds_decayed_confidence(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/state", params={"as_of": "2030-01-01T00:00:00+00:00"})
        assert response.status_code == 200
        facts = response.json()["state"]["facts"]
        assert "decayed_confidence" in facts[0]

    def test_bad_as_of_is_400(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/state", params={"as_of": "not-a-date"})
        assert response.status_code == 400


class TestDiff:
    def test_implicit_parent(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "first")
        repo.commit([make_fact(object="Rust")], "second")

        response = client.get("/api/diff")
        assert response.status_code == 200
        body = response.json()
        assert body["diff"]["type"] == "diff"
        assert "cardinality" in body["diff"]

    def test_explicit_pair(self, repo: Repository, client: TestClient) -> None:
        h1 = repo.commit([make_fact()], "first")
        h2 = repo.commit([make_fact(object="Rust")], "second")
        response = client.get("/api/diff", params={"before": h1, "after": h2})
        assert response.status_code == 200
        assert response.json()["resolved_before"] == h1
        assert response.json()["resolved_after"] == h2

    def test_use_merge_base_on_diverged_branches(self, repo: Repository, client: TestClient) -> None:
        base = repo.commit([make_fact()], "base")
        repo.create_branch("feature", at=base)
        main_tip = repo.commit([make_fact(object="Rust")], "on main")
        repo.checkout("feature")
        feature_tip = repo.commit([make_fact(subject="user", predicate="timezone", object="UTC")], "on feature")

        response = client.get(
            "/api/diff",
            params={"before": main_tip, "after": feature_tip, "use_merge_base": "true"},
        )
        assert response.status_code == 200

    def test_use_merge_base_without_explicit_before_is_400(
        self, repo: Repository, client: TestClient
    ) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/diff", params={"use_merge_base": "true"})
        assert response.status_code == 400

    def test_include_unchanged(self, repo: Repository, client: TestClient) -> None:
        # Repository.commit takes the whole fact set, not a delta -- so the
        # second commit must re-list the first fact for it to carry over
        # unchanged rather than reading as removed.
        base_fact = make_fact()
        h1 = repo.commit([base_fact], "first")
        h2 = repo.commit(
            [base_fact, make_fact(subject="user", predicate="timezone", object="UTC")], "second"
        )
        response = client.get(
            "/api/diff", params={"before": h1, "after": h2, "include_unchanged": "true"}
        )
        kinds = {kd["kind"] for kd in response.json()["diff"]["keys"]}
        assert "unchanged" in kinds


class TestRecall:
    def test_shape(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/recall", params={"query": "language"})
        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "language"
        assert isinstance(body["candidates"], int)
        if body["facts"]:
            fact = body["facts"][0]
            assert {"subject", "predicate", "object", "confidence", "score", "similarity"} <= fact.keys()

    def test_no_decay_ignores_confidence(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/recall", params={"query": "language", "decay": "false"})
        assert response.status_code == 200

    def test_unknown_rev_is_404(self, repo: Repository, client: TestClient) -> None:
        repo.commit([make_fact()], "seed")
        response = client.get("/api/recall", params={"query": "x", "rev": "nope"})
        assert response.status_code == 404
