"""Protocol-level tests for the MCP server, via the SDK's in-memory transport.

No subprocess, no port — ``mcp.Client(server, raise_exceptions=True)``
connects straight to the server object, per the SDK's own testing docs. These
tests exercise the tool surface as a real MCP host would see it: JSON-shaped
arguments in, ``CallToolResult`` out, including the ``is_error`` path for a
malformed call.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from memgit.core.fact import Fact  # noqa: E402
from memgit.core.repository import Repository  # noqa: E402
from memgit.mcp.server import build_server  # noqa: E402


def make_fact(**overrides) -> Fact:
    defaults = dict(
        subject="user",
        predicate="prefers_language",
        object="Python",
        confidence=0.9,
        asserted_at="2026-08-11T12:00:00+00:00",
    )
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.fixture
def repo(tmp_path) -> Repository:
    return Repository.init(tmp_path)


@pytest.fixture
def server(repo):
    return build_server(repo)


@pytest.fixture
async def client(server):
    async with Client(server, raise_exceptions=True) as c:
        yield c


class TestToolList:
    @pytest.mark.anyio
    async def test_lists_the_expected_tools(self, client):
        result = await client.list_tools()
        names = {t.name for t in result.tools}
        assert names == {
            "remember", "forget", "commit", "recall", "read_state", "diff", "log", "create_branch",
        }


class TestRememberAndCommit:
    @pytest.mark.anyio
    async def test_remember_then_commit_produces_a_real_commit(self, client, repo):
        result = await client.call_tool(
            "remember",
            {
                "subject": "user", "predicate": "prefers_language", "object": "Python",
                "confidence": 0.9, "source_text": "I like Python",
            },
        )
        assert result.is_error is not True
        assert "staged" in result.content[0].text

        # Nothing durable yet.
        assert repo.head_commit() is None

        result = await client.call_tool("commit", {"message": "learned a preference"})
        assert result.is_error is not True
        assert "committed" in result.content[0].text

        assert repo.head_commit() is not None
        state = repo.state("HEAD")
        assert state.get("user", "prefers_language")[0].object == "Python"

    @pytest.mark.anyio
    async def test_bad_confidence_is_a_tool_error_not_a_crash(self, client):
        result = await client.call_tool(
            "remember",
            {
                "subject": "user", "predicate": "likes", "object": "chess",
                "confidence": 5.0, "source_text": "x",
            },
        )
        assert result.is_error is True

    @pytest.mark.anyio
    async def test_commit_with_nothing_staged_is_not_an_error(self, client):
        result = await client.call_tool("commit", {"message": "no-op"})
        assert result.is_error is not True
        assert "nothing to commit" in result.content[0].text

    @pytest.mark.anyio
    async def test_forget_stages_a_retraction(self, client, repo):
        repo.commit([make_fact(predicate="likes", object="chess")], "seed")
        result = await client.call_tool(
            "forget", {"subject": "user", "predicate": "likes", "object": None, "reason": "changed mind"}
        )
        assert result.is_error is not True
        await client.call_tool("commit", {"message": "forgot"})
        assert repo.state("HEAD").get("user", "likes") == ()


class TestReadToolsAreScoped:
    @pytest.mark.anyio
    async def test_recall_on_a_past_rev_cannot_see_a_later_fact(self, client, repo):
        first = repo.commit([make_fact()], "seed")
        repo.commit([make_fact(), make_fact(predicate="timezone", object="UTC")], "later")

        result = await client.call_tool(
            "recall", {"query": "timezone", "subject": None, "limit": 10, "rev": first}
        )
        assert "timezone" not in result.content[0].text

    @pytest.mark.anyio
    async def test_read_state_defaults_to_head(self, client, repo):
        repo.commit([make_fact()], "seed")
        result = await client.call_tool("read_state", {})
        assert "Python" in result.content[0].text

    @pytest.mark.anyio
    async def test_diff_matches_repository_diff(self, client, repo):
        repo.commit([make_fact()], "seed")
        result = await client.call_tool("diff", {"before": None, "after": "HEAD"})
        direct = repo.diff(before=None, after="HEAD")
        assert len(direct.keys) == 1
        assert "prefers_language" in result.content[0].text

    @pytest.mark.anyio
    async def test_unresolvable_rev_is_a_tool_error(self, client):
        result = await client.call_tool("read_state", {"rev": "nonexistent-branch"})
        assert result.is_error is True


class TestCreateBranch:
    @pytest.mark.anyio
    async def test_create_branch_does_not_move_head(self, client, repo):
        head = repo.commit([make_fact()], "seed")
        result = await client.call_tool("create_branch", {"name": "experiment", "at": "HEAD"})
        assert result.is_error is not True
        assert repo.branches()["experiment"] == head
        assert repo.head_commit() == head
        assert repo.current_branch() == "main"
