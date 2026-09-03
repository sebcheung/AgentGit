"""One opt-in test against the real Anthropic API.

Excluded from the default run by ``addopts = "-q -m 'not live'"`` in
pyproject.toml. Run it explicitly with::

    py -m uv run pytest -m live

Deliberately asserts nothing about which triples the model chose to
remember — that is a model-behavior assertion, and it will flake. It only
asserts the structural claim this whole slice makes: a turn against a real
model produces a commit, and what it remembered is retrievable afterward.
"""

from __future__ import annotations

import os

import pytest

from memgit.core.repository import Repository

pytestmark = pytest.mark.live

pytest.importorskip("anthropic")

if not os.environ.get("ANTHROPIC_API_KEY"):
    pytest.skip("ANTHROPIC_API_KEY not set", allow_module_level=True)


def test_a_real_turn_commits_and_is_retrievable(tmp_path):
    from memgit.agent.runtime import MemoryAgent

    repo = Repository.init(tmp_path)
    agent = MemoryAgent(repo)

    result = agent.turn(
        "My name is Sebastian and I mostly write Python. Please remember both."
    )

    assert result.reply
    assert result.commit is not None
    assert repo.head_commit() == result.commit
    state = repo.state()
    assert len(state) > 0
