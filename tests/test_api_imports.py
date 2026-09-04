"""The API layer's central design claim, made mechanical.

``memgit.api`` is meant to run on a ``web``-only install: reads never need
the ``agent`` extra, and ``POST /replay`` only reaches for it lazily, inside
the request path, exactly as ``cli.py``'s own commands do. If a future
change makes ``memgit.api.app`` import ``memgit.agent`` or ``memgit.replay``
at module level -- even transitively, even just for a type hint -- that
property is gone and this test is what catches it.

Run in a fresh subprocess, not against this test session's own
``sys.modules``: other test modules in the same pytest run legitimately
import ``memgit.agent`` first (and ``test_api_app.py`` imports
``memgit.agent.client.AgentError`` directly), which would make an in-process
check pass or fail based on test order rather than on ``memgit.api``'s own
import graph. Mirrors ``tests/test_eval_imports.py``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("fastapi")


def test_api_app_never_imports_agent_or_replay_at_module_level():
    script = (
        "import sys\n"
        "import memgit.api.app\n"
        "leaked = sorted(\n"
        "    n for n in sys.modules\n"
        "    if n in ('memgit.agent', 'memgit.replay', 'anthropic')\n"
        "    or n.startswith('memgit.agent.')\n"
        "    or n.startswith('memgit.replay.')\n"
        "    or n.startswith('anthropic.')\n"
        ")\n"
        "assert not leaked, f'memgit.api.app pulled in: {leaked}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
