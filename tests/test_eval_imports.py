"""The eval suite's central design claim, made mechanical.

``memgit.eval`` is deterministic and offline: every check reads
``Repository.state``/``diff``/``retriever``, never the agent runtime or the
replay engine. If a future change makes ``memgit.eval`` import
``memgit.agent`` -- even transitively, even just for a type hint -- that
property is gone and this test is what catches it.

Run in a fresh subprocess, not against this test session's own
``sys.modules``: other test modules in the same pytest run legitimately
import ``memgit.agent`` first, which would make an in-process check pass or
fail based on test order rather than on ``memgit.eval``'s own import graph.
"""

from __future__ import annotations

import subprocess
import sys


def test_eval_package_never_imports_agent():
    script = (
        "import sys\n"
        "import memgit.eval\n"
        "leaked = sorted(n for n in sys.modules if n == 'memgit.agent' or n.startswith('memgit.agent.'))\n"
        "assert not leaked, f'memgit.eval pulled in memgit.agent: {leaked}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
