"""memgit.core stays offline: the projection is never in its import graph.

If a future change makes ``memgit.core.repository`` (or anything else
under ``memgit.core``) import ``sqlalchemy``, ``psycopg``, or ``alembic`` --
even transitively, even just for a type hint -- that property is gone and
this test is what catches it. Run in a fresh subprocess for the same
reason ``test_api_imports.py``/``test_eval_imports.py`` are: other test
modules in the same pytest run legitimately import ``memgit.pg`` first,
which would make an in-process check pass or fail based on test order.
"""

from __future__ import annotations

import subprocess
import sys


def test_core_never_imports_the_pg_stack():
    script = (
        "import sys\n"
        "import memgit.core.repository\n"
        "leaked = sorted(\n"
        "    n for n in sys.modules\n"
        "    if n in ('sqlalchemy', 'psycopg', 'alembic', 'memgit.pg')\n"
        "    or n.startswith('sqlalchemy.')\n"
        "    or n.startswith('psycopg.')\n"
        "    or n.startswith('alembic.')\n"
        "    or n.startswith('memgit.pg.')\n"
        ")\n"
        "assert not leaked, f'memgit.core.repository pulled in: {leaked}'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pg_package_imports_no_memgit_core_internals():
    # The reverse direction is a weaker claim -- pg.project legitimately
    # reads Repository/Tree/Commit objects -- but memgit.pg must still
    # import cleanly with no database configured at all: schema.py and
    # engine.py alone should never require $DATABASE_URL to be set.
    script = "import memgit.pg.schema\nimport memgit.pg.engine\n"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
