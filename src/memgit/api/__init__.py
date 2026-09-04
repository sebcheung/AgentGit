"""Slice 10 — the FastAPI REST layer and dashboard.

Read-only, plus one write-shaped exception (``POST /replay``, which
structurally cannot write — see ``replay.py``). Auth is deferred to slice
11; see PLAN.md's "REST surface" decision row for why nothing writable
ships here.

Only :func:`create_app` is exported: everything else is an implementation
detail a caller reaches through the app it builds, not through this
package's namespace.
"""

from __future__ import annotations

from memgit.api.app import create_app

__all__ = ["create_app"]
