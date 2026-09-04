"""Slice 8: the MCP server wrapper.

See ``src/memgit/mcp/server.py`` for the tool surface and
``src/memgit/mcp/session.py`` for how one MCP session's calls map onto a
:class:`~memgit.core.staging.StagingArea`. Importing this package (or
``server.py``) requires the ``mcp`` extra (``pip install memgit[mcp]``);
``memgit.core`` and ``memgit.agent.tools`` do not, by design — see
``core/staging.py``'s module docstring.
"""

from __future__ import annotations
