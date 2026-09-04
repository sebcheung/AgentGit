"""The shared API-key primitive for the REST API and the MCP server.

Top-level, not under ``api/``, because ``mcp/server.py`` must run the
identical check without the ``web`` extra installed — putting this under
``api/`` would make the MCP server import FastAPI merely to read a header.

The rule this module exists to let every caller enforce identically:
*unauthenticated is allowed only when a server cannot be reached from
outside the machine it runs on.* ``is_loopback`` is what a bind host is
checked against before a server is allowed to skip a key.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
from collections.abc import Mapping

__all__ = ["check", "extract_presented", "generate_key", "is_loopback", "resolve_key"]

_ENV_VAR = "MEMGIT_API_KEY"


def resolve_key(explicit: str | None) -> str | None:
    """The key a server should require: ``explicit``, else ``$MEMGIT_API_KEY``, else none."""
    if explicit:
        return explicit
    return os.environ.get(_ENV_VAR) or None


def extract_presented(headers: Mapping[str, str]) -> str | None:
    """The key a caller presented, via ``X-API-Key`` or ``Authorization: Bearer <key>``.

    Case-insensitive on both the header name and ``Bearer`` itself, since
    HTTP header names are case-insensitive by spec and not every caller
    (or ASGI layer) normalizes them the same way before this runs.
    """
    lowered = {key.lower(): value for key, value in headers.items()}
    api_key = lowered.get("x-api-key")
    if api_key:
        return api_key
    auth = lowered.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def check(presented: str | None, expected: str) -> bool:
    """Constant-time comparison — a timing side-channel here would leak the key one byte at a time."""
    if presented is None:
        return False
    return hmac.compare_digest(presented, expected)


def is_loopback(host: str) -> bool:
    """True if ``host`` can only be reached from the machine the server runs on."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def generate_key() -> str:
    """A fresh, high-entropy key — what a deployment with no key set should mint."""
    return secrets.token_urlsafe(32)
