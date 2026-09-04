"""Structured logging — configured once, by an entry point, never by a library import.

``memgit.cli``'s ``main()`` callback and the two ``serve``/``serve-web``
commands are the only callers of :func:`configure_logging`. Every other
module just does ``logging.getLogger(__name__)`` and logs — a library that
touches the root logger on import would hijack whatever logging setup its
host application already has, which is exactly the failure mode this split
avoids.

:data:`request_id_var` is the seam between this module and
``api/middleware.py``: a request's id is stashed here once, and
:class:`JsonFormatter` reads it back for every log line emitted while that
request is in flight, without threading a ``request_id`` parameter through
every intervening call.
"""

from __future__ import annotations

import json
import logging
import logging.config
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

__all__ = ["JsonFormatter", "configure_logging", "request_id_var"]

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# What a bare `logging.LogRecord` already carries — anything else on a
# record is a caller's `extra={...}`, and that's the part worth rendering.
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: timestamp, level, logger, message, request id, and any ``extra``."""

    def format(self, record: logging.LogRecord) -> str:
        """See :meth:`logging.Formatter.format`."""
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Configure the ``memgit`` logger tree. Call once, from an entry point only.

    Args:
        level: A standard level name (``"DEBUG"``, ``"INFO"``, ...).
        json_output: Render :class:`JsonFormatter` lines instead of a plain
            human-readable line — the shape a log aggregator or ``jq``
            expects, versus the shape a terminal expects.
    """
    formatter: dict[str, Any] = (
        {"()": JsonFormatter} if json_output else {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}
    )
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"memgit": formatter},
            "handlers": {
                "memgit": {
                    "class": "logging.StreamHandler",
                    "formatter": "memgit",
                    "stream": "ext://sys.stderr",
                }
            },
            "loggers": {
                "memgit": {
                    "handlers": ["memgit"],
                    "level": level.upper(),
                    "propagate": False,
                }
            },
        }
    )
