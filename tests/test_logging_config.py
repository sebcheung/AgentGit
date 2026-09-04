"""Tests for the JSON log formatter and the request-id contextvar it reads.

``configure_logging`` itself just wires ``logging.config.dictConfig`` --
not worth testing beyond "it runs" -- so this focuses on what's actually
load-bearing: :class:`JsonFormatter` produces valid, parseable JSON with the
right fields, folds ``extra=`` kwargs in, and picks up
:data:`request_id_var` without it being passed explicitly.
"""

from __future__ import annotations

import json
import logging

from memgit.logging_config import JsonFormatter, configure_logging, request_id_var


def _record(**extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="memgit.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="something happened",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestJsonFormatter:
    def test_renders_valid_json_with_core_fields(self) -> None:
        line = JsonFormatter().format(_record())
        payload = json.loads(line)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "memgit.test"
        assert payload["message"] == "something happened"
        assert "time" in payload

    def test_extra_fields_are_folded_in(self) -> None:
        line = JsonFormatter().format(_record(model="claude-opus-5", duration_ms=12.5))
        payload = json.loads(line)
        assert payload["model"] == "claude-opus-5"
        assert payload["duration_ms"] == 12.5

    def test_request_id_is_read_from_the_contextvar(self) -> None:
        token = request_id_var.set("req-123")
        try:
            payload = json.loads(JsonFormatter().format(_record()))
        finally:
            request_id_var.reset(token)
        assert payload["request_id"] == "req-123"

    def test_no_request_id_means_no_field(self) -> None:
        payload = json.loads(JsonFormatter().format(_record()))
        assert "request_id" not in payload

    def test_exc_info_is_rendered_as_a_string(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="memgit.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError: boom" in payload["exc_info"]


class TestConfigureLogging:
    def test_sets_the_memgit_logger_level(self) -> None:
        configure_logging("DEBUG")
        try:
            assert logging.getLogger("memgit").level == logging.DEBUG
        finally:
            configure_logging("WARNING")

    def test_json_output_installs_the_json_formatter(self) -> None:
        configure_logging("INFO", json_output=True)
        try:
            logger = logging.getLogger("memgit")
            assert isinstance(logger.handlers[0].formatter, JsonFormatter)
        finally:
            configure_logging("WARNING")
