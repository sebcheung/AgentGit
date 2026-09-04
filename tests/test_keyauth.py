"""Tests for the shared API-key primitive: header extraction, comparison, loopback detection.

Pure functions over plain dicts and strings -- no FastAPI, no MCP, no
network -- since both ``api/deps.py`` and ``mcp/transport.py`` build their
own header mapping and call straight into this module.
"""

from __future__ import annotations

import pytest

from memgit.keyauth import check, extract_presented, generate_key, is_loopback, resolve_key


class TestResolveKey:
    def test_explicit_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMGIT_API_KEY", "from-env")
        assert resolve_key("from-flag") == "from-flag"

    def test_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMGIT_API_KEY", "from-env")
        assert resolve_key(None) == "from-env"

    def test_none_when_neither_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MEMGIT_API_KEY", raising=False)
        assert resolve_key(None) is None

    def test_empty_env_value_is_treated_as_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMGIT_API_KEY", "")
        assert resolve_key(None) is None


class TestExtractPresented:
    def test_x_api_key_header(self) -> None:
        assert extract_presented({"X-API-Key": "secret"}) == "secret"

    def test_x_api_key_header_is_case_insensitive(self) -> None:
        assert extract_presented({"x-api-key": "secret"}) == "secret"

    def test_bearer_token(self) -> None:
        assert extract_presented({"Authorization": "Bearer secret"}) == "secret"

    def test_bearer_is_case_insensitive(self) -> None:
        assert extract_presented({"Authorization": "bearer secret"}) == "secret"

    def test_x_api_key_wins_over_authorization(self) -> None:
        headers = {"X-API-Key": "from-key-header", "Authorization": "Bearer from-auth-header"}
        assert extract_presented(headers) == "from-key-header"

    def test_no_headers_present(self) -> None:
        assert extract_presented({}) is None

    def test_authorization_without_bearer_scheme(self) -> None:
        assert extract_presented({"Authorization": "Basic dXNlcjpwYXNz"}) is None

    def test_bearer_with_no_token(self) -> None:
        assert extract_presented({"Authorization": "Bearer "}) is None


class TestCheck:
    def test_matching_key(self) -> None:
        assert check("secret", "secret") is True

    def test_mismatched_key(self) -> None:
        assert check("wrong", "secret") is False

    def test_no_key_presented(self) -> None:
        assert check(None, "secret") is False


class TestIsLoopback:
    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
    def test_loopback_hosts(self, host: str) -> None:
        assert is_loopback(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.1", "example.com", ""])
    def test_non_loopback_hosts(self, host: str) -> None:
        assert is_loopback(host) is False


class TestGenerateKey:
    def test_produces_a_nonempty_string(self) -> None:
        assert isinstance(generate_key(), str)
        assert len(generate_key()) > 20

    def test_two_calls_differ(self) -> None:
        assert generate_key() != generate_key()
