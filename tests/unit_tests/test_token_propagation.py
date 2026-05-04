"""Tests for per-request Taiga-JWT propagation via FastMCP AccessToken.claims.

Verifies two contracts:
1. Stdio mode (no HTTP request context) keeps working unchanged with ENV auth.
2. HTTP mode reads the verified Taiga JWT from the FastMCP AccessToken claims
   and never leaks cached data between users.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("TAIGA_URL", "https://taiga.example.test")
    monkeypatch.setenv("TAIGA_API_URL", "https://taiga.example.test")
    monkeypatch.setenv("TAIGA_USERNAME", "test_user")
    monkeypatch.setenv("TAIGA_PASSWORD", "test_pass")
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@contextmanager
def patched_access_token(monkeypatch, *, claims=None):
    """Stub fastmcp.server.dependencies.get_access_token.

    claims=None  → simulate "no active auth context" (stdio / unauthenticated)
    claims=dict  → simulate a verified incoming request with these claims
    """
    from langchain_taiga.tools import taiga_tools

    if claims is None:
        def _fake():
            raise LookupError("No access token in context")
    else:
        fake_token = MagicMock()
        fake_token.claims = claims
        def _fake():
            return fake_token

    monkeypatch.setattr(taiga_tools, "get_access_token", _fake)
    yield


def test_get_taiga_api_no_arg_uses_env_cache():
    from langchain_taiga.tools.taiga_tools import get_taiga_api, taiga_api_cache
    taiga_api_cache.clear()
    from unittest.mock import patch
    with patch("langchain_taiga.tools.taiga_tools.TaigaAPI") as MockAPI:
        instance = MagicMock()
        MockAPI.return_value = instance
        first = get_taiga_api()
        second = get_taiga_api()
        assert first is second
        MockAPI.assert_called_once()


def test_get_taiga_api_with_token_returns_fresh_client():
    from langchain_taiga.tools.taiga_tools import get_taiga_api
    from unittest.mock import patch
    with patch("langchain_taiga.tools.taiga_tools.TaigaAPI") as MockAPI:
        MockAPI.side_effect = [MagicMock(), MagicMock()]
        a = get_taiga_api(token="user_a")
        b = get_taiga_api(token="user_b")
        assert a is not b
        assert MockAPI.call_count == 2
        assert MockAPI.call_args_list[0].kwargs.get("token") == "user_a"


def test_get_taiga_api_per_request_not_cached():
    from langchain_taiga.tools.taiga_tools import get_taiga_api
    from unittest.mock import patch
    with patch("langchain_taiga.tools.taiga_tools.TaigaAPI") as MockAPI:
        MockAPI.side_effect = [MagicMock(), MagicMock()]
        a = get_taiga_api(token="same")
        b = get_taiga_api(token="same")
        assert a is not b
        assert MockAPI.call_count == 2


def test_current_taiga_jwt_none_outside_request(monkeypatch):
    from langchain_taiga.tools.taiga_tools import _current_taiga_jwt
    with patched_access_token(monkeypatch, claims=None):
        assert _current_taiga_jwt() is None


def test_current_taiga_jwt_reads_from_claims(monkeypatch):
    from langchain_taiga.tools.taiga_tools import _current_taiga_jwt
    with patched_access_token(
        monkeypatch, claims={"taiga_jwt": "user_jwt_xyz", "user_id": 7}
    ):
        assert _current_taiga_jwt() == "user_jwt_xyz"


def test_current_taiga_jwt_handles_missing_claim(monkeypatch):
    """If somehow the claim is absent on a verified token, return None — never crash."""
    from langchain_taiga.tools.taiga_tools import _current_taiga_jwt
    with patched_access_token(monkeypatch, claims={"user_id": 7}):
        assert _current_taiga_jwt() is None


def test_user_scoped_key_default_scope_outside_request(monkeypatch):
    from langchain_taiga.tools.taiga_tools import _user_scoped_key
    with patched_access_token(monkeypatch, claims=None):
        key = _user_scoped_key("slug")
    assert key[0] == "default"


def test_user_scoped_key_distinct_per_user(monkeypatch):
    from langchain_taiga.tools.taiga_tools import _user_scoped_key
    with patched_access_token(monkeypatch, claims={"user_id": 1}):
        a = _user_scoped_key("slug")
    with patched_access_token(monkeypatch, claims={"user_id": 2}):
        b = _user_scoped_key("slug")
    with patched_access_token(monkeypatch, claims=None):
        d = _user_scoped_key("slug")
    assert a != b != d != a


def test_user_scoped_key_64_bit_scope(monkeypatch):
    """16 hex chars (64 bits) keeps collision risk negligible."""
    from langchain_taiga.tools.taiga_tools import _user_scoped_key
    with patched_access_token(monkeypatch, claims={"user_id": 42}):
        key = _user_scoped_key("slug")
    assert len(key[0]) == 16
