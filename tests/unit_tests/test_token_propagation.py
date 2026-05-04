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


def test_current_taiga_jwt_raises_on_missing_taiga_jwt_claim(monkeypatch):
    """Verified token without taiga_jwt claim → fail-closed (no ENV fallback)."""
    from langchain_taiga.tools.taiga_tools import _current_taiga_jwt
    with patched_access_token(monkeypatch, claims={"user_id": 7}):
        with pytest.raises(PermissionError, match="missing taiga_jwt claim"):
            _current_taiga_jwt()


def test_current_user_scope_raises_on_missing_user_id_claim(monkeypatch):
    """Verified token without user_id claim → fail-closed (no default-scope leak)."""
    from langchain_taiga.tools.taiga_tools import _current_user_scope
    with patched_access_token(monkeypatch, claims={"taiga_jwt": "x"}):
        with pytest.raises(PermissionError, match="missing user_id claim"):
            _current_user_scope()


def test_user_scoped_key_propagates_scope_error(monkeypatch):
    """_user_scoped_key must propagate PermissionError so cachetools doesn't cache."""
    from langchain_taiga.tools.taiga_tools import _user_scoped_key
    with patched_access_token(monkeypatch, claims={"taiga_jwt": "x"}):
        with pytest.raises(PermissionError, match="missing user_id claim"):
            _user_scoped_key("slug")


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


def test_cache_isolation_get_project(monkeypatch):
    """Two different users looking up the same slug → two cache entries."""
    from langchain_taiga.tools import taiga_tools
    taiga_tools.project_cache.clear()

    api_clients_by_user = {}

    def fake_get_taiga_api(token=None):
        if token not in api_clients_by_user:
            api = MagicMock()
            proj = MagicMock(); proj.id = 999; proj._token_used = token
            api.projects.get_by_slug.return_value = proj
            api_clients_by_user[token] = api
        return api_clients_by_user[token]

    monkeypatch.setattr(taiga_tools, "get_taiga_api", fake_get_taiga_api)

    with patched_access_token(
        monkeypatch, claims={"user_id": 1, "taiga_jwt": "alice_jwt"}
    ):
        p_a = taiga_tools.get_project("shikenso-development")
    with patched_access_token(
        monkeypatch, claims={"user_id": 2, "taiga_jwt": "bob_jwt"}
    ):
        p_b = taiga_tools.get_project("shikenso-development")

    assert p_a is not p_b
    assert p_a._token_used == "alice_jwt"
    assert p_b._token_used == "bob_jwt"
    assert len(taiga_tools.project_cache) == 2


def test_default_scope_caches_when_no_request(monkeypatch):
    """Stdio path keeps single shared cache (today's behaviour preserved)."""
    from langchain_taiga.tools import taiga_tools
    taiga_tools.project_cache.clear()

    calls = {"n": 0}

    def fake_get_taiga_api(token=None):
        calls["n"] += 1
        api = MagicMock(); api.projects.get_by_slug.return_value = MagicMock(id=1)
        return api

    monkeypatch.setattr(taiga_tools, "get_taiga_api", fake_get_taiga_api)
    with patched_access_token(monkeypatch, claims=None):
        taiga_tools.get_project("shikenso-development")
        taiga_tools.get_project("shikenso-development")
    assert calls["n"] == 1, "stdio path should cache hit on second call"
