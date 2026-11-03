"""Functional smoke tests for the remote-server ASGI app.

Builds the full ASGI app via ``mcp.http_app()`` (no uvicorn) and drives it
with ``starlette.testclient.TestClient``. Catches route-wiring regressions
that would otherwise only surface in production:

- ``/health`` and ``/mcp/health`` — liveness probes K8s relies on.
- ``/oauth/login`` GET/POST input validation paths.
- Root ``/.well-known/oauth-protected-resource`` mirror with the
  ``scopes_supported`` field that matches the auto-mounted path-aware
  variant.

Also a direct test of ``_require_env`` so a missing-env regression
surfaces here, not as a 500 in production.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from tests.unit_tests.remote_app import build_remote_app


@pytest.fixture(scope="module")
def client():
    app, _ = build_remote_app()
    with TestClient(app) as c:
        yield c


def test_health_root(client):
    """K8s liveness probe at ``/health`` returns 200/ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.text == "ok"


def test_health_under_mcp(client):
    """K8s liveness probe at ``/mcp/health`` returns 200/ok.

    Belt-and-suspenders for ingresses that preserve the ``/mcp`` prefix.
    """
    response = client.get("/mcp/health")
    assert response.status_code == 200
    assert response.text == "ok"


def test_oauth_login_get_missing_state_is_400(client):
    """``GET /oauth/login`` without ``internal_state`` is a client error."""
    response = client.get("/mcp/oauth/login")
    assert response.status_code == 400
    assert "internal_state" in response.text


def test_oauth_login_get_with_state_renders_html(client):
    """``GET /oauth/login?internal_state=...`` returns the login form HTML."""
    response = client.get("/mcp/oauth/login?internal_state=fake")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_oauth_login_post_missing_fields_is_400(client):
    """``POST /oauth/login`` without state/username/password is a client error."""
    response = client.post("/mcp/oauth/login", data={})
    assert response.status_code == 400
    assert "Missing field" in response.text


def test_root_oauth_protected_resource_metadata_has_scopes(client):
    """Root ``/.well-known/oauth-protected-resource`` mirror exposes the
    ``scopes_supported`` field, matching the path-aware auto-mounted variant.

    ``authorization_servers`` is the server origin (root), NOT the
    path-aware MCP URL — strict RFC 8414 clients (VSCode 1.107+) validate
    that the AS metadata's ``issuer`` matches the auth-server URL from
    the PRM, and the AS metadata's issuer is built from ``base_url``
    (root). Mismatch → "DCR not supported" dialog.
    """
    response = client.get("/.well-known/oauth-protected-resource")
    assert response.status_code == 200
    body = response.json()
    assert body["resource"] == "https://taiga.example.com/mcp"
    # MUST be the origin, not .../mcp — see provider.py issuer_url override.
    assert body["authorization_servers"] == ["https://taiga.example.com/"]
    assert body["bearer_methods_supported"] == ["header"]
    assert body["scopes_supported"] == ["taiga"]


def test_path_aware_oauth_protected_resource_advertises_root_auth_server(client):
    """Path-aware PRM auto-mounted by FastMCP: same issuer-match rule.
    The ``authorization_servers`` value is taken from
    ``provider.issuer_url``, which we override to ``base_url`` (root)
    in TaigaOAuthProvider.__init__ exactly so this validation passes."""
    response = client.get("/.well-known/oauth-protected-resource/mcp")
    assert response.status_code == 200
    body = response.json()
    assert body["resource"] == "https://taiga.example.com/mcp"
    assert body["authorization_servers"] == ["https://taiga.example.com/"]


def test_oauth_authorization_server_issuer_matches_prm_auth_server(client):
    """End-to-end consistency: the AS-metadata ``issuer`` must equal the
    URL the PRM advertises as auth_server. This is the ENTIRE reason
    VSCode rejected DCR. Pin both sides to the same root URL."""
    prm = client.get("/.well-known/oauth-protected-resource/mcp").json()
    auth_server_url = prm["authorization_servers"][0].rstrip("/")
    as_meta = client.get("/.well-known/oauth-authorization-server").json()
    issuer = as_meta["issuer"].rstrip("/")
    assert auth_server_url == issuer, (
        f"PRM auth_server {auth_server_url!r} != AS issuer {issuer!r} — "
        "strict OAuth clients (VSCode) will reject the discovery as invalid"
    )


def test_require_env_raises_clear_error_on_missing(monkeypatch):
    """Direct test of ``_require_env``: missing var → RuntimeError naming the var.

    Testing ``_async_main``'s wrapping requires an ``asyncio.run`` dance;
    the helper is the load-bearing piece, so testing it directly is enough.
    """
    from langchain_taiga.remote_server import _require_env

    monkeypatch.delenv("TAIGA_API_URL", raising=False)
    with pytest.raises(RuntimeError, match="TAIGA_API_URL"):
        _require_env("TAIGA_API_URL")


def test_require_env_returns_value_when_set(monkeypatch):
    """Sanity: when the env var is set, ``_require_env`` returns it verbatim."""
    from langchain_taiga.remote_server import _require_env

    monkeypatch.setenv("TAIGA_API_URL", "https://taiga.example.test")
    assert _require_env("TAIGA_API_URL") == "https://taiga.example.test"
