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

import asyncio

import pytest
from starlette.testclient import TestClient


def _build_app():
    """Build the ASGI app the same way ``_async_main`` does, minus uvicorn.

    Uses real ``InMemoryStore`` + ``TaigaOAuthProvider`` because FastMCP's
    ``http_app()`` reads ``provider.resource_server_url`` to build the
    auto-mounted RFC 9728 metadata; a ``MagicMock(spec=OAuthProvider)``
    fails pydantic validation at app-build time.
    """
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.store import InMemoryStore
    from langchain_taiga.auth.taiga_client import TaigaClient
    from langchain_taiga.mcp import make_mcp
    from langchain_taiga.remote_server import _attach_custom_routes

    async def _build():
        store = InMemoryStore()
        provider = TaigaOAuthProvider(
            store=store,
            taiga_client=TaigaClient(api_url="https://taiga.example.test"),
            issuer_url="https://taiga.shikenso.org/mcp",
        )
        mcp_instance = make_mcp(auth=provider, streamable_http_path="/mcp")
        _attach_custom_routes(
            mcp_instance,
            provider,
            taiga_url="https://taiga.example.test",
            base_url="https://taiga.shikenso.org/mcp",
        )
        return mcp_instance.http_app()

    return asyncio.run(_build())


@pytest.fixture(scope="module")
def client():
    app = _build_app()
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
    assert body["resource"] == "https://taiga.shikenso.org/mcp"
    # MUST be the origin, not .../mcp — see provider.py issuer_url override.
    assert body["authorization_servers"] == ["https://taiga.shikenso.org"]
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
    assert body["resource"] == "https://taiga.shikenso.org/mcp"
    assert body["authorization_servers"] == ["https://taiga.shikenso.org/"]


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


# ---------------------------------------------------------------------------
# DCR for public clients: token_endpoint_auth_methods_supported MUST advertise
# ``"none"`` so RFC 7591 public clients (VSCode, Claude Desktop, …) accept
# the authorization server. The MCP SDK hard-codes the list to
# ``["client_secret_post"]`` (sometimes plus ``"client_secret_basic"``);
# ``_patch_metadata_to_advertise_none_auth`` rescues that.
# ---------------------------------------------------------------------------


def test_patch_metadata_adds_none_auth_method():
    """The patched ``build_metadata`` must include ``"none"`` so public
    clients can register via DCR. Regression for VSCode showing
    "Dynamic Client Registration not supported"."""
    from mcp.server.auth import routes as _routes
    from mcp.server.auth.settings import (
        ClientRegistrationOptions,
        RevocationOptions,
    )
    from pydantic import AnyHttpUrl

    from langchain_taiga.remote_server import (
        _patch_metadata_to_advertise_none_auth,
    )

    # Save the unpatched function so we can restore it after the test.
    saved = _routes.build_metadata
    try:
        # If a previous test (or import order) already wrapped the function,
        # walk back to the underlying original for a clean exercise.
        if getattr(_routes.build_metadata, "_taiga_patched", False):
            _routes.build_metadata = (
                _routes.build_metadata.__closure__[0].cell_contents  # type: ignore
            )

        _patch_metadata_to_advertise_none_auth()

        md = _routes.build_metadata(
            issuer_url=AnyHttpUrl("https://taiga.shikenso.org/mcp"),
            service_documentation_url=None,
            client_registration_options=ClientRegistrationOptions(
                enabled=True, valid_scopes=["taiga"], default_scopes=["taiga"]
            ),
            revocation_options=RevocationOptions(enabled=False),
        )
        assert "none" in (md.token_endpoint_auth_methods_supported or [])
    finally:
        _routes.build_metadata = saved


def test_patch_metadata_is_idempotent():
    """Re-importing remote_server (e.g. in tests) must not double-wrap
    the metadata builder — otherwise repeated app-builds within a process
    would inflate the auth-methods list with duplicate ``"none"`` entries."""
    from mcp.server.auth import routes as _routes

    from langchain_taiga.remote_server import (
        _patch_metadata_to_advertise_none_auth,
    )

    _patch_metadata_to_advertise_none_auth()
    first_ref = _routes.build_metadata
    _patch_metadata_to_advertise_none_auth()
    second_ref = _routes.build_metadata
    assert first_ref is second_ref, "patch should be a no-op on the second call"
