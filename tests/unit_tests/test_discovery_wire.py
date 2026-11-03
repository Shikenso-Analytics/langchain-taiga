"""Discovery over the wire: what the real app serves, not what a builder returns."""

from __future__ import annotations

from fastmcp.server.auth import OAuthProvider
from starlette.applications import Starlette
from starlette.testclient import TestClient

from tests.unit_tests.remote_app import build_remote_app

AS_META = "/.well-known/oauth-authorization-server"


def test_served_as_metadata_advertises_none():
    app, _ = build_remote_app()
    with TestClient(app) as client:
        resp = client.get(AS_META)
    assert resp.status_code == 200
    methods = resp.json()["token_endpoint_auth_methods_supported"]
    assert "none" in methods
    assert "client_secret_post" in methods
    assert methods.count("none") == 1


def test_served_as_metadata_matches_fastmcps_own_route():
    """Our override replaces fastmcp's AS-metadata route. Compare it with the
    document plain fastmcp serves for the same provider, so a field a future
    fastmcp adds to its route cannot be silently dropped by the override."""
    app, provider = build_remote_app()
    stock = Starlette(routes=OAuthProvider.get_routes(provider, "/mcp"))
    with TestClient(stock) as stock_client:
        expected = stock_client.get(AS_META).json()
    with TestClient(app) as client:
        served = client.get(AS_META).json()

    assert "none" not in expected["token_endpoint_auth_methods_supported"]
    expected["token_endpoint_auth_methods_supported"].append("none")
    assert served == expected


def test_served_as_metadata_issuer_matches_prm():
    app, _ = build_remote_app()
    with TestClient(app) as client:
        issuer = client.get(AS_META).json()["issuer"]

        # Path-aware PRM: exact equality, no normalisation.
        resp = client.get("/.well-known/oauth-protected-resource/mcp")
        assert resp.status_code == 200
        assert resp.json()["authorization_servers"][0] == issuer

        # Root mirror: its missing trailing slash is unchanged from 2.23.0.
        resp = client.get("/.well-known/oauth-protected-resource")
        assert resp.status_code == 200
        assert resp.json()["authorization_servers"][0].rstrip("/") == issuer.rstrip("/")


def test_public_client_dcr_round_trip():
    app, _ = build_remote_app()
    with TestClient(app) as client:
        resp = client.post(
            "/register",
            json={
                "token_endpoint_auth_method": "none",
                "redirect_uris": [
                    "http://127.0.0.1:33418",
                    "https://vscode.dev/redirect",
                    "https://insiders.vscode.dev/redirect",
                ],
                "client_name": "VS Code",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
        )
    assert resp.status_code == 201, resp.text
    assert "client_secret" not in resp.json()
