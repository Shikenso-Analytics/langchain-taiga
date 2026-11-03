"""Discovery over the wire: what the real app serves, not what a builder returns."""

from __future__ import annotations

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


def test_served_as_metadata_issuer_matches_prm():
    app, _ = build_remote_app()
    with TestClient(app) as client:
        issuer = client.get(AS_META).json()["issuer"].rstrip("/")
        for path in (
            "/.well-known/oauth-protected-resource/mcp",
            "/.well-known/oauth-protected-resource",
        ):
            resp = client.get(path)
            assert resp.status_code == 200, path
            prm_as = resp.json()["authorization_servers"][0].rstrip("/")
            assert issuer == prm_as, path


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
