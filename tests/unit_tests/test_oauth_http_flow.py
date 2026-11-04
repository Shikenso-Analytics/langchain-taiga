"""The whole OAuth flow over HTTP against the real remote app.

``remote_app.mint_bearer`` calls provider methods directly, so it never
touches the HTTP handlers of the mcp SDK and fastmcp (``/register``,
``/authorize``, ``/token``). This test drives them as a client would.
"""

from __future__ import annotations

import base64
import hashlib
from urllib.parse import parse_qs, urlparse

from httpx import Response
from starlette.testclient import TestClient

from tests.unit_tests.remote_app import (
    ISSUER,
    MODERN,
    TAIGA_API,
    build_remote_app,
    rpc,
    rpc_result,
)
from tests.unit_tests.test_mcp import EXPECTED_MCP_TOOL_NAMES

REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"


def _query(response) -> dict[str, str]:
    return {
        k: v[0] for k, v in parse_qs(urlparse(response.headers["location"]).query).items()
    }


def test_full_oauth_flow_over_http(taiga_auth_mock):
    taiga_auth_mock.post(f"{TAIGA_API}/api/v1/auth/refresh").mock(
        return_value=Response(200, json={"auth_token": "alice_jwt2", "refresh": "alice_ref2"})
    )
    verifier = "v" * 60
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    app, _ = build_remote_app()

    with TestClient(app, follow_redirects=False) as client:
        # 1. dynamic client registration: public client, no secret
        reg = client.post(
            "/register",
            json={
                "client_name": "Claude",
                "redirect_uris": [REDIRECT_URI],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            },
        )
        assert reg.status_code == 201, reg.text
        client_id = reg.json()["client_id"]
        assert not reg.json().get("client_secret")

        # 2. authorize redirects to the bridge's login page
        auth = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "csrf-state",
                "scope": "taiga",
                "resource": ISSUER,
            },
        )
        assert auth.status_code in (302, 307), auth.text
        assert urlparse(auth.headers["location"]).path.endswith("/oauth/login")
        internal_state = _query(auth)["internal_state"]

        # 3. login form -> redirect back to the client with code and state
        login = client.post(
            "/mcp/oauth/login",
            data={"state": internal_state, "username": "alice", "password": "pw"},
        )
        assert login.status_code in (302, 303, 307), login.text
        assert login.headers["location"].startswith(REDIRECT_URI)
        callback = _query(login)
        assert callback["state"] == "csrf-state"
        code = callback["code"]

        # 4. authorization_code grant
        token_form = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "resource": ISSUER,
        }
        tok = client.post("/token", data=token_form)
        assert tok.status_code == 200, tok.text
        first = tok.json()
        assert first["access_token"]
        assert first["refresh_token"]

        # 5. refresh_token grant rotates both tokens
        ref = client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": first["refresh_token"],
                "client_id": client_id,
            },
        )
        assert ref.status_code == 200, ref.text
        second = ref.json()
        assert second["access_token"]
        assert second["access_token"] != first["access_token"]
        assert second["refresh_token"] != first["refresh_token"]

        # 6. the refreshed token opens the MCP endpoint
        listed = rpc(client, "tools/list", version=MODERN, bearer=second["access_token"])
        names = {t["name"] for t in rpc_result(listed)["tools"]}
        assert names == EXPECTED_MCP_TOOL_NAMES

        # 7. the authorization code is single-use
        replay = client.post("/token", data=token_form)
        assert replay.status_code != 200
        assert replay.json()["error"] == "invalid_grant"
