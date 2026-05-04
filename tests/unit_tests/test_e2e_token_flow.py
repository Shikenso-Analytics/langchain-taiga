"""End-to-end Multi-Tenant security check.

Drives two users (alice, bob) through the full OAuth flow against a
``TaigaOAuthProvider`` instance, then simulates outbound python-taiga
calls using the JWT each provider mints into ``AccessToken.claims``.

The two killer assertions:
  - alice's outbound request carries ``Bearer alice_jwt``
  - bob's outbound request carries ``Bearer bob_jwt``
  - No single request header carries BOTH JWTs (token-leak detector)

**Dual-mock-stack required:** TaigaClient (the OAuth bridge) uses httpx →
respx_mock fixture. python-taiga (the tools' outbound) uses the
synchronous ``requests`` library → ``responses.RequestsMock``. A single
mocking layer would fail to intercept python-taiga.
"""

from __future__ import annotations

import base64
import hashlib
import json as _json

import pytest
import responses
from httpx import Response


@pytest.mark.asyncio
async def test_tool_call_carries_per_user_taiga_jwt(respx_mock):
    """Two users, two tool-style outbound calls. Headers must carry distinct
    JWTs and no single request must contain both tokens."""
    from mcp.server.auth.provider import AuthorizationParams
    from mcp.shared.auth import OAuthClientInformationFull
    from taiga import TaigaAPI

    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.store import InMemoryStore
    from langchain_taiga.auth.taiga_client import TaigaClient

    captured_headers: list[str] = []

    # ---- Mock layer 1: TaigaClient credential exchange (httpx via respx) ----
    def auth_handler(request):
        payload = _json.loads(request.content.decode())
        if payload["username"] == "alice":
            return Response(
                200,
                json={
                    "auth_token": "alice_jwt",
                    "refresh": "alice_ref",
                    "id": 1,
                    "username": "alice",
                },
            )
        if payload["username"] == "bob":
            return Response(
                200,
                json={
                    "auth_token": "bob_jwt",
                    "refresh": "bob_ref",
                    "id": 2,
                    "username": "bob",
                },
            )
        return Response(401)

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        side_effect=auth_handler
    )

    # ---- Mock layer 2: python-taiga outbound (requests via responses) ----
    # python-taiga calls /api/v1/projects/by_slug?slug=<slug> (querystring,
    # not path). Capture the Authorization header on each call.
    def capture_callback(request):
        captured_headers.append(request.headers.get("Authorization", ""))
        return (
            200,
            {},
            _json.dumps(
                {"id": 99, "slug": "shikenso-development", "name": "T", "members": []}
            ),
        )

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add_callback(
            responses.GET,
            "https://taiga.example.test/api/v1/projects/by_slug",
            callback=capture_callback,
        )

        # ---- Drive OAuth flow for alice and bob ----
        store = InMemoryStore()
        provider = TaigaOAuthProvider(
            store=store,
            taiga_client=TaigaClient(api_url="https://taiga.example.test"),
            issuer_url="https://taiga.shikenso.org/mcp",
        )

        # The mcp-sdk's RegistrationHandler mints client_id/client_secret
        # before invoking provider.register_client; mirror that shape here.
        client_info = await provider.register_client(
            OAuthClientInformationFull(
                client_id="cid_e2e_test",
                client_secret="sec_e2e_test",
                redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
                client_name="Claude",
                token_endpoint_auth_method="none",
            )
        )

        async def issue_mcp_token(username: str) -> str:
            verifier = f"v_{username}" + "x" * 50
            challenge = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(verifier.encode()).digest()
                )
                .rstrip(b"=")
                .decode()
            )
            redirect = await provider.authorize(
                client=client_info,
                params=AuthorizationParams(
                    state="csrf",
                    scopes=["taiga"],
                    code_challenge=challenge,
                    redirect_uri="https://claude.ai/api/mcp/auth_callback",
                    redirect_uri_provided_explicitly=True,
                ),
            )
            internal = redirect.split("internal_state=", 1)[1]
            code, _ = await provider.complete_login(
                internal_state=internal, username=username, password="x"
            )
            auth_obj = await provider.load_authorization_code(client_info, code)
            assert auth_obj is not None
            oauth_tok = await provider.exchange_authorization_code(
                client_info, auth_obj
            )
            return oauth_tok.access_token

        alice_mcp = await issue_mcp_token("alice")
        bob_mcp = await issue_mcp_token("bob")

        # ---- Simulate per-user tool calls ----
        # Each request maps the per-user MCP access token back to the
        # Taiga JWT in claims, then instantiates a fresh TaigaAPI with
        # that token. This is exactly what PR 1's
        # ``get_taiga_api(token=_current_taiga_jwt())`` does inside the
        # tools after FastMCP injects the AccessToken via the request
        # context. Here we drive it explicitly.

        alice_access = await provider.load_access_token(alice_mcp)
        assert alice_access is not None
        alice_jwt = alice_access.claims["taiga_jwt"]
        TaigaAPI(
            host="https://taiga.example.test", token=alice_jwt
        ).projects.get_by_slug("shikenso-development")

        bob_access = await provider.load_access_token(bob_mcp)
        assert bob_access is not None
        bob_jwt = bob_access.claims["taiga_jwt"]
        TaigaAPI(
            host="https://taiga.example.test", token=bob_jwt
        ).projects.get_by_slug("shikenso-development")

        # ---- Killer assertions ----
        assert any("alice_jwt" in h for h in captured_headers), (
            f"alice_jwt missing from outbound headers: {captured_headers}"
        )
        assert any("bob_jwt" in h for h in captured_headers), (
            f"bob_jwt missing from outbound headers: {captured_headers}"
        )
        for h in captured_headers:
            assert not ("alice_jwt" in h and "bob_jwt" in h), (
                "Token leakage — single request header contained BOTH "
                f"JWTs: {h}"
            )
