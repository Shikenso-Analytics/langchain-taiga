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

import json as _json

import pytest
import responses
from httpx import Response


@pytest.mark.asyncio
async def test_tool_call_carries_per_user_taiga_jwt(respx_mock):
    """Two users, two tool-style outbound calls. Headers must carry distinct
    JWTs and no single request must contain both tokens."""
    from taiga import TaigaAPI

    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.store import InMemoryStore
    from langchain_taiga.auth.taiga_client import TaigaClient
    from tests.unit_tests.remote_app import mint_bearer

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
                {"id": 99, "slug": "my-project", "name": "T", "members": []}
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
            issuer_url="https://taiga.example.com/mcp",
        )

        alice_mcp = await mint_bearer(provider, "alice")
        bob_mcp = await mint_bearer(provider, "bob")

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
        ).projects.get_by_slug("my-project")

        bob_access = await provider.load_access_token(bob_mcp)
        assert bob_access is not None
        bob_jwt = bob_access.claims["taiga_jwt"]
        TaigaAPI(
            host="https://taiga.example.test", token=bob_jwt
        ).projects.get_by_slug("my-project")

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
