"""Build and talk to the real remote MCP app in tests (not collected).

One responsibility: construct the ASGI app the way ``_async_main`` does and
speak both MCP protocol eras to ``/mcp`` over the wire.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from httpx import Response
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from starlette.applications import Starlette

from langchain_taiga.auth.provider import TaigaOAuthProvider
from langchain_taiga.auth.store import InMemoryStore
from langchain_taiga.auth.taiga_client import TaigaClient
from langchain_taiga.mcp import make_mcp
from langchain_taiga.remote_server import _attach_custom_routes, _http_app_kwargs

MODERN = "2026-07-28"
LEGACY = "2025-11-25"
ACCEPT = "application/json, text/event-stream"
TAIGA_API = "https://taiga.example.test"
ISSUER = "https://taiga.example.com/mcp"


def build_remote_app() -> tuple[Starlette, TaigaOAuthProvider]:
    """The production app, minus uvicorn."""
    store = InMemoryStore()
    provider = TaigaOAuthProvider(
        store=store,
        taiga_client=TaigaClient(api_url=TAIGA_API),
        issuer_url=ISSUER,
    )
    m = make_mcp(auth=provider, streamable_http_path="/mcp")
    _attach_custom_routes(m, provider, taiga_url=TAIGA_API, base_url=ISSUER)
    return m.http_app(**_http_app_kwargs(m)), provider


async def mint_bearer(provider: TaigaOAuthProvider, username: str) -> str:
    """Walk one user through register, authorize, login and code exchange."""
    client_info = await provider.register_client(
        OAuthClientInformationFull(
            client_id=f"cid_{username}",
            client_secret=f"sec_{username}",
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            client_name="Claude",
            token_endpoint_auth_method="none",
        )
    )
    verifier = f"v_{username}" + "x" * 50
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
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
    oauth_tok = await provider.exchange_authorization_code(client_info, auth_obj)
    return oauth_tok.access_token


def build_rpc_request(
    method: str,
    params: dict | None = None,
    *,
    version: str | None = MODERN,
    bearer: str | None = None,
    session_id: str | None = None,
    rpc_id: Any = 1,
) -> tuple[dict, dict]:
    """The ``(body, headers)`` of one JSON-RPC request to ``/mcp``.

    ``version`` is ``MODERN`` (per-request headers and ``_meta``), ``LEGACY``
    (only ``Mcp-Protocol-Version``) or ``None`` (no version header at all).
    """
    headers = {"Accept": ACCEPT, "Content-Type": "application/json"}
    body_params = dict(params) if params is not None else None
    if version is not None:
        headers["Mcp-Protocol-Version"] = version
    if version == MODERN:
        headers["Mcp-Method"] = method
        if method == "tools/call" and body_params:
            headers["Mcp-Name"] = body_params["name"]
        body_params = body_params or {}
        body_params["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": MODERN,
            "io.modelcontextprotocol/clientInfo": {
                "name": "test-client",
                "version": "0",
            },
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
    if body_params is not None:
        body["params"] = body_params
    return body, headers


def rpc(
    client,
    method: str,
    params: dict | None = None,
    *,
    version: str | None = MODERN,
    bearer: str | None = None,
    session_id: str | None = None,
    rpc_id: Any = 1,
) -> Response:
    """POST one JSON-RPC request to ``/mcp`` (see ``build_rpc_request``)."""
    body, headers = build_rpc_request(
        method,
        params,
        version=version,
        bearer=bearer,
        session_id=session_id,
        rpc_id=rpc_id,
    )
    return client.post("/mcp", json=body, headers=headers)


def rpc_result(response: Response, rpc_id: Any = 1) -> dict:
    """The ``result`` of a JSON or SSE answer; fails on HTTP or JSON-RPC errors."""
    assert response.status_code == 200, (response.status_code, response.text)
    if "text/event-stream" in response.headers.get("content-type", ""):
        messages = [
            json.loads(line[5:].strip())
            for line in response.text.splitlines()
            if line.startswith("data:") and line[5:].strip()
        ]
    else:
        messages = [response.json()]
    matching = [m for m in messages if m.get("id") == rpc_id]
    assert matching, f"no answer for id {rpc_id!r}: {response.text}"
    assert "error" not in matching[0], matching[0]["error"]
    return matching[0]["result"]
