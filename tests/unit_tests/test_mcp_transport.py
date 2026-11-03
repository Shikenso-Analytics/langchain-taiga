"""``/mcp`` served stateless, both protocol eras, over the wire.

Every test drives the real app built by ``remote_app.build_remote_app``
through ``TestClient``; the lifespan runs because the client is a context
manager. Taiga is mocked on two stacks (see ``test_e2e_token_flow``): httpx
via ``respx`` for the credential exchange, ``requests`` via ``responses`` for
python-taiga's outbound calls.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest
import responses
from starlette.testclient import TestClient

from langchain_taiga import remote_server
from langchain_taiga.auth.store import InMemoryStore
from langchain_taiga.toolkits import TaigaToolkit
from tests.unit_tests.remote_app import (
    LEGACY,
    MODERN,
    TAIGA_API,
    build_remote_app,
    mint_bearer,
    rpc,
    rpc_result,
)
from tests.unit_tests.test_mcp import EXPECTED_TOOL_NAMES

INIT_PARAMS = {
    "protocolVersion": LEGACY,
    "capabilities": {},
    "clientInfo": {"name": "t", "version": "0"},
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TAIGA_API_URL", TAIGA_API)
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.fixture
def served(taiga_auth_mock):
    """(client, alice_bearer) against the real app, lifespan running."""
    app, provider = build_remote_app()
    bearer = asyncio.run(mint_bearer(provider, "alice"))
    with TestClient(app) as client:
        yield client, bearer


def test_async_main_serves_mcp_stateless(monkeypatch):
    from fastmcp import FastMCP

    monkeypatch.setenv("TAIGA_URL", TAIGA_API)
    monkeypatch.setenv("TAIGA_MCP_BASE_URL", "https://taiga.example.com/mcp")
    app, provider = build_remote_app()

    async def fake_bootstrap(**_kw):
        return provider, InMemoryStore()

    seen: dict = {}

    async def fake_run_async(self, *args, **kw):
        seen.update(kw)

    monkeypatch.setattr(remote_server, "_bootstrap_provider", fake_bootstrap)
    monkeypatch.setattr(FastMCP, "run_async", fake_run_async)

    asyncio.run(remote_server._async_main("127.0.0.1", 8000))

    assert seen["transport"] == "streamable-http"
    assert seen["path"] == "/mcp"
    assert seen["stateless_http"] is True


def test_legacy_initialize_issues_no_session_id(served):
    client, bearer = served
    response = rpc(client, "initialize", INIT_PARAMS, version=None, bearer=bearer)
    assert response.status_code == 200, response.text
    assert "mcp-session-id" not in response.headers


def test_legacy_tools_list_needs_no_session(served):
    client, bearer = served
    result = rpc_result(rpc(client, "tools/list", version=LEGACY, bearer=bearer), 1)
    assert {t["name"] for t in result["tools"]} == EXPECTED_TOOL_NAMES


def test_stale_session_id_header_is_served(served):
    client, bearer = served
    response = rpc(
        client,
        "tools/list",
        version=LEGACY,
        bearer=bearer,
        session_id="0123456789abcdef",
    )
    result = rpc_result(response, 1)
    assert {t["name"] for t in result["tools"]} == EXPECTED_TOOL_NAMES


def test_modern_server_discover_answers(served):
    client, bearer = served
    result = rpc_result(rpc(client, "server/discover", bearer=bearer), 1)
    assert MODERN in result["supportedVersions"]


def test_modern_tools_list_publishes_every_parameter_description(served):
    client, bearer = served
    tools = rpc_result(rpc(client, "tools/list", bearer=bearer), 1)["tools"]
    assert {t["name"] for t in tools} == EXPECTED_TOOL_NAMES
    missing = [
        (t["name"], p)
        for t in tools
        for p, schema in t["inputSchema"].get("properties", {}).items()
        if not schema.get("description")
    ]
    assert missing == []

    # Descriptions are the full docstring, not FastMCP 4's first paragraph.
    source = {t.name: inspect.getdoc(t.func) for t in TaigaToolkit().get_tools()}
    assert all(source.values())
    wrong = [t["name"] for t in tools if t.get("description") != source[t["name"]]]
    assert wrong == []


def test_modern_tools_call_uses_the_callers_taiga_token(served):
    client, bearer = served
    seen: list[str] = []

    def me(request):
        seen.append(request.headers.get("Authorization", ""))
        return (
            200,
            {},
            json.dumps(
                {"id": 1, "username": "alice", "full_name": "Alice", "email": "a@x.y"}
            ),
        )

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add_callback(responses.GET, f"{TAIGA_API}/api/v1/users/me", callback=me)
        response = rpc(
            client,
            "tools/call",
            {"name": "whoami_tool", "arguments": {}},
            bearer=bearer,
        )
    result = rpc_result(response, 1)
    assert not result.get("isError"), result
    assert seen == ["Bearer alice_jwt"]


@pytest.mark.parametrize("version", [None, LEGACY, MODERN])
def test_unauthenticated_requests_get_401_with_prm_pointer(served, version):
    client, _ = served
    response = rpc(client, "tools/list", version=version)
    assert response.status_code == 401
    challenge = response.headers["www-authenticate"]
    assert "resource_metadata=" in challenge
    assert "/.well-known/oauth-protected-resource" in challenge
