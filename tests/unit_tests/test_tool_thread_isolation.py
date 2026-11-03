"""Two callers, one process, sync tools on worker threads.

fastmcp 4 runs every synchronous tool in a thread, so tool calls of different
tenants overlap. Each call must reach Taiga with its own caller's token and
never with another's. Multi-tenant isolation hangs on the token context
variable surviving the hop into the worker thread.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import httpx
import pytest
import responses

from tests.unit_tests.remote_app import (
    ACCEPT,
    MODERN,
    TAIGA_API,
    build_remote_app,
    mint_bearer,
    rpc_result,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("TAIGA_API_URL", TAIGA_API)
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


async def _whoami(client: httpx.AsyncClient, bearer: str, rpc_id: int) -> httpx.Response:
    """One MODERN ``tools/call whoami_tool`` (async twin of ``remote_app.rpc``)."""
    headers = {
        "Accept": ACCEPT,
        "Content-Type": "application/json",
        "Mcp-Protocol-Version": MODERN,
        "Mcp-Method": "tools/call",
        "Mcp-Name": "whoami_tool",
        "Authorization": f"Bearer {bearer}",
    }
    body = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/call",
        "params": {
            "name": "whoami_tool",
            "arguments": {},
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": MODERN,
                "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "0"},
                "io.modelcontextprotocol/clientCapabilities": {},
            },
        },
    }
    return await client.post("/mcp", json=body, headers=headers)


@pytest.mark.asyncio
async def test_parallel_tool_calls_keep_their_own_taiga_token(taiga_auth_mock):
    app, provider = build_remote_app()
    alice = await mint_bearer(provider, "alice")
    bob = await mint_bearer(provider, "bob")

    users = {
        "Bearer alice_jwt": {"id": 1, "username": "alice"},
        "Bearer bob_jwt": {"id": 2, "username": "bob"},
    }
    seen: list[str] = []
    live = 0
    peak = 0
    counter_lock = threading.Lock()

    def me(request):
        nonlocal live, peak
        auth = request.headers.get("Authorization", "")
        with counter_lock:
            seen.append(auth)
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)  # worker thread: never blocks the loop
        with counter_lock:
            live -= 1
        user = users.get(auth)
        if user is None:
            return (401, {}, json.dumps({"detail": "wrong or missing token"}))
        return (200, {}, json.dumps({**user, "full_name": user["username"], "email": "x@y.z"}))

    callers = [("alice", alice), ("bob", bob)] * 5  # interleaved
    transport = httpx.ASGITransport(app=app)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add_callback(responses.GET, f"{TAIGA_API}/api/v1/users/me", callback=me)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                answers = await asyncio.gather(
                    *(_whoami(c, bearer, i) for i, (_, bearer) in enumerate(callers))
                )

    for i, ((name, _), response) in enumerate(zip(callers, answers)):
        result = rpc_result(response, i)
        assert not result.get("isError"), result
        assert name in json.dumps(result), (name, result)
        other = "bob" if name == "alice" else "alice"
        assert other not in json.dumps(result), (name, result)

    assert sorted(seen) == ["Bearer alice_jwt"] * 5 + ["Bearer bob_jwt"] * 5
    # Overlap proof: without it the assertions above say nothing about threads.
    assert peak >= 2, peak
