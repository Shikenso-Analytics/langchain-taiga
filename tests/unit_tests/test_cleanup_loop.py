"""Test the periodic ``run_cleanup_loop`` helper.

The helper spawns as a background task in ``remote_server.py``'s lifespan
and sweeps expired records every ``period_seconds``. Stopping is done by
setting an ``asyncio.Event``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_cleanup_loop_purges_expired_records():
    from langchain_taiga.auth.provider import run_cleanup_loop
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    now = datetime.now(timezone.utc)

    # Two expired records and one live one
    await store.store_access_token(
        token="dead_1",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        client_id="c",
        scopes=["taiga"],
        expires_at=now - timedelta(seconds=1),
    )
    await store.store_access_token(
        token="live",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=2,
        taiga_username="y",
        client_id="c",
        scopes=["taiga"],
        expires_at=now + timedelta(hours=1),
    )
    await store.store_authorization_code(
        code="dead_code",
        client_id="c",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="cc",
        code_challenge_method="S256",
        taiga_auth_token="jwt",
        taiga_refresh_token="ref",
        taiga_user_id=42,
        taiga_username="alice",
        scopes=["taiga"],
        expires_at=now - timedelta(seconds=1),
    )

    stop = asyncio.Event()
    task = asyncio.create_task(
        run_cleanup_loop(store, period_seconds=0.1, stop=stop)
    )

    # Let the loop run a few iterations
    await asyncio.sleep(0.3)

    # Stop and await clean shutdown
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    # Expired records purged
    assert await store.lookup_access_token("dead_1") is None
    assert await store.consume_authorization_code("dead_code") is None
    # Live token remains
    assert await store.lookup_access_token("live") is not None
