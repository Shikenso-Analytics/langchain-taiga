"""Test the periodic ``run_cleanup_loop`` helper.

The helper spawns as a background task in ``remote_server.py``'s lifespan
and sweeps expired records every ``period_seconds``. Stopping is done by
setting an ``asyncio.Event``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from langchain_taiga.auth.provider import _PendingAuthorize


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


@pytest.mark.asyncio
async def test_cleanup_loop_prunes_provider_authorize_states():
    """When ``provider`` is supplied, ``run_cleanup_loop`` also prunes its
    ``_authorize_states`` entries with past ``expires_at``."""
    from langchain_taiga.auth.provider import (
        TaigaOAuthProvider,
        run_cleanup_loop,
    )
    from langchain_taiga.auth.store import InMemoryStore
    from langchain_taiga.auth.taiga_client import TaigaClient

    store = InMemoryStore()
    provider = TaigaOAuthProvider(
        store=store,
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )

    now = datetime.now(timezone.utc)
    # Seed one expired and one live authorize state
    provider._authorize_states["dead_state"] = _PendingAuthorize(
        client_id="c",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="cc",
        code_challenge_method="S256",
        scopes=["taiga"],
        claude_state="abc",
        expires_at=now - timedelta(seconds=1),
    )
    provider._authorize_states["live_state"] = _PendingAuthorize(
        client_id="c",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="cc",
        code_challenge_method="S256",
        scopes=["taiga"],
        claude_state="def",
        expires_at=now + timedelta(minutes=10),
    )

    stop = asyncio.Event()
    task = asyncio.create_task(
        run_cleanup_loop(
            store, provider=provider, period_seconds=0.1, stop=stop
        )
    )

    # Let the loop run a few iterations
    await asyncio.sleep(0.3)

    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert "dead_state" not in provider._authorize_states
    assert "live_state" in provider._authorize_states


@pytest.mark.asyncio
async def test_cleanup_authorize_states_directly():
    """Direct unit check of ``provider.cleanup_authorize_states`` (no loop)."""
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.store import InMemoryStore
    from langchain_taiga.auth.taiga_client import TaigaClient

    provider = TaigaOAuthProvider(
        store=InMemoryStore(),
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )
    now = datetime.now(timezone.utc)
    provider._authorize_states["expired"] = _PendingAuthorize(
        client_id="c",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="cc",
        code_challenge_method="S256",
        scopes=["taiga"],
        claude_state="x",
        expires_at=now - timedelta(minutes=5),
    )
    provider._authorize_states["fresh"] = _PendingAuthorize(
        client_id="c",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="cc",
        code_challenge_method="S256",
        scopes=["taiga"],
        claude_state="y",
        expires_at=now + timedelta(minutes=5),
    )
    purged = provider.cleanup_authorize_states()
    assert purged == 1
    assert "expired" not in provider._authorize_states
    assert "fresh" in provider._authorize_states


@pytest.mark.asyncio
async def test_cleanup_loop_purges_expired_refresh_tokens():
    """The store-level cleanup_expired (now sweeping refresh tokens) must
    surface through run_cleanup_loop: an expired refresh record is removed
    within one loop iteration."""
    import asyncio
    from langchain_taiga.auth.provider import run_cleanup_loop
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    await store.store_refresh_token(
        token="stale_ref",
        family_id="fam",
        client_id="c",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    stop = asyncio.Event()
    task = asyncio.create_task(
        run_cleanup_loop(store, period_seconds=0.05, stop=stop)
    )
    await asyncio.sleep(0.1)
    stop.set()
    await task

    assert await store.lookup_refresh_token("stale_ref") is None
