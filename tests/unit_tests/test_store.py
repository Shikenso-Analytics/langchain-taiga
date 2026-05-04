"""Unit tests for ``langchain_taiga.auth.store.InMemoryStore``.

Per Amendment v3.4 of the OAuth bridge plan: pure in-memory storage,
no Postgres, no at-rest encryption. Tests run on plain pytest +
pytest-asyncio with a fresh ``InMemoryStore()`` per test.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_store_and_lookup_access_token():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    await store.store_access_token(
        token="mcp_t1",
        taiga_auth_token="taiga_jwt_1",
        taiga_refresh_token="ref_1",
        taiga_user_id=42,
        taiga_username="alice",
        client_id="c1",
        scopes=["taiga"],
        expires_at=expires_at,
    )
    record = await store.lookup_access_token("mcp_t1")
    assert record is not None
    assert record.token == "mcp_t1"
    assert record.taiga_auth_token == "taiga_jwt_1"
    assert record.taiga_refresh_token == "ref_1"
    assert record.taiga_user_id == 42
    assert record.taiga_username == "alice"
    assert record.client_id == "c1"
    assert record.scopes == ["taiga"]
    assert record.expires_at == expires_at


@pytest.mark.asyncio
async def test_lookup_unknown_token_returns_none():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    assert await store.lookup_access_token("never_minted") is None


@pytest.mark.asyncio
async def test_lookup_expired_returns_none():
    """Defensive check — expired records must not be returned even if
    cleanup_expired hasn't yet swept them."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    expired_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    await store.store_access_token(
        token="dead",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        client_id="c",
        scopes=["taiga"],
        expires_at=expired_at,
    )
    assert await store.lookup_access_token("dead") is None


@pytest.mark.asyncio
async def test_update_taiga_token_for_refresh():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    initial_expires = datetime.now(timezone.utc) + timedelta(minutes=5)
    await store.store_access_token(
        token="t",
        taiga_auth_token="old_jwt",
        taiga_refresh_token="old_ref",
        taiga_user_id=1,
        taiga_username="alice",
        client_id="c",
        scopes=["taiga"],
        expires_at=initial_expires,
    )
    new_expires = datetime.now(timezone.utc) + timedelta(hours=1)
    await store.update_taiga_token(
        token="t",
        taiga_auth_token="new_jwt",
        taiga_refresh_token="new_ref",
        expires_at=new_expires,
    )
    record = await store.lookup_access_token("t")
    assert record is not None
    assert record.taiga_auth_token == "new_jwt"
    assert record.taiga_refresh_token == "new_ref"
    assert record.expires_at == new_expires


@pytest.mark.asyncio
async def test_authorization_code_single_use():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.store_authorization_code(
        code="auth_code_1",
        client_id="c",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="cc",
        code_challenge_method="S256",
        taiga_auth_token="jwt",
        taiga_refresh_token="ref",
        taiga_user_id=42,
        taiga_username="alice",
        scopes=["taiga"],
        expires_at=expires_at,
    )
    first = await store.consume_authorization_code("auth_code_1")
    assert first is not None
    assert first.code == "auth_code_1"
    second = await store.consume_authorization_code("auth_code_1")
    assert second is None  # single-use


@pytest.mark.asyncio
async def test_consume_expired_code_returns_none_and_purges():
    """Expired auth codes are dropped on first consume and stay gone afterwards."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    await store.store_authorization_code(
        code="expired_code",
        client_id="c",
        redirect_uri="r",
        code_challenge="cc",
        code_challenge_method="S256",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="u",
        scopes=["taiga"],
        expires_at=past,
    )
    # First consume returns None (expired) and purges from dict
    assert await store.consume_authorization_code("expired_code") is None
    # Confirm purge: a second call also returns None (would be the same anyway,
    # but this catches accidental dict re-population)
    assert await store.consume_authorization_code("expired_code") is None


@pytest.mark.asyncio
async def test_peek_authorization_code():
    """``peek_authorization_code`` returns a record when present, None when missing
    or expired, and never consumes — parallel to ``consume_authorization_code``."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()

    # Missing returns None
    assert await store.peek_authorization_code("nope") is None

    # Present returns the record (and does not consume it)
    future = datetime.now(timezone.utc) + timedelta(minutes=10)
    await store.store_authorization_code(
        code="live_code",
        client_id="c",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        code_challenge="cc",
        code_challenge_method="S256",
        taiga_auth_token="jwt",
        taiga_refresh_token="ref",
        taiga_user_id=42,
        taiga_username="alice",
        scopes=["taiga"],
        expires_at=future,
    )
    peeked = await store.peek_authorization_code("live_code")
    assert peeked is not None
    assert peeked.code == "live_code"
    assert peeked.client_id == "c"
    # Peek must NOT consume — a second peek still works
    again = await store.peek_authorization_code("live_code")
    assert again is not None
    # And consume still finds it after peeks
    consumed = await store.consume_authorization_code("live_code")
    assert consumed is not None
    # After consume, peek is empty
    assert await store.peek_authorization_code("live_code") is None

    # Expired returns None
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    await store.store_authorization_code(
        code="dead_code",
        client_id="c",
        redirect_uri="r",
        code_challenge="cc",
        code_challenge_method="S256",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="u",
        scopes=["taiga"],
        expires_at=past,
    )
    assert await store.peek_authorization_code("dead_code") is None


@pytest.mark.asyncio
async def test_dynamic_client_registration():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    await store.register_client(
        client_id="cid_1",
        client_secret="sup3r_secret",
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_name="Claude",
        token_endpoint_auth_method="none",
    )
    assert await store.verify_client_secret("cid_1", "sup3r_secret") is True
    assert await store.verify_client_secret("cid_1", "wrong") is False
    assert await store.verify_client_secret("nonexistent", "sup3r_secret") is False


@pytest.mark.asyncio
async def test_lookup_client_returns_no_secret():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    await store.register_client(
        client_id="cid_2",
        client_secret="sup3r_secret",
        redirect_uris=["https://claude.com/api/mcp/auth_callback"],
        client_name="Claude (com)",
        token_endpoint_auth_method="client_secret_basic",
    )
    record = await store.lookup_client("cid_2")
    assert record is not None
    assert record.client_id == "cid_2"
    assert record.client_secret is None  # never expose plaintext on lookup
    assert record.redirect_uris == ["https://claude.com/api/mcp/auth_callback"]
    assert record.client_name == "Claude (com)"
    assert record.token_endpoint_auth_method == "client_secret_basic"


@pytest.mark.asyncio
async def test_lookup_unknown_client_returns_none():
    """RFC 6749 invalid_client signal — caller (provider.get_client) returns
    None and FastMCP renders HTTP 400 invalid_client so claude.ai re-DCRs."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    assert await store.lookup_client("never_registered") is None


@pytest.mark.asyncio
async def test_concurrent_refresh_serialized():
    """Two coroutines acquire ``refresh_lock`` concurrently. Their writes must
    be serialized — final state is one of the two writes, never garbled."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    initial = datetime.now(timezone.utc) + timedelta(minutes=5)
    await store.store_access_token(
        token="shared",
        taiga_auth_token="initial_jwt",
        taiga_refresh_token="initial_ref",
        taiga_user_id=1,
        taiga_username="alice",
        client_id="c",
        scopes=["taiga"],
        expires_at=initial,
    )

    order: list[str] = []

    async def writer(label: str, jwt: str, ref: str, expires: datetime) -> None:
        async with store.refresh_lock("shared") as record:
            assert record is not None
            order.append(f"{label}:enter")
            # Simulate a network delay between read and write so a non-locked
            # implementation would interleave.
            await asyncio.sleep(0.01)
            await store.update_taiga_token(
                token="shared",
                taiga_auth_token=jwt,
                taiga_refresh_token=ref,
                expires_at=expires,
            )
            order.append(f"{label}:exit")

    new_a = datetime.now(timezone.utc) + timedelta(hours=1)
    new_b = datetime.now(timezone.utc) + timedelta(hours=2)
    await asyncio.gather(
        writer("A", "jwt_A", "ref_A", new_a),
        writer("B", "jwt_B", "ref_B", new_b),
    )

    # Verify serialization: one writer fully runs before the other starts.
    # Order must be A:enter A:exit B:enter B:exit OR B:enter B:exit A:enter A:exit
    assert order in (
        ["A:enter", "A:exit", "B:enter", "B:exit"],
        ["B:enter", "B:exit", "A:enter", "A:exit"],
    ), f"Expected serialized order, got: {order}"

    final = await store.lookup_access_token("shared")
    assert final is not None
    # Final state matches one of the two writers — never a mix.
    assert (final.taiga_auth_token, final.taiga_refresh_token) in (
        ("jwt_A", "ref_A"),
        ("jwt_B", "ref_B"),
    )


@pytest.mark.asyncio
async def test_cleanup_expired_purges_old_records():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    now = datetime.now(timezone.utc)
    # Two expired tokens
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
        token="dead_2",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=2,
        taiga_username="y",
        client_id="c",
        scopes=["taiga"],
        expires_at=now - timedelta(minutes=1),
    )
    # One live token
    await store.store_access_token(
        token="live",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=3,
        taiga_username="z",
        client_id="c",
        scopes=["taiga"],
        expires_at=now + timedelta(hours=1),
    )
    # Two expired auth codes
    await store.store_authorization_code(
        code="dead_code_1",
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

    # Also seed a refresh-lock for an expired token to confirm it gets cleaned
    async with store.refresh_lock("dead_1"):
        pass

    purged = await store.cleanup_expired()
    assert purged == 3  # two dead tokens + one dead code

    assert await store.lookup_access_token("dead_1") is None
    assert await store.lookup_access_token("dead_2") is None
    assert await store.lookup_access_token("live") is not None
    # Refresh lock for purged token must be removed
    assert "dead_1" not in store._refresh_locks
