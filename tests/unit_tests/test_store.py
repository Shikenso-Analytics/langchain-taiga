"""Unit tests for ``langchain_taiga.auth.store.InMemoryStore``.

Per Amendment v3.4 of the OAuth bridge plan: pure in-memory storage,
no Postgres, no at-rest encryption. Tests run on plain pytest +
pytest-asyncio with a fresh ``InMemoryStore()`` per test.
"""

from __future__ import annotations
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
    """In-memory store; secret is plaintext but never persisted to disk —
    mcp-sdk's auth middleware needs ``client_secret`` populated on
    ``lookup_client`` for ``client_secret_basic`` / ``client_secret_post``
    verification at /token. ``verify_client_secret`` remains for callers
    wanting a constant-time compare API."""
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
async def test_lookup_client_returns_secret():
    """``lookup_client`` returns the full record with ``client_secret`` populated.

    In-memory store; secret is plaintext but never persisted to disk.
    mcp-sdk's ``ClientAuthenticator`` middleware calls
    ``provider.get_client(client_id).client_secret`` to validate
    ``client_secret_basic`` / ``client_secret_post``. Returning ``None`` for
    the secret would short-circuit comparisons and let any caller knowing the
    ``client_id`` bypass authentication — confidential clients would silently
    downgrade to public.
    """
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
    assert record.client_secret == "sup3r_secret"
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

    purged = await store.cleanup_expired()
    assert purged == 3  # two dead tokens + one dead code

    assert await store.lookup_access_token("dead_1") is None
    assert await store.lookup_access_token("dead_2") is None
    assert await store.lookup_access_token("live") is not None


@pytest.mark.asyncio
async def test_access_token_record_has_family_id():
    """Each access token belongs to a refresh-family. The family_id is the
    revocation handle when reuse-detection fires."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    await store.store_access_token(
        token="mcp_acc_1",
        family_id="fam_xyz",
        taiga_auth_token="taiga_jwt",
        taiga_refresh_token="taiga_ref",
        taiga_user_id=1,
        taiga_username="alice",
        client_id="c1",
        scopes=["taiga"],
        expires_at=expires_at,
    )
    record = await store.lookup_access_token("mcp_acc_1")
    assert record is not None
    assert record.family_id == "fam_xyz"


@pytest.mark.asyncio
async def test_store_and_lookup_refresh_token():
    """Roundtrip: store then lookup returns the persisted record."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    await store.store_refresh_token(
        token="ref_1",
        family_id="fam_1",
        client_id="c1",
        taiga_auth_token="t_jwt",
        taiga_refresh_token="t_ref",
        taiga_user_id=42,
        taiga_username="alice",
        scopes=["taiga"],
        expires_at=expires_at,
    )
    record = await store.lookup_refresh_token("ref_1")
    assert record is not None
    assert record.token == "ref_1"
    assert record.family_id == "fam_1"
    assert record.client_id == "c1"
    assert record.taiga_auth_token == "t_jwt"
    assert record.taiga_refresh_token == "t_ref"
    assert record.taiga_user_id == 42
    assert record.taiga_username == "alice"
    assert record.scopes == ["taiga"]
    assert record.expires_at == expires_at
    assert record.rotated_out is False


@pytest.mark.asyncio
async def test_lookup_refresh_token_filters_expired():
    """Mirror of lookup_access_token: expired records are not returned."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    expired_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    await store.store_refresh_token(
        token="dead_ref",
        family_id="fam",
        client_id="c1",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=expired_at,
    )
    assert await store.lookup_refresh_token("dead_ref") is None


@pytest.mark.asyncio
async def test_lookup_refresh_token_returns_rotated_out_records():
    """Reuse-detection requires seeing rotated_out records on lookup so
    exchange_refresh_token can route to the family-revoke branch."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    await store.store_refresh_token(
        token="ref_rot",
        family_id="fam",
        client_id="c1",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=expires_at,
    )
    # Manually flip rotated_out to simulate post-rotation state
    store._refresh_tokens["ref_rot"].rotated_out = True
    record = await store.lookup_refresh_token("ref_rot")
    assert record is not None
    assert record.rotated_out is True


@pytest.mark.asyncio
async def test_lookup_unknown_refresh_token_returns_none():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    assert await store.lookup_refresh_token("never_minted") is None


@pytest.mark.asyncio
async def test_consume_refresh_token_active_marks_rotated_out():
    """First call on a fresh token returns status=active and flips the bit."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    await store.store_refresh_token(
        token="ref_active",
        family_id="fam",
        client_id="c1",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )
    result = await store.consume_refresh_token("ref_active")
    assert result.status == "active"
    assert result.record is not None
    assert result.record.token == "ref_active"
    # rotated_out flag was flipped in place
    assert store._refresh_tokens["ref_active"].rotated_out is True


@pytest.mark.asyncio
async def test_consume_refresh_token_already_rotated_returns_replay_status():
    """Second consume on the same token returns status=already_rotated."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    await store.store_refresh_token(
        token="ref_dup",
        family_id="fam",
        client_id="c1",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )
    first = await store.consume_refresh_token("ref_dup")
    assert first.status == "active"
    second = await store.consume_refresh_token("ref_dup")
    assert second.status == "already_rotated"
    assert second.record is not None
    assert second.record.family_id == "fam"


@pytest.mark.asyncio
async def test_consume_refresh_token_expired_returns_expired_status():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    await store.store_refresh_token(
        token="ref_exp",
        family_id="fam",
        client_id="c1",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    result = await store.consume_refresh_token("ref_exp")
    assert result.status == "expired"


@pytest.mark.asyncio
async def test_consume_refresh_token_not_found():
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    result = await store.consume_refresh_token("never_minted")
    assert result.status == "not_found"
    assert result.record is None


@pytest.mark.asyncio
async def test_revoke_token_family_removes_all_tokens_in_family():
    """Family-revoke wipes every access AND refresh token sharing family_id."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    exp_a = datetime.now(timezone.utc) + timedelta(hours=1)
    exp_r = datetime.now(timezone.utc) + timedelta(days=30)
    # Family A: 2 access tokens, 2 refresh tokens (one rotated_out, one active)
    for tok in ("acc_a1", "acc_a2"):
        await store.store_access_token(
            token=tok,
            family_id="fam_A",
            taiga_auth_token="t",
            taiga_refresh_token="r",
            taiga_user_id=1,
            taiga_username="x",
            client_id="c1",
            scopes=["taiga"],
            expires_at=exp_a,
        )
    for tok in ("ref_a1", "ref_a2"):
        await store.store_refresh_token(
            token=tok,
            family_id="fam_A",
            client_id="c1",
            taiga_auth_token="t",
            taiga_refresh_token="r",
            taiga_user_id=1,
            taiga_username="x",
            scopes=["taiga"],
            expires_at=exp_r,
        )
    store._refresh_tokens["ref_a1"].rotated_out = True
    # Unrelated family B: must survive
    await store.store_access_token(
        token="acc_b1",
        family_id="fam_B",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=2,
        taiga_username="bob",
        client_id="c2",
        scopes=["taiga"],
        expires_at=exp_a,
    )
    revoked = await store.revoke_token_family("fam_A")
    assert revoked == 4
    # Family A wiped
    for tok in ("acc_a1", "acc_a2"):
        assert await store.lookup_access_token(tok) is None
    for tok in ("ref_a1", "ref_a2"):
        assert await store.lookup_refresh_token(tok) is None
    # Family B intact
    assert await store.lookup_access_token("acc_b1") is not None


@pytest.mark.asyncio
async def test_revoke_token_family_with_empty_family_id_is_a_noop():
    """Defense: an empty-string family_id (e.g. from a legacy/unset field)
    must NOT purge records whose family_id also defaults to ''."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    # Legacy record with default ""
    await store.store_access_token(
        token="legacy",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        client_id="c",
        scopes=["taiga"],
        expires_at=expires_at,
    )  # family_id defaults to ""
    revoked = await store.revoke_token_family("")
    assert revoked == 0
    assert await store.lookup_access_token("legacy") is not None


@pytest.mark.asyncio
async def test_cleanup_expired_sweeps_refresh_tokens():
    """Both active and rotated_out refresh tokens should be purged when
    expired; fresh tokens survive."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    now = datetime.now(timezone.utc)
    fresh_exp = now + timedelta(days=30)
    stale_exp = now - timedelta(seconds=1)

    await store.store_refresh_token(
        token="fresh",
        family_id="f",
        client_id="c",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=fresh_exp,
    )
    await store.store_refresh_token(
        token="stale_active",
        family_id="f",
        client_id="c",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=stale_exp,
    )
    await store.store_refresh_token(
        token="stale_rotated",
        family_id="f",
        client_id="c",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=stale_exp,
    )
    store._refresh_tokens["stale_rotated"].rotated_out = True

    purged = await store.cleanup_expired()
    assert purged == 2
    assert "fresh" in store._refresh_tokens
    assert "stale_active" not in store._refresh_tokens
    assert "stale_rotated" not in store._refresh_tokens


@pytest.mark.asyncio
async def test_cleanup_expired_purges_old_revoked_family_tombstones():
    """Tombstones older than REFRESH_TOKEN_TTL are no longer useful for
    reuse-detection (every refresh token in the family has expired by
    then) and must be swept to bound memory growth."""
    from langchain_taiga.auth.store import InMemoryStore

    store = InMemoryStore()
    # Inject a stale and a fresh tombstone directly
    stale_ts = datetime.now(timezone.utc) - timedelta(days=31)
    fresh_ts = datetime.now(timezone.utc) - timedelta(days=1)
    store._revoked_families["fam_stale"] = stale_ts
    store._revoked_families["fam_fresh"] = fresh_ts

    purged = await store.cleanup_expired()
    assert purged >= 1
    assert "fam_stale" not in store._revoked_families
    assert "fam_fresh" in store._revoked_families
