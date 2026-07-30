"""Shared contract tests: the two stores must behave identically.

``PostgresStore`` is a drop-in replacement chosen at startup by environment
(``remote_server._build_store``), so any behavioural drift between the two
silently changes production OAuth semantics. Running one suite against both
implementations is what keeps them honest.

The Postgres parametrisation is **skipped unless ``TAIGA_MCP_TEST_DATABASE_URL``
is set**. CI sets it via a ``postgres`` service container; locally:

    docker run --rm -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:16
    export TAIGA_MCP_TEST_DATABASE_URL=postgres://postgres:pw@127.0.0.1:5432/postgres

The suite is deliberately run against a real database rather than a mocked
connection: the properties under test here are the SQL-level atomicity
guarantees (``FOR UPDATE`` row locks, ``DELETE ... RETURNING``, advisory
locks). A mock would assert that we send certain strings, not that the
semantics hold, which is the only thing that actually matters.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest

from langchain_taiga.auth.store import InMemoryStore

TEST_DSN_ENV = "TAIGA_MCP_TEST_DATABASE_URL"

_PG_TABLES = (
    "oauth_access_tokens",
    "oauth_refresh_tokens",
    "oauth_auth_codes",
    "oauth_clients",
    "oauth_revoked_families",
)


def _future(**kwargs) -> datetime:
    return datetime.now(timezone.utc) + timedelta(**kwargs)


def _past(**kwargs) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**kwargs)


@pytest.fixture
def pg_dsn():
    """The test database DSN, with sockets re-enabled for the test's duration.

    The suite runs under ``--disable-socket`` to catch accidental network
    calls; a database connection is the one deliberate exception. Skips when
    no database is configured.
    """
    import pytest_socket

    dsn = os.getenv(TEST_DSN_ENV)
    if not dsn:
        pytest.skip(f"{TEST_DSN_ENV} not set; skipping Postgres run")

    pytest_socket.enable_socket()
    try:
        yield dsn
    finally:
        pytest_socket.disable_socket(allow_unix_socket=True)


@pytest.fixture(params=["memory", "postgres"])
async def store(request, monkeypatch):
    """Yield each store implementation in turn."""
    if request.param == "memory":
        yield await InMemoryStore.from_env()
        return

    dsn = request.getfixturevalue("pg_dsn")

    from langchain_taiga.auth.postgres_store import DATABASE_URL_ENV, PostgresStore

    monkeypatch.setenv(DATABASE_URL_ENV, dsn)
    pg = await PostgresStore.from_env()
    # Start from a clean slate — these tests assert on counts returned by
    # cleanup/revoke, which leftovers from a previous test would corrupt.
    async with pg._pool.acquire() as conn:
        await conn.execute(f"TRUNCATE {', '.join(_PG_TABLES)}")
    try:
        yield pg
    finally:
        await pg.close()


async def _seed_access(store, token="at-1", *, expires_at=None, family_id=""):
    await store.store_access_token(
        token=token,
        taiga_auth_token="taiga-jwt",
        taiga_refresh_token="taiga-refresh",
        taiga_user_id=7,
        taiga_username="wahed",
        client_id="client-1",
        scopes=["taiga"],
        expires_at=expires_at or _future(hours=1),
        family_id=family_id,
    )


async def _seed_refresh(store, token="rt-1", *, family_id="fam-1", expires_at=None):
    await store.store_refresh_token(
        token=token,
        family_id=family_id,
        client_id="client-1",
        taiga_auth_token="taiga-jwt",
        taiga_refresh_token="taiga-refresh",
        taiga_user_id=7,
        taiga_username="wahed",
        scopes=["taiga"],
        expires_at=expires_at or _future(days=30),
    )


async def _seed_code(store, code="code-1", *, expires_at=None):
    await store.store_authorization_code(
        code=code,
        client_id="client-1",
        redirect_uri="https://claude.ai/cb",
        code_challenge="chal",
        code_challenge_method="S256",
        taiga_auth_token="taiga-jwt",
        taiga_refresh_token="taiga-refresh",
        taiga_user_id=7,
        taiga_username="wahed",
        scopes=["taiga"],
        expires_at=expires_at or _future(minutes=10),
    )


# --- Access tokens --------------------------------------------------------


async def test_access_token_roundtrip(store):
    await _seed_access(store)
    rec = await store.lookup_access_token("at-1")
    assert rec is not None
    assert rec.taiga_auth_token == "taiga-jwt"
    assert rec.taiga_user_id == 7
    assert rec.taiga_username == "wahed"
    assert rec.client_id == "client-1"
    assert rec.scopes == ["taiga"]


async def test_lookup_unknown_access_token_returns_none(store):
    assert await store.lookup_access_token("nope") is None


async def test_expired_access_token_not_returned(store):
    await _seed_access(store, "at-old", expires_at=_past(minutes=1))
    assert await store.lookup_access_token("at-old") is None


async def test_update_taiga_token_refreshes_in_place(store):
    await _seed_access(store)
    await store.update_taiga_token(
        token="at-1",
        taiga_auth_token="new-jwt",
        taiga_refresh_token="new-refresh",
        expires_at=_future(hours=2),
    )
    rec = await store.lookup_access_token("at-1")
    assert rec.taiga_auth_token == "new-jwt"
    assert rec.taiga_refresh_token == "new-refresh"


async def test_update_taiga_token_on_missing_row_is_noop(store):
    # InMemoryStore returns early; Postgres updates zero rows. Neither raises.
    await store.update_taiga_token(
        token="ghost",
        taiga_auth_token="x",
        taiga_refresh_token="y",
        expires_at=_future(hours=1),
    )
    assert await store.lookup_access_token("ghost") is None


# --- Authorization codes --------------------------------------------------


async def test_authorization_code_is_single_use(store):
    await _seed_code(store)
    first = await store.consume_authorization_code("code-1")
    assert first is not None
    assert first.redirect_uri == "https://claude.ai/cb"
    assert await store.consume_authorization_code("code-1") is None


async def test_peek_does_not_consume(store):
    await _seed_code(store)
    assert await store.peek_authorization_code("code-1") is not None
    assert await store.peek_authorization_code("code-1") is not None
    assert await store.consume_authorization_code("code-1") is not None


async def test_expired_code_is_rejected_and_purged(store):
    await _seed_code(store, "code-old", expires_at=_past(minutes=1))
    assert await store.peek_authorization_code("code-old") is None
    assert await store.consume_authorization_code("code-old") is None


# --- DCR ------------------------------------------------------------------


async def test_client_registration_roundtrip(store):
    await store.register_client(
        client_id="c-1",
        client_secret="s3cret",
        redirect_uris=["https://claude.ai/cb"],
        client_name="claude.ai",
        token_endpoint_auth_method="client_secret_post",
        scope="taiga",
    )
    rec = await store.lookup_client("c-1")
    assert rec is not None
    # Returning the secret is load-bearing: mcp-sdk's ClientAuthenticator
    # compares it directly. A None here silently downgrades confidential
    # clients to public.
    assert rec.client_secret == "s3cret"
    assert rec.redirect_uris == ["https://claude.ai/cb"]
    assert rec.token_endpoint_auth_method == "client_secret_post"
    assert rec.scope == "taiga"


async def test_lookup_unknown_client_returns_none(store):
    assert await store.lookup_client("nope") is None


async def test_verify_client_secret(store):
    await store.register_client(
        client_id="c-1",
        client_secret="s3cret",
        redirect_uris=["https://claude.ai/cb"],
        client_name="claude.ai",
    )
    assert await store.verify_client_secret("c-1", "s3cret") is True
    assert await store.verify_client_secret("c-1", "wrong") is False
    assert await store.verify_client_secret("ghost", "s3cret") is False


# --- Refresh tokens -------------------------------------------------------


async def test_refresh_token_roundtrip(store):
    await _seed_refresh(store)
    rec = await store.lookup_refresh_token("rt-1")
    assert rec is not None
    assert rec.family_id == "fam-1"
    assert rec.rotated_out is False


async def test_lookup_refresh_token_filters_expired(store):
    await _seed_refresh(store, "rt-old", expires_at=_past(days=1))
    assert await store.lookup_refresh_token("rt-old") is None


async def test_lookup_refresh_token_returns_rotated_out_records(store):
    """Rotated-out records must still be visible to ``load_refresh_token``.

    That is the primary reuse-detection point: it inspects ``rotated_out``
    itself and revokes the family. Filtering them out here would hide the
    replay signal entirely.
    """
    await _seed_refresh(store)
    await store.consume_refresh_token("rt-1")
    rec = await store.lookup_refresh_token("rt-1")
    assert rec is not None
    assert rec.rotated_out is True


async def test_consume_refresh_token_statuses(store):
    await _seed_refresh(store)
    first = await store.consume_refresh_token("rt-1")
    assert first.status == "active"
    assert first.record.rotated_out is True

    replay = await store.consume_refresh_token("rt-1")
    assert replay.status == "already_rotated"

    assert (await store.consume_refresh_token("ghost")).status == "not_found"

    await _seed_refresh(store, "rt-exp", expires_at=_past(days=1))
    assert (await store.consume_refresh_token("rt-exp")).status == "expired"


async def test_concurrent_consume_yields_exactly_one_active(store):
    """The core rotation invariant, under concurrency.

    ``InMemoryStore`` gets this from asyncio's single-threaded execution;
    ``PostgresStore`` from a ``SELECT ... FOR UPDATE`` row lock. If more than
    one caller ever saw ``active``, two valid refresh-token families would
    exist at once and reuse-detection would stop meaning anything.
    """
    await _seed_refresh(store)
    results = await asyncio.gather(
        *(store.consume_refresh_token("rt-1") for _ in range(8))
    )
    statuses = [r.status for r in results]
    assert statuses.count("active") == 1
    assert statuses.count("already_rotated") == 7


# --- Family revocation ----------------------------------------------------


async def test_revoke_family_purges_only_its_own_family(store):
    """Scoping matters as much as the purge itself.

    A revoke that dropped its WHERE clause would still pass a test that only
    checks the target family disappeared — while logging out every user on
    the server. The unrelated-family record is the assertion that catches it.
    """
    await _seed_access(store, "at-f", family_id="fam-1")
    await _seed_refresh(store, "rt-f", family_id="fam-1")
    await _seed_access(store, "at-other", family_id="fam-other")
    await _seed_refresh(store, "rt-other", family_id="fam-other")

    purged = await store.revoke_token_family("fam-1")
    assert purged == 2
    assert await store.lookup_access_token("at-f") is None
    assert await store.lookup_refresh_token("rt-f") is None

    assert await store.lookup_access_token("at-other") is not None
    assert await store.lookup_refresh_token("rt-other") is not None


async def test_revoke_empty_family_id_is_refused(store):
    """Legacy (pre-2.5.0) records carry ``family_id=""``.

    A bare ``revoke_token_family("")`` would mass-purge every one of them —
    a silent org-wide logout. Both backends must short-circuit.
    """
    await _seed_access(store, "at-legacy", family_id="")
    assert await store.revoke_token_family("") == 0
    assert await store.lookup_access_token("at-legacy") is not None


async def test_issue_new_generation_persists_pair(store):
    ok = await store.issue_new_generation(
        family_id="fam-2",
        access_token="at-new",
        refresh_token="rt-new",
        taiga_auth_token="jwt",
        taiga_refresh_token="tr",
        taiga_user_id=7,
        taiga_username="wahed",
        client_id="client-1",
        access_scopes=["taiga"],
        refresh_scopes=["taiga"],
        access_expires_at=_future(hours=1),
        refresh_expires_at=_future(days=30),
    )
    assert ok is True
    assert await store.lookup_access_token("at-new") is not None
    assert await store.lookup_refresh_token("rt-new") is not None


async def test_issue_new_generation_refused_after_revoke(store):
    """A coroutine suspended inside the Taiga refresh cascade must not be able
    to resurrect a family that was revoked while it was waiting."""
    await store.revoke_token_family("fam-3")
    ok = await store.issue_new_generation(
        family_id="fam-3",
        access_token="at-zombie",
        refresh_token="rt-zombie",
        taiga_auth_token="jwt",
        taiga_refresh_token="tr",
        taiga_user_id=7,
        taiga_username="wahed",
        client_id="client-1",
        access_scopes=["taiga"],
        refresh_scopes=["taiga"],
        access_expires_at=_future(hours=1),
        refresh_expires_at=_future(days=30),
    )
    assert ok is False
    assert await store.lookup_access_token("at-zombie") is None
    assert await store.lookup_refresh_token("rt-zombie") is None


# --- Cleanup --------------------------------------------------------------


async def test_cleanup_expired_purges_and_counts(store):
    await _seed_access(store, "at-live")
    await _seed_access(store, "at-dead", expires_at=_past(hours=1))
    await _seed_refresh(store, "rt-dead", expires_at=_past(days=1))
    await _seed_code(store, "code-dead", expires_at=_past(minutes=30))

    purged = await store.cleanup_expired()
    assert purged == 3
    assert await store.lookup_access_token("at-live") is not None


async def test_cleanup_keeps_recent_tombstones(store):
    """Tombstones younger than REFRESH_TOKEN_TTL still guard reuse-detection."""
    await store.revoke_token_family("fam-4")
    await store.cleanup_expired()
    ok = await store.issue_new_generation(
        family_id="fam-4",
        access_token="at-x",
        refresh_token="rt-x",
        taiga_auth_token="jwt",
        taiga_refresh_token="tr",
        taiga_user_id=7,
        taiga_username="wahed",
        client_id="client-1",
        access_scopes=["taiga"],
        refresh_scopes=["taiga"],
        access_expires_at=_future(hours=1),
        refresh_expires_at=_future(days=30),
    )
    assert ok is False


# --- Durability (Postgres only) -------------------------------------------


async def test_state_survives_a_process_restart(store, monkeypatch):
    """The whole point of the change: a new process must see existing state.

    Simulates a pod restart by dropping the store and building a fresh one
    over the same DSN. On the in-memory backend this is inherently impossible,
    so the assertion is inverted there — documenting exactly the behaviour
    that was breaking unattended agents.
    """
    await _seed_access(store, "at-survivor")
    await store.register_client(
        client_id="c-survivor",
        client_secret="s3cret",
        redirect_uris=["https://claude.ai/cb"],
        client_name="claude.ai",
    )

    if isinstance(store, InMemoryStore):
        reborn = await InMemoryStore.from_env()
        assert await reborn.lookup_access_token("at-survivor") is None
        assert await reborn.lookup_client("c-survivor") is None
        return

    from langchain_taiga.auth.postgres_store import PostgresStore

    reborn = await PostgresStore.from_env()
    try:
        assert await reborn.lookup_access_token("at-survivor") is not None
        assert await reborn.lookup_client("c-survivor") is not None
    finally:
        await reborn.close()


# --- Backend selection ----------------------------------------------------


def _clear_backend_env(monkeypatch):
    from langchain_taiga import remote_server as rs
    from langchain_taiga.auth import postgres_store as ps

    for var in (
        rs.STATE_BACKEND_ENV,
        ps.DATABASE_URL_ENV,
        ps.PG_HOST_ENV,
        ps.PG_USER_ENV,
        ps.PG_DATABASE_ENV,
    ):
        monkeypatch.delenv(var, raising=False)


async def test_backend_defaults_to_memory(monkeypatch):
    from langchain_taiga import remote_server as rs

    _clear_backend_env(monkeypatch)
    assert isinstance(await rs._build_store(), InMemoryStore)


async def test_postgres_backend_without_connection_config_raises(monkeypatch):
    """The regression guard: asking for durable state and not getting it must
    stop the pod, not boot green on the in-memory store."""
    from langchain_taiga import remote_server as rs

    _clear_backend_env(monkeypatch)
    monkeypatch.setenv(rs.STATE_BACKEND_ENV, "postgres")

    with pytest.raises(RuntimeError, match="no connection configuration"):
        await rs._build_store()


async def test_unknown_backend_raises(monkeypatch):
    from langchain_taiga import remote_server as rs

    _clear_backend_env(monkeypatch)
    monkeypatch.setenv(rs.STATE_BACKEND_ENV, "postgress")

    with pytest.raises(RuntimeError, match="not a valid backend"):
        await rs._build_store()


async def test_connection_vars_without_backend_switch_warns(monkeypatch, caplog):
    """Half-applied config: connection details present, switch absent.

    Falling back is correct here (the operator never asked for Postgres), but
    it must be loud — this is what a partially-reverted chart looks like.
    """
    from langchain_taiga import remote_server as rs
    from langchain_taiga.auth import postgres_store as ps

    _clear_backend_env(monkeypatch)
    monkeypatch.setenv(ps.PG_HOST_ENV, "taiga-postgresql")

    with caplog.at_level("WARNING"):
        store = await rs._build_store()

    assert isinstance(store, InMemoryStore)
    assert rs.STATE_BACKEND_ENV in caplog.text


def test_postgres_configured_reads_either_form(monkeypatch):
    from langchain_taiga.auth import postgres_store as ps

    for var in (
        ps.DATABASE_URL_ENV,
        ps.PG_HOST_ENV,
        ps.PG_USER_ENV,
        ps.PG_DATABASE_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    assert ps.postgres_configured() is False

    monkeypatch.setenv(ps.DATABASE_URL_ENV, "postgres://u@h/db")
    assert ps.postgres_configured() is True

    monkeypatch.delenv(ps.DATABASE_URL_ENV)
    monkeypatch.setenv(ps.PG_HOST_ENV, "taiga-postgresql")
    assert ps.postgres_configured() is True


def test_discrete_connect_kwargs(monkeypatch):
    """The chart passes credentials as parts precisely so a password
    containing ``@`` or ``/`` cannot corrupt the connection target."""
    from langchain_taiga.auth import postgres_store as ps

    monkeypatch.setenv(ps.PG_HOST_ENV, "taiga-postgresql")
    monkeypatch.setenv(ps.PG_USER_ENV, "taiga")
    monkeypatch.setenv(ps.PG_DATABASE_ENV, "taiga_mcp")
    monkeypatch.setenv(ps.PG_PASSWORD_ENV, "p@ss/w:rd?#")
    monkeypatch.delenv(ps.PG_PORT_ENV, raising=False)

    kwargs = ps._connect_kwargs_from_env()
    assert kwargs == {
        "host": "taiga-postgresql",
        "port": 5432,
        "user": "taiga",
        "password": "p@ss/w:rd?#",
        "database": "taiga_mcp",
    }


def test_schema_name_must_be_a_bare_identifier(monkeypatch):
    """The schema is interpolated into DDL, which asyncpg cannot parameterise."""
    from langchain_taiga.auth import postgres_store as ps

    monkeypatch.setenv(ps.PG_SCHEMA_ENV, "public; DROP TABLE users--")
    with pytest.raises(RuntimeError, match="not a valid identifier"):
        ps._schema_from_env()

    monkeypatch.delenv(ps.PG_SCHEMA_ENV)
    assert ps._schema_from_env() == ps.DEFAULT_SCHEMA


async def test_bootstraps_into_a_fresh_schema(pg_dsn, monkeypatch):
    """First boot must create its own schema, not require a manual step.

    Also pins the isolation property: the tables must NOT land in ``public``,
    where Taiga's own Django introspection would see them.
    """
    from langchain_taiga.auth.postgres_store import DATABASE_URL_ENV, PostgresStore

    schema = "mcp_oauth_freshboot"
    monkeypatch.setenv(DATABASE_URL_ENV, pg_dsn)
    monkeypatch.setenv("TAIGA_MCP_PG_SCHEMA", schema)

    bootstrap = await PostgresStore.from_env()
    try:
        async with bootstrap._pool.acquire() as conn:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    finally:
        await bootstrap.close()

    store = await PostgresStore.from_env()
    try:
        await _seed_access(store, "at-fresh")
        assert await store.lookup_access_token("at-fresh") is not None

        async with store._pool.acquire() as conn:
            # Qualify by schema: the default-schema tables from the other
            # tests in this module exist concurrently, so an unqualified
            # tablename lookup is ambiguous.
            in_ours = await conn.fetchval(
                "SELECT 1 FROM pg_tables WHERE tablename = $1 AND schemaname = $2",
                "oauth_access_tokens",
                schema,
            )
            assert in_ours == 1
            in_public = await conn.fetchval(
                "SELECT 1 FROM pg_tables "
                "WHERE tablename = $1 AND schemaname = 'public'",
                "oauth_access_tokens",
            )
            assert in_public is None, "tables must stay out of public schema"
            await conn.execute(f"DROP SCHEMA {schema} CASCADE")
    finally:
        await store.close()


async def test_boots_as_least_privilege_role_owning_only_its_schema(
    pg_dsn, monkeypatch
):
    """The production deployment path: a dedicated role with NO database-level
    CREATE, owning a pre-created schema.

    Regression guard. ``CREATE SCHEMA IF NOT EXISTS`` is NOT a no-op for such a
    role — Postgres checks CREATE on the *database* before the IF NOT EXISTS
    short-circuit and raises 42501 even though the schema is already there. An
    earlier version of ``_ensure_schema`` called it unconditionally and would
    have crash-looped the pod on every boot.
    """
    import asyncpg

    from langchain_taiga.auth.postgres_store import PostgresStore

    schema = "mcp_oauth_leastpriv"
    role = "taiga_mcp_leastpriv"
    admin = await asyncpg.connect(pg_dsn)
    dbname = await admin.fetchval("SELECT current_database()")
    # Capture the pre-test ACL: restoring unconditionally would GRANT CREATE to
    # PUBLIC on a database where it was never granted, permanently loosening a
    # long-lived local Postgres.
    public_had_create = await admin.fetchval(
        "SELECT has_database_privilege('public', $1, 'CREATE')", dbname
    )
    host_part = pg_dsn.rsplit("@", 1)[1]
    try:
        await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await admin.execute(f"DROP ROLE IF EXISTS {role}")
        # Exactly the two statements the deployment docs tell operators to run.
        await admin.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'pw'")
        await admin.execute(f"CREATE SCHEMA {schema} AUTHORIZATION {role}")
        await admin.execute(f'REVOKE CREATE ON DATABASE "{dbname}" FROM PUBLIC')
    finally:
        await admin.close()

    try:
        monkeypatch.delenv("TAIGA_MCP_DATABASE_URL", raising=False)
        monkeypatch.setenv("TAIGA_MCP_PG_HOST", host_part.split(":")[0])
        monkeypatch.setenv("TAIGA_MCP_PG_PORT", host_part.split(":")[1].split("/")[0])
        monkeypatch.setenv("TAIGA_MCP_PG_USER", role)
        monkeypatch.setenv("TAIGA_MCP_PG_PASSWORD", "pw")
        monkeypatch.setenv("TAIGA_MCP_PG_DATABASE", dbname)
        monkeypatch.setenv("TAIGA_MCP_PG_SCHEMA", schema)

        store = await PostgresStore.from_env()
        try:
            async with store._pool.acquire() as conn:
                assert await conn.fetchval(
                    "SELECT has_database_privilege(current_user, $1, 'CREATE')",
                    dbname,
                ) is False, "test setup: role must NOT have database CREATE"
                assert (
                    await conn.fetchval(
                        "SELECT schemaname FROM pg_tables "
                        "WHERE tablename = 'oauth_clients' AND schemaname = $1",
                        schema,
                    )
                    == schema
                )
            await _seed_access(store, "at-leastpriv")
            assert await store.lookup_access_token("at-leastpriv") is not None
        finally:
            await store.close()
    finally:
        admin = await asyncpg.connect(pg_dsn)
        try:
            await admin.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            await admin.execute(f"DROP ROLE IF EXISTS {role}")
            if public_had_create:
                await admin.execute(
                    f'GRANT CREATE ON DATABASE "{dbname}" TO PUBLIC'
                )
        finally:
            await admin.close()


async def test_missing_create_privilege_explains_the_fix(pg_dsn, monkeypatch):
    """A pod that can't create its schema must say why, not just 'permission denied'.

    Uses a real unprivileged role, because the whole point is that we map
    Postgres' SQLSTATE 42501 correctly.
    """
    import asyncpg

    from langchain_taiga.auth.postgres_store import DATABASE_URL_ENV, PostgresStore

    admin = await asyncpg.connect(pg_dsn)
    dbname = await admin.fetchval("SELECT current_database()")
    public_had_create = await admin.fetchval(
        "SELECT has_database_privilege('public', $1, 'CREATE')", dbname
    )
    try:
        await admin.execute("DROP ROLE IF EXISTS mcp_nopriv")
        await admin.execute(
            "CREATE ROLE mcp_nopriv LOGIN PASSWORD 'pw' NOCREATEDB NOSUPERUSER"
        )
        await admin.execute(f'REVOKE CREATE ON DATABASE "{dbname}" FROM PUBLIC')
    finally:
        await admin.close()

    try:
        unprivileged = f"postgres://mcp_nopriv:pw@{pg_dsn.rsplit('@', 1)[1]}"
        monkeypatch.setenv(DATABASE_URL_ENV, unprivileged)
        monkeypatch.setenv("TAIGA_MCP_PG_SCHEMA", "mcp_oauth_denied")

        with pytest.raises(RuntimeError, match="GRANT CREATE ON DATABASE"):
            await PostgresStore.from_env()
    finally:
        # Restore BOTH the role and the grant. Dropping the GRANT and walking
        # away is free in a throwaway CI container but permanent in the
        # long-lived local Postgres that AGENTS.md tells you to run.
        admin = await asyncpg.connect(pg_dsn)
        try:
            await admin.execute("DROP ROLE IF EXISTS mcp_nopriv")
            if public_had_create:
                await admin.execute(
                    f'GRANT CREATE ON DATABASE "{dbname}" TO PUBLIC'
                )
        finally:
            await admin.close()


def test_discrete_connect_kwargs_demands_user_and_database(monkeypatch):
    from langchain_taiga.auth import postgres_store as ps

    monkeypatch.setenv(ps.PG_HOST_ENV, "taiga-postgresql")
    monkeypatch.delenv(ps.PG_USER_ENV, raising=False)
    monkeypatch.delenv(ps.PG_DATABASE_ENV, raising=False)

    # Fail loudly at startup rather than connecting to an implicit database
    # named after the OS user, which would silently start with an empty store.
    with pytest.raises(RuntimeError, match="are required too"):
        ps._connect_kwargs_from_env()
