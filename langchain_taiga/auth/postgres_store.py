"""Postgres-backed OAuth state store for the Taiga MCP bridge.

Drop-in replacement for :class:`~langchain_taiga.auth.store.InMemoryStore`,
implementing the identical async interface over an ``asyncpg`` pool.

Why this exists
---------------
``InMemoryStore`` keeps DCR client registrations, access tokens and refresh
tokens in process-local dicts. Every pod start — a deploy *or* a liveness-probe
kill — wipes them, so claude.ai's stored ``client_id`` and refresh token both
become unknown and the connector demands a fresh browser login. Unattended
agents ("loops") die at that point because a human has to click through the
OAuth consent again.

Persisting the same state to Postgres means a restart is invisible to clients:
the access token still resolves, and once it expires the refresh token mints a
new generation without user interaction.

Atomicity
---------
``InMemoryStore`` gets its atomicity for free from asyncio's single-threaded
execution — its comments call out that each critical section contains no
``await`` between read and mutation. Those same sections are re-expressed here
as SQL transactions, per the note in ``store.py``:

    "For a future Redis / Postgres backend, this method must be implemented
    as a Lua script or row-locked transaction to preserve the semantics."

Concretely:

- ``consume_refresh_token`` — ``SELECT ... FOR UPDATE`` then a conditional
  ``UPDATE`` inside one transaction, so exactly one caller observes ``active``
  and the rest observe ``already_rotated`` (the reuse-detection signal).
- ``consume_authorization_code`` — ``DELETE ... RETURNING`` is atomic by
  itself, giving the single-use pop.
- ``revoke_token_family`` / ``issue_new_generation`` — both take the same
  per-family transaction-scoped advisory lock, which is the faithful
  translation of "no intervening awaits between the tombstone check and the
  dict writes". Without it a coroutine suspended inside the Taiga refresh
  cascade could resurrect a family that was just revoked.

Placement
---------
Tables live in their own schema (``mcp_oauth`` by default, see
``TAIGA_MCP_PG_SCHEMA``) inside an existing database rather than a database of
their own: a separate database would need a CREATEDB grant and would fall
outside the host application's ``pg_dump`` backup. Staying out of ``public``
also keeps these tables invisible to the host application's own migrations and
introspection.

In production the schema is pre-created and owned by a dedicated
least-privilege role, so the bridge holds no rights over anything else in the
database. Note that ``CREATE SCHEMA IF NOT EXISTS`` is **not** a no-op for such
a role when the schema exists — Postgres checks CREATE on the *database*
before the IF-NOT-EXISTS short-circuit and raises 42501 anyway — hence the
``pg_namespace`` probe in ``_ensure_schema`` before attempting creation.
Where the connecting role does own the database, first boot self-provisions.

Selection
---------
This backend is not chosen by the presence of these variables — it is chosen
explicitly by ``TAIGA_MCP_STATE_BACKEND=postgres`` (see
``remote_server._build_store``), so a dropped connection variable fails the
pod at startup instead of silently reverting production to the in-memory
store. Production supplies the discrete ``TAIGA_MCP_PG_*`` variables via the
Helm chart; the ``TAIGA_MCP_DATABASE_URL`` DSN form is what CI and local
tests use.

Security note
-------------
Unlike the in-memory store, this backend persists ``taiga_auth_token``,
``taiga_refresh_token`` and DCR ``client_secret`` values **at rest**, in
plaintext columns. This is an accepted risk, not an oversight: the same
database already holds Django session keys and password hashes for the same
users, so anyone able to read it has the application anyway — encrypting only
our four tables would move no real trust boundary while adding a key to
manage, rotate and migrate.

What does the work instead is scoping: the bridge connects as a dedicated role
that owns nothing but its own schema, so a foothold in this pod cannot reach
the host application's tables. Keep it that way — do not point this at the
database owner.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Optional

import asyncpg

from langchain_taiga.auth.store import (
    REFRESH_TOKEN_TTL,
    AccessTokenRecord,
    AuthCodeRecord,
    ClientRecord,
    ConsumeRefreshResult,
    RefreshTokenRecord,
)

_log = logging.getLogger(__name__)

#: Environment variable holding a full libpq/asyncpg DSN.
DATABASE_URL_ENV = "TAIGA_MCP_DATABASE_URL"

#: Discrete connection parameters, used when no DSN is given. These exist
#: because a DSN has to percent-encode any ``@ : / ? #`` in the password, and
#: the Helm chart assembles credentials from a Kubernetes Secret it does not
#: control the charset of. Passing the parts separately removes that whole
#: class of "works until someone rotates the password" breakage.
PG_HOST_ENV = "TAIGA_MCP_PG_HOST"
PG_PORT_ENV = "TAIGA_MCP_PG_PORT"
PG_USER_ENV = "TAIGA_MCP_PG_USER"
PG_PASSWORD_ENV = "TAIGA_MCP_PG_PASSWORD"
PG_DATABASE_ENV = "TAIGA_MCP_PG_DATABASE"

#: Dedicated schema for our tables. Defaulting to a non-``public`` schema lets
#: the bridge share the host application's database without its tables ever
#: appearing to that application's introspection or colliding with its
#: migrations. See the "Placement" section of the module docstring for why a
#: schema rather than a separate database, and for the ownership requirements.
PG_SCHEMA_ENV = "TAIGA_MCP_PG_SCHEMA"
DEFAULT_SCHEMA = "mcp_oauth"

#: Schema names are interpolated into DDL, which asyncpg cannot parameterise.
#: Restricting them to bare lowercase identifiers keeps that safe without
#: quoting gymnastics.
_SCHEMA_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _schema_from_env() -> str:
    schema = os.getenv(PG_SCHEMA_ENV) or DEFAULT_SCHEMA
    if not _SCHEMA_NAME_RE.match(schema):
        raise RuntimeError(
            f"{PG_SCHEMA_ENV}={schema!r} is not a valid identifier "
            "(expected ^[a-z_][a-z0-9_]*$)."
        )
    return schema


def postgres_configured() -> bool:
    """True when enough connection settings are present to build a pool.

    This reports **configuration presence only — it does not select the
    backend.** ``remote_server._build_store`` selects independently, via
    ``TAIGA_MCP_STATE_BACKEND``, and uses this for two different things: to
    reject ``STATE_BACKEND=postgres`` that has nothing to connect to, and to
    warn when connection settings are set but the backend was never asked for
    (a half-applied config). Conflating the two is exactly the inference this
    design avoids.
    """
    return bool(os.getenv(DATABASE_URL_ENV) or os.getenv(PG_HOST_ENV))


def _connect_kwargs_from_env() -> dict:
    """Build ``asyncpg.create_pool`` kwargs from the discrete env vars."""
    host = os.getenv(PG_HOST_ENV)
    if not host:
        raise RuntimeError(
            f"Neither {DATABASE_URL_ENV} nor {PG_HOST_ENV} is set; "
            "cannot build a PostgresStore."
        )
    database = os.getenv(PG_DATABASE_ENV)
    user = os.getenv(PG_USER_ENV)
    if not database or not user:
        raise RuntimeError(
            f"{PG_HOST_ENV} is set, so {PG_DATABASE_ENV} and {PG_USER_ENV} "
            "are required too."
        )
    return {
        "host": host,
        "port": int(os.getenv(PG_PORT_ENV) or 5432),
        "user": user,
        "password": os.getenv(PG_PASSWORD_ENV) or None,
        "database": database,
    }

#: Both writers of each table share one statement. They used to be inlined
#: twice with *different* ``DO UPDATE`` column lists (8 columns in
#: ``store_access_token``, 4 in ``issue_new_generation``) — same table, same
#: conflict target, no reason for the divergence. Because the primary keys are
#: freshly minted ``secrets.token_urlsafe`` values the conflict branch
#: effectively never fires, so the mismatch was untestable and would never have
#: surfaced as a failure; it would just have quietly gone stale. The upsert
#: (rather than a plain INSERT) is what keeps parity with ``InMemoryStore``,
#: whose dict assignment overwrites unconditionally.
_UPSERT_ACCESS_TOKEN = """
    INSERT INTO oauth_access_tokens (
        token, taiga_auth_token, taiga_refresh_token, taiga_user_id,
        taiga_username, client_id, scopes, expires_at, family_id
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    ON CONFLICT (token) DO UPDATE SET
        taiga_auth_token = EXCLUDED.taiga_auth_token,
        taiga_refresh_token = EXCLUDED.taiga_refresh_token,
        taiga_user_id = EXCLUDED.taiga_user_id,
        taiga_username = EXCLUDED.taiga_username,
        client_id = EXCLUDED.client_id,
        scopes = EXCLUDED.scopes,
        expires_at = EXCLUDED.expires_at,
        family_id = EXCLUDED.family_id
"""

_UPSERT_REFRESH_TOKEN = """
    INSERT INTO oauth_refresh_tokens (
        token, family_id, client_id, taiga_auth_token, taiga_refresh_token,
        taiga_user_id, taiga_username, scopes, expires_at, rotated_out
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, FALSE)
    ON CONFLICT (token) DO UPDATE SET
        family_id = EXCLUDED.family_id,
        client_id = EXCLUDED.client_id,
        taiga_auth_token = EXCLUDED.taiga_auth_token,
        taiga_refresh_token = EXCLUDED.taiga_refresh_token,
        taiga_user_id = EXCLUDED.taiga_user_id,
        taiga_username = EXCLUDED.taiga_username,
        scopes = EXCLUDED.scopes,
        expires_at = EXCLUDED.expires_at,
        rotated_out = FALSE
"""

# APPEND-ONLY. ``CREATE TABLE IF NOT EXISTS`` is a no-op against a table that
# already exists, so EDITING a CREATE below does nothing in an environment
# that has already booted once — the app then fails at runtime on the token
# path, with no version marker to say what is actually live. To add a column,
# APPEND an ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` statement to the end
# of this string instead. Real migration tooling isn't warranted until there
# is a second writer or more than one replica.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id                 TEXT PRIMARY KEY,
    client_secret             TEXT,
    redirect_uris             TEXT[] NOT NULL,
    client_name               TEXT NOT NULL,
    token_endpoint_auth_method TEXT NOT NULL,
    scope                     TEXT
);

CREATE TABLE IF NOT EXISTS oauth_access_tokens (
    token              TEXT PRIMARY KEY,
    taiga_auth_token   TEXT NOT NULL,
    taiga_refresh_token TEXT NOT NULL,
    taiga_user_id      BIGINT NOT NULL,
    taiga_username     TEXT NOT NULL,
    client_id          TEXT NOT NULL,
    scopes             TEXT[] NOT NULL,
    expires_at         TIMESTAMPTZ NOT NULL,
    family_id          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS oauth_access_tokens_expires_at_idx
    ON oauth_access_tokens (expires_at);
CREATE INDEX IF NOT EXISTS oauth_access_tokens_family_id_idx
    ON oauth_access_tokens (family_id);

CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
    token              TEXT PRIMARY KEY,
    family_id          TEXT NOT NULL,
    client_id          TEXT NOT NULL,
    taiga_auth_token   TEXT NOT NULL,
    taiga_refresh_token TEXT NOT NULL,
    taiga_user_id      BIGINT NOT NULL,
    taiga_username     TEXT NOT NULL,
    scopes             TEXT[] NOT NULL,
    expires_at         TIMESTAMPTZ NOT NULL,
    rotated_out        BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS oauth_refresh_tokens_expires_at_idx
    ON oauth_refresh_tokens (expires_at);
CREATE INDEX IF NOT EXISTS oauth_refresh_tokens_family_id_idx
    ON oauth_refresh_tokens (family_id);

CREATE TABLE IF NOT EXISTS oauth_auth_codes (
    code                 TEXT PRIMARY KEY,
    client_id            TEXT NOT NULL,
    redirect_uri         TEXT NOT NULL,
    code_challenge       TEXT NOT NULL,
    code_challenge_method TEXT NOT NULL,
    taiga_auth_token     TEXT NOT NULL,
    taiga_refresh_token  TEXT NOT NULL,
    taiga_user_id        BIGINT NOT NULL,
    taiga_username       TEXT NOT NULL,
    scopes               TEXT[] NOT NULL,
    expires_at           TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS oauth_auth_codes_expires_at_idx
    ON oauth_auth_codes (expires_at);

CREATE TABLE IF NOT EXISTS oauth_revoked_families (
    family_id  TEXT PRIMARY KEY,
    revoked_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS oauth_revoked_families_revoked_at_idx
    ON oauth_revoked_families (revoked_at);
"""


def _family_lock_key(family_id: str) -> int:
    """Map a family_id to a stable signed 64-bit advisory-lock key.

    Postgres' own ``hashtext()`` would do, but it is an undocumented internal
    with no cross-version stability guarantee. A blake2b digest is explicit and
    portable. Collisions merely serialise two unrelated families for the
    duration of one transaction, which is harmless.
    """
    digest = hashlib.blake2b(family_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


class PostgresStore:
    """Durable OAuth state. Survives pod restarts, deploys and rescheduling.

    Mirrors :class:`~langchain_taiga.auth.store.InMemoryStore` method for
    method — including the expiry-filtering semantics, the ``rotated_out``
    passthrough in ``lookup_refresh_token`` and the empty-``family_id`` guard
    in ``revoke_token_family`` — so the two are interchangeable behind
    ``_bootstrap_provider``.
    """

    def __init__(self, pool: "asyncpg.Pool", schema: str = DEFAULT_SCHEMA) -> None:
        self._pool = pool
        self._schema = schema

    @classmethod
    async def from_env(cls) -> "PostgresStore":
        """Connect from the environment and ensure the schema exists.

        Prefers a full DSN in ``TAIGA_MCP_DATABASE_URL``; otherwise assembles
        the connection from the discrete ``TAIGA_MCP_PG_*`` vars, which is
        what the Helm chart uses (no password escaping to get wrong).

        Matches ``InMemoryStore.from_env()`` so ``_bootstrap_provider`` can
        await either one.
        """
        schema = _schema_from_env()
        dsn = os.getenv(DATABASE_URL_ENV)
        pool = await asyncpg.create_pool(
            dsn,
            min_size=1,
            max_size=10,
            # ``min_size`` alone does NOT keep a connection warm: asyncpg arms a
            # per-holder timer from ``max_inactive_connection_lifetime``
            # (default 300s) and terminates idle connections with no min_size
            # floor. A connector idle overnight would then hold zero backends
            # and the next request — ``lookup_access_token``, the per-request
            # hot path — would pay TCP + auth + search_path + statement re-Parse.
            # 0 disables the timer entirely (the value is only checked for
            # truthiness before the timer is scheduled).
            max_inactive_connection_lifetime=0,
            # Pins search_path so every statement can use bare table names
            # while still landing in our own schema.
            server_settings={"search_path": schema},
            # Discrete host/user/password/database, only when no DSN was given.
            **({} if dsn else _connect_kwargs_from_env()),
        )
        store = cls(pool, schema=schema)
        try:
            await store._ensure_schema()
        except BaseException:
            # Don't strand the pool we just opened. The missing-privilege path
            # below raises by design, and a caller that catches the startup
            # error would otherwise hold open connections for the life of the
            # process.
            await pool.close()
            raise
        _log.info("PostgresStore ready (schema %r ensured)", schema)
        return store

    async def _ensure_schema(self) -> None:
        async with self._pool.acquire() as conn:
            # Must precede the table DDL: with search_path pointing at a
            # schema that doesn't exist yet, CREATE TABLE fails with
            # "no schema has been selected to create in".
            # Look before leaping: ``CREATE SCHEMA IF NOT EXISTS`` is NOT a
            # no-op for an unprivileged role when the schema already exists.
            # Postgres checks CREATE on the *database* before the IF NOT EXISTS
            # short-circuit, so it raises 42501 regardless. That would
            # crash-loop the production deployment, where the schema is
            # pre-created and owned by a role that deliberately has no
            # database-level CREATE. Verified against Postgres 16; covered by
            # test_boots_as_least_privilege_role_owning_only_its_schema.
            already_exists = await conn.fetchval(
                "SELECT 1 FROM pg_namespace WHERE nspname = $1", self._schema
            )
            try:
                if not already_exists:
                    await conn.execute(f"CREATE SCHEMA {self._schema}")
            except Exception as exc:  # asyncpg.InsufficientPrivilegeError & friends
                if getattr(exc, "sqlstate", None) != "42501":
                    raise
                # Creating a schema needs CREATE on the database, which the
                # database owner has. Spell out the fix rather than leaving a
                # bare "permission denied" in a crash-looping pod.
                raise RuntimeError(
                    f"Not allowed to create schema {self._schema!r}. The "
                    "configured user needs CREATE on the database (the owner "
                    "has it by default). Either grant it:\n"
                    f'  GRANT CREATE ON DATABASE "<db>" TO "<user>";\n'
                    "or pre-create the schema and hand it over:\n"
                    f'  CREATE SCHEMA {self._schema}; '
                    f'ALTER SCHEMA {self._schema} OWNER TO "<user>";'
                ) from exc
            await conn.execute(_SCHEMA)

    # --- Row mapping ---------------------------------------------------
    #
    # Every column in ``_SCHEMA`` is named exactly like its dataclass field and
    # every read is ``SELECT *``, so unpacking the Record straight into the
    # dataclass is equivalent to four hand-written mappers — minus the fourth
    # place to forget when a field is added. Drift still fails loudly, with
    # ``TypeError: unexpected keyword argument``, instead of silently
    # returning a half-populated record on the token path.

    @staticmethod
    def _row_to(record_cls, row):
        return record_cls(**row) if row is not None else None

    # --- Access tokens ----------------------------------------------------

    async def store_access_token(
        self,
        *,
        token: str,
        taiga_auth_token: str,
        taiga_refresh_token: str,
        taiga_user_id: int,
        taiga_username: str,
        client_id: str,
        scopes: List[str],
        expires_at: datetime,
        family_id: str = "",
    ) -> None:
        await self._pool.execute(
            _UPSERT_ACCESS_TOKEN,
            token,
            taiga_auth_token,
            taiga_refresh_token,
            taiga_user_id,
            taiga_username,
            client_id,
            list(scopes),
            expires_at,
            family_id,
        )

    async def lookup_access_token(self, token: str) -> Optional[AccessTokenRecord]:
        # Expiry filtered in SQL, matching InMemoryStore's defensive check that
        # never returns a record the cleanup loop hasn't swept yet.
        row = await self._pool.fetchrow(
            "SELECT * FROM oauth_access_tokens "
            "WHERE token = $1 AND expires_at >= now()",
            token,
        )
        return self._row_to(AccessTokenRecord, row)

    async def update_taiga_token(
        self,
        *,
        token: str,
        taiga_auth_token: str,
        taiga_refresh_token: str,
        expires_at: datetime,
    ) -> None:
        # No-op when the row is gone, matching InMemoryStore's early return.
        await self._pool.execute(
            """
            UPDATE oauth_access_tokens
               SET taiga_auth_token = $2,
                   taiga_refresh_token = $3,
                   expires_at = $4
             WHERE token = $1
            """,
            token,
            taiga_auth_token,
            taiga_refresh_token,
            expires_at,
        )

    # --- Refresh tokens ---------------------------------------------------

    async def store_refresh_token(
        self,
        *,
        token: str,
        family_id: str,
        client_id: str,
        taiga_auth_token: str,
        taiga_refresh_token: str,
        taiga_user_id: int,
        taiga_username: str,
        scopes: List[str],
        expires_at: datetime,
    ) -> None:
        await self._pool.execute(
            _UPSERT_REFRESH_TOKEN,
            token,
            family_id,
            client_id,
            taiga_auth_token,
            taiga_refresh_token,
            taiga_user_id,
            taiga_username,
            list(scopes),
            expires_at,
        )

    async def lookup_refresh_token(self, token: str) -> Optional[RefreshTokenRecord]:
        """Return the record only if it exists AND is not expired.

        Deliberately returns ``rotated_out`` records —
        ``TaigaOAuthProvider.load_refresh_token`` inspects that flag itself and
        revokes the family on the spot. Filtering it here would hide the replay
        signal from the primary reuse-detection path.
        """
        row = await self._pool.fetchrow(
            "SELECT * FROM oauth_refresh_tokens "
            "WHERE token = $1 AND expires_at >= now()",
            token,
        )
        return self._row_to(RefreshTokenRecord, row)

    async def consume_refresh_token(self, token: str) -> ConsumeRefreshResult:
        """Atomic lookup + ``rotated_out`` transition.

        The in-memory version relies on there being no ``await`` between the
        read and the mutation. Here the same guarantee comes from holding a row
        lock for the whole transaction: concurrent callers serialise on
        ``FOR UPDATE``, so exactly one sees ``active``.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    "SELECT * FROM oauth_refresh_tokens "
                    "WHERE token = $1 FOR UPDATE",
                    token,
                )
                if row is None:
                    return ConsumeRefreshResult(status="not_found", record=None)
                record = self._row_to(RefreshTokenRecord, row)
                # Expiry is checked before rotated_out, matching InMemoryStore.
                if record.expires_at < datetime.now(timezone.utc):
                    return ConsumeRefreshResult(status="expired", record=record)
                if record.rotated_out:
                    return ConsumeRefreshResult(
                        status="already_rotated", record=record
                    )
                await conn.execute(
                    "UPDATE oauth_refresh_tokens SET rotated_out = TRUE "
                    "WHERE token = $1",
                    token,
                )
                record.rotated_out = True
                return ConsumeRefreshResult(status="active", record=record)

    async def revoke_token_family(self, family_id: str) -> int:
        """Delete every access and refresh token sharing this family_id.

        Returns the total number of records purged.
        """
        # Same guard as InMemoryStore: AccessTokenRecord.family_id defaults to
        # "" for pre-2.5.0 records, so a bare revoke_token_family("") would
        # mass-purge every legacy row. Silent mass-logout for one line.
        if not family_id:
            return 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Serialises against issue_new_generation for this family so a
                # suspended refresh cascade cannot resurrect it post-revoke.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock($1)", _family_lock_key(family_id)
                )
                await conn.execute(
                    """
                    INSERT INTO oauth_revoked_families (family_id, revoked_at)
                    VALUES ($1, now())
                    ON CONFLICT (family_id) DO UPDATE SET revoked_at = now()
                    """,
                    family_id,
                )
                # Counted server-side in one statement. The transaction (and
                # with it the advisory lock above) stays — it is what serialises
                # this against issue_new_generation.
                return await conn.fetchval(
                    """
                    WITH a AS (
                        DELETE FROM oauth_access_tokens WHERE family_id = $1
                        RETURNING 1
                    ), r AS (
                        DELETE FROM oauth_refresh_tokens WHERE family_id = $1
                        RETURNING 1
                    )
                    SELECT (SELECT count(*) FROM a) + (SELECT count(*) FROM r)
                    """,
                    family_id,
                )

    async def issue_new_generation(
        self,
        *,
        family_id: str,
        access_token: str,
        refresh_token: str,
        taiga_auth_token: str,
        taiga_refresh_token: str,
        taiga_user_id: int,
        taiga_username: str,
        client_id: str,
        access_scopes: List[str],
        refresh_scopes: List[str],
        access_expires_at: datetime,
        refresh_expires_at: datetime,
    ) -> bool:
        """Atomically issue a new generation for a refresh family.

        Returns True if the new access+refresh pair was persisted; False if the
        family was revoked between consume and now (caller raises
        ``invalid_grant``). The tombstone check and both inserts share one
        transaction and the family's advisory lock, so a concurrent
        ``revoke_token_family`` cannot interleave.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock($1)", _family_lock_key(family_id)
                )
                revoked = await conn.fetchval(
                    "SELECT 1 FROM oauth_revoked_families WHERE family_id = $1",
                    family_id,
                )
                if revoked is not None:
                    return False
                await conn.execute(
                    _UPSERT_ACCESS_TOKEN,
                    access_token,
                    taiga_auth_token,
                    taiga_refresh_token,
                    taiga_user_id,
                    taiga_username,
                    client_id,
                    list(access_scopes),
                    access_expires_at,
                    family_id,
                )
                await conn.execute(
                    _UPSERT_REFRESH_TOKEN,
                    refresh_token,
                    family_id,
                    client_id,
                    taiga_auth_token,
                    taiga_refresh_token,
                    taiga_user_id,
                    taiga_username,
                    list(refresh_scopes),
                    refresh_expires_at,
                )
                return True

    # --- Auth codes -------------------------------------------------------

    async def store_authorization_code(
        self,
        *,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        code_challenge_method: str,
        taiga_auth_token: str,
        taiga_refresh_token: str,
        taiga_user_id: int,
        taiga_username: str,
        scopes: List[str],
        expires_at: datetime,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO oauth_auth_codes (
                code, client_id, redirect_uri, code_challenge,
                code_challenge_method, taiga_auth_token, taiga_refresh_token,
                taiga_user_id, taiga_username, scopes, expires_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (code) DO UPDATE SET
                client_id = EXCLUDED.client_id,
                redirect_uri = EXCLUDED.redirect_uri,
                code_challenge = EXCLUDED.code_challenge,
                code_challenge_method = EXCLUDED.code_challenge_method,
                taiga_auth_token = EXCLUDED.taiga_auth_token,
                taiga_refresh_token = EXCLUDED.taiga_refresh_token,
                taiga_user_id = EXCLUDED.taiga_user_id,
                taiga_username = EXCLUDED.taiga_username,
                scopes = EXCLUDED.scopes,
                expires_at = EXCLUDED.expires_at
            """,
            code,
            client_id,
            redirect_uri,
            code_challenge,
            code_challenge_method,
            taiga_auth_token,
            taiga_refresh_token,
            taiga_user_id,
            taiga_username,
            list(scopes),
            expires_at,
        )

    async def consume_authorization_code(self, code: str) -> Optional[AuthCodeRecord]:
        """Single-use atomic pop. Returns None if missing or expired.

        ``DELETE ... RETURNING`` is atomic on its own, so an expired code is
        still removed — matching InMemoryStore, which pops first and only then
        checks expiry.
        """
        row = await self._pool.fetchrow(
            "DELETE FROM oauth_auth_codes WHERE code = $1 RETURNING *", code
        )
        if row is None:
            return None
        record = self._row_to(AuthCodeRecord, row)
        if record.expires_at < datetime.now(timezone.utc):
            return None
        return record

    async def peek_authorization_code(self, code: str) -> Optional[AuthCodeRecord]:
        """Return the auth code without consuming it."""
        row = await self._pool.fetchrow(
            "SELECT * FROM oauth_auth_codes "
            "WHERE code = $1 AND expires_at >= now()",
            code,
        )
        return self._row_to(AuthCodeRecord, row)

    # --- DCR --------------------------------------------------------------

    async def register_client(
        self,
        *,
        client_id: str,
        client_secret: str,
        redirect_uris: List[str],
        client_name: str,
        token_endpoint_auth_method: str = "client_secret_basic",
        scope: Optional[str] = None,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO oauth_clients (
                client_id, client_secret, redirect_uris, client_name,
                token_endpoint_auth_method, scope
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (client_id) DO UPDATE SET
                client_secret = EXCLUDED.client_secret,
                redirect_uris = EXCLUDED.redirect_uris,
                client_name = EXCLUDED.client_name,
                token_endpoint_auth_method = EXCLUDED.token_endpoint_auth_method,
                scope = EXCLUDED.scope
            """,
            client_id,
            client_secret,
            list(redirect_uris),
            client_name,
            token_endpoint_auth_method,
            scope,
        )

    async def lookup_client(self, client_id: str) -> Optional[ClientRecord]:
        """Return the full client record, including ``client_secret``.

        mcp-sdk's ``ClientAuthenticator`` compares
        ``provider.get_client(client_id).client_secret`` for
        ``client_secret_basic`` / ``client_secret_post`` requests at /token.
        Returning None for the secret would downgrade confidential clients to
        public — any caller knowing the client_id would pass auth.
        """
        row = await self._pool.fetchrow(
            "SELECT * FROM oauth_clients WHERE client_id = $1", client_id
        )
        return self._row_to(ClientRecord, row)

    async def verify_client_secret(self, client_id: str, presented: str) -> bool:
        secret = await self._pool.fetchval(
            "SELECT client_secret FROM oauth_clients WHERE client_id = $1",
            client_id,
        )
        if secret is None:
            return False
        return hmac.compare_digest(secret, presented)

    # --- Cleanup ----------------------------------------------------------

    async def cleanup_expired(self) -> int:
        """Sweep expired access tokens, auth codes, refresh tokens, and
        revoked-family tombstones. Returns the total number of records purged.

        Tombstones older than the max refresh-token TTL are dropped too: by
        then every refresh token in the family has expired, so the tombstone
        serves no further reuse-detection purpose. Without this the table would
        grow unboundedly.
        """
        horizon = datetime.now(timezone.utc) - REFRESH_TOKEN_TTL
        # One statement rather than BEGIN + 4 DELETEs + COMMIT. The four
        # deletes are independent, so CTE evaluation order is irrelevant, and a
        # single statement is already atomic — which also makes the explicit
        # transaction unnecessary. Counting inside the CTEs keeps the deleted
        # rows on the server: the previous DELETE ... RETURNING 1 + len() built
        # one Record per row purely to count them, which is unbounded after any
        # downtime backlog.
        return await self._pool.fetchval(
            """
            WITH a AS (
                DELETE FROM oauth_access_tokens WHERE expires_at < now()
                RETURNING 1
            ), c AS (
                DELETE FROM oauth_auth_codes WHERE expires_at < now()
                RETURNING 1
            ), r AS (
                DELETE FROM oauth_refresh_tokens WHERE expires_at < now()
                RETURNING 1
            ), t AS (
                DELETE FROM oauth_revoked_families WHERE revoked_at < $1
                RETURNING 1
            )
            SELECT (SELECT count(*) FROM a) + (SELECT count(*) FROM c)
                 + (SELECT count(*) FROM r) + (SELECT count(*) FROM t)
            """,
            horizon,
        )

    async def close(self) -> None:
        """Close the connection pool. Called from the lifespan on shutdown."""
        await self._pool.close()
