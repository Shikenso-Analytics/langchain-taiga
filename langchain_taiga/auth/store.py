"""In-memory OAuth state store for the Taiga MCP bridge.

Per Amendment v3.4 of the Multi-Tenant OAuth Bridge plan: process-local
storage instead of Postgres. Pod restart wipes everything; users re-OAuth.
Acceptable for single-replica deployments at ~30-user scale.

For multi-replica or persistence requirements, swap to a Postgres-backed
implementation with the same async interface.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

#: Canonical refresh-token lifetime, defined here because ``store`` is the leaf
#: module: ``provider`` and ``postgres_store`` both import from it, so there is
#: no cycle to avoid. It doubles as the retention horizon for revoked-family
#: tombstones — once every refresh token in a family has expired, the tombstone
#: has nothing left to detect a replay against.
#:
#: Previously this value was written out three times (here, ``provider.py``,
#: ``postgres_store.py``), each copy defending itself with an import-cycle
#: argument that was never true. Raising the TTL while missing one copy would
#: silently sweep reuse-detection state out from under still-live tokens.
REFRESH_TOKEN_TTL = timedelta(days=30)


@dataclass
class AccessTokenRecord:
    token: str
    taiga_auth_token: str
    taiga_refresh_token: str
    taiga_user_id: int
    taiga_username: str
    client_id: str
    scopes: List[str]
    expires_at: datetime
    family_id: str = ""  # "" for pre-2.5.0 records


@dataclass
class AuthCodeRecord:
    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    taiga_auth_token: str
    taiga_refresh_token: str
    taiga_user_id: int
    taiga_username: str
    scopes: List[str]
    expires_at: datetime


@dataclass
class RefreshTokenRecord:
    """An MCP refresh token. Once exchanged, ``rotated_out`` is flipped to
    True so a subsequent presentation triggers reuse-detection in
    ``TaigaOAuthProvider.load_refresh_token`` (which is the primary
    detection point — ``exchange_refresh_token`` keeps a defensive branch
    for direct callers that bypass load)."""
    token: str
    family_id: str
    client_id: str
    taiga_auth_token: str
    taiga_refresh_token: str
    taiga_user_id: int
    taiga_username: str
    scopes: List[str]
    expires_at: datetime
    rotated_out: bool = False


@dataclass
class ConsumeRefreshResult:
    """Discriminated result of the atomic consume_refresh_token operation.

    - ``active`` — token was active; ``record.rotated_out`` has been flipped
      to True in place; the caller now owns the right to mint a new family
      generation.
    - ``already_rotated`` — token was already consumed; the caller MUST
      revoke the family (reuse-detection signal).
    - ``expired`` — record exists but past its ``expires_at``.
    - ``not_found`` — no record under this token.
    """
    status: Literal["active", "already_rotated", "expired", "not_found"]
    record: Optional[RefreshTokenRecord]


@dataclass
class ClientRecord:
    client_id: str
    client_secret: Optional[str]
    redirect_uris: List[str]
    client_name: str
    token_endpoint_auth_method: str
    scope: Optional[str] = None  # Space-separated scope string from DCR


class InMemoryStore:
    """Process-local OAuth state. Pod restart wipes everything; users re-OAuth.

    Acceptable for single-replica deployments at ~30-user scale. For
    multi-replica or persistence requirements, swap to a Postgres-backed
    implementation with the same interface.
    """

    def __init__(self) -> None:
        self._access_tokens: dict[str, AccessTokenRecord] = {}
        self._auth_codes: dict[str, AuthCodeRecord] = {}
        self._clients: dict[str, ClientRecord] = {}
        self._refresh_tokens: dict[str, RefreshTokenRecord] = {}
        # Tombstone map for revoked families (family_id → revoked_at). Used
        # by ``issue_new_generation`` to refuse a write-back from a coroutine
        # whose family was revoked while it was suspended inside the Taiga
        # refresh cascade (otherwise a resumed exchange could resurrect a
        # just-revoked family by storing new access + refresh tokens with
        # the revoked ``family_id``). Carrying a timestamp lets
        # ``cleanup_expired`` sweep entries older than REFRESH_TOKEN_TTL —
        # by then every refresh token in the family has expired, so the
        # tombstone serves no further reuse-detection purpose.
        self._revoked_families: dict[str, datetime] = {}

    @classmethod
    async def from_env(cls) -> "InMemoryStore":
        """Match the from_env() shape of the Postgres design for swap-out symmetry."""
        return cls()

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
        self._access_tokens[token] = AccessTokenRecord(
            token=token,
            taiga_auth_token=taiga_auth_token,
            taiga_refresh_token=taiga_refresh_token,
            taiga_user_id=taiga_user_id,
            taiga_username=taiga_username,
            client_id=client_id,
            scopes=list(scopes),
            expires_at=expires_at,
            family_id=family_id,
        )

    async def lookup_access_token(self, token: str) -> Optional[AccessTokenRecord]:
        record = self._access_tokens.get(token)
        if record is None:
            return None
        # Defensive: don't return expired records (cleanup may not have run yet)
        if record.expires_at < datetime.now(timezone.utc):
            return None
        return record

    async def update_taiga_token(
        self,
        *,
        token: str,
        taiga_auth_token: str,
        taiga_refresh_token: str,
        expires_at: datetime,
    ) -> None:
        record = self._access_tokens.get(token)
        if record is None:
            return
        record.taiga_auth_token = taiga_auth_token
        record.taiga_refresh_token = taiga_refresh_token
        record.expires_at = expires_at

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
        self._refresh_tokens[token] = RefreshTokenRecord(
            token=token,
            family_id=family_id,
            client_id=client_id,
            taiga_auth_token=taiga_auth_token,
            taiga_refresh_token=taiga_refresh_token,
            taiga_user_id=taiga_user_id,
            taiga_username=taiga_username,
            scopes=list(scopes),
            expires_at=expires_at,
        )

    async def lookup_refresh_token(
        self, token: str
    ) -> Optional[RefreshTokenRecord]:
        """Return the record only if it exists AND is not expired.

        Returns rotated_out records too — ``TaigaOAuthProvider.load_refresh_token``
        inspects the ``rotated_out`` flag itself: if True, it revokes the
        family on the spot and returns None to mcp-sdk's TokenHandler (which
        then surfaces ``invalid_grant``). Filtering rotated_out here would
        hide the replay signal from that detection path.
        """
        record = self._refresh_tokens.get(token)
        if record is None:
            return None
        if record.expires_at < datetime.now(timezone.utc):
            return None
        return record

    async def consume_refresh_token(self, token: str) -> ConsumeRefreshResult:
        """Atomic lookup + rotated_out transition.

        Atomicity comes from asyncio's single-threaded execution: this method
        contains no awaits between the read and the mutation, so concurrent
        callers cannot interleave inside it. The first observer of a fresh
        token sees status="active" and flips rotated_out; subsequent observers
        see status="already_rotated".

        For a future Redis / Postgres backend, this method must be implemented
        as a Lua script or row-locked transaction to preserve the semantics.
        """
        record = self._refresh_tokens.get(token)
        if record is None:
            return ConsumeRefreshResult(status="not_found", record=None)
        if record.expires_at < datetime.now(timezone.utc):
            return ConsumeRefreshResult(status="expired", record=record)
        if record.rotated_out:
            return ConsumeRefreshResult(status="already_rotated", record=record)
        record.rotated_out = True
        return ConsumeRefreshResult(status="active", record=record)

    def _sweep(self, mapping: dict, *, predicate) -> int:
        """Bulk-delete dict entries matching predicate. Returns count purged.

        Two-phase (collect keys, then pop) avoids dict-mutated-during-iteration.
        """
        stale = [k for k, v in mapping.items() if predicate(v)]
        for k in stale:
            mapping.pop(k, None)
        return len(stale)

    async def revoke_token_family(self, family_id: str) -> int:
        """Delete every access and refresh token sharing this family_id.

        Used by reuse-detection in TaigaOAuthProvider.exchange_refresh_token
        when a rotated_out refresh token is presented. Returns the total
        number of records purged.
        """
        # Defense: AccessTokenRecord.family_id defaults to "" for legacy
        # (pre-2.5.0) records. A bare revoke_token_family("") would mass-purge
        # every legacy record. Current call sites never pass "" (provider mints
        # via secrets.token_urlsafe(16)), but the guard is one line and the
        # failure mode is silent mass-logout.
        if not family_id:
            return 0
        # Flag the family BEFORE the pops. ``issue_new_generation`` checks
        # this map with no intervening awaits between the check and its
        # dict writes, so a coroutine suspended inside the Taiga cascade
        # cannot resurrect a family that was just revoked. The timestamp
        # lets ``cleanup_expired`` sweep stale tombstones.
        self._revoked_families[family_id] = datetime.now(timezone.utc)
        return (
            self._sweep(self._access_tokens, predicate=lambda v: v.family_id == family_id)
            + self._sweep(self._refresh_tokens, predicate=lambda v: v.family_id == family_id)
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

        Returns True if the new access+refresh pair was persisted; False if
        the family was revoked between consume and now (caller should raise
        invalid_grant). The check + both dict writes happen inside this
        method body with no intervening awaits, so asyncio's single-threaded
        execution makes the operation atomic against concurrent revokes
        that may have flipped _revoked_families.

        Without this atomicity, a request suspended inside a cascade call can
        resurrect a family that another coroutine just revoked.
        """
        if family_id in self._revoked_families:
            return False
        self._access_tokens[access_token] = AccessTokenRecord(
            token=access_token,
            taiga_auth_token=taiga_auth_token,
            taiga_refresh_token=taiga_refresh_token,
            taiga_user_id=taiga_user_id,
            taiga_username=taiga_username,
            client_id=client_id,
            scopes=list(access_scopes),
            expires_at=access_expires_at,
            family_id=family_id,
        )
        self._refresh_tokens[refresh_token] = RefreshTokenRecord(
            token=refresh_token,
            family_id=family_id,
            client_id=client_id,
            taiga_auth_token=taiga_auth_token,
            taiga_refresh_token=taiga_refresh_token,
            taiga_user_id=taiga_user_id,
            taiga_username=taiga_username,
            scopes=list(refresh_scopes),
            expires_at=refresh_expires_at,
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
        self._auth_codes[code] = AuthCodeRecord(
            code=code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            taiga_auth_token=taiga_auth_token,
            taiga_refresh_token=taiga_refresh_token,
            taiga_user_id=taiga_user_id,
            taiga_username=taiga_username,
            scopes=list(scopes),
            expires_at=expires_at,
        )

    async def consume_authorization_code(self, code: str) -> Optional[AuthCodeRecord]:
        """Single-use atomic pop. Returns None if missing or expired."""
        record = self._auth_codes.pop(code, None)
        if record is None:
            return None
        if record.expires_at < datetime.now(timezone.utc):
            return None
        return record

    async def peek_authorization_code(self, code: str) -> Optional[AuthCodeRecord]:
        """Return the auth code without consuming it. Use ``consume_authorization_code``
        for the single-use atomic pop. Returns None if missing or expired."""
        record = self._auth_codes.get(code)
        if record is None:
            return None
        if record.expires_at < datetime.now(timezone.utc):
            return None
        return record

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
        self._clients[client_id] = ClientRecord(
            client_id=client_id,
            client_secret=client_secret,  # plaintext in-memory; not at rest
            redirect_uris=list(redirect_uris),
            client_name=client_name,
            token_endpoint_auth_method=token_endpoint_auth_method,
            scope=scope,
        )

    async def lookup_client(self, client_id: str) -> Optional[ClientRecord]:
        """Return the full client record (including ``client_secret``).

        mcp-sdk's ``ClientAuthenticator`` middleware calls
        ``provider.get_client(client_id).client_secret`` to validate
        ``client_secret_basic`` / ``client_secret_post`` requests at /token.
        Returning ``None`` for the secret short-circuits that comparison and
        downgrades confidential clients to public — any caller knowing the
        ``client_id`` would pass auth. Per Amendment v3.4 the store is
        in-memory (no at-rest concerns), so the plaintext secret stays in
        process memory only and is never persisted to disk.

        Callers wanting a constant-time API still have ``verify_client_secret``.
        """
        record = self._clients.get(client_id)
        return replace(record) if record is not None else None

    async def verify_client_secret(self, client_id: str, presented: str) -> bool:
        record = self._clients.get(client_id)
        if record is None or record.client_secret is None:
            return False
        return hmac.compare_digest(record.client_secret, presented)

    # --- Cleanup ----------------------------------------------------------

    async def cleanup_expired(self) -> int:
        """Sweep expired access tokens, auth codes, refresh tokens, and
        revoked-family tombstones.

        Refresh-token sweep covers both active and rotated_out records — the
        rotated_out records don't have a separate retention policy, just the
        same REFRESH_TOKEN_TTL.

        Revoked-family tombstones older than REFRESH_TOKEN_TTL are also
        swept — by that point every refresh token in the family has expired
        (or been purged above), so the tombstone serves no further
        reuse-detection purpose. Without this sweep ``_revoked_families``
        would grow unboundedly across the pod's lifetime.

        Returns the total number of records purged.
        """
        now = datetime.now(timezone.utc)
        # Tombstones for revoked families older than the max refresh-token
        # TTL serve no purpose — every refresh token in that family has
        # expired by now, so there is nothing left for reuse-detection to
        # fire on.
        tombstone_horizon = now - REFRESH_TOKEN_TTL
        expired = lambda v: v.expires_at < now
        return (
            self._sweep(self._access_tokens, predicate=expired)
            + self._sweep(self._auth_codes, predicate=expired)
            + self._sweep(self._refresh_tokens, predicate=expired)
            + self._sweep(
                self._revoked_families,
                predicate=lambda ts: ts < tombstone_horizon,
            )
        )

    async def close(self) -> None:
        """No-op — kept for API symmetry with future persistent backends."""
        return
