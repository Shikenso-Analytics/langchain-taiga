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
from datetime import datetime, timezone
from typing import List, Optional


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
        """Sweep expired access tokens + auth codes. Returns total purged."""
        now = datetime.now(timezone.utc)
        purged_tokens = [k for k, v in self._access_tokens.items() if v.expires_at < now]
        for k in purged_tokens:
            self._access_tokens.pop(k, None)
        purged_codes = [k for k, v in self._auth_codes.items() if v.expires_at < now]
        for k in purged_codes:
            self._auth_codes.pop(k, None)
        return len(purged_tokens) + len(purged_codes)

    async def close(self) -> None:
        """No-op — kept for API symmetry with future persistent backends."""
        return
