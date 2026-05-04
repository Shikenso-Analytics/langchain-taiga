"""FastMCP OAuthProvider subclass for Taiga username/password backends.

When constructed and passed to ``FastMCP(auth=provider)``, this triggers
auto-mounting of:
  GET /.well-known/oauth-authorization-server  (RFC 8414)
  GET /.well-known/oauth-protected-resource    (RFC 9728)
  POST /register                              (RFC 7591 DCR)
  GET /authorize                              (response_type=code, PKCE)
  POST /token                                 (Authorization Code grant)

The custom HTML login page is NOT auto-mounted — it is registered in
remote_server.py via ``mcp.custom_route("/oauth/login", ...)``.

Phase 0 deltas applied (verified against fastmcp 2.14.5 + mcp SDK):
- ``AccessToken`` is imported from ``fastmcp.server.auth`` (it has the
  ``claims`` field; the mcp.server.auth.provider one does not).
- ``ClientRegistrationOptions`` lives in ``mcp.server.auth.settings``
  (not ``fastmcp.server.auth``).
- ``OAuthProvider.__init__`` requires both ``base_url`` and ``issuer_url``.
- ``AuthorizationCode`` requires ``redirect_uri_provided_explicitly: bool``.
- ``AuthorizationParams`` required fields: ``state``, ``scopes``,
  ``code_challenge``, ``redirect_uri``, ``redirect_uri_provided_explicitly``.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

from fastmcp.server.auth import AccessToken, OAuthProvider
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
)
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from langchain_taiga.auth.store import InMemoryStore
from langchain_taiga.auth.taiga_client import TaigaAuthenticationError, TaigaClient

ACCESS_TOKEN_TTL = timedelta(hours=1)
AUTH_CODE_TTL = timedelta(minutes=10)

_log = logging.getLogger(__name__)


@dataclass
class _PendingAuthorize:
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    scopes: List[str]
    claude_state: str
    expires_at: datetime


class TaigaOAuthProvider(OAuthProvider):
    """OAuth Authorization Server bound to Taiga as the credential source."""

    DEFAULT_ALLOWED_REDIRECT_PREFIXES: Tuple[str, ...] = (
        "https://claude.ai/",
        "https://claude.com/",
        "http://localhost:",  # MCP Inspector
    )

    def __init__(
        self,
        *,
        store: InMemoryStore,
        taiga_client: TaigaClient,
        issuer_url: str,
        allowed_redirect_uri_prefixes: Optional[Tuple[str, ...]] = None,
    ):
        # base_url and issuer_url are both required by FastMCP. In our
        # deployment they are the same value (the path-aware MCP-server
        # root), but the framework still wants both passed.
        super().__init__(
            base_url=issuer_url,
            issuer_url=issuer_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=["taiga"],
                default_scopes=["taiga"],
            ),
            required_scopes=["taiga"],
        )
        self._store = store
        self._taiga = taiga_client
        self._issuer = issuer_url.rstrip("/")
        self._allowed_redirect_prefixes = (
            allowed_redirect_uri_prefixes
            if allowed_redirect_uri_prefixes is not None
            else self.DEFAULT_ALLOWED_REDIRECT_PREFIXES
        )
        # In-memory pending-authorize state. Stays per-instance even when
        # the rest of the store moves to an external backend — different
        # scope (request-bound) than persistent OAuth state.
        self._authorize_states: Dict[str, _PendingAuthorize] = {}

    # ---- Allowlist ------------------------------------------------------

    def _validate_redirect_uri(self, uri: str) -> None:
        if not any(uri.startswith(p) for p in self._allowed_redirect_prefixes):
            raise ValueError(
                f"Redirect URI not allowed: {uri!r}. "
                f"Allowed prefixes: {self._allowed_redirect_prefixes}"
            )

    # ---- DCR ------------------------------------------------------------

    async def register_client(
        self, client_info: OAuthClientMetadata
    ) -> OAuthClientInformationFull:
        for uri in client_info.redirect_uris:
            self._validate_redirect_uri(str(uri))

        client_id = f"mcp_{secrets.token_urlsafe(16)}"
        # Public clients (token_endpoint_auth_method="none") still get a
        # client_secret minted but it's not required at /token.
        client_secret = secrets.token_urlsafe(32)
        method = client_info.token_endpoint_auth_method or "client_secret_basic"

        await self._store.register_client(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=[str(u) for u in client_info.redirect_uris],
            client_name=client_info.client_name or "unnamed",
            token_endpoint_auth_method=method,
        )
        return OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,  # only returned once
            redirect_uris=client_info.redirect_uris,
            client_name=client_info.client_name,
            token_endpoint_auth_method=method,
        )

    async def get_client(
        self, client_id: str
    ) -> Optional[OAuthClientInformationFull]:
        record = await self._store.lookup_client(client_id)
        if record is None:
            # Returning None makes FastMCP render HTTP 400 invalid_client per
            # RFC 6749 — Anthropic-required so claude.ai re-registers.
            return None
        # Propagate the actual client_secret. mcp-sdk's ClientAuthenticator
        # compares against this field to validate client_secret_basic /
        # client_secret_post; ``None`` would silently bypass auth on
        # confidential clients. The store is in-memory (Amendment v3.4) so
        # the secret never leaves process memory.
        return OAuthClientInformationFull(
            client_id=record.client_id,
            client_secret=record.client_secret,
            redirect_uris=record.redirect_uris,
            client_name=record.client_name,
            token_endpoint_auth_method=record.token_endpoint_auth_method,
        )

    # ---- Authorize ------------------------------------------------------

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        """Stash the authorize request and return URL to our HTML login page."""
        redirect_uri = str(params.redirect_uri)
        if redirect_uri not in [str(r) for r in client.redirect_uris]:
            raise ValueError("Redirect URI not registered for this client")
        self._validate_redirect_uri(redirect_uri)

        internal_state = secrets.token_urlsafe(24)
        self._authorize_states[internal_state] = _PendingAuthorize(
            client_id=client.client_id,
            redirect_uri=redirect_uri,
            code_challenge=params.code_challenge,
            # AuthorizationParams in 2.14.5 mcp SDK does NOT carry
            # code_challenge_method — S256 is assumed implicitly.
            code_challenge_method="S256",
            scopes=list(params.scopes or ["taiga"]),
            claude_state=params.state or "",
            expires_at=datetime.now(timezone.utc) + AUTH_CODE_TTL,
        )
        return f"{self._issuer}/oauth/login?internal_state={internal_state}"

    async def complete_login(
        self, *, internal_state: str, username: str, password: str
    ) -> Tuple[str, str]:
        """Called by remote_server.py's /oauth/login POST handler.

        Authenticates against Taiga, mints an authorization code, returns
        ``(code, redirect_url_with_code_and_claude_state)``. State is preserved
        on ``TaigaAuthenticationError`` so the user can retry their password.
        """
        st = self._authorize_states.get(internal_state)
        if st is None:
            raise ValueError("Invalid or expired internal_state")
        if st.expires_at < datetime.now(timezone.utc):
            self._authorize_states.pop(internal_state, None)
            raise ValueError("Authorize state expired")

        # Authenticate against Taiga FIRST. If this raises, leave state
        # intact so the user can retry without restarting the whole flow.
        creds = await self._taiga.authenticate_user(username, password)

        # Only consume the state on success.
        self._authorize_states.pop(internal_state, None)

        code = secrets.token_urlsafe(32)
        await self._store.store_authorization_code(
            code=code,
            client_id=st.client_id,
            redirect_uri=st.redirect_uri,
            code_challenge=st.code_challenge,
            code_challenge_method=st.code_challenge_method,
            taiga_auth_token=creds.auth_token,
            taiga_refresh_token=creds.refresh,
            taiga_user_id=creds.user_id,
            taiga_username=creds.username,
            scopes=st.scopes,
            expires_at=datetime.now(timezone.utc) + AUTH_CODE_TTL,
        )
        params = urlencode({"code": code, "state": st.claude_state})
        return code, f"{st.redirect_uri}?{params}"

    # ---- Authorization Code grant ---------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        """Peek at the auth code without consuming it.

        FastMCP convention: ``load_authorization_code`` returns the
        ``AuthorizationCode`` model; ``exchange_authorization_code`` is the
        single-use consumer. We satisfy this via the store's
        ``peek_authorization_code`` (no pop); the pop happens in ``exchange_*``.
        """
        record = await self._store.peek_authorization_code(authorization_code)
        if record is None:
            return None
        if record.client_id != client.client_id:
            return None
        return AuthorizationCode(
            code=record.code,
            client_id=record.client_id,
            redirect_uri=record.redirect_uri,
            code_challenge=record.code_challenge,
            scopes=list(record.scopes),
            expires_at=record.expires_at.timestamp(),
            # Phase 0 delta: required field. We received the redirect_uri
            # explicitly from the original /authorize request, not derived.
            redirect_uri_provided_explicitly=True,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        # mcp-sdk's token handler converts ``TokenError`` into RFC 6749
        # 400 responses with the right ``error`` code; ``ValueError`` would
        # become a generic 500. Use TokenError("invalid_grant" |
        # "invalid_client" | ...) for spec-correct surfaces.
        # Atomic single-use consume
        consumed = await self._store.consume_authorization_code(
            authorization_code.code
        )
        if consumed is None:
            raise TokenError(
                "invalid_grant",
                "Authorization code already used or expired",
            )
        if consumed.client_id != client.client_id:
            raise TokenError(
                "invalid_client",
                "Code was issued to a different client",
            )
        if str(consumed.redirect_uri) != str(authorization_code.redirect_uri):
            raise TokenError(
                "invalid_grant",
                "redirect_uri mismatch with the original authorization request",
            )

        mcp_access_token = secrets.token_urlsafe(32)
        await self._store.store_access_token(
            token=mcp_access_token,
            taiga_auth_token=consumed.taiga_auth_token,
            taiga_refresh_token=consumed.taiga_refresh_token,
            taiga_user_id=consumed.taiga_user_id,
            taiga_username=consumed.taiga_username,
            expires_at=datetime.now(timezone.utc) + ACCESS_TOKEN_TTL,
            client_id=consumed.client_id,
            scopes=consumed.scopes,
        )
        return OAuthToken(
            access_token=mcp_access_token,
            token_type="Bearer",
            expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
            scope=" ".join(consumed.scopes),
        )

    # ---- Refresh tokens (deferred to v2) --------------------------------

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> Optional[RefreshToken]:
        # v1 doesn't issue refresh tokens; claude.ai falls back to re-OAuth on expiry.
        return None

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: List[str],
    ) -> OAuthToken:
        raise NotImplementedError("Refresh tokens deferred to v2")

    # ---- Access token verification --------------------------------------

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        """Look up the MCP access token; expiry is enforced by the FastMCP layer.

        **No transparent Taiga-token refresh here** — the lifetime invariant
        (MCP TTL <= Taiga JWT TTL) makes this dead code at v1 TTLs.

        If your Taiga deployment shortens its JWT TTL below MCP's, the right
        fix is to ALSO shorten ``ACCESS_TOKEN_TTL`` to stay below it — not to
        add transparent refresh. Refresh-on-the-fly is a v2 feature that
        requires splitting the schema into ``mcp_expires_at`` +
        ``taiga_expires_at`` and a refresh-token grant flow.
        """
        record = await self._store.lookup_access_token(token)
        if record is None:
            return None
        return AccessToken(
            token=token,
            client_id=record.client_id,
            scopes=record.scopes,
            # Phase 0 delta: fastmcp AccessToken expires_at is int|None
            # (timestamp seconds), not datetime.
            expires_at=int(record.expires_at.timestamp()),
            claims={
                "taiga_jwt": record.taiga_auth_token,
                "user_id": record.taiga_user_id,
                "username": record.taiga_username,
            },
        )

    # ---- Optional revocation --------------------------------------------

    async def revoke_token(self, token: str) -> None:
        """RFC 7009 revocation endpoint. v1: no-op — rely on TTL expiry.

        Logging the call so an operator reading audit logs sees "yes, claude.ai
        asked us to revoke; we noted it but didn't act because v1 doesn't track
        revocation state separately from expiry."
        """
        _log.info(
            "revoke_token called (no-op in v1) for token=%s...",
            token[:8] if token else "<empty>",
        )

    # ---- Pending-authorize state cleanup --------------------------------

    def cleanup_authorize_states(self) -> int:
        """Drop ``_authorize_states`` entries whose ``expires_at`` is in the past.

        Without this, abandoned authorize-clicks (user opened ``/authorize``,
        never POSTed the form) leak indefinitely until pod restart. Mirrors
        the pattern in ``InMemoryStore.cleanup_expired``. Returns the number
        of entries purged.
        """
        now = datetime.now(timezone.utc)
        purged = [k for k, v in self._authorize_states.items() if v.expires_at < now]
        for k in purged:
            self._authorize_states.pop(k, None)
        return len(purged)


# ---- Cleanup loop (called from remote_server lifespan) ------------------


async def run_cleanup_loop(
    store: InMemoryStore,
    *,
    provider: Optional["TaigaOAuthProvider"] = None,
    period_seconds: float = 300.0,
    stop: Optional[asyncio.Event] = None,
) -> None:
    """Periodic sweeper for expired tokens, auth codes, and authorize states.

    Cancellation: pass ``stop`` (an ``asyncio.Event``) and call
    ``stop.set()`` from the lifespan shutdown. The loop wakes within
    ``period_seconds`` and exits cleanly.

    If ``provider`` is supplied, also prunes its ``_authorize_states`` so
    abandoned authorize-clicks don't leak until pod restart.
    """
    stop = stop or asyncio.Event()
    while not stop.is_set():
        try:
            deleted = await store.cleanup_expired()
            if deleted:
                _log.info("Cleanup deleted %d expired records", deleted)
            if provider is not None:
                pruned_states = provider.cleanup_authorize_states()
                if pruned_states:
                    _log.info(
                        "Cleanup pruned %d expired authorize states", pruned_states
                    )
        except Exception:
            _log.exception("Cleanup iteration failed; continuing")
        try:
            await asyncio.wait_for(stop.wait(), timeout=period_seconds)
        except asyncio.TimeoutError:
            pass


__all__ = [
    "TaigaOAuthProvider",
    "TaigaAuthenticationError",
    "run_cleanup_loop",
]
