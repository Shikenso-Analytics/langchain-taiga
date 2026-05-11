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
import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

from fastmcp.server.auth import AccessToken, OAuthProvider
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
)
from mcp.server.auth.provider import construct_redirect_uri
from mcp.server.auth.settings import ClientRegistrationOptions
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthToken,
)

from langchain_taiga.auth.store import InMemoryStore
from langchain_taiga.auth.taiga_client import (
    TaigaAuthenticationError,
    TaigaClient,
    TaigaRefreshError,
)

ACCESS_TOKEN_TTL = timedelta(hours=1)
AUTH_CODE_TTL = timedelta(minutes=10)
REFRESH_TOKEN_TTL = timedelta(days=30)

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

    # (scheme, hostname-lowercase, port|None) — port=None means any port allowed.
    # Hostname comparison uses urlparse().hostname (which discards userinfo),
    # closing the ``http://localhost:8080@evil.com/cb`` userinfo-bypass.
    DEFAULT_ALLOWED_REDIRECT_TARGETS: Tuple[Tuple[str, str, Optional[int]], ...] = (
        ("https", "claude.ai", None),
        ("https", "claude.com", None),
        # VSCode's MCP integration submits THREE redirect URIs in DCR:
        # ``http://127.0.0.1:<ephemeral>`` (covered by 127.0.0.1 below),
        # ``https://vscode.dev/redirect`` (Web VSCode stable), AND
        # ``https://insiders.vscode.dev/redirect`` (Web VSCode Insiders, sent
        # whenever the user has Insiders installed locally OR has Settings
        # Sync turned on with the Insiders side). All must be allowed; ANY
        # rejected URI raises ValueError mid-registration → 500 → VSCode
        # interprets that as "DCR not supported" and falls back to manual
        # client registration.
        ("https", "vscode.dev", None),
        ("https", "insiders.vscode.dev", None),
        ("http", "localhost", None),   # MCP Inspector + VSCode local
        ("http", "127.0.0.1", None),   # IPv4 loopback alias
    )

    def __init__(
        self,
        *,
        store: InMemoryStore,
        taiga_client: TaigaClient,
        issuer_url: str,
        allowed_redirect_targets: Optional[
            Tuple[Tuple[str, str, Optional[int]], ...]
        ] = None,
    ):
        # FastMCP convention (verified post-Phase 0):
        #
        # - ``base_url`` must be the SERVER ROOT — scheme://host[:port] only,
        #   no path. The framework appends the MCP path to it via
        #   ``_get_resource_url(mcp_path)`` to build the protected-resource
        #   URL. Passing a path-bearing base_url here would generate the
        #   double-pathed ``/.well-known/oauth-protected-resource/mcp/mcp``
        #   metadata route observed in local smoke tests.
        # - ``issuer_url`` is the path-aware OAuth identifier
        #   (``https://host/mcp``) used in metadata documents. The deployment
        #   passes it via ``TAIGA_MCP_BASE_URL``.
        parsed = urlparse(issuer_url)
        base_url = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

        super().__init__(
            base_url=base_url,
            issuer_url=issuer_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=["taiga"],
                default_scopes=["taiga"],
            ),
            required_scopes=["taiga"],
        )
        # RFC 8414 §3.3 strict-validation rescue. FastMCP's
        # ``OAuthProvider.get_routes`` builds:
        #   - AS metadata with ``issuer = self.base_url``  (root)
        #   - PRM with ``authorization_servers = [self.issuer_url]``
        #     (which we set to ``.../mcp`` above, because we want the
        #     OAuth identity scoped to the MCP path).
        # Strict clients (VSCode 1.107+) follow the PRM → fetch the auth
        # server's discovery → require that the metadata's ``issuer``
        # MATCHES the auth-server URL they came from. With our split
        # (PRM says auth_server=.../mcp, AS issuer=root) the validation
        # fails and clients show "DCR not supported". claude.ai is
        # lenient and ignores the mismatch.
        # Fix: pin ``self.issuer_url`` BACK to the root so PRM advertises
        # auth_server=root and the AS metadata's issuer (also root)
        # matches. Our internal ``self._issuer`` stays path-aware so the
        # /mcp/oauth/login redirect keeps working.
        self.issuer_url = self.base_url
        self._store = store
        self._taiga = taiga_client
        self._issuer = issuer_url.rstrip("/")
        self._allowed_redirect_targets = (
            allowed_redirect_targets
            if allowed_redirect_targets is not None
            else self.DEFAULT_ALLOWED_REDIRECT_TARGETS
        )
        # In-memory pending-authorize state. Stays per-instance even when
        # the rest of the store moves to an external backend — different
        # scope (request-bound) than persistent OAuth state.
        self._authorize_states: Dict[str, _PendingAuthorize] = {}

    # ---- Allowlist ------------------------------------------------------

    def _validate_redirect_uri(self, uri: str) -> None:
        """Strict-parse the redirect URI; compare scheme + hostname + port.

        ``startswith`` was unsafe because the URL ``http://localhost:8080@evil.com/cb``
        starts with ``http://localhost:`` but ``urlparse(...).hostname`` is
        ``evil.com`` (the ``localhost:8080`` portion is userinfo). An attacker who
        could DCR-register such a redirect_uri would have authorization codes
        delivered to their own host. This implementation extracts the actual
        hostname and only accepts an exact match against the allowlist.
        """
        parsed = urlparse(uri)
        host = (parsed.hostname or "").lower()
        for scheme, allowed_host, allowed_port in self._allowed_redirect_targets:
            if (
                parsed.scheme == scheme
                and host == allowed_host
                and (allowed_port is None or parsed.port == allowed_port)
            ):
                return
        raise ValueError(
            f"Redirect URI not allowed: {uri!r}. "
            f"Allowed targets: {self._allowed_redirect_targets}"
        )

    # ---- DCR ------------------------------------------------------------

    async def register_client(
        self, client_info: OAuthClientInformationFull
    ) -> OAuthClientInformationFull:
        """Persist a DCR client.

        FastMCP / mcp-sdk's ``RegistrationHandler`` mints the ``client_id``
        and ``client_secret`` BEFORE calling this method (with values it
        then echoes back to the client unchanged — our return value is
        discarded by the SDK). We must therefore persist the SDK-provided
        identifiers, not generate our own; otherwise subsequent
        ``get_client(<sdk-uuid>)`` calls return None and claude.ai loops on
        re-registration. Public-client semantics (``token_endpoint_auth_method
        == "none"``): SDK still mints a ``client_secret`` for storage
        symmetry, but the secret is never required at ``/token``.

        ``scope`` must be persisted because the SDK's authorize handler
        delegates to ``OAuthClientInformationFull.validate_scope``, which
        only allows requested scopes that the client was registered with.
        Returning a scope-less client info on lookup short-circuits to
        ``invalid_scope`` even when the request matches our valid_scopes.
        """
        for uri in client_info.redirect_uris:
            self._validate_redirect_uri(str(uri))

        method = client_info.token_endpoint_auth_method or "client_secret_basic"

        await self._store.register_client(
            client_id=client_info.client_id,
            client_secret=client_info.client_secret or "",
            redirect_uris=[str(u) for u in client_info.redirect_uris],
            client_name=client_info.client_name or "unnamed",
            token_endpoint_auth_method=method,
            scope=client_info.scope,
        )
        return client_info

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
            scope=record.scope,
        )

    # ---- Authorize ------------------------------------------------------

    #: Hard cap on concurrent in-flight Authorize states. Each entry lives ≤10
    #: minutes (AUTH_CODE_TTL); 1024 covers ~30 users with 30+ retries each
    #: and bounds memory under abuse from non-allowlisted IPs that somehow
    #: bypass the NetworkPolicy.
    MAX_IN_FLIGHT_AUTHORIZE_STATES = 1024

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

        if len(self._authorize_states) >= self.MAX_IN_FLIGHT_AUTHORIZE_STATES:
            self.cleanup_authorize_states()
            if len(self._authorize_states) >= self.MAX_IN_FLIGHT_AUTHORIZE_STATES:
                raise ValueError(
                    "Too many in-flight authorize requests; try again later"
                )

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
        # Merge ``code`` and ``state`` into the registered redirect_uri while
        # preserving any pre-existing query string the client embedded.
        redirect_url = construct_redirect_uri(
            st.redirect_uri, code=code, state=st.claude_state
        )
        return code, redirect_url

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

        now = datetime.now(timezone.utc)
        family_id = secrets.token_urlsafe(16)
        mcp_access_token = secrets.token_urlsafe(32)
        mcp_refresh_token = secrets.token_urlsafe(32)

        await self._store.store_access_token(
            token=mcp_access_token,
            family_id=family_id,
            taiga_auth_token=consumed.taiga_auth_token,
            taiga_refresh_token=consumed.taiga_refresh_token,
            taiga_user_id=consumed.taiga_user_id,
            taiga_username=consumed.taiga_username,
            expires_at=now + ACCESS_TOKEN_TTL,
            client_id=consumed.client_id,
            scopes=consumed.scopes,
        )
        await self._store.store_refresh_token(
            token=mcp_refresh_token,
            family_id=family_id,
            client_id=consumed.client_id,
            taiga_auth_token=consumed.taiga_auth_token,
            taiga_refresh_token=consumed.taiga_refresh_token,
            taiga_user_id=consumed.taiga_user_id,
            taiga_username=consumed.taiga_username,
            scopes=consumed.scopes,
            expires_at=now + REFRESH_TOKEN_TTL,
        )
        return OAuthToken(
            access_token=mcp_access_token,
            refresh_token=mcp_refresh_token,
            token_type="Bearer",
            expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
            scope=" ".join(consumed.scopes),
        )

    # ---- Refresh tokens (rotation + reuse-detection) --------------------

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> Optional[RefreshToken]:
        """Lookup a refresh token by string and return the RefreshToken model.

        Reuse-detection happens HERE rather than only in exchange_refresh_token:
        if we return the record to mcp-sdk's TokenHandler and the SDK then
        rejects the request on a pre-check (e.g. invalid_scope), we never get
        a chance to revoke the family. By revoking-and-returning-None on a
        rotated_out record up front, the SDK sees invalid_grant and the
        attacker can't bypass the security check by submitting bad scopes.
        """
        record = await self._store.lookup_refresh_token(refresh_token)
        if record is None or record.client_id != client.client_id:
            return None
        if record.rotated_out:
            # Reuse-detection. Single source of truth for "rotated_out replay
            # means revoke" — the same branch in exchange_refresh_token is
            # defensive-only (mcp-sdk doesn't reach it after we return None
            # here, but anyone calling exchange_refresh_token directly would).
            revoked = await self._store.revoke_token_family(record.family_id)
            _log.warning(
                "Refresh-token reuse detected (via load) for family=%s; "
                "revoked %d tokens",
                record.family_id, revoked,
            )
            return None
        return RefreshToken(
            token=record.token,
            client_id=record.client_id,
            scopes=list(record.scopes),
            expires_at=int(record.expires_at.timestamp()),
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: List[str],
    ) -> OAuthToken:
        """Exchange a refresh token for a new access+refresh pair.

        Cascade-first ordering (Codex P2 fix): validate + cascade BEFORE
        consuming the MCP refresh token. A transient Taiga failure
        (5xx, ReadTimeout, malformed-200) therefore leaves the refresh
        token usable for retry rather than forcing a full re-OAuth.

        Flow:
        1. Peek lookup — no state change. None → invalid_grant.
        2. Pre-validate (client_id match, scope subset, defensive
           rotated_out guard for direct callers that bypass
           ``load_refresh_token``).
        3. Cascade Taiga refresh. Failure raises invalid_grant; the MCP
           refresh token is NOT rotated and the client can retry.
        4. Atomic ``consume_refresh_token``. Race-window protection: any
           ``already_rotated`` result here is a concurrent winner — revoke
           the family per OAuth 2.1 reuse-detection semantics.
        5. Atomic ``issue_new_generation`` — guards against a revoke that
           may have happened between consume and issue (concurrent
           load/exchange replay racing this branch).
        """
        # Step 1: peek lookup (no state change)
        record = await self._store.lookup_refresh_token(refresh_token.token)
        if record is None:
            raise TokenError("invalid_grant", "Refresh token unknown")

        # Defense-in-depth: ``load_refresh_token`` normally filters
        # rotated_out (and revokes the family there). This branch handles
        # direct callers that bypass load, plus the load/consume race
        # window where two callers reach exchange before the first
        # finishes consuming.
        if record.rotated_out:
            revoked = await self._store.revoke_token_family(record.family_id)
            _log.warning(
                "Refresh-token reuse detected (via exchange peek) for "
                "family=%s; revoked %d tokens",
                record.family_id, revoked,
            )
            raise TokenError("invalid_grant", "Refresh token already used")

        if record.client_id != client.client_id:
            raise TokenError(
                "invalid_client",
                "Refresh token issued to different client",
            )

        requested_scopes = list(scopes) if scopes else list(record.scopes)
        if not set(requested_scopes).issubset(set(record.scopes)):
            raise TokenError(
                "invalid_scope",
                "Requested scopes exceed original grant",
            )

        # Step 2: cascade BEFORE consuming. On cascade failure, the MCP
        # refresh token stays active and the client can retry.
        try:
            taiga = await self._taiga.refresh_taiga_token(record.taiga_refresh_token)
        except TaigaRefreshError as exc:
            _log.info(
                "Taiga refresh failed for family=%s; refresh token not "
                "rotated, client can retry the same refresh token: %s",
                record.family_id, exc,
            )
            raise TokenError("invalid_grant", "Upstream auth refresh failed")

        # Step 3: atomic consume (only now flip rotated_out)
        result = await self._store.consume_refresh_token(refresh_token.token)
        if result.status == "not_found":
            # Vanished between peek and consume — likely revoked by a
            # concurrent reuse-detection. Treat as already-used.
            raise TokenError("invalid_grant", "Refresh token unknown")
        if result.status == "expired":
            raise TokenError("invalid_grant", "Refresh token expired")
        if result.status == "already_rotated":
            # Race: another concurrent request consumed first. Revoke
            # the family per OAuth 2.1 reuse-detection semantics.
            revoked = await self._store.revoke_token_family(result.record.family_id)
            _log.warning(
                "Refresh-token reuse detected (via consume) for family=%s; "
                "revoked %d tokens",
                result.record.family_id, revoked,
            )
            raise TokenError("invalid_grant", "Refresh token already used")

        # status == "active"; record is now rotated_out. Use the freshly
        # returned record for symmetry (same data as the earlier peek).
        record = result.record

        # Step 4: atomic issue. Race-protected against a revoke that
        # may have happened between our consume and now.
        now = datetime.now(timezone.utc)
        new_access = secrets.token_urlsafe(32)
        new_refresh = secrets.token_urlsafe(32)
        issued = await self._store.issue_new_generation(
            family_id=record.family_id,
            access_token=new_access,
            refresh_token=new_refresh,
            taiga_auth_token=taiga.auth_token,
            taiga_refresh_token=taiga.refresh,
            taiga_user_id=record.taiga_user_id,
            taiga_username=record.taiga_username,
            client_id=record.client_id,
            access_scopes=requested_scopes,
            refresh_scopes=record.scopes,
            access_expires_at=now + ACCESS_TOKEN_TTL,
            refresh_expires_at=now + REFRESH_TOKEN_TTL,
        )
        if not issued:
            _log.warning(
                "Family was revoked during exchange for family=%s; "
                "refusing to issue new generation",
                record.family_id,
            )
            raise TokenError("invalid_grant", "Refresh token already used")

        return OAuthToken(
            access_token=new_access,
            refresh_token=new_refresh,
            token_type="Bearer",
            expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
            scope=" ".join(requested_scopes),
        )

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

        Logs only a sha256 digest of the token (8 hex chars) so audit-log
        readers can correlate "claude.ai asked us to revoke this", without
        leaking any prefix of the token bearer that would help an attacker
        with shoulder-surfed log access.
        """
        digest = (
            hashlib.sha256(token.encode()).hexdigest()[:8] if token else "empty"
        )
        _log.info(
            "revoke_token called (no-op in v1) for token-digest=%s...", digest,
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
