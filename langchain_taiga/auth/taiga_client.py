"""HTTP client for Taiga's native auth endpoints.

Wraps Taiga's ``/api/v1/auth`` (credential exchange) and
``/api/v1/auth/refresh`` (refresh token) endpoints with async httpx and
typed dataclasses. Used by ``TaigaOAuthProvider`` for the credential
exchange step of the MCP OAuth flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


class TaigaAuthenticationError(Exception):
    """User credentials rejected by Taiga."""


class TaigaRefreshError(Exception):
    """Refresh token rejected — claude.ai must restart OAuth."""


class TaigaRefreshTokenInvalidError(TaigaRefreshError):
    """Taiga rejected the refresh token (4xx, typically 401).

    Distinguished from generic TaigaRefreshError because the provider
    treats it as a concurrent-replay signal and revokes the token family,
    rather than as a transient outage that should preserve the family.
    """


@dataclass
class TaigaCredentials:
    auth_token: str
    refresh: str
    user_id: int
    username: str


@dataclass
class RefreshedTokens:
    auth_token: str
    refresh: str


class TaigaClient:
    """Async client for Taiga's username/password auth endpoints."""

    def __init__(self, api_url: str, timeout: float = 10.0):
        self._api_url = api_url.rstrip("/")
        self._timeout = timeout

    async def authenticate_user(
        self, username: str, password: str
    ) -> TaigaCredentials:
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            resp = await http.post(
                f"{self._api_url}/api/v1/auth",
                json={
                    "type": "normal",
                    "username": username,
                    "password": password,
                },
            )
        if resp.status_code != 200:
            raise TaigaAuthenticationError(
                f"Taiga auth failed: {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()
        return TaigaCredentials(
            auth_token=body["auth_token"],
            refresh=body["refresh"],
            user_id=int(body["id"]),
            username=body["username"],
        )

    async def refresh_taiga_token(self, refresh_token: str) -> RefreshedTokens:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                resp = await http.post(
                    f"{self._api_url}/api/v1/auth/refresh",
                    json={"refresh": refresh_token},
                )
        except httpx.RequestError as exc:
            # Transient transport blips (ReadTimeout, ConnectError, ...) MUST
            # surface as TaigaRefreshError so the provider's cascade-failure
            # handler catches them and keeps the family alive. Letting these
            # propagate would skip the except-TaigaRefreshError branch, the
            # request would 500, and the user's retry — with the now
            # rotated_out refresh token — would trigger reuse-detection and
            # mass-revoke the family on every transient network blip.
            raise TaigaRefreshError(
                f"Taiga refresh transport failed: {exc!r}"
            ) from exc
        if resp.status_code == 200:
            # Same rationale as refresh-transport failures above: a malformed
            # 200 (bad JSON or missing auth_token/refresh fields) MUST surface
            # as TaigaRefreshError so the provider's cascade-failure branch
            # catches it and keeps the family alive. Otherwise the response
            # bubbles as a 500, the user retries with the now rotated_out
            # refresh token, and reuse-detection mass-revokes the family on
            # every Taiga blip.
            try:
                body = resp.json()
                return RefreshedTokens(
                    auth_token=body["auth_token"], refresh=body["refresh"]
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise TaigaRefreshError(
                    f"Taiga refresh response malformed: {exc!r}"
                ) from exc
        if resp.status_code == 429:
            # Rate-limited — transient even though it's a 4xx. Surfaces as
            # the generic TaigaRefreshError so the provider preserves the
            # family and lets the client retry the same refresh token.
            raise TaigaRefreshError(
                f"Taiga refresh rate-limited: {resp.status_code}"
            )
        if 400 <= resp.status_code < 500:
            # Other 4xx → Taiga is telling us the refresh token is invalid.
            # Typical cause: the token was already rotated server-side by a
            # concurrent request. Surface as the invalid-token subtype so
            # the provider treats it as a concurrent-replay signal and
            # revokes the family per OAuth 2.1 reuse-detection.
            raise TaigaRefreshTokenInvalidError(
                f"Taiga rejected refresh token: "
                f"{resp.status_code} {resp.text[:200]}"
            )
        # 5xx — server-side outage, transient. Preserve the family.
        raise TaigaRefreshError(
            f"Taiga refresh failed: {resp.status_code} {resp.text[:200]}"
        )
