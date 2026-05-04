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
                json={"type": "normal", "username": username, "password": password},
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
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            resp = await http.post(
                f"{self._api_url}/api/v1/auth/refresh",
                json={"refresh": refresh_token},
            )
        if resp.status_code != 200:
            raise TaigaRefreshError(
                f"Taiga refresh failed: {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()
        return RefreshedTokens(auth_token=body["auth_token"], refresh=body["refresh"])
