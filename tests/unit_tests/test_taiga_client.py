"""Unit tests for ``langchain_taiga.auth.taiga_client.TaigaClient``.

Uses the pytest-respx ``respx_mock`` fixture to mock httpx outbound — no
network required.
"""

from __future__ import annotations

import pytest
from httpx import Response


@pytest.mark.asyncio
async def test_authenticate_user_returns_credentials(respx_mock):
    from langchain_taiga.auth.taiga_client import TaigaClient

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(
            200,
            json={"auth_token": "jwt", "refresh": "ref", "id": 42, "username": "alice"},
        )
    )
    client = TaigaClient(api_url="https://taiga.example.test")
    creds = await client.authenticate_user("alice", "wonderland")
    assert creds.auth_token == "jwt"
    assert creds.refresh == "ref"
    assert creds.user_id == 42
    assert creds.username == "alice"


@pytest.mark.asyncio
async def test_authenticate_user_invalid_raises(respx_mock):
    from langchain_taiga.auth.taiga_client import (
        TaigaAuthenticationError,
        TaigaClient,
    )

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(400, json={"_error_message": "Invalid"})
    )
    client = TaigaClient(api_url="https://taiga.example.test")
    with pytest.raises(TaigaAuthenticationError):
        await client.authenticate_user("alice", "wrong")


@pytest.mark.asyncio
async def test_refresh_token(respx_mock):
    from langchain_taiga.auth.taiga_client import TaigaClient

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(200, json={"auth_token": "new", "refresh": "newref"})
    )
    client = TaigaClient(api_url="https://taiga.example.test")
    refreshed = await client.refresh_taiga_token("old_refresh")
    assert refreshed.auth_token == "new"
    assert refreshed.refresh == "newref"


@pytest.mark.asyncio
async def test_refresh_failure_raises(respx_mock):
    from langchain_taiga.auth.taiga_client import (
        TaigaClient,
        TaigaRefreshError,
    )

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(401, json={"_error_message": "Expired"})
    )
    client = TaigaClient(api_url="https://taiga.example.test")
    with pytest.raises(TaigaRefreshError):
        await client.refresh_taiga_token("dead")
