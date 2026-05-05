"""Unit tests for ``langchain_taiga.auth.provider.TaigaOAuthProvider``.

The centerpiece of PR 2: subclass of fastmcp's OAuthProvider that
authenticates against Taiga and surfaces the Taiga JWT via
``AccessToken.claims["taiga_jwt"]``.

Phase 0 deltas applied:
- AccessToken comes from fastmcp.server.auth (has claims field)
- ClientRegistrationOptions from mcp.server.auth.settings
- AuthorizationCode requires redirect_uri_provided_explicitly=True
- AuthorizationParams: state/scopes/code_challenge/redirect_uri/redirect_uri_provided_explicitly
- OAuthProvider __init__ requires both base_url and issuer_url
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from httpx import Response


def _make_provider(store):
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.taiga_client import TaigaClient

    return TaigaOAuthProvider(
        store=store,
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )


@pytest.fixture
def fresh_store():
    from langchain_taiga.auth.store import InMemoryStore

    return InMemoryStore()


def _make_client_info(
    redirect_uris,
    *,
    client_id="cid_test_1",
    client_secret="sec_test_1",
    client_name="Test",
    token_endpoint_auth_method="client_secret_post",
):
    """Construct an ``OAuthClientInformationFull`` matching the shape the
    mcp-sdk's ``RegistrationHandler`` produces in production (where
    ``client_id`` and ``client_secret`` are SDK-minted before
    ``provider.register_client`` is called)."""
    from mcp.shared.auth import OAuthClientInformationFull

    return OAuthClientInformationFull(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uris=redirect_uris,
        client_name=client_name,
        token_endpoint_auth_method=token_endpoint_auth_method,
    )


@pytest.mark.asyncio
async def test_register_client_rejects_userinfo_bypass(fresh_store):
    """Regression: ``http://localhost:8080@evil.com/cb`` must NOT pass the
    allowlist. With a naive ``startswith("http://localhost:")`` check, the URL
    parsed by browsers as host=evil.com would have been accepted, letting an
    attacker who DCR-registers such a redirect_uri receive victim auth codes.
    Also reject substring-tricks like ``https://claude.ai.attacker.com/cb``.
    """
    provider = _make_provider(fresh_store)
    with pytest.raises(ValueError, match="Redirect URI not allowed"):
        await provider.register_client(
            _make_client_info(
                redirect_uris=["http://localhost:8080@evil.com/cb"],
                client_name="userinfo-attacker",
            )
        )
    with pytest.raises(ValueError, match="Redirect URI not allowed"):
        await provider.register_client(
            _make_client_info(
                redirect_uris=["https://claude.ai.attacker.com/cb"],
                client_name="suffix-attacker",
            )
        )


@pytest.mark.asyncio
async def test_register_client_rejects_unallowed_redirect(fresh_store):
    """Open-redirect protection."""
    provider = _make_provider(fresh_store)
    with pytest.raises(ValueError, match="Redirect URI not allowed"):
        await provider.register_client(
            _make_client_info(
                redirect_uris=["https://attacker.example.com/steal"],
                client_name="Attacker",
            )
        )


@pytest.mark.asyncio
async def test_register_client_accepts_claude_ai_and_claude_com(fresh_store):
    provider = _make_provider(fresh_store)
    info_a = await provider.register_client(
        _make_client_info(
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            client_id="cid_a",
            client_secret="sec_a",
            client_name="Claude (claude.ai)",
            token_endpoint_auth_method="none",
        )
    )
    info_b = await provider.register_client(
        _make_client_info(
            redirect_uris=["https://claude.com/api/mcp/auth_callback"],
            client_id="cid_b",
            client_secret="sec_b",
            client_name="Claude (claude.com)",
            token_endpoint_auth_method="none",
        )
    )
    assert info_a.client_id != info_b.client_id


@pytest.mark.asyncio
async def test_register_client_accepts_vscode_redirect_uris(fresh_store):
    """Regression: VSCode's MCP integration submits BOTH a 127.0.0.1:<port>
    callback AND ``https://vscode.dev/redirect``. The latter used to be
    rejected by the redirect-URI allowlist, surfacing in VSCode as
    "Dynamic Client Registration not supported"."""
    provider = _make_provider(fresh_store)
    info = await provider.register_client(
        _make_client_info(
            redirect_uris=[
                "http://127.0.0.1:33418",
                "https://vscode.dev/redirect",
            ],
            client_id="vscode_cid",
            client_secret="sec",
            client_name="VSCode",
            token_endpoint_auth_method="none",
        )
    )
    # Both URIs round-tripped — the allowlist accepted both.
    assert "https://vscode.dev/redirect" in [str(u) for u in info.redirect_uris]


@pytest.mark.asyncio
async def test_get_client_returns_none_for_unknown(fresh_store):
    """Anthropic requires HTTP 400 invalid_client — provider returns None and
    FastMCP renders the error so claude.ai re-DCRs."""
    provider = _make_provider(fresh_store)
    assert await provider.get_client("never_registered") is None


@pytest.mark.asyncio
async def test_load_access_token_attaches_taiga_jwt_to_claims(fresh_store):
    """The load_access_token contract: returned AccessToken.claims must carry
    the Taiga JWT — what tools see via get_access_token().claims."""
    provider = _make_provider(fresh_store)
    await fresh_store.store_access_token(
        token="mcp_xyz",
        taiga_auth_token="taiga_jwt_456",
        taiga_refresh_token="ref",
        taiga_user_id=42,
        taiga_username="alice",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        client_id="c",
        scopes=["taiga"],
    )
    result = await provider.load_access_token("mcp_xyz")
    assert result is not None
    assert result.claims["taiga_jwt"] == "taiga_jwt_456"
    assert result.claims["user_id"] == 42
    assert result.claims["username"] == "alice"
    # expires_at is int seconds-since-epoch (Phase 0 delta)
    assert isinstance(result.expires_at, int)


@pytest.mark.asyncio
async def test_load_access_token_returns_none_for_unknown(fresh_store):
    provider = _make_provider(fresh_store)
    assert await provider.load_access_token("never_minted") is None


@pytest.mark.asyncio
async def test_failed_login_preserves_authorize_state(fresh_store, respx_mock):
    """A TaigaAuthenticationError on complete_login must NOT consume the
    authorize state, so the user can retry their password."""
    from langchain_taiga.auth.taiga_client import TaigaAuthenticationError
    from mcp.server.auth.provider import AuthorizationParams

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(400, json={"_error_message": "Bad creds"})
    )

    provider = _make_provider(fresh_store)
    client_info = await provider.register_client(
        _make_client_info(
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            client_name="Claude",
            token_endpoint_auth_method="none",
        )
    )
    redirect = await provider.authorize(
        client=client_info,
        params=AuthorizationParams(
            state="csrf",
            scopes=["taiga"],
            code_challenge="cc",
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            redirect_uri_provided_explicitly=True,
        ),
    )
    internal_state = redirect.split("internal_state=", 1)[1]

    with pytest.raises(TaigaAuthenticationError):
        await provider.complete_login(
            internal_state=internal_state, username="alice", password="wrong"
        )

    # State must still be there for retry
    assert internal_state in provider._authorize_states


@pytest.mark.asyncio
async def test_metadata_endpoint_includes_path_aware_issuer(fresh_store):
    """If FastMCP exposes a metadata helper, verify the issuer matches our
    construction. Otherwise skip — the route is still auto-mounted by
    FastMCP at runtime."""
    provider = _make_provider(fresh_store)
    if not hasattr(provider, "authorization_server_metadata"):
        pytest.skip(
            "fastmcp.OAuthProvider has no authorization_server_metadata "
            "helper; route is mounted by FastMCP at runtime."
        )
    metadata = provider.authorization_server_metadata()
    # If somehow it does exist, check the issuer:
    assert "taiga.shikenso.org/mcp" in str(metadata)


@pytest.mark.asyncio
async def test_full_auth_flow(fresh_store, respx_mock):
    """Authorize → login → exchange_authorization_code → AccessToken with
    taiga_jwt in claims. Uses real PKCE pair."""
    from mcp.server.auth.provider import AuthorizationParams

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(
            200,
            json={
                "auth_token": "alice_jwt",
                "refresh": "alice_ref",
                "id": 42,
                "username": "alice",
            },
        )
    )

    provider = _make_provider(fresh_store)
    client_info = await provider.register_client(
        _make_client_info(
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            client_name="Claude",
            token_endpoint_auth_method="none",
        )
    )

    # Real PKCE pair (S256)
    verifier = "verifier_for_alice_with_enough_entropy_xx"
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )

    redirect = await provider.authorize(
        client=client_info,
        params=AuthorizationParams(
            state="claude_csrf",
            scopes=["taiga"],
            code_challenge=challenge,
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            redirect_uri_provided_explicitly=True,
        ),
    )
    assert redirect.startswith(
        "https://taiga.shikenso.org/mcp/oauth/login?internal_state="
    )

    internal_state = redirect.split("internal_state=", 1)[1]
    code, redirect_url = await provider.complete_login(
        internal_state=internal_state, username="alice", password="x"
    )
    assert "code=" in redirect_url
    assert "state=claude_csrf" in redirect_url

    # FastMCP uses load_authorization_code first, then exchange_*
    auth_code_obj = await provider.load_authorization_code(client_info, code)
    assert auth_code_obj is not None
    assert auth_code_obj.redirect_uri_provided_explicitly is True

    oauth_token = await provider.exchange_authorization_code(
        client=client_info, authorization_code=auth_code_obj
    )
    assert oauth_token.access_token

    # The minted MCP token, looked up via load_access_token, must carry the
    # Taiga JWT in claims.
    access = await provider.load_access_token(oauth_token.access_token)
    assert access is not None
    assert access.claims["taiga_jwt"] == "alice_jwt"
    assert access.claims["user_id"] == 42
    assert access.claims["username"] == "alice"


async def _drive_to_authorization_code(provider, fresh_store, *, username="alice"):
    """Helper: register a client, run authorize → complete_login →
    load_authorization_code, returning ``(client_info, auth_code_obj, code)``.

    Caller is responsible for mocking the Taiga ``/api/v1/auth`` endpoint."""
    from mcp.server.auth.provider import AuthorizationParams

    client_info = _make_client_info(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_name="Claude",
        token_endpoint_auth_method="none",
    )
    await provider.register_client(client_info)
    verifier = "v_" + username + "x" * 50
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    redirect = await provider.authorize(
        client=client_info,
        params=AuthorizationParams(
            state="csrf",
            scopes=["taiga"],
            code_challenge=challenge,
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            redirect_uri_provided_explicitly=True,
        ),
    )
    internal_state = redirect.split("internal_state=", 1)[1]
    code, _redirect_url = await provider.complete_login(
        internal_state=internal_state, username=username, password="x"
    )
    auth_code_obj = await provider.load_authorization_code(client_info, code)
    assert auth_code_obj is not None
    return client_info, auth_code_obj, code


@pytest.mark.asyncio
async def test_exchange_code_with_used_code_raises_invalid_grant(
    fresh_store, respx_mock
):
    """A second exchange of the same code must surface ``TokenError`` with
    ``error="invalid_grant"`` so mcp-sdk's TokenHandler renders an
    RFC 6749 § 5.2 400 response instead of a generic 500."""
    from mcp.server.auth.provider import TokenError

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(
            200,
            json={
                "auth_token": "alice_jwt",
                "refresh": "alice_ref",
                "id": 42,
                "username": "alice",
            },
        )
    )

    provider = _make_provider(fresh_store)
    client_info, auth_code_obj, _ = await _drive_to_authorization_code(
        provider, fresh_store
    )

    # First exchange succeeds and consumes the code.
    await provider.exchange_authorization_code(client_info, auth_code_obj)

    # Second exchange of the same code must raise TokenError("invalid_grant").
    with pytest.raises(TokenError) as excinfo:
        await provider.exchange_authorization_code(client_info, auth_code_obj)
    assert excinfo.value.error == "invalid_grant"


@pytest.mark.asyncio
async def test_exchange_code_with_wrong_client_raises_invalid_client(
    fresh_store, respx_mock
):
    """A confused-deputy attacker presenting a code with the wrong
    ``client_id`` must trip ``TokenError("invalid_client")`` so the handler
    returns RFC 6749 § 5.2 invalid_client instead of a generic 500."""
    from mcp.shared.auth import OAuthClientInformationFull
    from mcp.server.auth.provider import TokenError

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(
            200,
            json={
                "auth_token": "alice_jwt",
                "refresh": "alice_ref",
                "id": 42,
                "username": "alice",
            },
        )
    )

    provider = _make_provider(fresh_store)
    legitimate_client, auth_code_obj, _ = await _drive_to_authorization_code(
        provider, fresh_store
    )

    # Forge a different client object with the same redirect_uri
    forged_client = OAuthClientInformationFull(
        client_id="some_other_client",
        client_secret=None,
        redirect_uris=legitimate_client.redirect_uris,
        client_name="Forged",
        token_endpoint_auth_method="none",
    )
    with pytest.raises(TokenError) as excinfo:
        await provider.exchange_authorization_code(forged_client, auth_code_obj)
    assert excinfo.value.error == "invalid_client"


@pytest.mark.asyncio
async def test_complete_login_preserves_existing_query_in_redirect_uri(
    fresh_store, respx_mock
):
    """If the registered ``redirect_uri`` already carries a query string,
    ``complete_login`` must merge ``code`` + ``state`` into it without
    producing a malformed double-``?`` URL like ``...cb?foo=bar?code=...``."""
    from mcp.server.auth.provider import AuthorizationParams

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(
            200,
            json={
                "auth_token": "alice_jwt",
                "refresh": "alice_ref",
                "id": 42,
                "username": "alice",
            },
        )
    )

    provider = _make_provider(fresh_store)
    client_info = _make_client_info(
        redirect_uris=["https://claude.ai/cb?foo=bar"],
        client_name="Claude with query",
        token_endpoint_auth_method="none",
    )
    await provider.register_client(client_info)

    redirect = await provider.authorize(
        client=client_info,
        params=AuthorizationParams(
            state="csrf_state",
            scopes=["taiga"],
            code_challenge="cc",
            redirect_uri="https://claude.ai/cb?foo=bar",
            redirect_uri_provided_explicitly=True,
        ),
    )
    internal_state = redirect.split("internal_state=", 1)[1]
    _, redirect_url = await provider.complete_login(
        internal_state=internal_state, username="alice", password="x"
    )

    # Single ``?`` separator only.
    assert redirect_url.count("?") == 1, redirect_url
    # Pre-existing query parameter survives.
    assert "foo=bar" in redirect_url
    # Newly-injected parameters present.
    assert "code=" in redirect_url
    assert "state=csrf_state" in redirect_url


@pytest.mark.asyncio
async def test_exchange_code_with_redirect_uri_mismatch_raises_invalid_grant(
    fresh_store, respx_mock
):
    """If the AuthorizationCode passed in has a different redirect_uri than
    what was stored at /authorize, raise ``TokenError("invalid_grant")``.
    Defense-in-depth: mcp-sdk's TokenHandler already enforces this against
    the token-request body, but our internal check guards the direct path
    in ``exchange_authorization_code`` itself."""
    from mcp.server.auth.provider import AuthorizationCode, TokenError

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(
            200,
            json={
                "auth_token": "alice_jwt",
                "refresh": "alice_ref",
                "id": 42,
                "username": "alice",
            },
        )
    )

    provider = _make_provider(fresh_store)
    client_info, auth_code_obj, _ = await _drive_to_authorization_code(
        provider, fresh_store
    )

    tampered = AuthorizationCode(
        code=auth_code_obj.code,
        client_id=auth_code_obj.client_id,
        redirect_uri="https://claude.ai/different/callback",
        code_challenge=auth_code_obj.code_challenge,
        scopes=list(auth_code_obj.scopes),
        expires_at=auth_code_obj.expires_at,
        redirect_uri_provided_explicitly=True,
    )
    with pytest.raises(TokenError) as excinfo:
        await provider.exchange_authorization_code(client_info, tampered)
    assert excinfo.value.error == "invalid_grant"
