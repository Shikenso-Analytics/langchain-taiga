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
    """Regression: VSCode's MCP integration submits all three redirect
    URIs (loopback + vscode.dev + insiders.vscode.dev) in a single DCR
    request — including users who have Settings Sync turned on with
    Insiders. Any rejected URI fails the entire registration."""
    provider = _make_provider(fresh_store)
    info = await provider.register_client(
        _make_client_info(
            redirect_uris=[
                "http://127.0.0.1:33418",
                "https://vscode.dev/redirect",
                "https://insiders.vscode.dev/redirect",
            ],
            client_id="vscode_cid",
            client_secret="sec",
            client_name="VSCode",
            token_endpoint_auth_method="none",
        )
    )
    # All three URIs round-tripped — the allowlist accepted them all.
    redirects = [str(u) for u in info.redirect_uris]
    assert "https://vscode.dev/redirect" in redirects
    assert "https://insiders.vscode.dev/redirect" in redirects


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


@pytest.mark.asyncio
async def test_authorization_code_exchange_issues_refresh_token(
    fresh_store, respx_mock
):
    """v2.5.0: exchange_authorization_code must include a refresh_token in
    the OAuthToken response, persist it in the store, and link both access
    and refresh under the same family_id."""
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

    verifier = "verifier_for_alice_with_enough_entropy_xx"
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
    code, _ = await provider.complete_login(
        internal_state=internal_state, username="alice", password="x"
    )
    auth_code_obj = await provider.load_authorization_code(client_info, code)
    oauth_token = await provider.exchange_authorization_code(
        client=client_info, authorization_code=auth_code_obj
    )

    assert oauth_token.access_token
    assert oauth_token.refresh_token is not None
    assert oauth_token.refresh_token != oauth_token.access_token

    # Both tokens must be in the store with the SAME family_id
    access_record = await fresh_store.lookup_access_token(oauth_token.access_token)
    refresh_record = await fresh_store.lookup_refresh_token(oauth_token.refresh_token)
    assert access_record is not None
    assert refresh_record is not None
    assert access_record.family_id == refresh_record.family_id
    assert access_record.family_id != ""


@pytest.mark.asyncio
async def test_load_refresh_token_returns_token_record_for_known(fresh_store):
    """load_refresh_token must return a RefreshToken model for a token
    stored by the provider, no longer the v1 stub-None."""
    # Seed a refresh token directly into the store
    await fresh_store.store_refresh_token(
        token="ref_seed",
        family_id="fam",
        client_id="cid_test_1",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )
    provider = _make_provider(fresh_store)
    client_info = _make_client_info(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_id="cid_test_1",
        token_endpoint_auth_method="none",
    )
    token = await provider.load_refresh_token(client_info, "ref_seed")
    assert token is not None
    assert token.token == "ref_seed"
    assert "taiga" in token.scopes


@pytest.mark.asyncio
async def test_load_refresh_token_returns_none_for_unknown(fresh_store):
    provider = _make_provider(fresh_store)
    client_info = _make_client_info(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_id="cid_test_1",
        token_endpoint_auth_method="none",
    )
    assert await provider.load_refresh_token(client_info, "ghost") is None


@pytest.mark.asyncio
async def test_load_refresh_token_returns_none_for_cross_client(fresh_store):
    """A refresh issued to client A must not be loadable by client B."""
    await fresh_store.store_refresh_token(
        token="ref_other",
        family_id="fam",
        client_id="cid_owner",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga"],
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )
    provider = _make_provider(fresh_store)
    other = _make_client_info(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_id="cid_intruder",
        token_endpoint_auth_method="none",
    )
    assert await provider.load_refresh_token(other, "ref_other") is None


async def _do_full_auth_flow(provider, fresh_store, respx_mock):
    """Walk a registered client through authorize → login → exchange to
    arrive at (oauth_token, family_id). Helper used by refresh tests below."""
    from mcp.server.auth.provider import AuthorizationParams

    respx_mock.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(
            200,
            json={
                "auth_token": "alice_jwt_v1",
                "refresh": "alice_ref_v1",
                "id": 42,
                "username": "alice",
            },
        )
    )
    client_info = await provider.register_client(
        _make_client_info(
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            client_name="Claude",
            token_endpoint_auth_method="none",
        )
    )
    verifier = "verifier_for_alice_with_enough_entropy_xx"
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
    code, _ = await provider.complete_login(
        internal_state=internal_state, username="alice", password="x"
    )
    auth_code_obj = await provider.load_authorization_code(client_info, code)
    oauth_token = await provider.exchange_authorization_code(
        client=client_info, authorization_code=auth_code_obj
    )
    return client_info, oauth_token


@pytest.mark.asyncio
async def test_refresh_token_exchange_returns_new_tokens(
    fresh_store, respx_mock
):
    """Refresh roundtrip: new access + new refresh, both different from the
    originals, with refresh_token field populated in the response."""
    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    # Mock the Taiga refresh endpoint for the cascade
    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(
            200,
            json={"auth_token": "alice_jwt_v2", "refresh": "alice_ref_v2"},
        )
    )

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    new_oauth = await provider.exchange_refresh_token(
        client=client_info, refresh_token=refresh_obj, scopes=[]
    )
    assert new_oauth.access_token
    assert new_oauth.access_token != oauth.access_token
    assert new_oauth.refresh_token
    assert new_oauth.refresh_token != oauth.refresh_token


@pytest.mark.asyncio
async def test_refresh_token_cascades_to_taiga_refresh(
    fresh_store, respx_mock
):
    """The /api/v1/auth/refresh endpoint must be called with the stored
    taiga_refresh_token, and the rotated Taiga JWT must land in the new
    access-token record."""
    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    refresh_route = respx_mock.post(
        "https://taiga.example.test/api/v1/auth/refresh"
    ).mock(
        return_value=Response(
            200,
            json={"auth_token": "alice_jwt_v2", "refresh": "alice_ref_v2"},
        )
    )

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    new_oauth = await provider.exchange_refresh_token(
        client=client_info, refresh_token=refresh_obj, scopes=[]
    )

    assert refresh_route.called
    # Body must carry the originally-stored taiga refresh token
    body = refresh_route.calls[0].request.read().decode()
    assert "alice_ref_v1" in body

    # The new MCP access-token record must hold the rotated Taiga JWT
    new_access_record = await fresh_store.lookup_access_token(new_oauth.access_token)
    assert new_access_record.taiga_auth_token == "alice_jwt_v2"
    assert new_access_record.taiga_refresh_token == "alice_ref_v2"


@pytest.mark.asyncio
async def test_refresh_token_family_persists_across_multiple_refreshes(
    fresh_store, respx_mock
):
    """3× consecutive refreshes; all generations share the same family_id."""
    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(
            200,
            json={"auth_token": "jwt_v2", "refresh": "ref_v2"},
        )
    )

    initial_access = await fresh_store.lookup_access_token(oauth.access_token)
    expected_family = initial_access.family_id

    current = oauth
    for _ in range(3):
        refresh_obj = await provider.load_refresh_token(client_info, current.refresh_token)
        current = await provider.exchange_refresh_token(
            client=client_info, refresh_token=refresh_obj, scopes=[]
        )
        access_record = await fresh_store.lookup_access_token(current.access_token)
        refresh_record = await fresh_store.lookup_refresh_token(current.refresh_token)
        assert access_record.family_id == expected_family
        assert refresh_record.family_id == expected_family


@pytest.mark.asyncio
async def test_old_refresh_token_invalid_after_rotation(
    fresh_store, respx_mock
):
    """After exchange, the original refresh token must be rejected on
    second presentation. Reuse-detection (covered by a separate test) now
    fires inside ``load_refresh_token`` itself, so the second load
    short-circuits to None instead of returning a stale RefreshToken."""
    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(
            200, json={"auth_token": "jwt_v2", "refresh": "ref_v2"}
        )
    )

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    await provider.exchange_refresh_token(
        client=client_info, refresh_token=refresh_obj, scopes=[]
    )
    # Second presentation of the SAME refresh_token: load returns None,
    # which is mcp-sdk's signal to surface invalid_grant to the OAuth client.
    refresh_obj_2 = await provider.load_refresh_token(client_info, oauth.refresh_token)
    assert refresh_obj_2 is None


@pytest.mark.asyncio
async def test_taiga_refresh_failure_preserves_family_and_old_access_token(
    fresh_store, respx_mock, caplog
):
    """Cascade failure must NOT revoke the family. Previously-issued access
    token continues to resolve. Under the cascade-first ordering (Codex P2),
    the refresh token is also NOT rotated on cascade failure — the client
    can retry the same refresh token. The "retry succeeds" leg is covered
    by ``test_transient_cascade_failure_allows_retry_with_same_refresh_token``;
    here we just assert the family + access-token survival contract."""
    import logging
    from mcp.server.auth.provider import TokenError

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(500, text="taiga down")
    )

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    with caplog.at_level(logging.INFO, logger="langchain_taiga.auth.provider"):
        with pytest.raises(TokenError) as excinfo:
            await provider.exchange_refresh_token(
                client=client_info, refresh_token=refresh_obj, scopes=[]
            )
    assert excinfo.value.error == "invalid_grant"
    # Defense: Taiga's response text must NOT leak into the OAuth client's error
    # response — operator gets it via the INFO log, client sees a generic message.
    assert "taiga down" not in str(excinfo.value)

    # The original access_token must STILL be valid (family preserved)
    assert await fresh_store.lookup_access_token(oauth.access_token) is not None
    # Only INFO log, no WARNING (not a security event)
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records == []


@pytest.mark.asyncio
async def test_reusing_rotated_out_refresh_revokes_entire_family(
    fresh_store, respx_mock, caplog
):
    """Replay of a rotated_out refresh token: family revoke covers ALL
    access + refresh tokens that share the family_id, including the
    just-issued new ones.

    Reuse-detection now fires inside ``load_refresh_token`` (so the SDK's
    pre-checks can't bypass it). The replay leg is therefore JUST the load
    call — no exchange_refresh_token is needed to trigger the revoke.
    """
    import logging

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(
            200, json={"auth_token": "jwt_v2", "refresh": "ref_v2"}
        )
    )

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    rotated = await provider.exchange_refresh_token(
        client=client_info, refresh_token=refresh_obj, scopes=[]
    )
    # All four tokens currently exist in the store
    assert await fresh_store.lookup_access_token(oauth.access_token) is not None
    assert await fresh_store.lookup_access_token(rotated.access_token) is not None
    assert await fresh_store.lookup_refresh_token(rotated.refresh_token) is not None

    # Replay: re-present the original (now rotated_out) refresh. The load
    # call alone must trigger reuse-detection, revoke the family, and
    # return None (so mcp-sdk surfaces invalid_grant on the SDK side).
    with caplog.at_level(logging.WARNING, logger="langchain_taiga.auth.provider"):
        refresh_replay = await provider.load_refresh_token(
            client_info, oauth.refresh_token
        )
    assert refresh_replay is None

    # ALL four tokens (across both generations of the family) are now gone
    assert await fresh_store.lookup_access_token(oauth.access_token) is None
    assert await fresh_store.lookup_access_token(rotated.access_token) is None
    assert await fresh_store.lookup_refresh_token(oauth.refresh_token) is None
    assert await fresh_store.lookup_refresh_token(rotated.refresh_token) is None


@pytest.mark.asyncio
async def test_reuse_detection_logs_security_warning(
    fresh_store, respx_mock, caplog
):
    """The replay event must surface at WARNING level for audit logs.

    The warning is emitted by ``load_refresh_token`` (the load-path
    reuse-detection branch added in v2.5.0 — see ``Bug A`` rationale).
    """
    import logging

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(
            200, json={"auth_token": "jwt_v2", "refresh": "ref_v2"}
        )
    )

    first = await provider.load_refresh_token(client_info, oauth.refresh_token)
    await provider.exchange_refresh_token(
        client=client_info, refresh_token=first, scopes=[]
    )
    with caplog.at_level(logging.WARNING, logger="langchain_taiga.auth.provider"):
        second = await provider.load_refresh_token(client_info, oauth.refresh_token)
    assert second is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("reuse detected" in r.getMessage() for r in warnings)


@pytest.mark.asyncio
async def test_concurrent_exchange_same_refresh_token_one_wins_other_revokes_family(
    fresh_store, respx_mock
):
    """asyncio.gather two exchange calls on the same refresh token.

    Atomicity of consume_refresh_token: exactly one observes status=active,
    the other observes status=already_rotated and revokes the family.
    Net result: even the winner's tokens are revoked by the
    ``issue_new_generation`` guard (family was revoked between consume and
    issue).

    Both attempts share a SINGLE pre-fetched RefreshToken object — this
    matches the mcp-sdk production trace (TokenHandler calls
    load_refresh_token once, then exchange_refresh_token), and exercises the
    consume-side defense-in-depth branch rather than the load-side
    reuse-detection branch.
    """
    import asyncio
    from mcp.server.auth.provider import TokenError

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(
            200, json={"auth_token": "jwt_v2", "refresh": "ref_v2"}
        )
    )

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)

    async def attempt():
        return await provider.exchange_refresh_token(
            client=client_info, refresh_token=refresh_obj, scopes=[]
        )

    results = await asyncio.gather(attempt(), attempt(), return_exceptions=True)
    # Exactly one success, exactly one TokenError
    successes = [r for r in results if not isinstance(r, BaseException)]
    failures = [r for r in results if isinstance(r, TokenError)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert failures[0].error == "invalid_grant"

    # The winner's just-issued tokens are gone — family was revoked by the
    # losing call's reuse-detection branch.
    winner_oauth = successes[0]
    assert await fresh_store.lookup_access_token(winner_oauth.access_token) is None
    assert await fresh_store.lookup_refresh_token(winner_oauth.refresh_token) is None


@pytest.mark.asyncio
async def test_refresh_cannot_escalate_scopes(fresh_store, respx_mock):
    """A request for a scope superset of the original grant is rejected
    with invalid_scope. Under the cascade-first ordering (Codex P2), the
    scope check happens before consume, so the refresh token remains
    active — the client may retry with a valid scope subset."""
    from mcp.server.auth.provider import TokenError

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    with pytest.raises(TokenError) as excinfo:
        await provider.exchange_refresh_token(
            client=client_info,
            refresh_token=refresh_obj,
            scopes=["taiga", "admin"],
        )
    assert excinfo.value.error == "invalid_scope"


@pytest.mark.asyncio
async def test_refresh_with_subset_scopes_allowed(fresh_store, respx_mock):
    """Requesting fewer scopes than the original grant succeeds; the new
    access token carries only the requested subset."""
    provider = _make_provider(fresh_store)
    # Seed a multi-scope grant directly into the store (bypass auth flow for
    # scope variety — the helper hardcodes ["taiga"]).
    fam = "fam_subset"
    expires_a = datetime.now(timezone.utc) + timedelta(hours=1)
    expires_r = datetime.now(timezone.utc) + timedelta(days=30)
    await fresh_store.store_access_token(
        token="acc_seed",
        family_id=fam,
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        client_id="cid_test_1",
        scopes=["taiga", "read"],
        expires_at=expires_a,
    )
    await fresh_store.store_refresh_token(
        token="ref_seed",
        family_id=fam,
        client_id="cid_test_1",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="x",
        scopes=["taiga", "read"],
        expires_at=expires_r,
    )

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(
            200, json={"auth_token": "jwt_v2", "refresh": "ref_v2"}
        )
    )

    client_info = _make_client_info(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_id="cid_test_1",
        token_endpoint_auth_method="none",
    )
    refresh_obj = await provider.load_refresh_token(client_info, "ref_seed")
    new_oauth = await provider.exchange_refresh_token(
        client=client_info, refresh_token=refresh_obj, scopes=["read"]
    )
    assert "read" in new_oauth.scope
    assert "taiga" not in new_oauth.scope


@pytest.mark.asyncio
async def test_refresh_token_from_different_client_rejected(
    fresh_store, respx_mock
):
    """Client A's refresh + Client B's identity → invalid_client.

    Note: in production, mcp-sdk's load_refresh_token filters this earlier
    (returns None for cross-client), but exchange_refresh_token has a
    defensive check too. We exercise that defensive path by seeding the
    record directly and bypassing the load step's client filter.
    """
    from mcp.server.auth.provider import RefreshToken, TokenError

    await fresh_store.store_refresh_token(
        token="ref_alice",
        family_id="fam",
        client_id="cid_alice",
        taiga_auth_token="t",
        taiga_refresh_token="r",
        taiga_user_id=1,
        taiga_username="alice",
        scopes=["taiga"],
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )

    provider = _make_provider(fresh_store)
    bob_info = _make_client_info(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_id="cid_bob",
        token_endpoint_auth_method="none",
    )
    # Construct a RefreshToken model directly to bypass load_refresh_token's
    # cross-client filter and reach exchange_refresh_token's defensive check.
    rt = RefreshToken(
        token="ref_alice",
        client_id="cid_alice",
        scopes=["taiga"],
        expires_at=int(
            (datetime.now(timezone.utc) + timedelta(days=30)).timestamp()
        ),
    )
    with pytest.raises(TokenError) as excinfo:
        await provider.exchange_refresh_token(
            client=bob_info, refresh_token=rt, scopes=[]
        )
    assert excinfo.value.error == "invalid_client"


@pytest.mark.asyncio
async def test_concurrent_exchange_with_cascade_does_not_resurrect_revoked_family(
    fresh_store, respx_mock
):
    """Race protection: a refresh suspended in Taiga cascade must NOT
    write back tokens after another coroutine revoked the family.

    Under the cascade-first ordering (Codex P2 fix) the cascade happens
    BEFORE consume_refresh_token, so suspending at the cascade gate leaves
    the refresh token active in the store. A concurrent replay therefore
    cannot trigger load-path reuse-detection. We instead simulate the
    "family revoked while a refresh is in flight" race directly via
    ``revoke_token_family``. The slow task must abort cleanly — either
    via the consume's not_found branch (token was purged by revoke) or
    via ``issue_new_generation`` refusing to write into a revoked family.
    Either way, ``invalid_grant`` and no resurrection.
    """
    import asyncio
    from mcp.server.auth.provider import TokenError

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    # Use a gate to deterministically suspend the cascade
    cascade_gate = asyncio.Event()

    async def slow_taiga_refresh(taiga_refresh_token: str):
        from langchain_taiga.auth.taiga_client import RefreshedTokens
        await cascade_gate.wait()
        return RefreshedTokens(auth_token="jwt_v2", refresh="ref_v2")

    provider._taiga.refresh_taiga_token = slow_taiga_refresh  # type: ignore

    # Kick off the slow refresh; it suspends at cascade_gate.wait() BEFORE
    # the consume call, so the refresh token is still active in the store.
    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    slow_task = asyncio.create_task(
        provider.exchange_refresh_token(
            client=client_info, refresh_token=refresh_obj, scopes=[]
        )
    )
    await asyncio.sleep(0)  # let slow_task suspend at the cascade gate

    # Simulate a concurrent revoke of the entire family while the slow
    # task is parked in the cascade. This is the security property we
    # care about: in-flight refreshes must not resurrect a revoked family.
    access_record = await fresh_store.lookup_access_token(oauth.access_token)
    assert access_record is not None
    await fresh_store.revoke_token_family(access_record.family_id)

    # Release the slow task. After cascade returns it calls
    # consume_refresh_token (the token has been purged → "not_found") or
    # issue_new_generation (family is in _revoked_families → False).
    # Either path raises invalid_grant.
    cascade_gate.set()
    with pytest.raises(TokenError) as excinfo:
        await slow_task
    assert excinfo.value.error == "invalid_grant"

    # The family stays revoked: original access token gone, no resurrection.
    assert await fresh_store.lookup_access_token(oauth.access_token) is None


@pytest.mark.asyncio
async def test_taiga_refresh_transport_error_does_not_revoke_family(
    fresh_store, respx_mock
):
    """Transient network errors must be caught as TaigaRefreshError so the
    family stays alive (only the user has to re-OAuth on next attempt)."""
    import httpx
    from mcp.server.auth.provider import TokenError

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    # Simulate a ReadTimeout on the refresh endpoint
    respx_mock.post(
        "https://taiga.example.test/api/v1/auth/refresh"
    ).mock(side_effect=httpx.ReadTimeout("simulated read timeout"))

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    with pytest.raises(TokenError) as excinfo:
        await provider.exchange_refresh_token(
            client=client_info, refresh_token=refresh_obj, scopes=[]
        )
    assert excinfo.value.error == "invalid_grant"
    # Family must still be alive: original access token still resolves
    assert await fresh_store.lookup_access_token(oauth.access_token) is not None


@pytest.mark.asyncio
async def test_reuse_detection_fires_via_load_before_sdk_scope_check(
    fresh_store, respx_mock, caplog
):
    """Replay with INVALID scope should still revoke the family.

    Before the fix: SDK rejected invalid_scope before reaching our
    exchange_refresh_token, so reuse-detection never fired and the family
    stayed alive — an attacker could replay-and-probe indefinitely.

    After the fix: load_refresh_token revokes the family up front on any
    rotated_out record, regardless of what the SDK does next.
    """
    import logging

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    respx_mock.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(
            200, json={"auth_token": "jwt_v2", "refresh": "ref_v2"}
        )
    )

    # Legitimate first refresh — rotates the original token out
    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    rotated = await provider.exchange_refresh_token(
        client=client_info, refresh_token=refresh_obj, scopes=[]
    )

    # Replay attempt — just load_refresh_token, which is what the SDK calls
    # first. Even without calling exchange_refresh_token, the family must
    # be revoked.
    with caplog.at_level(logging.WARNING, logger="langchain_taiga.auth.provider"):
        result = await provider.load_refresh_token(
            client_info, oauth.refresh_token
        )

    assert result is None
    # Family must be wiped: even the just-issued new generation is gone
    assert await fresh_store.lookup_access_token(rotated.access_token) is None
    assert await fresh_store.lookup_refresh_token(rotated.refresh_token) is None
    # WARNING emitted
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("reuse detected" in r.getMessage() for r in warnings)


@pytest.mark.asyncio
async def test_taiga_refresh_malformed_200_does_not_revoke_family(
    fresh_store, respx_mock
):
    """Malformed 200 response from Taiga (missing fields) is treated like
    any other TaigaRefreshError: family preserved, user re-OAuths."""
    from mcp.server.auth.provider import TokenError

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    # Taiga returns 200 but missing auth_token / refresh fields
    respx_mock.post(
        "https://taiga.example.test/api/v1/auth/refresh"
    ).mock(return_value=Response(200, json={"unexpected": "shape"}))

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    with pytest.raises(TokenError) as excinfo:
        await provider.exchange_refresh_token(
            client=client_info, refresh_token=refresh_obj, scopes=[]
        )
    assert excinfo.value.error == "invalid_grant"
    # Family preserved
    assert await fresh_store.lookup_access_token(oauth.access_token) is not None


@pytest.mark.asyncio
async def test_transient_cascade_failure_allows_retry_with_same_refresh_token(
    fresh_store, respx_mock
):
    """A transient Taiga 5xx must NOT rotate the MCP refresh token, so the
    client can retry with the same refresh token instead of being forced
    through a full OAuth flow.

    Codex P2 fix: cascade-first ordering moves ``consume_refresh_token``
    after the Taiga cascade succeeds. Without this, the previous fix to
    revoke families on rotated-out replay turned every transient Taiga
    blip into a forced re-OAuth (because the second retry attempt would
    look like a replay)."""
    from mcp.server.auth.provider import TokenError

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    # First attempt: Taiga returns 500
    fail_route = respx_mock.post(
        "https://taiga.example.test/api/v1/auth/refresh"
    ).mock(return_value=Response(500, text="temporarily unavailable"))

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    with pytest.raises(TokenError) as excinfo:
        await provider.exchange_refresh_token(
            client=client_info, refresh_token=refresh_obj, scopes=[]
        )
    assert excinfo.value.error == "invalid_grant"

    # Refresh token must still be active (NOT rotated_out)
    stored = await fresh_store.lookup_refresh_token(oauth.refresh_token)
    assert stored is not None
    assert stored.rotated_out is False

    # Second attempt: same refresh token, Taiga now returns 200
    fail_route.mock(
        return_value=Response(
            200, json={"auth_token": "jwt_v2", "refresh": "ref_v2"}
        )
    )
    refresh_obj_retry = await provider.load_refresh_token(
        client_info, oauth.refresh_token
    )
    new_oauth = await provider.exchange_refresh_token(
        client=client_info, refresh_token=refresh_obj_retry, scopes=[]
    )
    assert new_oauth.access_token
    assert new_oauth.refresh_token


@pytest.mark.asyncio
async def test_malformed_taiga_response_allows_retry_with_same_refresh_token(
    fresh_store, respx_mock
):
    """Same contract as transient-5xx, but for malformed-200 (missing
    fields). The user retries the same refresh token once Taiga recovers,
    no full re-OAuth required."""
    from mcp.server.auth.provider import TokenError

    provider = _make_provider(fresh_store)
    client_info, oauth = await _do_full_auth_flow(provider, fresh_store, respx_mock)

    fail_route = respx_mock.post(
        "https://taiga.example.test/api/v1/auth/refresh"
    ).mock(return_value=Response(200, json={"unexpected": "shape"}))

    refresh_obj = await provider.load_refresh_token(client_info, oauth.refresh_token)
    with pytest.raises(TokenError):
        await provider.exchange_refresh_token(
            client=client_info, refresh_token=refresh_obj, scopes=[]
        )

    # Refresh token must still be active (NOT rotated_out)
    stored = await fresh_store.lookup_refresh_token(oauth.refresh_token)
    assert stored is not None
    assert stored.rotated_out is False

    # Recovery: retry succeeds
    fail_route.mock(
        return_value=Response(
            200, json={"auth_token": "jwt_v2", "refresh": "ref_v2"}
        )
    )
    refresh_obj_retry = await provider.load_refresh_token(
        client_info, oauth.refresh_token
    )
    new_oauth = await provider.exchange_refresh_token(
        client=client_info, refresh_token=refresh_obj_retry, scopes=[]
    )
    assert new_oauth.access_token
