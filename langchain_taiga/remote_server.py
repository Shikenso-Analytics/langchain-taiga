"""HTTP entry point for the multi-tenant remote MCP server.

Wires PR 1 (per-request JWT propagation in tool helpers) and PR 2
(``TaigaOAuthProvider`` + ``InMemoryStore``) into a runnable FastMCP HTTP
server. The flow:

1. ``_bootstrap_provider()`` eagerly creates the in-memory store + OAuth
   provider before the ASGI app is built. FastMCP auto-mounts OAuth and
   discovery routes at construction time, so the provider must exist by
   then — setting ``mcp.auth`` after construction is too late and silently
   misroutes the OAuth surface.
2. ``make_mcp(auth=provider, lifespan=...)`` constructs the FastMCP and
   registers all tools.
3. ``_attach_custom_routes(mcp, provider)`` adds ``/oauth/login``
   (GET + POST), ``/health``, ``/mcp/health``, and root-path well-known
   mirrors (defensive against non-spec-compliant MCP clients).
4. ``mcp.run_async(transport="streamable-http", host=, port=, path=...)``
   starts the server.

Storage is in-memory per Amendment v3.4 of the plan: pod restart wipes
state, users re-OAuth. Single-replica deployment at ~30-user scale.

Required environment:

    TAIGA_API_URL          Taiga API URL (cluster-internal in K8s)
    TAIGA_URL              Taiga UI URL (used in the login page CTA)
    TAIGA_MCP_BASE_URL     Public URL, e.g. https://taiga.shikenso.org/mcp
    OPENAI_API_KEY         For LLM-powered tool helpers
    TAIGA_MCP_HOST         Bind host (default 0.0.0.0)
    TAIGA_MCP_PORT         Bind port (default 8000)

NOTE: ``TAIGA_MCP_DB_URL`` / ``TAIGA_MCP_TOKEN_SECRET`` /
``TAIGA_MCP_FERNET_KEY`` from the Postgres-era plan are NOT required —
``InMemoryStore`` is process-local and unencrypted.

URL surface (FastMCP auto-mounts the OAuth + discovery routes):

  GET  /.well-known/oauth-authorization-server[/mcp]   AS metadata
  GET  /.well-known/oauth-protected-resource[/mcp]     RS metadata
  POST /register                                       DCR
  GET  /authorize                                      redirects to /oauth/login
  POST /token                                          code exchange
  GET  /oauth/login                                    OUR custom HTML form
  POST /oauth/login                                    OUR credential handler
  GET  /health, /mcp/health                            OUR K8s probes
  POST /mcp/...                                        Bearer-protected tool calls
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)

from langchain_taiga.auth.login_page import render_login_page
from langchain_taiga.auth.provider import (
    TaigaAuthenticationError,
    TaigaOAuthProvider,
    run_cleanup_loop,
)
from langchain_taiga.auth.store import InMemoryStore
from langchain_taiga.auth.taiga_client import TaigaClient
from langchain_taiga.mcp import make_mcp

_log = logging.getLogger(__name__)


async def _bootstrap_provider() -> tuple[TaigaOAuthProvider, InMemoryStore]:
    """Eagerly create the InMemoryStore + Provider before FastMCP is built.

    This is the architectural fix for FastMCP's auto-mount-at-construction
    behaviour: setting ``mcp.auth`` AFTER construction is too late, the OAuth
    and discovery routes are never mounted. The provider must exist before
    ``make_mcp(auth=provider, ...)`` is called.

    ``InMemoryStore.from_env()`` is a no-op factory (no DB connection,
    nothing to await on) — kept on the same shape as the Postgres-era
    design for swap-out symmetry.
    """
    store = await InMemoryStore.from_env()
    provider = TaigaOAuthProvider(
        store=store,
        taiga_client=TaigaClient(api_url=os.environ["TAIGA_API_URL"]),
        issuer_url=os.environ["TAIGA_MCP_BASE_URL"],
    )
    return provider, store


def _make_lifespan(store: InMemoryStore, provider: TaigaOAuthProvider):
    """Build a lifespan context manager closed over the booted store + provider.

    The lifespan owns ONLY the cleanup-loop task and the store-close on
    shutdown. Provider construction happened in ``_bootstrap_provider()``
    before us. The cleanup loop also prunes the provider's
    ``_authorize_states`` to keep abandoned authorize-clicks from leaking
    until pod restart (PR 2 quality fix).
    """

    @asynccontextmanager
    async def _lifespan(_app):
        stop = asyncio.Event()
        cleanup_task = asyncio.create_task(
            run_cleanup_loop(store, provider=provider, stop=stop)
        )
        try:
            yield
        finally:
            stop.set()
            try:
                await cleanup_task
            except Exception:
                _log.exception("cleanup teardown error; continuing")
            try:
                await store.close()
            except Exception:
                _log.exception("store.close() error; continuing")

    return _lifespan


def _attach_custom_routes(mcp, provider: TaigaOAuthProvider) -> None:
    """Bind /oauth/login, /health, and defensive root well-known mirrors.

    Called AFTER ``make_mcp()`` returns so the decorators bind to the
    correct FastMCP instance (the one with ``auth=provider`` attached at
    construction).
    """

    @mcp.custom_route("/health", methods=["GET"])
    async def _health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    @mcp.custom_route("/mcp/health", methods=["GET"])
    async def _mcp_health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    @mcp.custom_route("/oauth/login", methods=["GET"])
    async def _login_get(request: Request) -> Response:
        internal_state = request.query_params.get("internal_state", "")
        if not internal_state:
            return PlainTextResponse("Missing internal_state", status_code=400)
        html = render_login_page(
            state=internal_state,
            error=None,
            taiga_url=os.environ["TAIGA_URL"],
        )
        return Response(html, media_type="text/html")

    @mcp.custom_route("/oauth/login", methods=["POST"])
    async def _login_post(request: Request) -> Response:
        form = await request.form()
        internal_state = form.get("state", "")
        username = form.get("username", "")
        password = form.get("password", "")
        if not (internal_state and username and password):
            return PlainTextResponse("Missing field(s)", status_code=400)
        try:
            _, redirect_url = await provider.complete_login(
                internal_state=internal_state,
                username=username,
                password=password,
            )
        except TaigaAuthenticationError:
            # Re-render with error; state is preserved per the
            # complete_login contract (see provider.py).
            html = render_login_page(
                state=internal_state,
                error="Invalid Taiga username or password.",
                taiga_url=os.environ["TAIGA_URL"],
            )
            return Response(html, media_type="text/html", status_code=401)
        except ValueError as exc:
            return PlainTextResponse(str(exc), status_code=400)
        return RedirectResponse(redirect_url, status_code=303)

    # Defensive: mirror well-known discovery metadata at the root path.
    # MCP clients have an open RFC-8414 conformance issue (TS SDK #822);
    # some look at root /.well-known/ regardless of issuer path.
    # Belt-and-suspenders.
    @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    async def _as_metadata_root(_request: Request) -> JSONResponse:
        if hasattr(provider, "authorization_server_metadata"):
            try:
                return JSONResponse(provider.authorization_server_metadata())
            except Exception:  # pylint: disable=broad-except
                _log.exception(
                    "provider.authorization_server_metadata() raised; "
                    "falling back to hand-built doc"
                )
        # Hand-built minimal RFC 8414 doc.
        base = os.environ["TAIGA_MCP_BASE_URL"].rstrip("/")
        return JSONResponse(
            {
                "issuer": base,
                "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token",
                "registration_endpoint": f"{base}/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": [
                    "none",
                    "client_secret_basic",
                    "client_secret_post",
                ],
            }
        )

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def _rs_metadata_root(_request: Request) -> JSONResponse:
        base = os.environ["TAIGA_MCP_BASE_URL"].rstrip("/")
        return JSONResponse(
            {
                "resource": base,
                "authorization_servers": [base],
                "bearer_methods_supported": ["header"],
            }
        )


# ---- Main --------------------------------------------------------------


async def _async_main(host: str, port: int) -> None:
    """Run inside a single event loop so async store + ``run_async`` share it."""
    provider, store = await _bootstrap_provider()
    lifespan = _make_lifespan(store, provider)
    mcp = make_mcp(
        auth=provider,
        lifespan=lifespan,
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )
    _attach_custom_routes(mcp, provider)

    if not hasattr(mcp, "run_async"):
        raise RuntimeError(
            "FastMCP missing run_async — escalate, do not deploy. "
            "This violates the >=2.14.0,<3.0.0 pin contract; re-run the "
            "Phase 0 probe and update the plan rather than working around "
            "with a manual ASGI mount (would silently misroute OAuth + "
            "well-known under /mcp/... per RFC 8414 §3.1)."
        )

    # ``run_async`` real signature is ``(transport, show_banner,
    # **transport_kwargs)``. host/port/path travel via the kwargs.
    await mcp.run_async(
        transport="streamable-http",
        host=host,
        port=port,
        path=getattr(mcp, "streamable_http_path", "/mcp"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("TAIGA_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("TAIGA_MCP_PORT", "8000"))
    asyncio.run(_async_main(host, port))


if __name__ == "__main__":
    main()
