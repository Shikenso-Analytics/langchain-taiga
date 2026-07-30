"""HTTP entry point for the multi-tenant remote MCP server.

Wires PR 1 (per-request JWT propagation in tool helpers) and PR 2
(``TaigaOAuthProvider`` + ``InMemoryStore``) into a runnable FastMCP HTTP
server. The flow:

1. ``_bootstrap_provider()`` eagerly creates the configured state store (see
   ``_build_store``) + OAuth provider before the ASGI app is built. FastMCP
   auto-mounts OAuth and discovery routes at construction time, so the
   provider must exist by then — setting ``mcp.auth`` after construction is
   too late and silently misroutes the OAuth surface.
2. ``make_mcp(auth=provider, lifespan=...)`` constructs the FastMCP and
   registers all tools.
3. ``_attach_custom_routes(mcp, provider)`` adds ``/oauth/login``
   (GET + POST), ``/health``, ``/mcp/health``, and root-path well-known
   mirrors (defensive against non-spec-compliant MCP clients).
4. ``mcp.run_async(transport="streamable-http", host=, port=, path=...)``
   starts the server.

The OAuth state backend is chosen by ``TAIGA_MCP_STATE_BACKEND``; the full
rationale lives on ``postgres_store`` (module docstring) and ``_build_store``.

Required environment:

    TAIGA_API_URL          Taiga API URL (cluster-internal in K8s)
    TAIGA_URL              Taiga UI URL (used in the login page CTA)
    TAIGA_MCP_BASE_URL     Public URL, e.g. https://taiga.shikenso.org/mcp
    OPENAI_API_KEY         For LLM-powered tool helpers
    TAIGA_MCP_HOST         Bind host (default 0.0.0.0)
    TAIGA_MCP_PORT         Bind port (default 8000)

Durable OAuth state — ``TAIGA_MCP_STATE_BACKEND=postgres`` plus EITHER the
discrete vars (what the Helm chart passes, and therefore what production
runs) OR a full DSN (used by CI and local tests):

    TAIGA_MCP_PG_HOST      discrete form, avoids DSN password escaping
    TAIGA_MCP_PG_PORT      (default 5432)
    TAIGA_MCP_PG_USER
    TAIGA_MCP_PG_PASSWORD
    TAIGA_MCP_PG_DATABASE
    TAIGA_MCP_PG_SCHEMA    (default mcp_oauth)
    TAIGA_MCP_DATABASE_URL full asyncpg DSN; wins if both forms are set

Default is ``TAIGA_MCP_STATE_BACKEND=memory`` — state lost on every restart.
Asking for ``postgres`` without connection config is a startup error, never a
silent downgrade.

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
from urllib.parse import urlparse, urlunparse

from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)

from langchain_taiga.auth.login_page import render_login_page
from langchain_taiga.auth.postgres_store import (
    DATABASE_URL_ENV,
    PG_DATABASE_ENV,
    PG_HOST_ENV,
    PG_USER_ENV,
    PostgresStore,
    postgres_configured,
)
from langchain_taiga.auth.provider import (
    StateStore,
    TaigaAuthenticationError,
    TaigaOAuthProvider,
    run_cleanup_loop,
)
from langchain_taiga.auth.store import InMemoryStore
from langchain_taiga.auth.taiga_client import TaigaClient
from langchain_taiga.mcp import make_mcp

_log = logging.getLogger(__name__)

#: Explicit OAuth-state backend switch. See ``_build_store`` for why this is a
#: switch rather than something inferred from the connection variables.
STATE_BACKEND_ENV = "TAIGA_MCP_STATE_BACKEND"
_BACKEND_POSTGRES = "postgres"
_BACKEND_MEMORY = "memory"


def _require_env(name: str) -> str:
    """Fail fast at startup if a required env var is missing.

    Lifted out of the various closures so a missing env crashes ``main()``
    with a clear message instead of surfacing as a 500 on the first request.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set. "
            f"See remote_server.py module docstring for the full list."
        )
    return value


async def _build_store() -> StateStore:
    """Pick the OAuth state backend from the environment.

    ``TAIGA_MCP_STATE_BACKEND`` is the switch: ``postgres`` for durable state
    that survives pod restarts, ``memory`` (the default) for the process-local
    store that local development and the unit-test suite run on.

    The choice is **explicit rather than inferred** on purpose. Inferring it
    from the presence of connection variables means the single most likely
    operator error — a dropped env var, a chart refactor that moves the ``env``
    block, ``--set postgres.enabled=false`` — silently downgrades production to
    the in-memory store. The pod boots green, the probes pass, and the only
    trace is a log line nobody reads, while every user is forced through a
    fresh browser login on each restart. That is precisely the regression this
    backend exists to remove, so it must fail loudly instead.

    For the same reason a connection failure is not caught: a misconfigured
    DSN raises here, at startup, rather than surfacing as a 500 on the first
    token request.

    Both store classes expose the identical async interface and the same
    ``from_env()`` factory shape, so everything downstream is agnostic.
    """
    backend = (os.getenv(STATE_BACKEND_ENV) or _BACKEND_MEMORY).strip().lower()

    if backend == _BACKEND_POSTGRES:
        if not postgres_configured():
            raise RuntimeError(
                f"{STATE_BACKEND_ENV}={_BACKEND_POSTGRES!r} but no connection "
                f"configuration was found. Set {DATABASE_URL_ENV}, or the "
                f"discrete {PG_HOST_ENV}/{PG_DATABASE_ENV}/{PG_USER_ENV} vars."
            )
        _log.info("OAuth state backend: Postgres (durable across restarts)")
        return await PostgresStore.from_env()

    if backend != _BACKEND_MEMORY:
        raise RuntimeError(
            f"{STATE_BACKEND_ENV}={backend!r} is not a valid backend "
            f"(expected {_BACKEND_POSTGRES!r} or {_BACKEND_MEMORY!r})."
        )

    if postgres_configured():
        # Connection details present but the backend wasn't asked for — almost
        # certainly a half-applied config rather than an intentional choice.
        _log.warning(
            "Postgres connection variables are set but %s is not %r — running "
            "on the in-memory store, so all tokens and client registrations "
            "will be lost on restart.",
            STATE_BACKEND_ENV,
            _BACKEND_POSTGRES,
        )
    else:
        _log.warning(
            "OAuth state backend: in-memory. All tokens and client "
            "registrations will be lost on restart."
        )
    return await InMemoryStore.from_env()


async def _bootstrap_provider(
    *, api_url: str, base_url: str
) -> tuple[TaigaOAuthProvider, StateStore]:
    """Eagerly create the state store + Provider before FastMCP is built.

    This is the architectural fix for FastMCP's auto-mount-at-construction
    behaviour: setting ``mcp.auth`` AFTER construction is too late, the OAuth
    and discovery routes are never mounted. The provider must exist before
    ``make_mcp(auth=provider, ...)`` is called.

    Env values are passed in as explicit kwargs (validated upfront in
    ``_async_main``) so a missing var crashes at startup with a clear
    message instead of as a 500 on the first request.
    """
    store = await _build_store()
    provider = TaigaOAuthProvider(
        store=store,
        taiga_client=TaigaClient(api_url=api_url),
        issuer_url=base_url,
    )
    return provider, store


def _make_lifespan(store: StateStore, provider: TaigaOAuthProvider):
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


def _attach_custom_routes(
    mcp,
    provider: TaigaOAuthProvider,
    *,
    taiga_url: str,
    base_url: str,
) -> None:
    """Bind /oauth/login, /health, and the root oauth-protected-resource mirror.

    Called AFTER ``make_mcp()`` returns so the decorators bind to the
    correct FastMCP instance (the one with ``auth=provider`` attached at
    construction).

    FastMCP auto-mounts oauth-authorization-server at root; we only need
    to mirror oauth-protected-resource because the framework only
    auto-generates the path-aware variant of that one. Env values are
    threaded in as explicit kwargs (validated in ``_async_main``).
    """

    @mcp.custom_route("/health", methods=["GET"])
    async def _health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    @mcp.custom_route("/mcp/health", methods=["GET"])
    async def _mcp_health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    async def _do_login_get(request: Request) -> Response:
        internal_state = request.query_params.get("internal_state", "")
        if not internal_state:
            return PlainTextResponse("Missing internal_state", status_code=400)
        html = render_login_page(
            state=internal_state,
            error=None,
            taiga_url=taiga_url,
        )
        return Response(html, media_type="text/html")

    async def _do_login_post(request: Request) -> Response:
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
                taiga_url=taiga_url,
            )
            return Response(html, media_type="text/html", status_code=401)
        except ValueError as exc:
            return PlainTextResponse(str(exc), status_code=400)
        return RedirectResponse(redirect_url, status_code=303)

    # The provider's ``authorize()`` redirects to ``f"{issuer_url}/oauth/login"``
    # — and ``issuer_url`` already includes the ``/mcp`` path prefix. So the
    # form is always reached at ``/mcp/oauth/login``. The Helm ingress is
    # configured to forward ``/mcp/...`` to this pod without stripping, so a
    # root mount would be unreachable in production and is omitted.
    mcp.custom_route("/mcp/oauth/login", methods=["GET"])(_do_login_get)
    mcp.custom_route("/mcp/oauth/login", methods=["POST"])(_do_login_post)

    # Defensive: mirror oauth-protected-resource at root.
    # FastMCP auto-mounts oauth-authorization-server at root; we only need
    # to mirror oauth-protected-resource because the framework only
    # auto-generates the path-aware variant of that one. MCP clients have
    # an open RFC-8414 conformance issue (TS SDK #822); some look at root
    # /.well-known/ regardless of issuer path. Belt-and-suspenders.
    #
    # ``authorization_servers`` MUST be the server origin (root), not
    # the path-aware MCP URL — strict RFC 8414 clients (VSCode 1.107+)
    # validate that the AS metadata's ``issuer`` matches the auth-server
    # URL from the PRM, and FastMCP's auto-mounted AS metadata uses the
    # base_url (root) as issuer. See provider.py for the matching
    # ``self.issuer_url = self.base_url`` override.
    parsed_base = urlparse(base_url)
    origin = urlunparse((parsed_base.scheme, parsed_base.netloc, "", "", "", ""))
    resource = base_url.rstrip("/")

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def _rs_metadata_root(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "resource": resource,
                "authorization_servers": [origin],
                "bearer_methods_supported": ["header"],
                "scopes_supported": ["taiga"],
            }
        )


# ---- Main --------------------------------------------------------------


def _patch_metadata_to_advertise_none_auth() -> None:
    """Add ``"none"`` to ``token_endpoint_auth_methods_supported``.

    The MCP SDK's ``build_metadata`` hard-codes the list to
    ``["client_secret_post"]`` (sometimes with ``"client_secret_basic"``
    in newer releases) — neither includes ``"none"``, which RFC 7591
    public clients (VSCode, Claude Desktop, etc.) require to register
    via DCR. Without it, strict clients fall back to manual client
    registration ("Dynamic Client Registration not supported").

    The server actually accepts ``"none"`` at the registration endpoint
    — DCR with ``"token_endpoint_auth_method": "none"`` returns 201 with
    a valid client_id (verified live for claude.ai). Only the discovery
    doc was under-claiming.

    We patch ``build_metadata`` BEFORE ``make_mcp()`` because FastMCP
    captures the metadata object at construction time. After that the
    auto-mounted routes serve a frozen response.
    """
    from mcp.server.auth import routes as _routes

    if getattr(_routes.build_metadata, "_taiga_patched", False):
        return  # idempotent — re-importing the module shouldn't double-wrap

    _orig_build = _routes.build_metadata

    def _patched_build(*args, **kwargs):
        md = _orig_build(*args, **kwargs)
        existing = list(md.token_endpoint_auth_methods_supported or [])
        if "none" not in existing:
            md.token_endpoint_auth_methods_supported = existing + ["none"]
        return md

    _patched_build._taiga_patched = True  # type: ignore[attr-defined]
    _routes.build_metadata = _patched_build


async def _async_main(host: str, port: int) -> None:
    """Run inside a single event loop so async store + ``run_async`` share it.

    Required env vars are validated upfront so a misconfigured deployment
    crashes here with a clear message, not on first request with a 500.
    """
    api_url = _require_env("TAIGA_API_URL")
    taiga_url = _require_env("TAIGA_URL")
    base_url = _require_env("TAIGA_MCP_BASE_URL")
    _require_env("OPENAI_API_KEY")  # not used here but validated upfront

    # MUST run before make_mcp — the auto-mounted auth-server-metadata
    # route freezes its response at construction time.
    _patch_metadata_to_advertise_none_auth()

    provider, store = await _bootstrap_provider(api_url=api_url, base_url=base_url)
    lifespan = _make_lifespan(store, provider)
    mcp_instance = make_mcp(
        auth=provider,
        lifespan=lifespan,
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )
    _attach_custom_routes(
        mcp_instance, provider, taiga_url=taiga_url, base_url=base_url
    )

    if not hasattr(mcp_instance, "run_async"):
        raise RuntimeError(
            "FastMCP missing run_async — escalate, do not deploy. "
            "This violates the >=2.14.0,<3.0.0 pin contract; re-run the "
            "Phase 0 probe and update the plan rather than working around "
            "with a manual ASGI mount (would silently misroute OAuth + "
            "well-known under /mcp/... per RFC 8414 §3.1)."
        )

    # ``run_async`` real signature is ``(transport, show_banner,
    # **transport_kwargs)``. host/port/path travel via the kwargs and are
    # forwarded into ``run_http_async`` → ``http_app(path=...)`` →
    # ``create_streamable_http_app(streamable_http_path=...)`` →
    # ``auth.get_routes(mcp_path=...)`` (the latter generates the
    # path-aware ``/.well-known/oauth-protected-resource/mcp`` route).
    await mcp_instance.run_async(
        transport="streamable-http",
        host=getattr(mcp_instance, "streamable_http_host", host),
        port=getattr(mcp_instance, "streamable_http_port", port),
        path=getattr(mcp_instance, "streamable_http_path", "/mcp"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("TAIGA_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("TAIGA_MCP_PORT", "8000"))
    asyncio.run(_async_main(host, port))


if __name__ == "__main__":
    main()
