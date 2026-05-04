"""FastMCP factory for langchain-taiga.

Two consumers:

- ``mcp_server.py`` (stdio) imports the module-level ``mcp`` singleton.
- ``remote_server.py`` (HTTP+OAuth) calls
  ``make_mcp(auth=provider, lifespan=..., host=..., port=...,
  streamable_http_path="/mcp")`` to build a fresh instance with the OAuth
  provider attached at construction time. This is **mandatory** because
  FastMCP auto-mounts OAuth + discovery routes when the underlying ASGI app
  is built — setting ``.auth`` after construction is too late.

Phase 0 deltas applied (verified against fastmcp 2.14.5):

- ``FastMCP.__init__`` accepts ``auth=`` and ``lifespan=`` directly.
- Constructor-time ``host``/``port``/``streamable_http_path`` ARE in the
  ``__init__`` signature but trigger DeprecationWarnings — fastmcp 2.14.5
  wants them via ``run_async(host=, port=, path=)`` (which become
  ``transport_kwargs`` per the real ``run_async`` signature).
  The factory therefore stashes ``host``, ``port``, and
  ``streamable_http_path`` as plain attributes on the returned instance;
  ``remote_server.py`` reads them back when calling ``run_async``.
- ``mcp.run_async`` real signature is
  ``(transport, show_banner, **transport_kwargs)``. Passing host/port/path
  inside ``transport_kwargs`` flows down through ``run_http_async``.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Optional

from fastmcp import FastMCP

try:
    __version__ = metadata.version("langchain-taiga")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"


def make_mcp(
    *,
    auth: Any = None,
    lifespan: Any = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    streamable_http_path: str = "/mcp",
) -> FastMCP:
    """Construct a fresh FastMCP with all langchain-taiga tools registered.

    Args:
        auth: ``OAuthProvider``, ``TokenVerifier``, or ``None``. Triggers
            FastMCP's auto-mount of OAuth + discovery routes when non-None.
        lifespan: async context manager wired into the underlying ASGI app.
            Used by remote_server.py for cleanup-loop start/stop.
        host: bind host for streamable-http transport. Stashed on the
            instance for ``remote_server.py``'s ``run_async(host=...)`` call.
        port: bind port for streamable-http transport. Same stashing
            semantics as ``host``.
        streamable_http_path: URL path for the MCP-protocol endpoint.
            Defaults to ``"/mcp"`` so OAuth + well-known routes mount at
            root (RFC 8414 §3.1) and protocol traffic lives under
            ``/mcp/...``. Stashed on the instance for
            ``run_async(path=...)``.

    Storing host/port/path as attributes — instead of passing to the
    constructor — avoids fastmcp 2.14.5's DeprecationWarnings. The contract
    for ``auth=`` and ``lifespan=`` is enforced by ``pyproject.toml``'s
    ``fastmcp = ">=2.14.0,<3.0.0"`` pin; if a future patch within the pin
    range removes either kwarg, the build fails loud with a TypeError.
    """
    init_kwargs: dict[str, Any] = {
        "name": "langchain-taiga",
        "version": __version__ or "0.0.0",
        "instructions": (
            "MCP server that surfaces Taiga project management tools from the "
            "langchain-taiga package."
        ),
    }
    if auth is not None:
        init_kwargs["auth"] = auth
    if lifespan is not None:
        init_kwargs["lifespan"] = lifespan

    mcp_instance = FastMCP(**init_kwargs)
    # Stash transport details for ``remote_server.py``'s ``run_async`` call.
    mcp_instance.streamable_http_path = streamable_http_path
    if host is not None:
        mcp_instance.streamable_http_host = host
    if port is not None:
        mcp_instance.streamable_http_port = port

    # Register tools against this specific instance. ``_register_mcp_tools``
    # is idempotent per-instance (tracks via ``id()``).
    from langchain_taiga.tools.taiga_tools import _register_mcp_tools

    _register_mcp_tools(mcp_instance)
    return mcp_instance


# Module-level singleton — used by stdio entry, by the LangChain Toolkit, and
# by every existing test that does ``from langchain_taiga.mcp import mcp``.
mcp = make_mcp()
