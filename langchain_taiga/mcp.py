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

- ``FastMCP.__init__`` accepts ``auth=``, ``lifespan=``, ``host=``, and
  ``port=`` directly.
- Constructor-time ``streamable_http_path`` is deprecated in 2.14.5 (warning
  printed); fastmcp wants it on ``run_async`` (or ``http_app``) instead.
  The factory stashes it on the returned instance as
  ``mcp_instance.streamable_http_path`` so ``remote_server.py`` can hand it
  to ``run_async(path=...)``.
- ``mcp.run_async`` real signature is ``(transport, show_banner,
  **transport_kwargs)`` — host/port/path travel via the kwargs at runtime.
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
        host: bind host for streamable-http transport (constructor-time).
        port: bind port for streamable-http transport (constructor-time).
        streamable_http_path: URL path for the MCP-protocol endpoint.
            Defaults to ``"/mcp"`` so OAuth + well-known routes mount at
            root (RFC 8414 §3.1) and protocol traffic lives under
            ``/mcp/...``. Stashed as
            ``mcp_instance.streamable_http_path``; ``remote_server.py``
            reads it back when calling ``run_async(path=...)``.

    The contract for ``auth=``, ``lifespan=``, ``host=``, and ``port=`` is
    enforced by ``pyproject.toml``'s ``fastmcp = ">=2.14.0,<3.0.0"`` pin —
    Phase 0 verifies that the installed patch version honours all four. If
    a future patch within the pin range removes one, the build fails loud
    with a TypeError (which is what we want — better than a silent
    misroute). Do not add try/except wrappers; the version pin is the
    contract.
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
    if host is not None:
        init_kwargs["host"] = host
    if port is not None:
        init_kwargs["port"] = port

    mcp_instance = FastMCP(**init_kwargs)
    # Stash the protocol-endpoint path for remote_server's run_async call.
    # Setting it as an attribute (not constructor arg) avoids the
    # deprecation warning fastmcp 2.14.5 emits.
    mcp_instance.streamable_http_path = streamable_http_path

    # Register tools against this specific instance. ``_register_mcp_tools``
    # is idempotent per-instance (tracks via ``id()``).
    from langchain_taiga.tools.taiga_tools import _register_mcp_tools

    _register_mcp_tools(mcp_instance)
    return mcp_instance


# Module-level singleton — used by stdio entry, by the LangChain Toolkit, and
# by every existing test that does ``from langchain_taiga.mcp import mcp``.
mcp = make_mcp()
