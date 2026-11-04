"""The factory-style FastMCP construction must (a) keep the stdio singleton
intact and (b) allow a fresh instance with auth attached at construction.

Phase 0 confirmed ``FastMCP.__init__`` accepts ``auth=`` directly. The
``MagicMock(spec=OAuthProvider)`` shape is required because FastMCP
isinstance-checks ``auth=``; bare MagicMock would be rejected with TypeError,
masking the real assertion.
"""
from __future__ import annotations

from unittest.mock import MagicMock


def test_module_level_mcp_remains_importable():
    """Stdio path: importing ``langchain_taiga.mcp`` gives a usable FastMCP.

    Functional verification (tools really registered) is covered by the
    existing ``test_mcp.py::test_mcp_registers_taiga_tools`` test which
    inspects ``mcp.list_tools()`` directly.
    """
    from langchain_taiga.mcp import mcp

    assert mcp is not None


def test_make_mcp_returns_fresh_instance():
    """Two calls return distinct objects so remote_server can hold its own."""
    from langchain_taiga.mcp import make_mcp

    a = make_mcp()
    b = make_mcp()
    assert a is not b


def test_make_mcp_accepts_auth_kwarg():
    """The whole point of the factory: pass an OAuthProvider at construction."""
    from fastmcp.server.auth import OAuthProvider

    from langchain_taiga.mcp import make_mcp

    fake_provider = MagicMock(spec=OAuthProvider)
    fresh = make_mcp(auth=fake_provider)
    attached = getattr(fresh, "auth", None) or getattr(fresh, "_auth", None)
    assert attached is fake_provider


def test_a_new_instance_at_a_reused_address_still_gets_its_tools():
    """Registration used to be remembered by ``id(instance)``. CPython hands a
    freed instance's address to the next one, so most fresh instances came
    back with no tools at all (285 of 300 in a loop) and tool calls failed
    with "Unknown tool" in tests that build their own server."""
    import asyncio
    import gc

    from langchain_taiga.mcp import make_mcp
    from langchain_taiga.tools.taiga_tools import _register_mcp_tools

    for _ in range(5):
        instance = make_mcp()
        names = [t.name for t in asyncio.run(instance.list_tools())]
        assert "whoami_tool" in names
        _register_mcp_tools(instance)  # same instance again: a no-op
        assert [t.name for t in asyncio.run(instance.list_tools())] == names
        del instance
        gc.collect()
