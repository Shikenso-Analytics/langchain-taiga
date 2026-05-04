"""Model Context Protocol server exposing Taiga tools via fastmcp (stdio mode)."""

from __future__ import annotations

from langchain_taiga.mcp import mcp


def main() -> None:
    """Stdio entry point. Tools are registered against the module-level
    singleton at import time (via ``make_mcp()`` in ``mcp.py``)."""
    mcp.run()


# Legacy alias kept for any existing callers.
run = mcp.run


if __name__ == "__main__":
    main()
