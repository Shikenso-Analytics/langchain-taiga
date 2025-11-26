"""Model Context Protocol server exposing Taiga tools via fastmcp."""
from __future__ import annotations

from langchain_taiga.mcp import mcp
# Import Taiga tools for side effects so the MCP decorators register them.
from langchain_taiga.tools import taiga_tools  # noqa: F401


run = mcp.run


if __name__ == "__main__":
    run()
