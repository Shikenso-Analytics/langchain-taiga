"""Model Context Protocol server exposing Taiga tools via fastmcp."""

from __future__ import annotations

from langchain_taiga.mcp import mcp

# Import Taiga tools for side effects so the MCP decorators register them.


run = mcp.run


if __name__ == "__main__":
    run()
