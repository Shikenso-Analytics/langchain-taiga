"""Shared FastMCP instance for exposing Taiga tools."""

from __future__ import annotations

from importlib import metadata

from fastmcp import FastMCP

try:
    __version__ = metadata.version("langchain-taiga")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"


mcp = FastMCP(
    name="langchain-taiga",
    version=__version__ or "0.0.0",
    instructions=(
        "MCP server that surfaces Taiga project management tools from the "
        "langchain-taiga package."
    ),
)

