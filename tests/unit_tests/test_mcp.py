
import pytest

from langchain_taiga.mcp import mcp
from langchain_taiga.tools import taiga_tools  # noqa: F401


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """Ensure OpenAI credentials are present for tool initialization."""

    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.mark.asyncio
async def test_mcp_registers_taiga_tools():
    tool_map = await mcp.get_tools()

    assert {
        "create_entity_tool",
        "search_entities_tool",
        "get_entity_by_ref_tool",
        "update_entity_by_ref_tool",
        "add_comment_by_ref_tool",
        "add_attachment_by_ref_tool",
    }.issubset(tool_map.keys())


@pytest.mark.asyncio
async def test_mcp_metadata_defaults():
    assert mcp.name == "langchain-taiga"

    tools = await mcp.get_tools()
    create_tool = tools["create_entity_tool"]

    assert "langchain-taiga" in mcp.instructions
    assert create_tool.description
