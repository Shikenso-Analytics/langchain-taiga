
import pytest

from langchain_taiga.mcp import mcp
from langchain_taiga.toolkits import TaigaToolkit
from langchain_taiga.tools import taiga_tools  # noqa: F401


# Single source of truth for the tool surface this package exposes.
# Both the LangChain Toolkit and the MCP server are expected to advertise
# exactly this set — anything else is a registration drift like the
# 10-vs-15 gap that v2.1.0 fixed.
EXPECTED_TOOL_NAMES = frozenset({
    "create_entity_tool",
    "search_entities_tool",
    "get_entity_by_ref_tool",
    "update_entity_by_ref_tool",
    "add_comment_by_ref_tool",
    "add_attachment_by_ref_tool",
    "promote_issue_to_userstory_tool",
    "list_custom_attributes_tool",
    "set_custom_attributes_tool",
    "get_custom_attributes_tool",
    "sort_kanban_by_rice_tool",
    "list_wiki_pages_tool",
    "get_wiki_page_tool",
    "create_wiki_page_tool",
    "update_wiki_page_tool",
    "whoami_tool",
    "list_project_members_tool",
})


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """Ensure OpenAI credentials are present for tool initialization."""

    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.mark.asyncio
async def test_mcp_registers_taiga_tools():
    tool_map = await mcp.get_tools()
    # Exact-set assertion (not subset): catches both regressions —
    # a tool getting silently dropped, AND a tool getting added without
    # being listed in the README / EXPECTED_TOOL_NAMES.
    assert set(tool_map.keys()) == EXPECTED_TOOL_NAMES


def test_toolkit_matches_mcp_tool_set():
    # The LangChain Toolkit and the MCP server should advertise the
    # SAME tool surface. This guards against the 10-vs-15 drift that
    # existed before v2.1.0 (Toolkit had only 10 of the 15 MCP tools).
    toolkit_names = {t.name for t in TaigaToolkit().get_tools()}
    assert toolkit_names == EXPECTED_TOOL_NAMES


@pytest.mark.asyncio
async def test_mcp_metadata_defaults():
    assert mcp.name == "langchain-taiga"

    tools = await mcp.get_tools()
    create_tool = tools["create_entity_tool"]

    assert "langchain-taiga" in mcp.instructions
    assert create_tool.description
