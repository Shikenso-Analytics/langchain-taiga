import pytest

from langchain_taiga.mcp import mcp
from langchain_taiga.toolkits import TaigaToolkit
from langchain_taiga.tools import taiga_tools  # noqa: F401

# Single source of truth for the agent-callable tool surface — the
# LangChain Toolkit and the MCP server must advertise exactly this set
# (anything else is a registration drift like the 10-vs-15 gap that
# v2.1.0 fixed). This is NOT the complete set of importable tool
# functions: some tools (e.g. ``add_attachment_inline_by_ref_tool``
# since 2.9.0) remain importable from ``langchain_taiga`` /
# ``langchain_taiga.tools.taiga_tools`` for library consumers while
# being intentionally absent from the MCP + Toolkit surface.
EXPECTED_TOOL_NAMES = frozenset(
    {
        "create_entity_tool",
        "search_entities_tool",
        "get_kanban_board_tool",
        "get_entity_by_ref_tool",
        "update_entity_by_ref_tool",
        "add_comment_by_ref_tool",
        "add_attachment_by_ref_tool",
        "list_attachments_by_ref_tool",
        "get_attachment_by_ref_tool",
        "promote_issue_to_userstory_tool",
        "list_custom_attributes_tool",
        "set_custom_attributes_tool",
        "get_custom_attributes_tool",
        "sort_kanban_by_rice_tool",
        "set_userstory_points_tool",
        "list_wiki_pages_tool",
        "get_wiki_page_tool",
        "create_wiki_page_tool",
        "update_wiki_page_tool",
        "whoami_tool",
        "list_project_members_tool",
    }
)


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


@pytest.mark.asyncio
async def test_sort_kanban_registered_as_async_coroutine():
    """Regression guard for 2.3.3 (PR #14): sort_kanban_by_rice_tool's
    sync body must be offloaded to a worker thread via
    ``asyncio.to_thread`` at the FastMCP registration layer.
    Otherwise the FastMCP event loop is blocked while the tool runs
    its parallel HTTP fetches, and k8s liveness probe kills the pod
    mid-call (production symptom on the wahed project).

    A future contributor "simplifying" the registration back to sync
    would reintroduce the bug — this test fails loud if the
    registered handler is not a coroutine function.
    """
    import inspect

    tools = await mcp.get_tools()
    sort_tool = tools["sort_kanban_by_rice_tool"]

    # FastMCP's FunctionTool exposes the registered callable via
    # ``.fn``. After 2.3.3 it must be a coroutine function (the
    # async wrapper from _async_offload). Other tools register as
    # plain sync functions because their HTTP cost is <1s and
    # doesn't trip the liveness probe.
    assert inspect.iscoroutinefunction(sort_tool.fn), (
        "sort_kanban_by_rice_tool must be registered as an async "
        "coroutine so its sync body is offloaded via "
        "asyncio.to_thread — otherwise the FastMCP event loop is "
        "blocked while the tool runs and k8s liveness probe times out."
    )

    # Sanity check the contrapositive: a known-fast tool stays sync.
    create_tool = tools["create_entity_tool"]
    assert not inspect.iscoroutinefunction(create_tool.fn), (
        "create_entity_tool should NOT be wrapped — only slow tools "
        "(currently just sort_kanban_by_rice_tool) need the offload. "
        "If this assertion changed intentionally, update the "
        "_TOOLS_NEEDING_ASYNC_OFFLOAD set in _register_mcp_tools."
    )
