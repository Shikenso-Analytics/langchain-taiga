import inspect
from typing import Any

import pytest

from langchain_taiga.mcp import mcp
from langchain_taiga.toolkits import TaigaToolkit
from langchain_taiga.tools import taiga_tools  # noqa: F401


async def _tool_map(m) -> dict[str, Any]:
    return {t.name: t for t in await m.list_tools()}


async def _taiga_tool_map(m) -> dict[str, Any]:
    """The listed tools this package defines (docstrings it owns)."""
    return {n: t for n, t in (await _tool_map(m)).items() if n in EXPECTED_TOOL_NAMES}


# Single source of truth for the agent-callable tool surface — the
# LangChain Toolkit and the MCP server must advertise exactly this set
# (anything else is a registration drift like the 10-vs-15 gap that
# v2.1.0 fixed).
EXPECTED_TOOL_NAMES = frozenset(
    {
        "create_entity_tool",
        "search_entities_tool",
        "get_kanban_board_tool",
        "get_kanban_boards_tool",
        "get_entity_by_ref_tool",
        "update_entity_by_ref_tool",
        "update_entities_by_ref_tool",
        "manage_watchers_by_ref_tool",
        "manage_tags_by_ref_tool",
        "add_comment_by_ref_tool",
        "create_attachment_upload_by_ref_tool",
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
        "list_projects_tool",
    }
)

# The MCP server also lists the in-chat file upload's model-facing tools
# (``langchain_taiga.file_upload``, the ``[apps]`` extra). They are fastmcp's,
# not LangChain tools, so the Toolkit does not carry them.
FILE_UPLOAD_TOOL_NAMES = frozenset({"file_manager", "list_files", "read_file"})
EXPECTED_MCP_TOOL_NAMES = EXPECTED_TOOL_NAMES | FILE_UPLOAD_TOOL_NAMES


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """Ensure OpenAI credentials are present for tool initialization."""

    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.mark.asyncio
async def test_mcp_registers_taiga_tools():
    tool_map = await _tool_map(mcp)
    # Exact-set assertion (not subset): catches both regressions —
    # a tool getting silently dropped, AND a tool getting added without
    # being listed in the README / EXPECTED_TOOL_NAMES.
    assert set(tool_map.keys()) == EXPECTED_MCP_TOOL_NAMES


def test_toolkit_matches_mcp_tool_set():
    # The LangChain Toolkit and the MCP server should advertise the
    # SAME tool surface. This guards against the 10-vs-15 drift that
    # existed before v2.1.0 (Toolkit had only 10 of the 15 MCP tools).
    toolkit_names = {t.name for t in TaigaToolkit().get_tools()}
    assert toolkit_names == EXPECTED_TOOL_NAMES


@pytest.mark.asyncio
async def test_mcp_metadata_defaults():
    assert mcp.name == "langchain-taiga"

    tools = await _tool_map(mcp)
    create_tool = tools["create_entity_tool"]

    assert "langchain-taiga" in mcp.instructions
    assert create_tool.description


@pytest.mark.asyncio
async def test_every_taiga_tool_runs_in_a_worker_thread():
    """Every Taiga call blocks. On the event loop it would stall
    ``/mcp/health`` until the liveness probe kills the pod mid-call (seen in
    production before 2.3.3), and the ``asyncio.run`` fetchers of the board,
    RICE sort and task histories need a thread without a running loop."""
    tools = await _taiga_tool_map(mcp)
    assert len(tools) == len(EXPECTED_TOOL_NAMES)
    assert [n for n, t in tools.items() if not t.run_in_thread] == []
    assert [n for n, t in tools.items() if inspect.iscoroutinefunction(t.fn)] == []


@pytest.mark.asyncio
async def test_every_mcp_parameter_publishes_its_docstring_description():
    """Every MCP parameter publishes a description from the ``Args:`` block.

    fastmcp 2.x built its schema from the raw function and dropped them all
    (90 of 90 parameters, 23 tools, under 2.13/2.14); fastmcp 4 parses
    ``Args:`` itself. This pins that no parameter of any tool loses one.
    """
    tools = await _taiga_tool_map(mcp)
    missing = [
        f"{name}.{param}"
        for name, tool in tools.items()
        for param, spec in tool.parameters.get("properties", {}).items()
        if not spec.get("description")
    ]
    assert missing == [], f"parameters with no description: {missing}"

    # Non-vacuity: the check above passes on an empty schema. Every MCP
    # parameter must be matched by a LangChain args_schema property, so a
    # tool that lost its parameters (or a schema that came back empty)
    # fails here instead of passing with nothing checked.
    lc_tools = {t.name: t for t in TaigaToolkit().get_tools()}
    checked = sum(len(t.parameters.get("properties", {})) for t in tools.values())
    expected = sum(
        len(lc_tools[name].args_schema.model_json_schema().get("properties", {}))
        for name in tools
    )
    assert expected > 0
    assert checked == expected, (
        f"MCP publishes {checked} parameters, the LangChain args_schemas "
        f"declare {expected}"
    )


@pytest.mark.asyncio
async def test_arg_descriptions_match_the_langchain_source_of_truth():
    """fastmcp parses the same ``Args:`` block LangChain does, so the two
    cannot drift; fastmcp only keeps the docstring's line breaks."""
    tools = await _tool_map(mcp)
    published = tools["search_entities_tool"].parameters["properties"]
    source = taiga_tools.search_entities_tool.args_schema.model_json_schema()["properties"]

    for param in ("open_only", "max_results", "include_custom_attributes"):
        assert published[param]["description"].split() == source[param]["description"].split()


@pytest.mark.asyncio
async def test_mcp_tool_descriptions_are_the_full_docstring():
    """FastMCP 4 keeps only the first docstring section as a tool's
    description, which would drop the "Use when" guidance, ``Returns:``,
    ``Examples:`` and constraints. Registration passes the whole docstring."""
    tools = await _taiga_tool_map(mcp)
    lc_tools = {t.name: t for t in TaigaToolkit().get_tools()}
    assert set(tools) == set(lc_tools) and tools
    wrong = [
        name
        for name, lc in lc_tools.items()
        if tools[name].description != inspect.getdoc(lc.func)
    ]
    assert wrong == [], f"descriptions that are not the full docstring: {wrong}"
