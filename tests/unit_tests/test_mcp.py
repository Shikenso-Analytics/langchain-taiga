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
        "get_kanban_boards_tool",
        "get_entity_by_ref_tool",
        "update_entity_by_ref_tool",
        "update_entities_by_ref_tool",
        "manage_watchers_by_ref_tool",
        "manage_tags_by_ref_tool",
        "add_comment_by_ref_tool",
        "add_attachment_by_ref_tool",
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
async def test_sort_kanban_registered_as_async_coroutine():
    """Regression guard for 2.3.3 (PR #14): sort_kanban_by_rice_tool's
    sync body must be offloaded to a worker thread via
    ``asyncio.to_thread`` at the FastMCP registration layer.
    Otherwise the FastMCP event loop is blocked while the tool runs
    its parallel HTTP fetches, and k8s liveness probe kills the pod
    mid-call (seen in production).

    fastmcp 4 runs plain sync tools in worker threads, so the other tools
    (plain sync functions, HTTP cost under 1s) are safe as they are; this one
    keeps the explicit offload because of its long fan-out.

    A future contributor "simplifying" the registration back to sync
    would reintroduce the bug — this test fails loud if the
    registered handler is not a coroutine function.
    """

    tools = await _tool_map(mcp)
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


@pytest.mark.asyncio
async def test_kanban_board_tool_is_async_offloaded():
    # get_kanban_board_tool does the same uncapped list_user_stories()
    # fetch as sort_kanban_by_rice_tool (plus per-ex-member get_user
    # fallbacks), so it must be in _TOOLS_NEEDING_ASYNC_OFFLOAD and
    # register as a coroutine — otherwise a large board blocks the
    # FastMCP event loop and the liveness probe times out.

    tools = await _tool_map(mcp)
    board_tool = tools["get_kanban_board_tool"]
    assert inspect.iscoroutinefunction(board_tool.fn), (
        "get_kanban_board_tool must be registered as an async coroutine "
        "so its sync body is offloaded via asyncio.to_thread — add "
        "id(get_kanban_board_tool) to _TOOLS_NEEDING_ASYNC_OFFLOAD."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        # Reads each touched task's history through its own event loop (2.22.0).
        "get_entity_by_ref_tool",
        # Up to 100 entities, each with its own PATCH (2.22.0).
        "update_entities_by_ref_tool",
    ],
)
async def test_tools_added_in_2_22_are_async_offloaded(name):

    tools = await _tool_map(mcp)
    assert inspect.iscoroutinefunction(tools[name].fn), (
        f"{name} must be registered as an async coroutine so its sync body "
        f"runs on a worker thread — add id({name}) to _TOOLS_NEEDING_ASYNC_OFFLOAD."
    )


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
    """Copied, not restated — so the docstring stays the single place to
    edit and the two cannot drift."""
    tools = await _tool_map(mcp)
    published = tools["search_entities_tool"].parameters["properties"]
    source = taiga_tools.search_entities_tool.args_schema.model_json_schema()["properties"]

    for param in ("open_only", "max_results", "include_custom_attributes"):
        assert published[param]["description"] == source[param]["description"]


def test_copy_arg_descriptions_never_overwrites_an_explicit_one():
    """A future ``Annotated[..., Field(description=...)]`` must win over the
    docstring, not be clobbered by it."""
    registered = type("T", (), {"parameters": {"properties": {"query": {"description": "explicit"}}}})()
    copied = taiga_tools._copy_arg_descriptions(taiga_tools.search_entities_tool, registered)

    assert copied == 0
    assert registered.parameters["properties"]["query"]["description"] == "explicit"


def test_copy_arg_descriptions_is_inert_on_a_toolless_object():
    """Must never take the server down at import time."""
    assert taiga_tools._copy_arg_descriptions(object(), object()) == 0


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
