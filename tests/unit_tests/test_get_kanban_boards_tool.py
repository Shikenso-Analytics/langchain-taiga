"""``get_kanban_boards_tool``: several boards in one call (2.23.0).

Each board is exactly what ``get_kanban_board_tool`` returns for it; boards are read one after
another so the load on Taiga stays one board's worth. A board that fails is an entry with its
error and is listed in ``failed`` — it must never read as an empty board.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_kanban_boards_tool
from tests.unit_tests.test_kanban_task_attributes import _Project as _AttrBoard
from tests.unit_tests.test_list_projects_tool import _project

API = "https://taiga.test/api/v1"


class _Board(_AttrBoard):
    def __init__(self, name):
        super().__init__()
        self.name = name


BOARDS = {
    "team-frontend": _Board("Team · Frontend"),
    "team-backend": _Board("Team · Backend"),
}
VISIBLE = [
    _project(37, "team-frontend", "Team · Frontend"),
    _project(39, "team-backend", "Team · Backend"),
    _project(3, "my-project", "My Project"),
]


@pytest.fixture
def env(monkeypatch):
    reads = []

    def get_project(slug):
        reads.append(slug)
        return BOARDS.get(slug)

    taiga_tools.custom_attr_definitions_cache.clear()
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "TAIGA_API_URL", "https://taiga.test")
    monkeypatch.setattr(
        taiga_tools,
        "get_taiga_api",
        lambda token=None: SimpleNamespace(token="tok", projects=SimpleNamespace(list=lambda: list(VISIBLE))),
    )
    monkeypatch.setattr(taiga_tools, "get_project", get_project)
    yield reads
    taiga_tools.custom_attr_definitions_cache.clear()


def _boards(**kw):
    return json.loads(get_kanban_boards_tool.invoke({"compact": True, **kw}))


def test_a_prefix_reads_every_matching_board_in_slug_order(env):
    out = _boards(slug_prefix="team-", fields=["columns.status", "columns.cards.ref"])
    assert [b["project_slug"] for b in out["boards"]] == ["team-backend", "team-frontend"]
    assert out["boards"][0]["columns"] == [{"status": "Later", "cards": [{"ref": 1}, {"ref": 2}]}]
    assert out["failed"] == []
    assert env == ["team-backend", "team-frontend"]


def test_slugs_keep_their_order_once_each_and_blanks_are_ignored(env):
    out = _boards(
        project_slugs=["team-backend", " ", "team-frontend", "team-backend"],
        fields=["columns.status"],
    )
    assert [b["project_slug"] for b in out["boards"]] == ["team-backend", "team-frontend"]


def test_a_failing_board_is_an_entry_and_listed(env):
    out = _boards(project_slugs=["team-frontend", "team-nope"], fields=["columns.status"])
    assert out["failed"] == ["team-nope"]
    assert out["boards"][1]["project_slug"] == "team-nope"
    assert out["boards"][1]["code"] == 404 and "error" in out["boards"][1]
    assert "columns" in out["boards"][0]


@pytest.mark.parametrize("kw", [{}, {"slug_prefix": "  "}, {"project_slugs": ["a"], "slug_prefix": "b"}])
def test_exactly_one_selector(env, kw):
    assert _boards(**kw)["code"] == 400


def test_more_than_twenty_boards_is_refused_before_any_read(env):
    out = _boards(project_slugs=[f"b{i}" for i in range(21)])
    assert out["code"] == 400 and env == []


def test_without_compact_each_board_keeps_its_nulls(env):
    """Each board must read exactly like get_kanban_board_tool's answer: unassigned stays null."""
    out = json.loads(get_kanban_boards_tool.invoke({"project_slugs": ["team-frontend"]}))
    board = out["boards"][0]
    assert board["columns"][0]["wip_limit"] is None
    assert board["columns"][0]["cards"][0]["assigned_to"] is None


def test_no_matching_project_is_a_404(env):
    assert _boards(slug_prefix="nothing-")["code"] == 404


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=False)
async def test_runs_off_the_event_loop_through_the_mcp_server(env, respx_mock):
    """The per-board attribute fetch runs ``asyncio.run``, which raises on FastMCP's loop thread."""
    from langchain_taiga.mcp import mcp

    respx_mock.get(url__regex=rf"{API}/userstories/custom-attributes-values/\d+").mock(
        return_value=httpx.Response(200, json={"attributes_values": {"91": "Frontend"}, "version": 1})
    )
    tool = (await mcp.get_tools())["get_kanban_boards_tool"]
    result = await tool.run(
        {"slug_prefix": "team-", "fields": ["columns.cards.custom_attributes"], "compact": True}
    )
    out = json.loads(result.content[0].text)
    assert out["failed"] == []
    assert out["boards"][0]["columns"][0]["cards"][0]["custom_attributes"] == {"91": "Frontend"}
