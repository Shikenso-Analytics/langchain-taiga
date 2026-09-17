"""``get_kanban_board_tool``: fields, compact and the open-only pushdown (2.19.0).

The download-sourcing sweep lists the open stories of one project. Listing every story and
hiding the closed columns afterwards costs a request per 100 stories of the whole project, so
``include_closed=False`` and ``statuses`` are now asked of Taiga.
"""

import json

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_kanban_board_tool
from tests.unit_tests.test_get_kanban_board_tool import (  # noqa: F401 - stub_get_user is an autouse fixture
    _FakeProject,
    _FakeStatus,
    _FakeUS,
    _FakeUser,
    stub_get_user,
)


def _board(**extra):
    project = _FakeProject(
        statuses=[_FakeStatus(1, "New", order=1), _FakeStatus(2, "Done", order=2, is_closed=True)],
        stories=[_FakeUS(ref=1, status=1, assigned_to=7), _FakeUS(ref=2, status=2)],
        members=[_FakeUser(9, "alice")],
    )
    for story in project._stories:
        story.modified_date = f"2026-09-1{story.ref}T10:00:00Z"
        story.tags = [["voice", None]]
    for key, value in extra.items():
        setattr(project, key, value)
    return project


def _invoke(monkeypatch, project, **kw):
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    return json.loads(get_kanban_board_tool.invoke({"project_slug": "sourcing", **kw}))


def test_open_only_is_asked_of_taiga_not_filtered_afterwards(monkeypatch):
    project = _board()
    out = _invoke(monkeypatch, project, include_closed=False)
    assert project.queries == [{"status__is_closed": "false"}]
    assert [column["status"] for column in out["columns"]] == ["New"]


def test_control_the_full_board_still_lists_everything(monkeypatch):
    project = _board()
    out = _invoke(monkeypatch, project)
    assert project.queries == [{}]
    assert [len(column["cards"]) for column in out["columns"]] == [1, 1]


def test_fields_project_columns_and_cards(monkeypatch):
    out = _invoke(monkeypatch, _board(), include_closed=False, fields=["columns.status", "columns.cards.ref"])
    assert out["columns"] == [{"status": "New", "cards": [{"ref": 1}]}]
    assert out["query"] == {
        "project_slug": "sourcing",
        "include_closed": False,
        "statuses": None,
        "fields": ["columns.status", "columns.cards.ref"],
    }


def test_an_assignee_nobody_asked_for_is_not_resolved(monkeypatch, stub_get_user):
    _invoke(monkeypatch, _board(), fields=["columns.cards.ref"])
    assert stub_get_user == []


def test_control_a_requested_ex_member_assignee_is_resolved(monkeypatch, stub_get_user):
    out = _invoke(monkeypatch, _board(), fields=["columns.cards.assigned_to"])
    assert stub_get_user == [7]
    assert out["columns"][0]["cards"] == [{"assigned_to": "user7"}]


def test_extra_card_fields_come_only_on_request(monkeypatch):
    out = _invoke(monkeypatch, _board(), include_closed=False, fields=["columns.cards.modified_date", "columns.cards.tags"])
    assert out["columns"][0]["cards"] == [{"modified_date": "2026-09-11T10:00:00Z", "tags": ["voice"]}]


def test_the_default_card_shape_is_unchanged(monkeypatch):
    out = _invoke(monkeypatch, _board(), include_closed=False)
    assert set(out["columns"][0]["cards"][0]) == {"ref", "subject", "assigned_to", "kanban_order"}
    assert "query" not in out


def test_an_unknown_card_field_is_an_error(monkeypatch):
    out = _invoke(monkeypatch, _board(), fields=["columns.cards.modified"])
    assert out["code"] == 400
    assert "modified_date" in out["error"]


def test_an_unknown_top_level_field_is_an_error(monkeypatch):
    out = _invoke(monkeypatch, _board(), fields=["cards"])
    assert out["code"] == 400
    assert "columns" in out["error"]


def test_compact_is_one_line(monkeypatch):
    project = _board()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    raw = get_kanban_board_tool.invoke({"project_slug": "sourcing", "fields": ["columns.status"], "compact": True})
    assert "\n" not in raw
    assert json.loads(raw)["columns"] == [{"status": "New"}, {"status": "Done"}]


# -- statuses --------------------------------------------------------------------------------------


def _three_column_board():
    project = _FakeProject(
        statuses=[
            _FakeStatus(1, "New", order=1),
            _FakeStatus(2, "Ongoing", order=2),
            _FakeStatus(3, "Done", order=3, is_closed=True),
        ],
        stories=[_FakeUS(ref=1, status=1), _FakeUS(ref=2, status=2), _FakeUS(ref=3, status=3)],
    )
    original = project.list_user_stories

    def list_user_stories(**queryparams):
        rows = original(**queryparams)
        if "status" in queryparams:
            wanted = {int(part) for part in queryparams["status"].split(",")}
            rows = [story for story in rows if story.status in wanted]
        return rows

    project.list_user_stories = list_user_stories
    return project


def test_statuses_are_asked_of_taiga_by_id(monkeypatch):
    project = _three_column_board()
    out = _invoke(monkeypatch, project, statuses=["ongoing", "1"], fields=["columns.status", "columns.cards.ref"])
    assert project.queries == [{"status": "1,2"}]
    assert out["columns"] == [{"status": "New", "cards": [{"ref": 1}]}, {"status": "Ongoing", "cards": [{"ref": 2}]}]
    assert out["query"]["statuses"] == ["ongoing", "1"]


def test_an_unknown_status_is_an_error_naming_the_valid_ones(monkeypatch):
    project = _three_column_board()
    out = _invoke(monkeypatch, project, statuses=["Ongo"])
    assert out["code"] == 400 and "Ongoing" in out["error"]
    assert getattr(project, "queries", []) == []


def test_a_closed_status_while_excluding_closed_ones_is_an_error(monkeypatch):
    out = _invoke(monkeypatch, _three_column_board(), statuses=["Done"], include_closed=False)
    assert out["code"] == 400 and "include_closed" in out["error"]


def test_an_empty_status_list_is_an_error(monkeypatch):
    assert _invoke(monkeypatch, _three_column_board(), statuses=[])["code"] == 400


def test_control_statuses_with_include_closed_can_list_a_closed_column(monkeypatch):
    project = _three_column_board()
    out = _invoke(monkeypatch, project, statuses=["Done"], fields=["columns.cards.ref"])
    assert project.queries == [{"status": "3"}]
    assert out["columns"] == [{"cards": [{"ref": 3}]}]


def test_an_ancestor_path_keeps_whole_cards_extras_included(monkeypatch):
    out = _invoke(monkeypatch, _board(), include_closed=False, fields=["columns.cards"])
    card = out["columns"][0]["cards"][0]
    assert card["modified_date"] == "2026-09-11T10:00:00Z" and card["tags"] == ["voice"]
    assert {"ref", "subject", "assigned_to", "kanban_order"} <= set(card)


def test_control_a_sibling_path_adds_no_extras(monkeypatch):
    out = _invoke(monkeypatch, _board(), include_closed=False, fields=["columns.cards.ref", "columns.status"])
    assert out["columns"][0]["cards"] == [{"ref": 1}]
