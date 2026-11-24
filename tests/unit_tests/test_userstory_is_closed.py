"""A user story's ``is_closed`` is its own, not its status's (5.7.0).

Taiga closes a story when its last task closes and reopens it when a task reopens, without
moving the story to another status, and its web client labels the story "Closed" or "Open"
from that field. The tools used to read the status's flag instead: a story whose tasks were
all done still read as open, and ``open_only`` kept it.
"""

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    _entity_is_closed,
    _read_back_state,
    get_entity_by_ref_tool,
    search_entities_tool,
)
from tests.unit_tests.test_lean_entity_reads import _story, _Stub
from tests.unit_tests.test_search_entities_tool import (  # noqa: F401 - fake_search_env is a fixture
    _FakeEntity,
    _FakeProject,
    _patch_llm,
    fake_search_env,
)

OPEN_STATUS = {"name": "In progress", "is_closed": False}
CLOSED_STATUS = {"name": "Done", "is_closed": True}


def _row(ref, *, is_closed, status):
    entity = _FakeEntity(ref=ref, subject=f"story {ref}", status_extra_info=dict(status))
    entity.is_closed = is_closed
    return entity


class _StoryProject(_FakeProject):
    """``/userstories`` as Taiga answers it: both closed params are applied, each to its own flag."""

    def __init__(self, entities, enforce=True):
        super().__init__(entities)
        self._enforce = enforce

    def list_user_stories(self, **kwargs):
        self.list_user_stories_kwargs.append(kwargs)
        rows = self._entities
        if self._enforce and kwargs.get("is_closed") == "false":
            rows = [e for e in rows if e.is_closed is not True]
        if self._enforce and kwargs.get("status__is_closed") == "false":
            rows = [e for e in rows if e.status_extra_info["is_closed"] is not True]
        return rows


def _rows():
    return [
        _row(1, is_closed=True, status=OPEN_STATUS),  # every task closed, status still an open one
        _row(2, is_closed=False, status=CLOSED_STATUS),  # closed status, but a task is still open
        _row(3, is_closed=False, status=OPEN_STATUS),
        _row(4, is_closed=True, status=CLOSED_STATUS),
    ]


def _search_env(monkeypatch, project):
    """``fake_search_env``'s stubs, with ``project`` in place of its issues."""
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _patch_llm(monkeypatch, {})
    return project


@pytest.fixture
def stories(request, monkeypatch):
    request.getfixturevalue("fake_search_env")
    return _search_env(monkeypatch, _StoryProject(_rows()))


def _search_stories(**params):
    raw = search_entities_tool.invoke({"project_slug": "p", "query": "q", "entity_type": "userstory", **params})
    return json.loads(raw)


# -- search_entities_tool ---------------------------------------------------------------------------


def test_a_match_carries_the_storys_own_flag_next_to_its_status(stories):
    matches = _search_stories()["matches"]
    assert [(m["ref"], m["status"], m["is_closed"]) for m in matches] == [
        (1, "In progress", True),
        (2, "Done", False),
        (3, "In progress", False),
        (4, "Done", True),
    ]


def test_open_only_keeps_the_stories_the_web_client_shows_as_open(stories):
    assert [m["ref"] for m in _search_stories(open_only=True)["matches"]] == [2, 3]


def test_open_only_asks_taiga_for_the_storys_own_flag(stories):
    _search_stories(open_only=True)
    assert stories.list_user_stories_kwargs == [{"is_closed": "false"}]


@pytest.mark.usefixtures("fake_search_env")
def test_open_only_holds_even_where_the_server_ignores_the_param(monkeypatch):
    """The client-side predicate decides on the same flag, so it does not depend on the pushdown."""
    project = _search_env(monkeypatch, _StoryProject(_rows(), enforce=False))
    assert [m["ref"] for m in _search_stories(open_only=True)["matches"]] == [2, 3]
    assert project.list_user_stories_kwargs == [{"is_closed": "false"}]


# -- the flag itself --------------------------------------------------------------------------------


def test_the_rows_own_flag_wins_over_the_status():
    assert _entity_is_closed(_Stub(is_closed=True), {"is_closed": False}) is True
    assert _entity_is_closed(_Stub(is_closed=False), {"is_closed": True}) is False


def test_without_a_flag_of_its_own_the_status_decides():
    assert _entity_is_closed(_Stub(), {"is_closed": True}) is True
    assert _entity_is_closed(_Stub(is_closed=None), {"is_closed": False}) is False
    assert _entity_is_closed(_Stub(), {"is_closed": None}) is None


# -- get_entity_by_ref_tool -------------------------------------------------------------------------


@pytest.fixture
def closed_story(monkeypatch):
    story = _story(is_closed=True, status_extra_info=dict(OPEN_STATUS))
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(name="P", slug=slug))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: story)
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: dict(OPEN_STATUS))
    monkeypatch.setattr(taiga_tools, "get_formatted_custom_attributes", lambda *a, **kw: [])
    monkeypatch.setattr(taiga_tools, "fetch_history", lambda *a, **kw: [])
    monkeypatch.setattr(taiga_tools, "list_milestones", lambda slug: [])
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: {"id": uid, "username": f"user{uid}"})
    monkeypatch.setattr(taiga_tools, "_format_userstory_points", lambda *a, **kw: {})
    monkeypatch.setattr(taiga_tools, "_current_user_id", lambda: 5)
    return story


def _get(entity_type="userstory", **params):
    args = {"project_slug": "p", "entity_ref": 1, "entity_type": entity_type, **params}
    return json.loads(get_entity_by_ref_tool.invoke(args))


def test_a_story_reads_closed_while_its_status_is_open(closed_story):
    out = _get(include_history=False)
    assert (out["status"], out["is_closed"]) == ("In progress", True)


def test_is_closed_can_be_asked_for_by_name(closed_story):
    out = _get(fields=["status", "is_closed"], compact=True)
    assert {key: out[key] for key in ("status", "is_closed")} == {"status": "In progress", "is_closed": True}


@pytest.mark.parametrize("entity_type", ["task", "issue", "epic"])
def test_every_type_accepts_is_closed_as_a_field(closed_story, entity_type):
    closed_story.user_story_extra_info = None
    assert _get(entity_type, fields=["is_closed"])["is_closed"] is True


# -- update_entity_by_ref_tool(read_back=True) ------------------------------------------------------


def test_read_back_reports_the_storys_own_flag(monkeypatch):
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: None)
    story = _Stub(
        is_closed=True,
        status=7,
        status_extra_info=dict(OPEN_STATUS),
        assigned_to=None,
        watchers=[],
        tags=[],
        version=3,
        assigned_users=[],
    )
    state = _read_back_state("p", "us", story, _Stub(members=[]))
    assert (state["status"], state["is_closed"]) == ("In progress", True)
