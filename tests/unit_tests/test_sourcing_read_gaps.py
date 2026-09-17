"""What the download-sourcing scripts still read over REST, now through the tools (2.20.0).

* The daily sweep needs every open story's tasks. ``include_tasks=1`` makes Taiga embed a summary
  of each task (id, ref, subject, status_id, ...) in the story row — and WITHOUT that flag the
  row still carries ``tasks``, as an empty list. So a card that asks for ``tasks`` must send the
  flag, or every story silently reads as having no tasks.
* The write gates compare status ids, and a hand-over assigns several PMs at once
  (``assigned_users``), so both are readable on request and ``assigned_users`` is writable.
"""

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    get_entity_by_ref_tool,
    get_kanban_board_tool,
    update_entity_by_ref_tool,
)
from tests.unit_tests.test_get_kanban_board_tool import _FakeStatus, _FakeUS, _FakeUser
from tests.unit_tests.test_update_single_patch import MEMBERS, _Entity, _NoLLM, _Stub

SUMMARY = {"id": 501, "ref": 90661, "subject": "16.09.2026 - Round 1: A vs B", "status_id": 29,
           "is_closed": False, "is_blocked": False, "is_iocaine": False}


# -- kanban: task summaries ------------------------------------------------------------------------


class _Board:
    """Answers like Taiga: ``tasks`` is always there, and filled only under include_tasks."""

    name = "Sourcing"

    def __init__(self):
        self.members = [_FakeUser(9, "alice")]
        self.queries = []
        self.statuses = [_FakeStatus(1, "Ongoing", order=1), _FakeStatus(2, "Done", order=2, is_closed=True)]
        self.stories = [_FakeUS(ref=7, status=1), _FakeUS(ref=8, status=2)]

    def list_user_story_statuses(self):
        return self.statuses

    def list_user_stories(self, **queryparams):
        self.queries.append(queryparams)
        embed = str(queryparams.get("include_tasks")) == "1"
        rows = [story for story in self.stories if "status" not in queryparams or str(story.status) in queryparams["status"].split(",")]
        for story in rows:
            story.tasks = [dict(SUMMARY)] if embed and story.ref == 7 else []
        return rows


@pytest.fixture
def board(monkeypatch):
    project = _Board()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: {"username": f"user{uid}"})
    return project


def _kanban(**kw):
    return json.loads(get_kanban_board_tool.invoke({"project_slug": "sourcing", **kw}))


def test_asking_for_task_summaries_sends_include_tasks(board):
    out = _kanban(statuses=["Ongoing"], fields=["columns.cards.ref", "columns.cards.tasks.ref", "columns.cards.tasks.status_id"])
    assert board.queries == [{"status": "1", "include_tasks": 1}]
    assert out["columns"] == [{"cards": [{"ref": 7, "tasks": [{"ref": 90661, "status_id": 29}]}]}]


def test_control_without_a_task_path_the_flag_is_not_sent(board):
    _kanban(statuses=["Ongoing"], fields=["columns.cards.ref"])
    assert board.queries == [{"status": "1"}]


def test_a_whole_card_carries_the_full_summaries(board):
    out = _kanban(include_closed=False, fields=["columns.cards"])
    assert board.queries == [{"status__is_closed": "false", "include_tasks": 1}]
    assert out["columns"][0]["cards"][0]["tasks"] == [SUMMARY]


def test_the_default_board_has_no_task_summaries(board):
    out = _kanban()
    assert board.queries == [{}]
    assert all("tasks" not in card for column in out["columns"] for card in column["cards"])


def test_an_unknown_summary_key_is_an_error(board):
    out = _kanban(fields=["columns.cards.tasks.status"])
    assert out["code"] == 400 and "status_id" in out["error"]
    assert board.queries == []


# -- get_entity_by_ref_tool: ids on request --------------------------------------------------------


class _Task(_Entity):
    def to_dict(self):
        return {"subject": self.subject, "tags": self.tags, "status": self.status, "assigned_to": self.assigned_to}


@pytest.fixture
def story_env(monkeypatch):
    story = _Entity(
        ref=3, subject="League", description="", status=87, assigned_to=2, assigned_users=[2, 7],
        due_date=None, milestone=None, owner=5, owner_extra_info={"id": 5, "username": "w"},
    )
    story.list_tasks = lambda: [_Task(id=501, ref=90661, subject="t", status=29, modified_date="2026-09-17T08:00:00Z",
                                      tags=[], assigned_to=None)]
    lookups = {"status": [], "users": []}

    def status(*args, **_kw):
        lookups["status"].append(args)
        return {"name": "Ongoing"}

    def user(uid):
        lookups["users"].append(uid)
        return {"id": uid, "username": f"user{uid}", "full_name": f"User {uid}"}

    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(name="P", slug=slug, members=MEMBERS))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: story)
    monkeypatch.setattr(taiga_tools, "get_status", status)
    monkeypatch.setattr(taiga_tools, "get_user", user)
    return lookups


def _entity(entity_type="userstory", **kw):
    args = {"project_slug": "p", "entity_ref": 3, "entity_type": entity_type, "include_history": False, **kw}
    return json.loads(get_entity_by_ref_tool.invoke(args))


def test_status_id_comes_on_request_without_a_lookup(story_env):
    out = _entity(fields=["status_id"])
    assert out["status_id"] == 87
    assert story_env["status"] == []


def test_assigned_users_resolve_against_the_members_first(story_env):
    out = _entity(fields=["assigned_users"])
    assert out["assigned_users"] == [
        {"id": 2, "username": "ben", "full_name": "Ben B"},
        {"id": 7, "username": "user7", "full_name": "User 7"},
    ]
    assert story_env["users"] == [7]


def test_assigned_users_exist_only_on_user_stories(story_env):
    out = _entity(entity_type="task", fields=["assigned_users"])
    assert out["code"] == 400 and "assigned_users" in out["error"]


def test_related_task_ids_dates_and_status_ids_come_on_request(story_env):
    out = _entity(fields=["related.tasks.id", "related.tasks.status_id", "related.tasks.modified_date"])
    assert out["related"]["tasks"] == [{"id": 501, "status_id": 29, "modified_date": "2026-09-17T08:00:00Z"}]
    assert story_env["status"] == []


def test_control_the_default_answer_has_none_of_the_new_keys(story_env):
    out = _entity()
    assert "status_id" not in out and "assigned_users" not in out
    assert set(out["related"]["tasks"][0]) == {"subject", "tags", "status", "assigned_to", "ref"}


# -- update_entity_by_ref_tool: assigned_users -----------------------------------------------------


@pytest.fixture
def us_env(monkeypatch):
    entity = _Entity(assigned_users=[1], status=87)
    project = _Stub(id=6, members=MEMBERS, list_user_story_statuses=lambda: [_Stub(id=85, name="QA/Feedback")])
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)
    monkeypatch.setattr(taiga_tools, "small_llm", _NoLLM())
    return entity


def _update(entity_type="userstory", **params):
    args = {"project_slug": "p", "entity_ref": 5, "entity_type": entity_type, **params}
    return json.loads(update_entity_by_ref_tool.invoke(args))


def test_assigned_users_go_out_in_the_same_patch(us_env):
    out = _update(status="QA/Feedback", assign_to="ben", assigned_users=["ben", "Anna A"], comment="PMs", strict=True)
    assert us_env.patches == [
        (["version"], {"status": 85, "assigned_to": 2, "assigned_users": [2, 1], "comment": "PMs"})
    ]
    assert out["applied"] == ["status", "assigned_to", "assigned_users", "comment"]


def test_an_empty_list_clears_the_assignees(us_env):
    out = _update(assigned_users=[])
    assert us_env.patches == [(["version"], {"assigned_users": []})]
    assert out["applied"] == ["assigned_users"]


def test_unchanged_assignees_are_not_sent(us_env):
    _update(assigned_users=["anna"], comment="x")
    assert us_env.patches == [(["version"], {"comment": "x"})]


@pytest.mark.parametrize(("given", "code"), [(["nobody"], 404), (["Ben B"], 409)])
def test_an_unresolvable_assignee_stops_the_write(us_env, given, code):
    out = _update(assigned_users=given, comment="x")
    assert out["code"] == code
    assert us_env.patches == []


def test_assigned_users_exist_only_on_user_stories_for_writes(us_env):
    out = _update(entity_type="task", assigned_users=["ben"])
    assert out["code"] == 400
    assert us_env.patches == []


def test_read_back_reports_the_stored_assignees(us_env, monkeypatch):
    after = _Entity(assigned_users=[2, 7], status=85, version=4,
                    status_extra_info={"name": "QA/Feedback", "is_closed": False})
    fetched = []

    def fetch(_project, _norm, _ref):
        fetched.append(1)
        return us_env if len(fetched) == 1 else after

    monkeypatch.setattr(taiga_tools, "fetch_entity", fetch)
    monkeypatch.setattr(taiga_tools, "fetch_history", lambda entity, norm: [])
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: {"username": f"user{uid}"})
    out = _update(assigned_users=["ben"], read_back=True)
    assert out["state"]["assigned_users"] == ["ben", "user7"]
    assert out["state"]["ids"] == {"status": 85, "assigned_to": None, "watchers": [1], "assigned_users": [2, 7]}


def test_control_read_back_of_a_task_has_no_assigned_users(us_env, monkeypatch):
    monkeypatch.setattr(taiga_tools, "fetch_history", lambda entity, norm: [])
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: {"name": "New", "is_closed": False})
    out = _update(entity_type="task", comment="x", read_back=True)
    assert "assigned_users" not in out["state"]
    assert "assigned_users" not in out["state"]["ids"]
