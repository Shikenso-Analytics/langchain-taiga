"""``update_entity_by_ref_tool(user_story_ref=...)``: move a task to another user story (5.4.0).

Splitting a story at the end of a sprint means moving its open tasks to the follow-up story. Before
this there was no way to do that through the bridge: ``user_story`` was not a parameter, and
``milestone`` is refused on tasks because a task's sprint is denormalised from its story. Taiga
refuses a task whose sprint differs from its story's, so the move sends the new story's sprint
along in the same PATCH; a story in the backlog takes the task out of its sprint.
"""

import json

import pytest
from taiga.exceptions import TaigaRestException

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    update_entities_by_ref_tool,
    update_entity_by_ref_tool,
)


class _Stub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Task:
    def __init__(self, **kwargs):
        self.id = 11
        self.ref = 101
        self.version = 2
        self.status = 1
        self.assigned_to = None
        self.watchers = []
        self.tags = []
        self.user_story = 400
        self.milestone = 7
        self.patches = []
        self.__dict__.update(kwargs)

    def patch(self, fields, **kwargs):
        self.patches.append((fields, kwargs))
        return self

    def update(self, **kwargs):
        raise AssertionError("a full PUT must never happen")


STORIES = {
    40: _Stub(id=400, ref=40, milestone=7),
    41: _Stub(id=410, ref=41, milestone=8),
    42: _Stub(id=420, ref=42, milestone=None),
}
TASK_STATUSES = [_Stub(id=29, name="New"), _Stub(id=30, name="Done")]


class _NoLLM:
    def invoke(self, _messages):
        raise AssertionError("moving a task must not ask the language model")


@pytest.fixture
def env(monkeypatch):
    tasks = {}
    lookups = []
    members = [_Stub(id=20, username="anna", full_name="Anna A")]
    project = _Stub(id=3, members=members, list_task_statuses=lambda: TASK_STATUSES)

    def fetch(_project, norm, ref):
        if norm == "us":
            lookups.append(ref)
            if ref == 43:
                raise TaigaRestException("/userstories/by_ref", 403, "You do not have permission.")
            return STORIES.get(ref)  # fetch_entity answers None for a 404
        return tasks.setdefault(ref, _Task(ref=ref, id=ref))

    monkeypatch.setenv("OPENAI_API_KEY", "FAKE")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    monkeypatch.setattr(taiga_tools, "fetch_entity", fetch)
    monkeypatch.setattr(taiga_tools, "small_llm", _NoLLM())
    return {"tasks": tasks, "lookups": lookups}


def _update(entity_type="task", **params):
    args = {"project_slug": "p", "entity_ref": 101, "entity_type": entity_type, **params}
    return json.loads(update_entity_by_ref_tool.invoke(args))


def _patches(env, ref=101):
    return env["tasks"][ref].patches if ref in env["tasks"] else []


def test_the_task_moves_with_the_new_storys_sprint_in_one_patch(env):
    out = _update(user_story_ref=41)
    assert _patches(env) == [(["version"], {"user_story": 410, "milestone": 8})]
    assert "error" not in out


def test_a_story_in_the_backlog_takes_the_task_out_of_its_sprint(env):
    _update(user_story_ref=42)
    assert _patches(env) == [(["version"], {"user_story": 420, "milestone": None})]


def test_the_move_goes_out_with_the_other_fields(env):
    out = _update(user_story_ref=41, status="Done", comment="Moved with the split", strict=True)
    assert _patches(env) == [
        (["version"], {"status": 30, "user_story": 410, "milestone": 8, "comment": "Moved with the split"})
    ]
    assert out["applied"] == ["status", "user_story", "milestone", "comment"]


def test_an_unknown_story_is_a_404_and_nothing_is_written(env):
    out = _update(user_story_ref=12345, status="Done", strict=True)
    assert out["code"] == 404 and "12345" in out["error"]
    assert _patches(env) == []


def test_a_task_already_under_that_story_is_not_moved(env):
    out = _update(user_story_ref=40, comment="x")
    assert _patches(env) == [(["version"], {"comment": "x"})]
    assert out["applied"] == ["comment"]


@pytest.mark.parametrize("entity_type", ["userstory", "issue", "epic"])
def test_only_a_task_can_be_moved_to_a_story(env, entity_type):
    out = _update(entity_type=entity_type, user_story_ref=41)
    assert out["code"] == 400 and "task" in out["error"]
    assert env["lookups"] == []
    assert _patches(env) == []


def test_a_refused_story_lookup_keeps_taigas_code_and_writes_nothing(env):
    out = _update(user_story_ref=43)
    assert out["code"] == 403
    assert _patches(env) == []


def test_control_a_sprint_on_a_task_is_still_refused_and_points_at_the_move(env):
    out = _update(milestone="Sprint 8")
    assert out["code"] == 400
    assert "user_story_ref" in out["error"]
    assert _patches(env) == []


def test_read_back_shows_the_tasks_story_and_sprint(env, monkeypatch):
    moved = _Task(
        version=3, user_story=410, milestone=8,
        user_story_extra_info={"id": 410, "ref": 41, "subject": "Follow-up story"},
    )
    task_reads = []

    def fetch(_project, norm, ref):
        if norm == "us":
            return STORIES.get(ref)
        task_reads.append(ref)
        return env["tasks"].setdefault(ref, _Task(ref=ref, id=ref)) if len(task_reads) == 1 else moved

    monkeypatch.setattr(taiga_tools, "fetch_entity", fetch)
    monkeypatch.setattr(taiga_tools, "fetch_history", lambda entity, norm: [])
    monkeypatch.setattr(taiga_tools, "get_status", lambda slug, typ, sid: {"name": "New", "is_closed": False})
    out = _update(user_story_ref=41, read_back=True)
    state = out["state"]
    assert (state["user_story_ref"], state["ids"]["user_story"], state["ids"]["milestone"]) == (41, 410, 8)


# -- update_entities_by_ref_tool -------------------------------------------------------------------


def _bulk(**params):
    args = {"project_slug": "p", "entity_type": "task", "read_back": False, **params}
    return json.loads(update_entities_by_ref_tool.invoke(args))


def test_bulk_moves_every_task_to_the_story(env):
    out = _bulk(items=[{"entity_ref": 101}, {"entity_ref": 102}], user_story_ref=41)
    assert out["succeeded"] == [101, 102] and out["failed"] == []
    for ref in (101, 102):
        assert _patches(env, ref) == [(["version"], {"user_story": 410, "milestone": 8})]


def test_bulk_an_unknown_story_fails_every_item_and_writes_nothing(env):
    out = _bulk(items=[{"entity_ref": 101}, {"entity_ref": 102}], user_story_ref=12345)
    assert out["succeeded"] == [] and out["failed"] == [101, 102]
    assert _patches(env, 101) == [] and _patches(env, 102) == []


def test_control_bulk_without_any_change_is_still_refused(env):
    out = _bulk(items=[{"entity_ref": 101}])
    assert out["code"] == 400 and "user_story_ref" in out["error"]


# -- a boolean is not a reference ------------------------------------------------------------------


def test_a_boolean_story_ref_is_refused_not_read_as_story_1(env):
    """Pydantic's lax mode would turn True into 1 and move the task to story #1."""
    with pytest.raises(ValueError):
        _update(user_story_ref=True)
    with pytest.raises(ValueError):
        _bulk(items=[{"entity_ref": 101}], user_story_ref=True)
    assert env["lookups"] == []
    assert _patches(env) == []


def test_the_mcp_schema_still_publishes_an_integer_ref():
    schema = update_entity_by_ref_tool.args_schema.model_json_schema()["properties"]["user_story_ref"]
    assert {"type": "integer"} in schema["anyOf"]


def test_a_digit_string_moves_the_task_like_an_integer(env):
    """A client whose cached schema predates user_story_ref sends the number as a string.

    The other integer parameters (entity_ref, expected_version) accept that, and so must this one.
    """
    _update(user_story_ref="41")
    assert _patches(env) == [(["version"], {"user_story": 410, "milestone": 8})]
    out = _bulk(items=[{"entity_ref": 102}], user_story_ref="41")
    assert out["succeeded"] == [102]
    assert _patches(env, 102) == [(["version"], {"user_story": 410, "milestone": 8})]


def test_control_a_non_numeric_string_is_still_refused(env):
    with pytest.raises(ValueError):
        _update(user_story_ref="Follow-up story")
    assert env["lookups"] == []
