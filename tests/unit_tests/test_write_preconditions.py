"""Scripted writes tied to the state they were planned on (2.21.0).

A script reads an entity, decides, and a model makes the write later — possibly after somebody
else changed the entity, possibly twice. Taiga's own optimistic lock does not help there: the tool
fetches the CURRENT version right before its PATCH, so the lock only covers the tool's own gap.
``expected_version`` carries the version the decision was made on and refuses the write when the
entity has moved on. ``target`` names what an update answer is about, so an answer saved for one
entity cannot verify a call made for another.
"""

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_entity_by_ref_tool, update_entity_by_ref_tool
from tests.unit_tests.test_update_single_patch import MEMBERS, TASK_STATUSES, _Entity, _NoLLM, _Stub


@pytest.fixture
def env(monkeypatch):
    entity = _Entity(version=7)
    project = _Stub(id=6, slug="p", name="P", members=MEMBERS, list_task_statuses=lambda: TASK_STATUSES)
    epic_links = []
    project.get_epic_by_ref = lambda ref: _Stub(id=90)
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)
    monkeypatch.setattr(taiga_tools, "small_llm", _NoLLM())
    raw = _Stub(post=lambda *a, **kw: epic_links.append(kw))
    monkeypatch.setattr(taiga_tools, "get_taiga_api", lambda token=None: _Stub(raw_request=raw))
    return {"entity": entity, "epic_links": epic_links}


def _update(entity_type="task", **params):
    args = {"project_slug": "p", "entity_ref": 5, "entity_type": entity_type, **params}
    return json.loads(update_entity_by_ref_tool.invoke(args))


def test_a_matching_version_writes(env):
    out = _update(status="30", strict=True, expected_version=7)
    assert env["entity"].patches == [(["version"], {"status": 30})]
    assert out["applied"] == ["status"]


def test_a_moved_entity_is_refused_before_anything_is_written(env):
    out = _update(status="30", comment="again", strict=True, expected_version=6)
    assert out["code"] == 409
    assert out["current_version"] == 7 and out["expected_version"] == 6
    assert env["entity"].patches == []


def test_the_precondition_also_guards_the_separate_epic_link(env):
    out = _update(entity_type="userstory", epic_ref=3, expected_version=6)
    assert out["code"] == 409
    assert env["epic_links"] == [] and env["entity"].patches == []


@pytest.mark.parametrize("bad", [0, -1])
def test_a_version_below_one_is_refused(env, bad):
    assert _update(comment="x", expected_version=bad)["code"] == 400
    assert env["entity"].patches == []


def test_the_answer_names_its_target(env):
    out = _update(comment="x", expected_version=7)
    assert out["target"] == {"project_slug": "p", "entity_ref": 5, "entity_type": "task"}


def test_control_a_legacy_call_answers_as_before(env):
    out = _update(subject="New")
    assert out == {"message": "Task 5 updated successfully."}


# -- the version is readable -----------------------------------------------------------------------


@pytest.fixture
def read_env(monkeypatch):
    story = _Entity(ref=3, subject="S", description="", status=87, version=12, due_date=None, milestone=None,
                    owner=5, owner_extra_info={"id": 5, "username": "w"})
    story.list_tasks = lambda: []
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(name="P", slug=slug, members=MEMBERS))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: story)
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: {"name": "Ongoing"})
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: {"id": uid, "username": f"user{uid}"})


def _read(**kw):
    args = {"project_slug": "p", "entity_ref": 3, "entity_type": "userstory", "include_history": False, **kw}
    return json.loads(get_entity_by_ref_tool.invoke(args))


def test_the_version_comes_on_request(read_env):
    assert _read(fields=["version", "status_id"]) == {
        "version": 12,
        "status_id": 87,
        "query": {
            "project_slug": "p",
            "entity_ref": 3,
            "entity_type": "us",
            "fields": ["version", "status_id"],
            "include_history": False,
            "history_since": None,
            "history_user": None,
            "history_comments_only": False,
            "history_limit": None,
        },
    }


def test_control_the_default_answer_has_no_version(read_env):
    assert "version" not in _read()


def test_expected_version_alone_gets_the_new_answer_shape(env):
    out = _update(subject="New", expected_version=7)
    assert out["applied"] == ["subject"]
    assert out["target"]["entity_ref"] == 5
