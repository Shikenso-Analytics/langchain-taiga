"""Status resolution on create, and sprint membership on create/update.

Both were gaps rather than regressions:

- ``create_entity_tool`` resolved ``status`` per entity type, three different
  ways, and only the issue branch was right. User stories dropped it silently
  (US #8130 sat in ``New`` for a whole sprint, then jumped to ``Done``).
- Sprint membership was not expressible through the MCP at all; it needed a
  hand-rolled REST PATCH.
"""

from __future__ import annotations

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    create_entity_tool,
    resolve_milestone,
    update_entity_by_ref_tool,
)


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")


class _Created:
    ref = 42
    subject = "s"
    id = 1


class _Project:
    """Records the payload each add_* would send to Taiga."""

    def __init__(self, captured):
        self._captured = captured

    def add_user_story(self, **kw):
        self._captured.update(kw)
        return _Created()

    def add_issue(self, **kw):
        self._captured.update(kw)
        return _Created()

    def add_epic(self, **kw):
        self._captured.update(kw)
        return _Created()

    def get_userstory_by_ref(self, ref):
        return _ParentUs(self._captured)

    # The issue branch falls back to the project's first entry for each of
    # these when the optional argument is omitted.
    def list_issue_types(self):
        return [_Attr(11)]

    def list_severities(self):
        return [_Attr(12)]

    def list_priorities(self):
        return [_Attr(13)]


class _Attr:
    def __init__(self, id):
        self.id = id


class _ParentUs:
    def __init__(self, captured):
        self._captured = captured

    def add_task(self, **kw):
        self._captured.update(kw)
        return _Created()


@pytest.fixture
def captured(monkeypatch):
    seen = {}
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project(seen))
    monkeypatch.setattr(taiga_tools, "_invalidate_tag_cache", lambda *a, **k: None)
    return seen


# ---------------------------------------------------------------------------
# Status on create.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entity_type,extra",
    [
        ("userstory", {}),
        ("issue", {}),
        ("epic", {}),
        ("task", {"parent_ref": 7}),
    ],
)
def test_create_sends_the_resolved_status_for_every_type(
    monkeypatch, captured, entity_type, extra
):
    """The user story case is the regression: it used to send no status at all
    and the entity landed in the project default, silently."""
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda **kw: [99])
    monkeypatch.setattr(taiga_tools, "find_issue_type_ids", lambda *a, **k: [1])
    monkeypatch.setattr(taiga_tools, "find_severity_ids", lambda *a, **k: [1])
    monkeypatch.setattr(taiga_tools, "find_priority_ids", lambda *a, **k: [1])

    out = json.loads(
        create_entity_tool.invoke(
            {
                "project_slug": "p",
                "entity_type": entity_type,
                "subject": "s",
                "status": "Ready for test",
                **extra,
            }
        )
    )
    assert out.get("created") is True, out
    assert captured.get("status") == 99, f"{entity_type} dropped the status"


@pytest.mark.parametrize(
    "entity_type,extra",
    [
        ("userstory", {}),
        ("issue", {}),
        ("epic", {}),
        ("task", {"parent_ref": 7}),
    ],
)
def test_create_rejects_an_unknown_status_for_every_type(
    monkeypatch, captured, entity_type, extra
):
    """Previously: user stories and epics ignored it, tasks raised IndexError
    and surfaced as 'Creation failed: list index out of range'."""
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda **kw: [])

    out = json.loads(
        create_entity_tool.invoke(
            {
                "project_slug": "p",
                "entity_type": entity_type,
                "subject": "s",
                "status": "Nope",
                **extra,
            }
        )
    )
    assert out.get("code") == 404, out
    assert "Nope" in out["error"]
    assert captured == {}, "nothing may be created when the status is unknown"


# ---------------------------------------------------------------------------
# Sprint resolution.
# ---------------------------------------------------------------------------


@pytest.fixture
def sprints(monkeypatch):
    data = [
        {
            "id": 229,
            "name": "Sprint 95",
            "closed": False,
            "estimated_start": "2026-08-18",
            "estimated_finish": "2026-09-01",
        },
        {
            "id": 228,
            "name": "Sprint 94",
            "closed": False,
            "estimated_start": "2026-08-04",
            "estimated_finish": "2026-08-18",
        },
        {
            "id": 224,
            "name": "Sprintless",
            "closed": True,
            "estimated_start": None,
            "estimated_finish": None,
        },
    ]
    monkeypatch.setattr(taiga_tools, "list_milestones", lambda slug: data)
    monkeypatch.setattr(taiga_tools, "get_current_milestone", lambda slug: data[0])
    return data


def test_resolve_milestone_by_name_and_current(sprints):
    assert resolve_milestone("p", "Sprint 94") == (228, None)
    assert resolve_milestone("p", "current sprint") == (229, None)
    assert resolve_milestone("p", "229") == (229, None)


@pytest.mark.parametrize("clear", ["", "none", "None", "null", "backlog"])
def test_resolve_milestone_clear_words_mean_remove_from_sprint(sprints, clear):
    assert resolve_milestone("p", clear) == (None, None)


def test_an_unknown_sprint_errors_instead_of_clearing(sprints):
    """The dangerous case. ``find_milestone_id`` answers None for BOTH 'no
    match' and 'empty', so reusing it directly would turn a typo'd sprint name
    into 'remove from sprint' — silently, on a write."""
    milestone_id, err = resolve_milestone("p", "Sprint 9999")
    assert milestone_id is None
    assert err["code"] == 404
    assert "Sprint 95" in err["open_sprints"]
    assert "Sprintless" not in err["open_sprints"], "closed sprints are noise here"


# ---------------------------------------------------------------------------
# Sprint on create / update.
# ---------------------------------------------------------------------------


def test_create_puts_a_userstory_in_a_sprint(monkeypatch, captured, sprints):
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda **kw: [1])
    out = json.loads(
        create_entity_tool.invoke(
            {
                "project_slug": "p",
                "entity_type": "userstory",
                "subject": "s",
                "status": "New",
                "milestone": "current sprint",
            }
        )
    )
    assert out.get("created") is True, out
    assert captured["milestone"] == 229


@pytest.mark.parametrize(
    "entity_type,extra", [("task", {"parent_ref": 7}), ("epic", {})]
)
def test_create_refuses_a_sprint_where_taiga_has_none(
    monkeypatch, captured, sprints, entity_type, extra
):
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda **kw: [1])
    monkeypatch.setattr(taiga_tools, "find_issue_type_ids", lambda *a, **k: [1])
    out = json.loads(
        create_entity_tool.invoke(
            {
                "project_slug": "p",
                "entity_type": entity_type,
                "subject": "s",
                "status": "New",
                "milestone": "Sprint 95",
                **extra,
            }
        )
    )
    assert out.get("code") == 400, out
    assert captured == {}


class _Entity:
    id = 6820
    ref = 8459
    version = 3

    def __init__(self, patched):
        self._patched = patched

    def patch(self, fields, **kw):
        self._patched["fields"] = fields
        self._patched.update(kw)


@pytest.fixture
def patched(monkeypatch):
    seen = {}
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: object())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda p, t, r: _Entity(seen))
    return seen


def test_update_moves_an_entity_into_a_sprint(patched, sprints):
    out = json.loads(
        update_entity_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 8459,
                "entity_type": "userstory",
                "milestone": "Sprint 94",
            }
        )
    )
    assert "updated successfully" in out["message"]
    assert patched["milestone"] == 228
    assert "version" in patched["fields"], "optimistic-lock field must ride along"


def test_update_can_take_an_entity_out_of_its_sprint(patched, sprints):
    """An empty string is the documented 'remove from sprint'. A truthiness
    check on the argument would silently ignore it."""
    json.loads(
        update_entity_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 8459,
                "entity_type": "userstory",
                "milestone": "",
            }
        )
    )
    assert "milestone" in patched and patched["milestone"] is None


def test_update_leaves_the_sprint_alone_when_not_asked(patched, sprints):
    json.loads(
        update_entity_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 8459,
                "entity_type": "userstory",
                "subject": "new subject",
            }
        )
    )
    assert "milestone" not in patched


def test_update_refuses_a_typod_sprint_without_touching_the_entity(patched, sprints):
    out = json.loads(
        update_entity_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 8459,
                "entity_type": "userstory",
                "milestone": "Sprnit 95",
            }
        )
    )
    assert out["code"] == 404
    assert patched == {}, "a failed sprint lookup must not write anything"
