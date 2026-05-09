"""Regression tests for the ``sort_kanban_by_rice_tool`` effort refactor.

Before 2.3.0, effort was read only from the Developer role with a
hard-coded ``role_id = "19"`` lookup. After 2.3.0, effort = sum of every
role's points. These tests lock that contract and exercise it against a
project where Developer's role-id is NOT 19.
"""

import json
from types import SimpleNamespace

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import sort_kanban_by_rice_tool


@pytest.fixture(autouse=True)
def fake_env_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")
    monkeypatch.setenv("TAIGA_URL", "https://taiga.test")


class _FakeAttr:
    def __init__(self, aid, name):
        self.id = aid
        self.name = name


class _FakePoint:
    def __init__(self, pid, value):
        self.id = pid
        self.value = value


class _FakeUS:
    def __init__(self, ref, points, attr_values, status=100):
        self.ref = ref
        self.id = ref * 1000
        self.subject = f"US {ref}"
        self.points = points
        self._attr_values = attr_values
        self.status = status
        self.epics = None
        self.due_date = None
        self.is_closed = False

    def get_attributes(self):
        return {"attributes_values": self._attr_values, "version": 1}


class _FakeProject:
    name = "Test"
    id = 7

    def __init__(self, stories, roles, points, us_attrs, epic_attrs=None):
        self._stories = stories
        self._roles = roles
        self._points = points
        self._us_attrs = us_attrs
        self._epic_attrs = epic_attrs or []

    def list_user_stories(self):
        return self._stories

    def list_roles(self):
        return self._roles

    def list_points(self):
        return self._points

    def list_user_story_attributes(self):
        return self._us_attrs

    def list_epic_attributes(self):
        return self._epic_attrs

    def list_epics(self):
        return []


@pytest.fixture
def patched_http(monkeypatch):
    """``--disable-socket`` blocks the bulk-update POST. We stub out
    ``requests.post`` to a 200 no-op so the tool can complete; the test
    inspects the JSON response, not the wire effect."""
    fake_response = SimpleNamespace(status_code=200, json=lambda: {})
    monkeypatch.setattr(
        taiga_tools.requests, "post", lambda *a, **kw: fake_response
    )
    # Also stub the ``get_taiga_api`` call so no real HTTP happens.
    fake_api = SimpleNamespace(token="fake-token")
    monkeypatch.setattr(taiga_tools, "get_taiga_api", lambda token=None: fake_api)


def _baseline_attrs():
    """Custom attribute definitions: 1=reach, 2=impact, 3=confidence."""
    return [
        _FakeAttr(1, "Reach"),
        _FakeAttr(2, "Impact"),
        _FakeAttr(3, "Confidence"),
    ]


def _baseline_points():
    return [
        _FakePoint(100, 1),
        _FakePoint(101, 2),
        _FakePoint(102, 3),
        _FakePoint(103, 5),
        _FakePoint(104, 8),
    ]


def test_effort_is_sum_of_all_role_points(monkeypatch, patched_http):
    """A story with Developer=5 (point id 103) AND UX=2 (point id 101)
    must produce effort=7 in the response, not effort=5."""
    story = _FakeUS(
        ref=34,
        points={"19": 103, "20": 101},  # Developer=5, UX=2
        attr_values={"1": 4, "2": 3, "3": 1},  # reach, impact, confidence
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer"), _FakeAttr(20, "UX")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    cols = payload["columns_updated"]
    assert len(cols) == 1
    order = cols[0]["order"]
    assert len(order) == 1
    assert order[0]["effort"] == 7  # 5 (Developer) + 2 (UX)


def test_works_when_developer_role_is_not_id_19(monkeypatch, patched_http):
    """The pre-2.3.0 hard-coded ``developer_role_id = "19"`` lookup
    silently produced effort=0 on any project where Developer was a
    different role-id. This test proves the bug is gone: project where
    Developer's role-id is 42; story has only Developer=5; effort
    correctly reads as 5."""
    story = _FakeUS(
        ref=10,
        points={"42": 103},  # Developer (id=42) = 5
        attr_values={"1": 1, "2": 1, "3": 1},
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(42, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    order = payload["columns_updated"][0]["order"]
    assert order[0]["effort"] == 5
