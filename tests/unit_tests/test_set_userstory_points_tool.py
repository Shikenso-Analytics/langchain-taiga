"""Unit tests for ``set_userstory_points_tool``.

Splits into two halves:

- ``TestSetUserstoryPointsUnit`` is the standard ``ToolsUnitTests`` scaffold
  that every other tool in this package has — it asserts the tool object
  is well-formed (callable, schema parses, etc.).
- The behavioural tests below use a hand-rolled ``_FakeProject`` /
  ``_FakeUS`` pair to exercise the role-name / point-value resolution
  logic without spinning up a real ``taiga.TaigaAPI`` client.
"""

import json

import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import set_userstory_points_tool
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """``OPENAI_API_KEY`` is read at module import time by the small_llm
    helper inside ``taiga_tools.py`` — same pattern as every other tool
    test in this directory."""
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestSetUserstoryPointsUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return set_userstory_points_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "project_slug": "slug",
            "user_story_ref": 34,
            "points": {"Developer": 5},
        }


# ---------------------------------------------------------------------------
# Behavioural tests with a hand-rolled fake project. We avoid spinning up
# python-taiga's Resource hierarchy because the only contract that matters
# here is: name -> role.id, value -> point.id, then us.points mutated, then
# us.patch(["points"]) called.
# ---------------------------------------------------------------------------


class _FakeRole:
    def __init__(self, rid, name):
        self.id = rid
        self.name = name


class _FakePoint:
    def __init__(self, pid, value):
        self.id = pid
        self.value = value


class _FakeUS:
    def __init__(self, ref, version=1, points=None):
        self.ref = ref
        self.id = 999
        self.version = version
        self.points = dict(points or {})
        self.patch_calls = []
        self.update_calls = []

    def patch(self, fields):
        self.patch_calls.append((tuple(fields), dict(self.points)))
        self.version += 1
        return self

    def update(self):
        # The implementation MUST use patch(["points"]), never update().
        # update() sends a full PUT and risks clobbering concurrent edits
        # on other fields. Failing loudly here guards against regression.
        self.update_calls.append(dict(self.points))
        raise AssertionError(
            "set_userstory_points_tool must call us.patch([\"points\"]), "
            "not us.update() — see spec rationale."
        )


class _FakeProject:
    name = "Wahed"
    slug = "wahed"
    id = 1

    def __init__(self, us, roles, point_scale):
        self._us = us
        self._roles = roles
        self._points = point_scale

    def get_userstory_by_ref(self, ref):
        return self._us if (self._us and self._us.ref == ref) else None

    def list_roles(self):
        return self._roles

    def list_points(self):
        return self._points


@pytest.fixture
def fake_env(monkeypatch):
    us = _FakeUS(ref=34, version=2, points={})
    project = _FakeProject(
        us=us,
        roles=[
            _FakeRole(19, "Developer"),
            _FakeRole(20, "UX"),
            _FakeRole(21, "Design"),
        ],
        point_scale=[
            _FakePoint(100, 1),
            _FakePoint(101, 2),
            _FakePoint(102, 3),
            _FakePoint(103, 5),
            _FakePoint(104, 8),
            _FakePoint(999, None),  # the "?" unestimated point
        ],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    return project, us


def test_sets_developer_points_by_role_name(fake_env):
    _, us = fake_env
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 34,
        "points": {"Developer": 5},
    })
    payload = json.loads(raw)
    assert payload["updated"] is True
    assert payload["points_set"] == {"Developer": 5}
    # role_id "19" -> point_id 103 (whose value is 5)
    assert us.points["19"] == 103
    assert len(us.patch_calls) == 1
    fields, _ = us.patch_calls[0]
    assert fields == ("points",)


def test_preserves_other_role_points(fake_env):
    _, us = fake_env
    # Pre-existing UX assignment must stay untouched when only Developer
    # is being set.
    us.points["20"] = 102  # UX = value 3
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 34,
        "points": {"Developer": 5},
    })
    json.loads(raw)
    assert us.points["20"] == 102
    assert us.points["19"] == 103


def test_role_name_case_insensitive(fake_env):
    _, us = fake_env
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 34,
        "points": {"developer": 5},
    })
    assert json.loads(raw)["updated"] is True
    # Lock that the lowercase name resolved to the SAME role-id as
    # "Developer" would, not to a no-op or a different role.
    assert us.points["19"] == 103


def test_multiple_roles_in_one_call(fake_env):
    _, us = fake_env
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 34,
        "points": {"Developer": 5, "UX": 2},
    })
    payload = json.loads(raw)
    assert payload["updated"] is True
    assert us.points["19"] == 103  # Developer = 5
    assert us.points["20"] == 101  # UX = 2
    # One patch call per tool invocation, regardless of how many roles.
    assert len(us.patch_calls) == 1


def test_unknown_role_returns_400_with_diagnostics(fake_env):
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 34,
        "points": {"Marketing": 5},
    })
    payload = json.loads(raw)
    assert payload["code"] == 400
    assert "Marketing" in payload["unresolved_roles"]
    # available_roles is alphabetically sorted, deterministic for LLM
    # consumption and assertions.
    assert payload["available_roles"] == ["Design", "Developer", "UX"]


def test_unknown_value_returns_400_with_diagnostics(fake_env):
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 34,
        "points": {"Developer": 99},
    })
    payload = json.loads(raw)
    assert payload["code"] == 400
    assert payload["unresolved_values"] == [{"role": "Developer", "value": 99}]
    # available_point_values excludes the "?" point (value None) and is
    # sorted ascending.
    assert payload["available_point_values"] == [1, 2, 3, 5, 8]


def test_no_partial_write_when_validation_fails(fake_env):
    """If any role or value fails to resolve, NO partial write happens —
    even if other entries in the same call are valid."""
    _, us = fake_env
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 34,
        # Developer:5 is valid, Marketing:1 is not. Whole call must fail.
        "points": {"Developer": 5, "Marketing": 1},
    })
    payload = json.loads(raw)
    assert payload["code"] == 400
    assert us.patch_calls == []
    assert "19" not in us.points  # the valid half was NOT applied


def test_us_not_found_returns_404(monkeypatch):
    project = _FakeProject(
        us=_FakeUS(ref=34),
        roles=[_FakeRole(19, "Developer")],
        point_scale=[_FakePoint(100, 1)],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 999,
        "points": {"Developer": 1},
    })
    assert json.loads(raw)["code"] == 404


def test_calls_patch_not_update(fake_env):
    """Regression guard: the implementation must call us.patch(['points']),
    NOT us.update(). update() sends a full PUT and risks clobbering
    concurrent edits on other fields. ``_FakeUS.update`` raises, so any
    accidental switch to update() in the impl will fail this test loudly."""
    _, us = fake_env
    raw = set_userstory_points_tool.invoke({
        "project_slug": "wahed",
        "user_story_ref": 34,
        "points": {"Developer": 5},
    })
    assert json.loads(raw)["updated"] is True
    assert us.update_calls == []
    assert len(us.patch_calls) == 1
    fields, _ = us.patch_calls[0]
    assert fields == ("points",)
