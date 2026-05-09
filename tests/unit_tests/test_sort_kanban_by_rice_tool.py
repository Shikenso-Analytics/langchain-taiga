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
    # ``taiga_tools.TAIGA_URL`` is captured at import time via
    # ``os.getenv("TAIGA_URL")``, so ``monkeypatch.setenv`` would be a
    # no-op here (the module was already imported at the top of this
    # file). Patch the module attribute directly so
    # ``TAIGA_URL.rstrip("/")`` inside ``sort_kanban_by_rice_tool``
    # doesn't ``AttributeError`` on ``None`` in CI environments where
    # ``TAIGA_URL`` is unset.
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.test")


class _FakeAttr:
    def __init__(self, aid, name):
        self.id = aid
        self.name = name


class _FakePoint:
    def __init__(self, pid, value):
        self.id = pid
        self.value = value


class _FakeUS:
    def __init__(self, ref, points, attr_values, status=100, total_points=None):
        self.ref = ref
        self.id = ref * 1000
        self.subject = f"US {ref}"
        self.points = points
        # Taiga's `/userstories?project=X` list endpoint inlines
        # ``total_points`` (sum across computable roles' assignments) so
        # since 2.3.2 the tool reads it directly instead of re-summing
        # ``points.values()`` against a separate ``list_points()`` lookup.
        # Fakes mirror that contract: the test sets the canonical sum.
        self.total_points = total_points
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


def test_effort_takes_us_total_points(monkeypatch, patched_http):
    """Effort = ``us.total_points`` (Taiga-computed sum across all
    computable roles' point assignments). Pre-2.3.2 we computed this
    locally from ``us.points`` against a separate ``list_points()``
    lookup; since 2.3.2 we trust Taiga's own pre-computed field, which
    is inlined on every list-userstories response."""
    story = _FakeUS(
        ref=34,
        points={"19": 103, "20": 101},  # Developer=5, UX=2 (legacy field)
        total_points=7.0,                # what Taiga computed
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
    assert order[0]["effort"] == 7.0


def test_works_regardless_of_role_id(monkeypatch, patched_http):
    """Pre-2.3.0 the tool hard-coded ``developer_role_id = "19"`` and
    silently zeroed effort on projects where Developer was a different
    role-id. Pre-2.3.2 it summed ``us.points.values()`` itself which
    needed a separate role/points lookup. Now it reads ``total_points``
    directly — completely role-id-agnostic, no project-wide lookup,
    works on any project regardless of how Taiga numbers its roles."""
    story = _FakeUS(
        ref=10,
        points={"42": 103},  # Developer at role-id 42 (legacy field)
        total_points=5.0,
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
    assert order[0]["effort"] == 5.0


def test_effort_zero_when_no_points_assigned(monkeypatch, patched_http):
    """Stories without any role-points assigned (``total_points=None``
    from Taiga) get effort=0 and consequently rice_score=0 — they sort
    to the bottom of their column instead of crashing the tool. Edge
    case for newly-created stories that haven't been estimated yet."""
    story = _FakeUS(
        ref=99,
        points={},
        total_points=None,
        attr_values={"1": 4, "2": 3, "3": 1},
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    order = payload["columns_updated"][0]["order"]
    assert order[0]["effort"] == 0
    assert order[0]["rice"] == 0


def test_outer_try_returns_json_on_unexpected_failure(monkeypatch):
    """Pre-2.3.2 the tool had inner try/excepts only — uncaught failures
    bubbled up to the FastMCP harness as the generic 'Error occurred
    during tool execution' with no diagnostic. The 2.3.2 outer
    try/except catches every uncaught exception and returns a JSON 500
    with ``trace_tail`` so the LLM (and the next debugger) sees what
    actually broke."""
    class _Boom:
        # Looks project-shaped enough for the early-validation gate to
        # pass, but explodes when we walk into list_user_story_attributes.
        name = "wahed"
        slug = "wahed"
        id = 1

        def list_user_story_attributes(self):
            raise RuntimeError("simulated network failure mid-fetch")

    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Boom())

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    assert payload["code"] == 500
    assert "RuntimeError" in payload["error"]
    assert "simulated network failure mid-fetch" in payload["error"]
    # Trace tail must be a list of strings (last 5 lines of the
    # traceback) — NOT a single newline-joined blob, so the harness's
    # JSON pretty-printer keeps each line readable.
    assert isinstance(payload["trace_tail"], list)
    assert len(payload["trace_tail"]) <= 5
