"""Task (milestone) custom attributes and attribute names in the board view (2.23.0).

Links between projects often live in task attributes ("Supplier team", "Supplier status").
Before, every one of them cost a separate call per task. Named-only like the card attributes: a task attribute costs a
request per task, so ``fields=["columns"]`` must not fan out.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_kanban_board_tool
from tests.unit_tests.test_get_kanban_board_tool import _FakeStatus

API = "https://taiga.test/api/v1"


def _summary(task_id, ref):
    return {
        "id": task_id,
        "ref": ref,
        "subject": f"M{ref}",
        "status_id": 1,
        "is_closed": False,
        "is_blocked": False,
        "is_iocaine": False,
    }


class _Story:
    def __init__(self, sid, ref, tasks):
        self.id = sid
        self.ref = ref
        self.subject = f"US {ref}"
        self.status = 1
        self.kanban_order = ref
        self.assigned_to = None
        self._tasks = tasks
        self.tasks = []


class _Attr(SimpleNamespace):
    pass


class _Project:
    id = 39
    name = "Board"
    members = []

    def __init__(self):
        self.stories = [_Story(101, 1, [_summary(501, 11), _summary(502, 12)]), _Story(102, 2, [])]

    def list_user_story_statuses(self):
        return [_FakeStatus(1, "Later", order=1)]

    def list_user_stories(self, **queryparams):
        for story in self.stories:
            story.tasks = [dict(t) for t in story._tasks] if str(queryparams.get("include_tasks")) == "1" else []
        return list(self.stories)

    def list_user_story_attributes(self):
        return [_Attr(id=91, name="Next team", type="dropdown", extra="Frontend", description="")]

    def list_task_attributes(self):
        return [_Attr(id=23, name="Supplier team", type="dropdown", extra="Frontend", description="")]


@pytest.fixture
def env(monkeypatch):
    taiga_tools.custom_attr_definitions_cache.clear()
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "TAIGA_API_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "get_taiga_api", lambda token=None: SimpleNamespace(token="tok"))
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    yield
    taiga_tools.custom_attr_definitions_cache.clear()


def _routes(respx_mock, fail_task=None):
    routes = {}
    for task_id in (501, 502):
        route = respx_mock.get(f"{API}/tasks/custom-attributes-values/{task_id}")
        if task_id == fail_task:
            route.mock(return_value=httpx.Response(500))
        else:
            route.mock(
                return_value=httpx.Response(200, json={"attributes_values": {"23": f"D{task_id}"}, "version": 1})
            )
        routes[task_id] = route
    routes["story"] = respx_mock.get(url__regex=rf"{API}/userstories/custom-attributes-values/\d+").mock(
        return_value=httpx.Response(200, json={"attributes_values": {"91": "Frontend"}, "version": 1})
    )
    return routes


def _board(**kw):
    return json.loads(get_kanban_board_tool.invoke({"project_slug": "b", "compact": True, **kw}))


@pytest.mark.respx(assert_all_called=False)
def test_task_attributes_come_per_task_with_names(env, respx_mock):
    routes = _routes(respx_mock)
    out = _board(fields=["columns.cards.ref", "columns.cards.tasks.ref", "columns.cards.tasks.custom_attributes"])
    assert out["columns"][0]["cards"] == [
        {
            "ref": 1,
            "tasks": [
                {"ref": 11, "custom_attributes": {"23": "D501"}},
                {"ref": 12, "custom_attributes": {"23": "D502"}},
            ],
        },
        {"ref": 2, "tasks": []},
    ]
    assert out["custom_attribute_names"] == {"task": {"23": "Supplier team"}}
    assert not routes["story"].called
    assert routes[501].calls[0].request.headers["Authorization"] == "Bearer tok"


@pytest.mark.respx(assert_all_called=False)
def test_both_kinds_bring_both_name_maps(env, respx_mock):
    _routes(respx_mock)
    out = _board(fields=["columns.cards.custom_attributes", "columns.cards.tasks.custom_attributes"])
    assert out["custom_attribute_names"] == {"userstory": {"91": "Next team"}, "task": {"23": "Supplier team"}}


@pytest.mark.respx(assert_all_called=False)
def test_a_task_that_fails_twice_fails_the_call_naming_it(env, respx_mock):
    _routes(respx_mock, fail_task=502)
    out = _board(fields=["columns.cards.tasks.custom_attributes"])
    assert out["code"] == 502 and "#12" in out["error"]


@pytest.mark.respx(assert_all_called=False)
def test_too_many_tasks_are_refused_before_any_request(env, respx_mock, monkeypatch):
    """One GET per task: a large board would fire thousands at the shared Taiga in one call."""
    monkeypatch.setattr(taiga_tools, "_KANBAN_TASK_ATTRIBUTES_MAX", 1)
    routes = _routes(respx_mock)
    out = _board(fields=["columns.cards.tasks.custom_attributes"])
    assert out["code"] == 400 and "statuses" in out["error"]
    assert not routes[501].called and not routes[502].called


@pytest.mark.respx(assert_all_called=False)
def test_control_an_ancestor_path_fetches_no_task_attributes(env, respx_mock):
    routes = _routes(respx_mock)
    out = _board(fields=["columns"])
    assert not routes[501].called and not routes[502].called
    assert "custom_attribute_names" not in out


@pytest.mark.respx(assert_all_called=False)
def test_control_the_default_answer_has_no_names(env, respx_mock):
    _routes(respx_mock)
    out = _board()
    assert "custom_attribute_names" not in out
