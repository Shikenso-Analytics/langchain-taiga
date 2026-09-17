"""Many tasks in one call: their recent history, and one change applied to all of them (2.22.0).

A morning run handed 53 tasks of one story over to a PM. Checking that hand-over (one comment per
task) cost one ``get_entity_by_ref_tool`` call per task, and making it cost one
``update_entity_by_ref_tool`` call per task — through a model, each call is a turn.

* ``related.tasks.history`` embeds each task's history, filtered like the story's. It is read
  only when a path names it and only with ``history_since``, and only for tasks modified since
  then: a task's history entry comes with a save, and a save stamps ``modified_date``.
* ``update_entities_by_ref_tool`` applies one change set to many entities, each with its own
  PATCH, its own ``expected_version`` and its own answer — the same answer
  ``update_entity_by_ref_tool`` gives for one.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_entity_by_ref_tool, update_entities_by_ref_tool
from tests.unit_tests.test_update_single_patch import MEMBERS, TASK_STATUSES, _Entity, _NoLLM, _Stub

API = "https://taiga.test/api/v1"
TODAY = "2026-09-17"

TASK_HISTORY = {
    501: [
        {"created_at": "2026-09-17T06:19:00Z", "user": {"pk": 51}, "comment": "Handed over", "diff": {"status": [27, 29]}},
        {"created_at": "2026-09-16T10:00:00Z", "user": {"pk": 9}, "comment": "", "diff": {"tags": [[], ["x"]]}},
    ],
    503: [{"created_at": "2026-09-17T07:00:00Z", "user": {"pk": 9}, "comment": "", "diff": {"subject": ["a", "b"]}}],
}


class _Task(_Entity):
    def to_dict(self):
        return {"subject": self.subject, "status": self.status, "assigned_to": self.assigned_to, "version": self.version}


@pytest.fixture
def story_env(monkeypatch):
    story = _Entity(ref=3, subject="League", description="", status=87, version=9, due_date=None, milestone=None,
                    owner=5, owner_extra_info={"id": 5, "username": "w"})
    story.list_tasks = lambda: [
        _Task(id=501, ref=91, subject="R1", status=29, assigned_to=82, modified_date="2026-09-17T06:19:00.010Z"),
        _Task(id=502, ref=92, subject="R2", status=27, assigned_to=None, modified_date="2026-09-16T23:59:59.000Z"),
        _Task(id=503, ref=93, subject="R3", status=27, assigned_to=None, modified_date="2026-09-17T07:00:00.000Z"),
    ]
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "TAIGA_API_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "get_taiga_api", lambda token=None: SimpleNamespace(token="tok"))
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(name="P", slug=slug, members=MEMBERS))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: story)
    monkeypatch.setattr(taiga_tools, "fetch_history", lambda entity, norm: [])
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: {"name": "x"})
    return story


def _routes(respx_mock, failing=()):
    routes = {}
    for task_id in (501, 502, 503):
        route = respx_mock.get(f"{API}/history/task/{task_id}")
        if task_id in failing:
            route.mock(return_value=httpx.Response(500, json={}))
        else:
            route.mock(return_value=httpx.Response(200, json=TASK_HISTORY.get(task_id, [])))
        routes[task_id] = route
    return routes


def _read(**kw):
    args = {"project_slug": "p", "entity_ref": 3, "entity_type": "userstory", "include_history": False,
            "compact": True, **kw}
    return json.loads(get_entity_by_ref_tool.invoke(args))


TASK_FIELDS = ["related.tasks.ref", "related.tasks.history.created_at", "related.tasks.history.comment"]


@pytest.mark.respx(assert_all_called=False)
def test_task_histories_come_filtered_and_only_for_tasks_touched_since(story_env, respx_mock):
    routes = _routes(respx_mock)
    out = _read(fields=TASK_FIELDS, history_since=TODAY)
    assert out["related"]["tasks"] == [
        {"ref": 91, "history": [{"created_at": "2026-09-17T06:19:00Z", "comment": "Handed over"}]},
        {"ref": 92, "history": []},
        {"ref": 93, "history": [{"created_at": "2026-09-17T07:00:00Z", "comment": ""}]},
    ]
    assert not routes[502].called
    assert routes[501].calls[0].request.headers["x-disable-pagination"] == "True"
    assert routes[501].calls[0].request.headers["Authorization"] == "Bearer tok"


@pytest.mark.respx(assert_all_called=False)
def test_the_other_history_filters_apply_per_task(story_env, respx_mock):
    _routes(respx_mock)
    by_51 = _read(fields=TASK_FIELDS, history_since=TODAY, history_user="51")
    assert [t["history"] for t in by_51["related"]["tasks"]] == [
        [{"created_at": "2026-09-17T06:19:00Z", "comment": "Handed over"}], [], [],
    ]
    with_comment = _read(fields=TASK_FIELDS, history_since=TODAY, history_comments_only=True)
    assert [t["history"] for t in with_comment["related"]["tasks"]] == [
        [{"created_at": "2026-09-17T06:19:00Z", "comment": "Handed over"}], [], [],
    ]


def test_task_histories_need_history_since(story_env):
    out = _read(fields=TASK_FIELDS)
    assert out["code"] == 400 and "history_since" in out["error"]


@pytest.mark.respx(assert_all_called=False)
def test_an_ancestor_path_does_not_fan_out(story_env, respx_mock):
    routes = _routes(respx_mock)
    out = _read(fields=["related.tasks", "history.created_at"], history_since=TODAY, include_history=True)
    assert "history" not in out["related"]["tasks"][0]
    assert not any(route.called for route in routes.values())


@pytest.mark.respx(assert_all_called=False)
def test_a_task_history_that_cannot_be_read_fails_the_call(story_env, respx_mock):
    _routes(respx_mock, failing=(503,))
    out = _read(fields=TASK_FIELDS, history_since=TODAY)
    assert out["code"] == 502 and "#93" in out["error"]


@pytest.mark.respx(assert_all_called=False)
def test_control_the_story_history_filter_still_needs_a_history_path(story_env, respx_mock):
    _routes(respx_mock)
    out = _read(fields=["related.tasks.ref"], history_since=TODAY)
    assert out["code"] == 400


# -- update_entities_by_ref_tool ------------------------------------------------------------------


@pytest.fixture
def bulk_env(monkeypatch):
    tasks = {ref: _Entity(ref=ref, version=version) for ref, version in ((1, 3), (2, 5), (3, 8))}
    project = _Stub(id=6, slug="p", members=MEMBERS, list_task_statuses=lambda: TASK_STATUSES)
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: tasks.get(ref))
    monkeypatch.setattr(taiga_tools, "small_llm", _NoLLM())
    return tasks


def _bulk(**params):
    args = {"project_slug": "p", "entity_type": "task", "read_back": False, **params}
    return json.loads(update_entities_by_ref_tool.invoke(args))


def test_one_change_set_one_patch_per_entity(bulk_env):
    out = _bulk(
        items=[{"entity_ref": 1, "expected_version": 3}, {"entity_ref": 2, "expected_version": 5, "comment": "own note"}],
        status="Preparing Processing", assign_to="ben", watchers=["ben"], comment="Handed over",
    )
    assert bulk_env[1].patches == [(["version"], {"status": 29, "assigned_to": 2, "watchers": [1, 2], "comment": "Handed over"})]
    assert bulk_env[2].patches == [(["version"], {"status": 29, "assigned_to": 2, "watchers": [1, 2], "comment": "own note"})]
    assert bulk_env[3].patches == []
    assert [r["target"]["entity_ref"] for r in out["results"]] == [1, 2]
    assert out["succeeded"] == [1, 2] and out["failed"] == []


def test_a_moved_entity_fails_alone(bulk_env):
    out = _bulk(items=[{"entity_ref": 1, "expected_version": 2}, {"entity_ref": 2, "expected_version": 5}], comment="x")
    assert out["results"][0]["code"] == 409 and out["results"][0]["target"]["entity_ref"] == 1
    assert bulk_env[1].patches == [] and bulk_env[2].patches == [(["version"], {"comment": "x"})]
    assert out["succeeded"] == [2] and out["failed"] == [1]


def test_resolution_is_always_strict(bulk_env):
    out = _bulk(items=[{"entity_ref": 1}], status="Preparing")
    assert out["results"][0]["code"] == 404
    assert bulk_env[1].patches == []


def test_read_back_is_on_by_default(bulk_env, monkeypatch):
    monkeypatch.setattr(taiga_tools, "fetch_history", lambda entity, norm: [])
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: {"name": "Preparing Processing", "is_closed": False})
    out = json.loads(update_entities_by_ref_tool.invoke(
        {"project_slug": "p", "entity_type": "task", "items": [{"entity_ref": 1}], "comment": "x"}
    ))
    assert out["results"][0]["state"]["ids"]["status"] == 1


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"entity_ref": 1}, {"entity_ref": 1}],
        [{"expected_version": 3}],
        [{"entity_ref": 1, "status": "Done"}],
        [{"entity_ref": 1, "expected_version": "3"}],
        [{"entity_ref": n} for n in range(1, 102)],
    ],
    ids=["empty", "duplicate", "no-ref", "unknown-key", "string-version", "too-many"],
)
def test_malformed_items_are_refused_before_any_write(bulk_env, items):
    out = _bulk(items=items, comment="x")
    assert out["code"] == 400
    assert all(task.patches == [] for task in bulk_env.values())


def test_results_keep_the_item_order(bulk_env):
    out = _bulk(items=[{"entity_ref": 3}, {"entity_ref": 1}, {"entity_ref": 2}], comment="x")
    assert [r["target"]["entity_ref"] for r in out["results"]] == [3, 1, 2]


def test_an_unknown_entity_is_a_failed_item(bulk_env):
    out = _bulk(items=[{"entity_ref": 1}, {"entity_ref": 77}], comment="x")
    assert out["results"][1]["code"] == 404
    assert out["failed"] == [77] and out["succeeded"] == [1]


def test_a_call_that_changes_nothing_is_refused(bulk_env):
    out = _bulk(items=[{"entity_ref": 1}])
    assert out["code"] == 400 and "Nothing to change" in out["error"]


def test_control_an_item_comment_alone_is_a_change(bulk_env):
    out = _bulk(items=[{"entity_ref": 1, "comment": "only here"}])
    assert bulk_env[1].patches == [(["version"], {"comment": "only here"})]
    assert out["succeeded"] == [1]


def test_every_worker_runs_in_the_callers_context(bulk_env, monkeypatch):
    """The caller's Taiga token lives in a context variable; a worker without it would act as nobody."""
    import contextvars

    probe = contextvars.ContextVar("probe", default="missing")
    seen = []
    original = taiga_tools.fetch_entity

    def fetch(project, norm, ref):
        seen.append(probe.get())
        return original(project, norm, ref)

    monkeypatch.setattr(taiga_tools, "fetch_entity", fetch)
    token = probe.set("caller")
    try:
        _bulk(items=[{"entity_ref": ref} for ref in (1, 2, 3)], comment="x")
    finally:
        probe.reset(token)
    assert seen == ["caller"] * 3
