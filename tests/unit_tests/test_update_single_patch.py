"""``update_entity_by_ref_tool``: one PATCH for status, assignee, watchers, tags and comment (2.19.0).

A sourcing hand-over moves a task, assigns it, adds watchers and leaves a comment. Through three
separate tools that is three writes and three history entries, and a script that checks "exactly
one entry carries my comment" cannot tell a retry from the design. Here it is one PATCH. ``strict``
takes the language model out of name resolution: a scripted write must never land in a
"semantically similar" status. ``read_back`` returns what Taiga stored, because a PATCH answers
200 whether or not it kept what was sent.
"""

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import update_entity_by_ref_tool


class _Stub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Entity:
    def __init__(self, **kwargs):
        self.id = 11
        self.ref = 5
        self.version = 3
        self.status = 1
        self.assigned_to = None
        self.watchers = [1]
        self.tags = [["voice", None]]
        self.patches = []
        self.__dict__.update(kwargs)

    def patch(self, fields, **kwargs):
        self.patches.append((fields, kwargs))
        return self

    def update(self, **kwargs):
        raise AssertionError("a full PUT must never happen")


MEMBERS = [
    _Stub(id=1, username="anna", full_name="Anna A"),
    _Stub(id=2, username="ben", full_name="Ben B"),
    _Stub(id=3, username="cara", full_name="Ben B"),
]
TASK_STATUSES = [_Stub(id=29, name="Preparing Processing"), _Stub(id=30, name="Done")]


class _NoLLM:
    def invoke(self, _messages):
        raise AssertionError("strict resolution must not ask the language model")


@pytest.fixture
def env(monkeypatch):
    entity = _Entity()
    project = _Stub(id=6, members=MEMBERS, list_task_statuses=lambda: TASK_STATUSES)
    state = {"entity": entity, "fetches": 0, "invalidated": [], "registry": ["voice", "Ops"]}

    def fetch(_project, _norm, _ref):
        state["fetches"] += 1
        return state["entity"]

    monkeypatch.setenv("OPENAI_API_KEY", "FAKE")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    monkeypatch.setattr(taiga_tools, "fetch_entity", fetch)
    monkeypatch.setattr(taiga_tools, "list_all_tags", lambda slug: state["registry"])
    monkeypatch.setattr(taiga_tools, "_invalidate_tag_cache", lambda slug: state["invalidated"].append(slug))
    monkeypatch.setattr(taiga_tools, "small_llm", _NoLLM())
    return state


def _update(**params):
    args = {"project_slug": "p", "entity_ref": 5, "entity_type": "task", **params}
    return json.loads(update_entity_by_ref_tool.invoke(args))


def _only_patch(env):
    assert len(env["entity"].patches) == 1, env["entity"].patches
    fields, kwargs = env["entity"].patches[0]
    assert fields == ["version"]
    return kwargs


# -- one PATCH -------------------------------------------------------------------------------------


def test_everything_goes_out_in_one_patch(env):
    out = _update(
        status="Preparing Processing",
        assign_to="ben",
        watchers=["cara", "anna"],
        tags=["ops"],
        comment="Handed over: jobs A, B",
        strict=True,
    )
    assert _only_patch(env) == {
        "status": 29,
        "assigned_to": 2,
        "watchers": [1, 3],
        "tags": ["voice", "Ops"],
        "comment": "Handed over: jobs A, B",
    }
    assert out["applied"] == ["status", "assigned_to", "watchers", "tags", "comment"]
    assert out["created_tags"] == []


def test_a_comment_alone_is_a_patch_with_the_version(env):
    _update(comment="ping")
    assert _only_patch(env) == {"comment": "ping"}


def test_a_blank_comment_is_refused_before_any_write(env):
    out = _update(comment="   ")
    assert out["code"] == 400
    assert env["entity"].patches == []


@pytest.mark.parametrize(
    ("mode", "given", "expected"),
    [("add", ["ben"], [1, 2]), ("replace", ["ben"], [2]), ("remove", ["anna"], []), ("replace", [], [])],
)
def test_watcher_modes(env, mode, given, expected):
    _update(watchers=given, watchers_mode=mode)
    assert _only_patch(env)["watchers"] == expected


def test_unchanged_watchers_are_not_sent(env):
    _update(watchers=["anna"], comment="x")
    assert "watchers" not in _only_patch(env)


def test_an_unknown_watcher_stops_the_whole_write(env):
    out = _update(watchers=["nobody"], status="Done", strict=True)
    assert out["code"] == 404 and out["unresolved"] == ["nobody"]
    assert env["entity"].patches == []


def test_an_empty_watcher_list_needs_replace(env):
    assert _update(watchers=[])["code"] == 400
    assert env["entity"].patches == []


@pytest.mark.parametrize(
    ("mode", "given", "expected"),
    [("add", ["VOICE", "ops"], None), ("replace", ["ops"], ["Ops"]), ("remove", ["voice"], [])],
)
def test_tag_modes_keep_the_known_spelling(env, mode, given, expected):
    _update(tags=given, tags_mode=mode, comment="x")
    sent = _only_patch(env)
    if expected is None:
        assert sent["tags"] == ["voice", "Ops"]
    else:
        assert sent["tags"] == expected


def test_a_new_project_tag_is_reported_and_evicts_the_cache(env):
    out = _update(tags=["brandnew"])
    assert out["created_tags"] == ["brandnew"]
    assert env["invalidated"] == ["p"]


def test_unchanged_tags_are_not_sent(env):
    out = _update(tags=["Voice"], comment="x")
    assert "tags" not in _only_patch(env)
    assert out["created_tags"] == []


def test_an_invalid_mode_is_refused(env):
    assert _update(watchers=["ben"], watchers_mode="merge")["code"] == 400
    assert _update(tags=["x"], tags_mode="merge")["code"] == 400
    assert env["entity"].patches == []


# -- strict ----------------------------------------------------------------------------------------


def test_strict_status_matches_the_name_exactly_ignoring_case(env):
    _update(status="preparing processing", strict=True)
    assert _only_patch(env) == {"status": 29}


def test_strict_status_accepts_a_numeric_id(env):
    _update(status="30", strict=True)
    assert _only_patch(env) == {"status": 30}


def test_strict_refuses_a_status_that_only_resembles_one(env):
    out = _update(status="Preparing", strict=True)
    assert out["code"] == 404 and "Preparing Processing" in out["error"]
    assert env["entity"].patches == []


@pytest.mark.parametrize(("given", "expected"), [("BEN", 2), ("2", 2), ("Anna A", 1)])
def test_strict_assignee_is_resolved_against_the_members(env, given, expected):
    _update(assign_to=given, strict=True)
    assert _only_patch(env) == {"assigned_to": expected}


def test_strict_refuses_an_ambiguous_assignee(env):
    out = _update(assign_to="Ben B", strict=True)
    assert out["code"] == 409
    assert env["entity"].patches == []


def test_strict_refuses_an_unknown_assignee(env):
    out = _update(assign_to="dora", strict=True)
    assert out["code"] == 404
    assert env["entity"].patches == []


def test_control_without_strict_the_old_resolution_still_runs(env, monkeypatch):
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda slug, etype, q: [30])
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q: [{"id": 2}])
    _update(status="finished", assign_to="Benjamin")
    assert _only_patch(env) == {"status": 30, "assigned_to": 2}


# -- read_back -------------------------------------------------------------------------------------


@pytest.fixture
def stored(env, monkeypatch):
    after = _Entity(
        version=4,
        status=29,
        assigned_to=2,
        assigned_to_extra_info={"id": 2, "username": "ben", "full_name_display": "Ben B"},
        watchers=[1, 3],
        tags=[["voice", None], ["Ops", "#fff"]],
    )
    history = [
        {"id": "new", "created_at": "2026-09-17T09:00:00Z", "comment": "Handed over", "user": {"pk": 5},
         "diff": {"status": [1, 29], "assigned_to": [None, 2]}},
        {"id": "old", "created_at": "2026-09-16T09:00:00Z", "comment": "Handed over", "user": {"pk": 5}, "diff": {}},
    ]
    fetched = []

    def fetch(_project, _norm, _ref):
        fetched.append(1)
        return env["entity"] if len(fetched) == 1 else after

    monkeypatch.setattr(taiga_tools, "fetch_entity", fetch)
    monkeypatch.setattr(taiga_tools, "fetch_history", lambda entity, norm: history)
    monkeypatch.setattr(taiga_tools, "get_status", lambda slug, typ, sid: {"name": "Preparing Processing", "is_closed": False})
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: {"username": f"user{uid}"})
    return fetched


def test_read_back_reports_what_taiga_stored(env, stored):
    out = _update(status="Preparing Processing", assign_to="ben", comment="Handed over", strict=True, read_back=True)
    assert out["state"] == {
        "status": "Preparing Processing",
        "is_closed": False,
        "assigned_to": "ben",
        "watchers": ["anna", "cara"],
        "tags": ["voice", "Ops"],
        "version": 4,
    }
    assert out["history_entry"] == {
        "id": "new",
        "created_at": "2026-09-17T09:00:00Z",
        "user_id": 5,
        "comment": "Handed over",
        "changed": ["assigned_to", "status"],
    }
    assert out["comment_entries"] == 2
    assert len(stored) == 2


def test_without_read_back_nothing_is_read_again(env, stored):
    out = _update(comment="Handed over")
    assert "state" not in out and "history_entry" not in out
    assert len(stored) == 1


def test_compact_write_answer_is_one_line(env):
    raw = update_entity_by_ref_tool.invoke(
        {"project_slug": "p", "entity_ref": 5, "entity_type": "task", "comment": "x", "compact": True}
    )
    assert "\n" not in raw and json.loads(raw)["applied"] == ["comment"]
