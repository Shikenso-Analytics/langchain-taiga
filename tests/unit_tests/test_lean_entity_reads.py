"""``get_entity_by_ref_tool``: fields, compact and history filters (2.19.0).

A script that needs a story's subject, its tasks' refs and its own comments used to receive the
whole ticket — description, every task object, each watcher's full user record and up to ~190 KB
of history — and paid a Taiga request per watcher on top. These tests pin the three things the
new parameters promise: the payload holds what was asked for, the parts nobody asked for are not
fetched at all, and a filtered history still says how much it left out.
"""

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_entity_by_ref_tool


class _Stub:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Entity(_Stub):
    def to_dict(self):
        return {"subject": self.subject, "tags": self.tags}

    def list_tasks(self):
        return getattr(self, "_tasks", [])


HISTORY = [
    {"id": "h3", "created_at": "2026-09-17T08:00:00.000Z", "comment": "", "user": {"pk": 9, "username": "Anna"}},
    {"id": "h2", "created_at": "2026-09-16T10:00:00.000Z", "comment": "ping", "user": {"pk": 5, "username": "wahed"}},
    {"id": "h1", "created_at": "2026-09-01T10:00:00.000Z", "comment": "old", "user": {"pk": 5, "username": "wahed"}},
]


def _story(**kwargs):
    base = dict(
        id=11,
        ref=1,
        subject="Stage 3",
        description="long text",
        status=1,
        milestone=42,
        assigned_to=None,
        watchers=[3, 4],
        tags=[["voice", "#845EF7"]],
        owner=5,
        owner_extra_info={"id": 5, "username": "wahed", "full_name_display": "W"},
        _tasks=[_Entity(ref=70, subject="t1", status=2, tags=[])],
    )
    base.update(kwargs)
    return _Entity(**base)


@pytest.fixture
def calls(monkeypatch):
    """The fetches the tool makes, recorded, with the heavy ones counted."""
    seen = {"history": 0, "users": [], "custom": 0, "milestones": 0, "points": 0, "tasks": 0, "me": 0}
    story = _story()
    original_list_tasks = story.list_tasks

    def list_tasks():
        seen["tasks"] += 1
        return original_list_tasks()

    story.list_tasks = list_tasks

    def history(*_a, **_kw):
        seen["history"] += 1
        return [dict(entry) for entry in HISTORY]

    def user(uid):
        seen["users"].append(uid)
        return {"id": uid, "username": f"user{uid}", "email": "secret@example.org"}

    def custom(*_a, **_kw):
        seen["custom"] += 1
        return [{"id": 1, "name": "Event end", "value": "2026-09-20"}]

    def milestones(_slug):
        seen["milestones"] += 1
        return []

    def points(*_a, **_kw):
        seen["points"] += 1
        return {}

    def me():
        seen["me"] += 1
        return 5

    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(name="P", slug=slug))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: story)
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: {"name": "In progress"})
    monkeypatch.setattr(taiga_tools, "get_formatted_custom_attributes", custom)
    monkeypatch.setattr(taiga_tools, "fetch_history", history)
    monkeypatch.setattr(taiga_tools, "list_milestones", milestones)
    monkeypatch.setattr(taiga_tools, "get_user", user)
    monkeypatch.setattr(taiga_tools, "_format_userstory_points", points)
    monkeypatch.setattr(taiga_tools, "_current_user_id", me)
    return seen


def _get(**params):
    args = {"project_slug": "p", "entity_ref": 1, "entity_type": "userstory", **params}
    return json.loads(get_entity_by_ref_tool.invoke(args))


# -- fields ----------------------------------------------------------------------------------------


def test_fields_keep_only_the_requested_paths(calls):
    out = _get(fields=["subject", "related.tasks.ref"])
    assert {key: out[key] for key in ("subject", "related")} == {"subject": "Stage 3", "related": {"tasks": [{"ref": 70}]}}
    assert set(out) == {"subject", "related", "query"}


def test_the_answer_says_which_question_it_answers(calls):
    out = _get(fields=["subject"], compact=True)
    assert out["query"] == {
        "project_slug": "p",
        "entity_ref": 1,
        "entity_type": "us",
        "fields": ["subject"],
        "include_history": True,
        "history_since": None,
        "history_user": None,
        "history_comments_only": False,
        "history_limit": None,
    }


def test_without_new_parameters_there_is_no_query_echo(calls):
    assert "query" not in _get()


def test_parts_nobody_asked_for_are_not_fetched(calls):
    _get(fields=["subject", "tags"])
    assert calls["history"] == 0
    assert calls["users"] == []
    assert calls["custom"] == 0
    assert calls["milestones"] == 0
    assert calls["points"] == 0
    assert calls["tasks"] == 0


def test_control_without_fields_everything_is_fetched(calls):
    _get()
    assert calls["history"] == 1
    assert sorted(calls["users"]) == [3, 4]
    assert calls["custom"] == calls["milestones"] == calls["points"] == calls["tasks"] == 1


def test_watchers_can_be_cut_down_to_usernames(calls):
    out = _get(fields=["watchers.username"])
    assert out["watchers"] == [{"username": "user3"}, {"username": "user4"}]


def test_an_unknown_field_is_an_error_naming_the_valid_ones(calls):
    out = _get(fields=["subjekt"])
    assert out["code"] == 400
    assert "subjekt" in out["error"] and "subject" in out["error"]


def test_a_field_of_another_entity_type_is_unknown_here(calls):
    out = _get(entity_type="task", fields=["points"])
    assert out["code"] == 400 and "points" in out["error"]


def test_a_blank_field_path_is_an_error(calls):
    out = _get(fields=["related..ref"])
    assert out["code"] == 400


def test_asking_for_history_while_excluding_it_is_an_error(calls):
    out = _get(fields=["history"], include_history=False)
    assert out["code"] == 400
    assert calls["history"] == 0


def test_compact_output_is_single_line_json(calls):
    raw = get_entity_by_ref_tool.invoke(
        {"project_slug": "p", "entity_ref": 1, "entity_type": "userstory", "fields": ["subject"], "compact": True}
    )
    assert "\n" not in raw and ": " not in raw
    assert json.loads(raw)["subject"] == "Stage 3"


def test_compact_keeps_a_requested_null_assignee(calls):
    assert _get(fields=["assigned_to"], compact=True)["assigned_to"] is None


# -- history filters -------------------------------------------------------------------------------


def test_history_comments_only_drops_entries_without_a_comment(calls):
    out = _get(fields=["history.id"], history_comments_only=True)
    assert out["history"] == [{"id": "h2"}, {"id": "h1"}]
    assert (out["history_total"], out["history_returned"]) == (3, 2)


def test_history_since_keeps_entries_at_or_after_the_moment(calls):
    out = _get(fields=["history.id"], history_since="2026-09-16")
    assert [entry["id"] for entry in out["history"]] == ["h3", "h2"]


def test_history_user_me_resolves_the_caller_once(calls):
    out = _get(fields=["history.id"], history_user="me")
    assert [entry["id"] for entry in out["history"]] == ["h2", "h1"]
    assert calls["me"] == 1


def test_history_user_matches_a_username_case_insensitively(calls):
    out = _get(fields=["history.id"], history_user="anna")
    assert [entry["id"] for entry in out["history"]] == ["h3"]


def test_history_user_matches_a_numeric_id(calls):
    out = _get(fields=["history.id"], history_user="5")
    assert [entry["id"] for entry in out["history"]] == ["h2", "h1"]
    assert calls["me"] == 0


def test_history_limit_keeps_the_newest_after_filtering(calls):
    out = _get(fields=["history.id"], history_user="me", history_limit=1)
    assert out["history"] == [{"id": "h2"}]
    assert (out["history_total"], out["history_returned"]) == (3, 1)


def test_filters_combine(calls):
    out = _get(fields=["history.id"], history_user="me", history_comments_only=True, history_since="2026-09-10T00:00:00Z")
    assert out["history"] == [{"id": "h2"}]


def test_the_counts_come_along_without_being_asked_for(calls):
    out = _get(fields=["history.comment"])
    assert (out["history_total"], out["history_returned"]) == (3, 3)


def test_an_unparseable_since_is_an_error(calls):
    out = _get(fields=["history.id"], history_since="yesterday-ish")
    assert out["code"] == 400
    assert calls["history"] == 0


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_is_an_error(calls, limit):
    assert _get(fields=["history.id"], history_limit=limit)["code"] == 400


def test_a_history_filter_with_history_projected_away_is_an_error(calls):
    out = _get(fields=["subject"], history_comments_only=True)
    assert out["code"] == 400
    assert calls["history"] == 0


def test_a_history_filter_with_history_excluded_is_an_error(calls):
    assert _get(include_history=False, history_user="me")["code"] == 400


def test_control_unfiltered_history_is_unchanged_and_uncounted(calls):
    out = _get()
    assert [entry["id"] for entry in out["history"]] == ["h3", "h2", "h1"]
    assert "history_total" not in out


def test_the_callers_own_id_is_asked_of_taiga_once_per_user_scope(monkeypatch):
    requests = []

    class _Api:
        def me(self):
            requests.append(1)
            return _Stub(id=5)

    monkeypatch.setattr(taiga_tools, "_current_taiga_jwt", lambda: None)
    monkeypatch.setattr(taiga_tools, "get_taiga_api", lambda token=None: _Api())
    taiga_tools.current_user_id_cache.clear()
    try:
        assert [taiga_tools._current_user_id() for _ in range(3)] == [5, 5, 5]
    finally:
        taiga_tools.current_user_id_cache.clear()
    assert requests == [1]


def test_a_failed_me_lookup_is_an_error_answer_not_a_raise(calls, monkeypatch):
    def broken():
        raise RuntimeError("token expired")

    monkeypatch.setattr(taiga_tools, "_current_user_id", broken)
    out = _get(fields=["history.id"], history_user="me")
    assert out["code"] == 500 and "token expired" in out["error"]
    assert calls["history"] == 0


def test_related_task_statuses_are_resolved_only_when_asked_for(calls, monkeypatch):
    lookups = []
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: lookups.append(a) or {"name": "New"})
    assert _get(fields=["related.tasks.ref"])["related"] == {"tasks": [{"ref": 70}]}
    assert lookups == []
    assert _get(fields=["related.tasks.status"])["related"] == {"tasks": [{"status": "New"}]}
    assert lookups == [("p", "task", 2)]


def test_epic_story_statuses_are_resolved_only_when_asked_for(monkeypatch):
    lookups = []
    epic = _Entity(
        id=3, ref=9, subject="E", description="", status=1, assigned_to=None, watchers=[], tags=[],
        owner=5, owner_extra_info={"id": 5, "username": "w"}, color="#fff", is_closed=False,
    )
    epic.list_user_stories = lambda: [_Stub(ref=12, subject="s", status=4)]
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(name="P", slug=slug))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: epic)
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: lookups.append(a) or {"name": "Ready"})
    args = {"project_slug": "p", "entity_ref": 9, "entity_type": "epic", "include_history": False}
    out = json.loads(get_entity_by_ref_tool.invoke({**args, "fields": ["related.user_stories.ref"]}))
    assert out["related"] == {"user_stories": [{"ref": 12}]} and lookups == []
    out = json.loads(get_entity_by_ref_tool.invoke({**args, "fields": ["related.user_stories.status"]}))
    assert out["related"] == {"user_stories": [{"status": "Ready"}]} and lookups == [("p", "us", 4)]
