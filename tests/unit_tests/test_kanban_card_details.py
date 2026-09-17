"""``get_kanban_board_tool``: custom attributes and last activity per card (2.20.0).

The download-sourcing sweep needs, for every open story, its custom attributes and when one of
"our" accounts last touched or commented on it. Asked story by story that is three tool calls per
story, ~140 a day. Here the server fetches both per card in parallel, inside the one board call.

These keys cost a request per card, so unlike the cheap extras they come only when a path names
them — ``fields=["columns"]`` does not quietly fan out. A card whose fetch fails twice fails the
whole call: a board with some attributes missing reads as "those stories have none".
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_kanban_board_tool
from tests.unit_tests.test_get_kanban_board_tool import _FakeStatus

API = "https://taiga.test/api/v1"

HISTORY = {
    101: [
        {"created_at": "2026-09-16T10:00:00Z", "user": {"pk": 9}, "comment": "PM note"},
        {"created_at": "2026-09-15T08:00:00Z", "user": {"pk": 51}, "comment": ""},
        {"created_at": "2026-09-10T08:00:00Z", "user": {"pk": 5}, "comment": "asked the PM"},
    ],
    102: [{"created_at": "2026-09-12T08:00:00Z", "user": {"pk": 9}, "comment": "x"}],
}


class _Story:
    def __init__(self, sid, ref, status):
        self.id = sid
        self.ref = ref
        self.subject = f"US {ref}"
        self.status = status
        self.kanban_order = ref
        self.assigned_to = None


class _Project:
    name = "Sourcing"

    def __init__(self):
        self.members = [SimpleNamespace(id=51, username="SourcerBot"), SimpleNamespace(id=5, username="Wahed")]
        self.stories = [_Story(101, 1, 1), _Story(102, 2, 1), _Story(103, 3, 2)]

    def list_user_story_statuses(self):
        return [_FakeStatus(1, "Ongoing", order=1), _FakeStatus(2, "Done", order=2, is_closed=True)]

    def list_user_stories(self, **queryparams):
        if queryparams.get("status__is_closed") == "false":
            return [s for s in self.stories if s.status == 1]
        return list(self.stories)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "TAIGA_API_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "get_taiga_api", lambda token=None: SimpleNamespace(token="tok"))
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "_current_user_id", lambda: 5)


def _routes(respx_mock, history_status=None):
    routes = {}
    for sid in (101, 102, 103):
        routes[("attrs", sid)] = respx_mock.get(f"{API}/userstories/custom-attributes-values/{sid}").mock(
            return_value=httpx.Response(200, json={"attributes_values": {"38": f"M{sid}"}, "version": 1})
        )
        responses = history_status.get(sid) if history_status else None
        route = respx_mock.get(f"{API}/history/userstory/{sid}")
        if responses:
            route.mock(side_effect=[httpx.Response(code, json=HISTORY.get(sid, [])) for code in responses])
        else:
            route.mock(return_value=httpx.Response(200, json=HISTORY.get(sid, [])))
        routes[("history", sid)] = route
    return routes


def _board(**kw):
    return json.loads(get_kanban_board_tool.invoke({"project_slug": "sourcing", "compact": True, **kw}))


def _cards(out):
    return [card for column in out["columns"] for card in column["cards"]]


@pytest.mark.respx(assert_all_called=False)
def test_custom_attributes_are_fetched_per_visible_card(env, respx_mock):
    routes = _routes(respx_mock)
    out = _board(include_closed=False, fields=["columns.cards.ref", "columns.cards.custom_attributes"])
    assert _cards(out) == [
        {"ref": 1, "custom_attributes": {"38": "M101"}},
        {"ref": 2, "custom_attributes": {"38": "M102"}},
    ]
    assert not routes[("attrs", 103)].called
    assert not any(routes[("history", sid)].called for sid in (101, 102, 103))
    request = routes[("attrs", 101)].calls[0].request
    assert request.headers["Authorization"] == "Bearer tok"


@pytest.mark.respx(assert_all_called=False)
def test_last_activity_counts_only_the_named_users(env, respx_mock):
    routes = _routes(respx_mock)
    out = _board(
        include_closed=False,
        activity_users=["51", "5"],
        fields=["columns.cards.ref", "columns.cards.last_activity_at", "columns.cards.last_comment_at"],
    )
    assert _cards(out) == [
        {"ref": 1, "last_activity_at": "2026-09-15T08:00:00Z", "last_comment_at": "2026-09-10T08:00:00Z"},
        {"ref": 2, "last_activity_at": None, "last_comment_at": None},
    ]
    assert out["query"]["activity_users"] == ["51", "5"]
    assert routes[("history", 101)].calls[0].request.headers["x-disable-pagination"] == "True"
    assert not any(routes[("attrs", sid)].called for sid in (101, 102, 103))


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize(("users", "expected"), [(["me"], "2026-09-10T08:00:00Z"), (["sourcerbot"], "2026-09-15T08:00:00Z")])
def test_activity_users_resolve_me_and_usernames(env, respx_mock, users, expected):
    _routes(respx_mock)
    out = _board(include_closed=False, activity_users=users, fields=["columns.cards.last_activity_at"])
    assert _cards(out)[0]["last_activity_at"] == expected


@pytest.mark.respx(assert_all_called=False)
def test_an_unknown_activity_user_is_an_error_before_any_fetch(env, respx_mock):
    routes = _routes(respx_mock)
    out = _board(activity_users=["nobody"], fields=["columns.cards.last_activity_at"])
    assert out["code"] == 404 and "nobody" in out["error"]
    assert not any(route.called for route in routes.values())


def test_activity_fields_need_activity_users(env):
    out = _board(fields=["columns.cards.last_comment_at"])
    assert out["code"] == 400 and "activity_users" in out["error"]


def test_activity_users_need_an_activity_field(env):
    out = _board(activity_users=["me"], fields=["columns.cards.ref"])
    assert out["code"] == 400 and "last_activity_at" in out["error"]


@pytest.mark.respx(assert_all_called=False)
def test_a_whole_card_does_not_fan_out(env, respx_mock):
    routes = _routes(respx_mock)
    out = _board(include_closed=False, fields=["columns"])
    assert "custom_attributes" not in _cards(out)[0]
    assert not any(route.called for route in routes.values())


@pytest.mark.respx(assert_all_called=False)
def test_a_card_that_fails_twice_fails_the_board(env, respx_mock):
    _routes(respx_mock, history_status={102: [500, 502]})
    out = _board(include_closed=False, activity_users=["me"], fields=["columns.cards.last_activity_at"])
    assert out["code"] == 502 and "#2" in out["error"]


@pytest.mark.respx(assert_all_called=False)
def test_control_one_retry_absorbs_a_single_failure(env, respx_mock):
    routes = _routes(respx_mock, history_status={102: [503, 200]})
    out = _board(include_closed=False, activity_users=["me"], fields=["columns.cards.last_activity_at"])
    assert [card["last_activity_at"] for card in _cards(out)] == ["2026-09-10T08:00:00Z", None]
    assert routes[("history", 102)].call_count == 2
