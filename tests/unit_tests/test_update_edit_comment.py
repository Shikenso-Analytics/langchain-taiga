"""``update_entity_by_ref_tool(edit_comment_id=...)``: correct a posted comment instead of adding one.

Taiga edits a comment through ``POST /history/<type>/<pk>/edit_comment?id=<entry id>`` with
``{"comment": ...}``. It writes no new history entry, so the reader is not notified twice, and a
correction does not grow longer than what it corrects. The edit goes out before any other write, so
an unknown id or someone else's comment stops the whole call.
"""

import json

import pytest
import requests
from taiga.exceptions import TaigaRestException

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import update_entity_by_ref_tool


class _Entity:
    def __init__(self, requester):
        self.requester = requester
        self.id = 4242
        self.ref = 5
        self.version = 4
        self.watchers = []
        self.tags = []
        self.patches = []

    def patch(self, fields, **kwargs):
        self.patches.append((fields, kwargs))
        return self


class _Requester:
    def __init__(self, calls, fail):
        self.calls = calls
        self.fail = fail
        self.status = 200

    def post(self, uri, payload=None, query=None, files=None, **parameters):
        self.calls.append({"url": uri.format(**parameters), "query": query, "payload": payload})
        if self.fail:
            raise self.fail
        response = requests.Response()
        response.status_code = self.status
        return response


@pytest.fixture
def env(monkeypatch):
    requester = _Requester([], None)
    state = {"entity": _Entity(requester), "calls": requester.calls, "requester": requester}

    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: object())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda _p, _t, _r: state["entity"])
    return state


def _update(**params):
    args = {"project_slug": "p", "entity_ref": 5, "entity_type": "issue", **params}
    return json.loads(update_entity_by_ref_tool.invoke(args))


@pytest.mark.parametrize(
    "entity_type, path", [("issue", "issue"), ("userstory", "userstory"), ("task", "task"), ("epic", "epic")]
)
def test_the_edit_goes_to_the_history_endpoint_of_the_type(env, entity_type, path):
    out = _update(entity_type=entity_type, comment="  new text \r\n", edit_comment_id=" c0ffee-id ")
    assert env["calls"] == [
        {"url": f"/history/{path}/4242/edit_comment", "query": {"id": "c0ffee-id"}, "payload": {"comment": "new text"}}
    ]
    assert out["applied"] == ["edit_comment"]


def test_an_edit_adds_no_comment_to_the_patch(env):
    # Control: without edit_comment_id the same text is a NEW comment in the PATCH.
    _update(comment="new text")
    assert env["entity"].patches == [(["version"], {"comment": "new text"})]
    assert env["calls"] == []

    env["entity"].patches.clear()
    _update(comment="new text", edit_comment_id="c0ffee")
    assert env["entity"].patches == []


def test_an_edit_combines_with_other_fields(env):
    out = _update(comment="new text", edit_comment_id="c0ffee", due_date="2026-12-01")
    assert len(env["calls"]) == 1
    assert env["entity"].patches == [(["version"], {"due_date": "2026-12-01"})]
    assert out["applied"] == ["edit_comment", "due_date"]


def test_a_refused_edit_writes_nothing_else(env):
    env["requester"].fail = TaigaRestException("https://taiga.example.org/api/v1/history", 403, "not the author", "POST")
    out = _update(comment="new text", edit_comment_id="c0ffee", due_date="2026-12-01")
    assert out["code"] == 403
    assert env["entity"].patches == []


def test_an_unknown_epic_stops_the_call_before_the_edit(env, monkeypatch):
    project = type("P", (), {"get_epic_by_ref": lambda self, ref: None})()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    out = _update(entity_type="userstory", comment="new text", edit_comment_id="c0ffee", epic_ref=999)
    assert out["code"] == 404
    assert env["calls"] == []


def test_a_gateway_error_python_taiga_lets_through_still_fails_the_call(env):
    # python-taiga raises only for 400-500; a 503 comes back as a normal response.
    env["requester"].status = 503
    out = _update(comment="new text", edit_comment_id="c0ffee", due_date="2026-12-01")
    assert out["code"] == 500
    assert env["entity"].patches == []


@pytest.mark.parametrize(
    "params", [{"edit_comment_id": "c0ffee"}, {"edit_comment_id": "  ", "comment": "x"}], ids=["no-comment", "blank-id"]
)
def test_bad_edit_input_is_refused_before_any_request(env, params):
    out = _update(**params)
    assert out["code"] == 400
    assert env["calls"] == []
    assert env["entity"].patches == []
