"""A write Taiga refuses (403 on a board you may only read) keeps Taiga's code (2.23.0).

Before, every refused write came back as ``code: 500`` — it read like a bug in the bridge,
not like "you may not write here". The message was already Taiga's own ``_error_message``.
"""

import json

import pytest
from taiga.exceptions import TaigaRestException

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    _taiga_error_code,
    add_comment_by_ref_tool,
    update_entity_by_ref_tool,
    update_wiki_page_tool,
)
from tests.unit_tests.test_update_single_patch import _Entity, _Stub

DENIED = TaigaRestException(
    "https://taiga.test/api/v1/tasks/11",
    403,
    '{"_error_message": "You do not have permission to perform this action."}',
    "PATCH",
)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (DENIED, 403),
        (TaigaRestException("u", 404, "", "GET"), 404),
        (TaigaRestException("u", 502, "", "GET"), 500),
        # python-taiga wraps every requests.RequestException (refused, DNS, timeout) as this 400.
        (TaigaRestException("u", 400, "Network error!", "PATCH"), 500),
        (TaigaRestException("u", 400, '{"_error_message": "Invalid due date"}', "PATCH"), 400),
        (ValueError("boom"), 500),
    ],
)
def test_only_taiga_4xx_keeps_its_code(exc, expected):
    assert _taiga_error_code(exc) == expected


class _RefusingEntity(_Entity):
    def patch(self, fields, **kwargs):
        raise DENIED

    def add_comment(self, comment):
        raise DENIED


@pytest.fixture
def refusing(monkeypatch):
    entity = _RefusingEntity()
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(id=6, members=[]))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)
    return entity


def test_update_keeps_403(refusing):
    out = json.loads(
        update_entity_by_ref_tool.invoke({"project_slug": "p", "entity_ref": 5, "entity_type": "task", "comment": "x"})
    )
    assert out["code"] == 403
    assert "permission" in out["error"]


def test_comment_keeps_403(refusing):
    out = json.loads(
        add_comment_by_ref_tool.invoke({"project_slug": "p", "entity_ref": 5, "entity_type": "task", "comment": "x"})
    )
    assert out["code"] == 403


def test_wiki_update_keeps_403(monkeypatch):
    class _Page:
        slug = "home"

        def update(self):
            raise DENIED

    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(id=6))
    monkeypatch.setattr(taiga_tools, "_get_wiki_page", lambda project, slug: _Page())
    out = json.loads(
        update_wiki_page_tool.invoke({"project_slug": "p", "wiki_slug": "home", "content": "x", "version": 1})
    )
    assert out["code"] == 403


def test_control_a_server_error_stays_500(refusing, monkeypatch):
    def broken(fields, **kwargs):
        raise TaigaRestException("u", 502, "", "PATCH")

    monkeypatch.setattr(refusing, "patch", broken)
    out = json.loads(
        update_entity_by_ref_tool.invoke({"project_slug": "p", "entity_ref": 5, "entity_type": "task", "comment": "x"})
    )
    assert out["code"] == 500
