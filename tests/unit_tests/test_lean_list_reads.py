"""fields/compact on the member list, the custom-attribute read and the search (2.19.0).

Same contract as ``get_entity_by_ref_tool``: requested paths only, unknown keys refused, and the
keys that say whether an answer is complete (``count``/``truncated``) survive every projection.
"""

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    get_custom_attributes_tool,
    list_project_members_tool,
    search_entities_tool,
)
from tests.unit_tests.test_list_project_members_tool import fake_project  # noqa: F401 - fixture
from tests.unit_tests.test_search_entities_tool import _patch_llm, fake_search_env  # noqa: F401 - fixture


# -- list_project_members_tool ---------------------------------------------------------------------


def _members(**kw):
    return json.loads(list_project_members_tool.invoke({"project_slug": "p", **kw}))


def test_member_fields_cut_every_entry(fake_project):
    assert _members(fields=["username", "role"]) == [
        {"username": "alice", "role": "Product Owner"},
        {"username": "bob", "role": "Developer"},
    ]


def test_member_email_needs_include_email(fake_project):
    out = _members(fields=["username", "email"])
    assert out["code"] == 400 and "include_email" in out["error"]


def test_control_member_email_with_include_email(fake_project):
    assert _members(fields=["email"], include_email=True) == [
        {"email": "alice@example.com"},
        {"email": "bob@example.com"},
    ]


def test_an_unknown_member_field_is_an_error(fake_project):
    out = _members(fields=["name"])
    assert out["code"] == 400 and "username" in out["error"]


def test_members_compact_is_one_line(fake_project):
    raw = list_project_members_tool.invoke({"project_slug": "p", "fields": ["user_id"], "compact": True})
    assert raw == '[{"user_id":1},{"user_id":2}]'


# -- get_custom_attributes_tool --------------------------------------------------------------------


class _Stub:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def attribute_env(monkeypatch):
    entity = _Stub(subject="Stage 3", get_attributes=lambda: {"attributes_values": {"12": "2026-09-20"}, "version": 3})
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(name="P"))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)


def _attributes(**kw):
    args = {"project_slug": "p", "entity_ref": 4, "entity_type": "userstory", **kw}
    return json.loads(get_custom_attributes_tool.invoke(args))


def test_attribute_fields_keep_the_values_and_echo_the_question(attribute_env):
    out = _attributes(fields=["attributes_values"])
    assert out == {
        "attributes_values": {"12": "2026-09-20"},
        "query": {"project_slug": "p", "entity_ref": 4, "entity_type": "us", "fields": ["attributes_values"]},
    }


def test_control_attributes_without_fields_are_unchanged(attribute_env):
    out = _attributes()
    assert set(out) == {"project", "entity_type", "ref", "subject", "url", "attributes_values", "version"}
    assert out["entity_type"] == "userstory"


def test_an_unknown_attribute_field_is_an_error(attribute_env):
    assert _attributes(fields=["values"])["code"] == 400


def test_attributes_compact_is_one_line(attribute_env):
    raw = get_custom_attributes_tool.invoke(
        {"project_slug": "p", "entity_ref": 4, "entity_type": "userstory", "fields": ["version"], "compact": True}
    )
    assert "\n" not in raw and json.loads(raw)["version"] == 3


# -- search_entities_tool --------------------------------------------------------------------------


def _search(monkeypatch, **kw):
    _patch_llm(monkeypatch, {})
    args = {"project_slug": "p", "query": "all", "entity_type": "issue", **kw}
    return json.loads(search_entities_tool.invoke(args))


def test_search_fields_cut_the_matches_and_keep_the_completeness_keys(fake_search_env, monkeypatch):
    out = _search(monkeypatch, fields=["matches.ref"], max_results=2)
    assert out == {"matches": [{"ref": 1}, {"ref": 2}], "count": 2, "max_results": 2, "truncated": True}


def test_an_unknown_match_field_is_an_error(fake_search_env, monkeypatch):
    out = _search(monkeypatch, fields=["matches.reference"])
    assert out["code"] == 400 and "subject" in out["error"]


def test_custom_attributes_need_include_custom_attributes(fake_search_env, monkeypatch):
    out = _search(monkeypatch, fields=["matches.custom_attributes"])
    assert out["code"] == 400 and "include_custom_attributes" in out["error"]


def test_search_compact_is_one_line(fake_search_env, monkeypatch):
    _patch_llm(monkeypatch, {})
    raw = search_entities_tool.invoke(
        {"project_slug": "p", "query": "all", "entity_type": "issue", "fields": ["matches.subject"], "compact": True}
    )
    assert "\n" not in raw
    assert json.loads(raw)["matches"] == [{"subject": "alpha"}, {"subject": "beta"}, {"subject": "gamma"}]


def test_control_search_without_fields_is_unchanged(fake_search_env, monkeypatch):
    out = _search(monkeypatch)
    assert set(out) == {"matches", "count", "max_results", "truncated"}
    assert "url" in out["matches"][0]


# -- compact errors (review #33) -------------------------------------------------------------------


def test_search_errors_honour_compact(monkeypatch):
    raw = search_entities_tool.invoke({"project_slug": "p", "query": "x", "entity_type": "bogus", "compact": True})
    assert "\n" not in raw and json.loads(raw)["code"] == 400
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: None)
    raw = search_entities_tool.invoke({"project_slug": "p", "query": "x", "entity_type": "issue", "compact": True})
    assert "\n" not in raw and json.loads(raw)["code"] == 404


def test_control_search_errors_stay_indented_without_compact(monkeypatch):
    raw = search_entities_tool.invoke({"project_slug": "p", "query": "x", "entity_type": "bogus"})
    assert raw == json.dumps({"error": "Invalid entity type 'bogus'", "code": 400}, indent=2)


def test_attribute_and_member_errors_honour_compact(monkeypatch):
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: None)
    raw = get_custom_attributes_tool.invoke(
        {"project_slug": "p", "entity_ref": 4, "entity_type": "userstory", "compact": True}
    )
    assert "\n" not in raw and json.loads(raw)["code"] == 404
    raw = list_project_members_tool.invoke({"project_slug": "p", "compact": True})
    assert "\n" not in raw and json.loads(raw)["code"] == 404


def test_attributes_are_not_read_when_only_metadata_is_asked_for(monkeypatch):
    reads = []
    entity = _Stub(subject="Stage 3", get_attributes=lambda: reads.append(1) or {"attributes_values": {}, "version": 3})
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Stub(name="P"))
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)
    assert _attributes(fields=["subject"])["subject"] == "Stage 3"
    assert reads == []
    assert _attributes(fields=["version"])["version"] == 3
    assert reads == [1]
