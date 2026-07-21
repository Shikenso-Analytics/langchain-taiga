import json

import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    _resolve_watcher_ids,
    manage_watchers_by_ref_tool,
)
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """
    Automatically apply a fake OPENAI_API_KEY environment variable
    for each test function. That way, login() won't raise ValueError.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestManageWatchersUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return manage_watchers_by_ref_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {
            "project_slug": "slug",
            "entity_ref": 555,
            "entity_type": "us",
            "watchers": ["alice"],
            "mode": "add",
        }


# ---------------------------------------------------------------------------
# Lightweight fakes so behavioural tests stay focused on the tool's watcher
# set-math and identifier resolution, without touching the Taiga API.
# ---------------------------------------------------------------------------


class _Member:
    def __init__(self, id, username, full_name):
        self.id = id
        self.username = username
        self.full_name = full_name


class _Entity:
    """Minimal stand-in for a python-taiga entity: records the watcher
    list it is updated with so tests can assert the exact wire payload."""

    def __init__(self, watchers=None):
        self.watchers = list(watchers) if watchers is not None else None
        self.updated_with = None

    def update(self, **kwargs):
        self.updated_with = kwargs
        if "watchers" in kwargs:
            self.watchers = kwargs["watchers"]
        return self


_MEMBERS = [
    _Member(1, "alice", "Alice Anderson"),
    _Member(2, "bob", "Bob Brown"),
    _Member(3, "carol", "Carol Clark"),
    # Two distinct members that share a full_name -> ambiguous by name.
    _Member(4, "dave1", "Dave Davis"),
    _Member(5, "dave2", "Dave Davis"),
]


class _Project:
    members = _MEMBERS


def _patch_common(monkeypatch, entity):
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)
    # get_user only feeds the human-readable response; keep it trivial.
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: {"id": uid})


def _invoke(**kwargs):
    return json.loads(manage_watchers_by_ref_tool.invoke(kwargs))


# --- _resolve_watcher_ids -------------------------------------------------


def test_resolve_matches_username_id_and_full_name():
    ids, unresolved, ambiguous = _resolve_watcher_ids(
        _MEMBERS, ["alice", "2", "Carol Clark"]
    )
    assert ids == [1, 2, 3]
    assert unresolved == []
    assert ambiguous == []


def test_resolve_is_case_insensitive_and_dedupes():
    ids, unresolved, ambiguous = _resolve_watcher_ids(_MEMBERS, ["ALICE", "alice", "1"])
    assert ids == [1]
    assert unresolved == []
    assert ambiguous == []


def test_resolve_reports_unresolved_and_ambiguous():
    ids, unresolved, ambiguous = _resolve_watcher_ids(
        _MEMBERS, ["nobody", "Dave Davis"]
    )
    assert ids == []
    assert unresolved == ["nobody"]
    assert ambiguous == ["Dave Davis"]


# --- manage_watchers_by_ref_tool ------------------------------------------


def test_add_merges_into_existing(monkeypatch):
    entity = _Entity(watchers=[1])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=["bob"], mode="add")
    assert entity.updated_with == {"watchers": [1, 2]}
    assert "updated" in out["message"]


def test_add_existing_watcher_is_noop(monkeypatch):
    entity = _Entity(watchers=[1, 2])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=["alice"], mode="add")
    assert entity.updated_with is None  # no PUT issued
    assert "No watcher change" in out["message"]


def test_replace_sets_exact_list(monkeypatch):
    entity = _Entity(watchers=[1, 2])
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=["carol"], mode="replace")
    assert entity.updated_with == {"watchers": [3]}


def test_replace_empty_clears_all(monkeypatch):
    entity = _Entity(watchers=[1, 2])
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=[], mode="replace")
    assert entity.updated_with == {"watchers": []}


def test_remove_drops_given(monkeypatch):
    entity = _Entity(watchers=[1, 2, 3])
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=["bob"], mode="remove")
    assert entity.updated_with == {"watchers": [1, 3]}


def test_remove_absent_is_noop(monkeypatch):
    entity = _Entity(watchers=[1])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=["bob"], mode="remove")
    assert entity.updated_with is None
    assert "No watcher change" in out["message"]


def test_add_empty_is_rejected(monkeypatch):
    entity = _Entity(watchers=[1])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=[], mode="add")
    assert out["code"] == 400
    assert entity.updated_with is None


def test_unknown_watcher_is_rejected(monkeypatch):
    entity = _Entity(watchers=[1])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=["ghost"], mode="add")
    assert out["code"] == 404
    assert out["unresolved"] == ["ghost"]
    assert entity.updated_with is None


def test_ambiguous_watcher_is_rejected(monkeypatch):
    entity = _Entity(watchers=[1])
    _patch_common(monkeypatch, entity)
    out = _invoke(
        project_slug="s", entity_ref=1, entity_type="us", watchers=["Dave Davis"], mode="add"
    )
    assert out["code"] == 404
    assert out["ambiguous"] == ["Dave Davis"]
    assert entity.updated_with is None


def test_invalid_mode_is_rejected(monkeypatch):
    entity = _Entity(watchers=[1])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", watchers=["bob"], mode="toggle")
    assert out["code"] == 400
    assert entity.updated_with is None


def test_unsupported_entity_type_is_rejected(monkeypatch):
    entity = _Entity(watchers=[1])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="banana", watchers=["bob"], mode="add")
    assert out["code"] == 400


def test_entity_not_found_is_rejected(monkeypatch):
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: None)
    out = _invoke(project_slug="s", entity_ref=9, entity_type="us", watchers=["bob"], mode="add")
    assert out["code"] == 404
