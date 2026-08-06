import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools.taiga_tools import (
    _format_userstory_points,
    get_entity_by_ref_tool,
)
from langchain_tests.unit_tests import ToolsUnitTests

@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """
    Automatically apply a fake OPENAI_API_KEY environment variable
    for each test function. That way, login() won't raise ValueError.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestGetEntityUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return get_entity_by_ref_tool

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"project_slug": "slug", "entity_ref": 555, "entity_type": "us"}


# ---------------------------------------------------------------------------
# Behavioural tests for ``_format_userstory_points`` — the helper that
# produces the ``points`` field added to the ``get_entity_by_ref_tool``
# response for user stories. Lightweight fakes keep the test focused on
# the {role_id: point_id} -> {role_name: point_value} translation; the
# tool's other heavy machinery (history, tasks, custom attributes) is
# tested elsewhere via integration paths.
# ---------------------------------------------------------------------------


class _Stub:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _project(roles, points):
    return _Stub(list_roles=lambda: roles, list_points=lambda: points)


def test_format_userstory_points_returns_role_name_to_value(monkeypatch):
    entity = _Stub(points={"19": 103, "20": 101})
    project = _project(
        roles=[_Stub(id=19, name="Developer"), _Stub(id=20, name="UX")],
        points=[
            _Stub(id=100, value=1),
            _Stub(id=101, value=2),
            _Stub(id=103, value=5),
        ],
    )
    assert _format_userstory_points(entity, project) == {
        "Developer": 5,
        "UX": 2,
    }


def test_format_userstory_points_empty_when_unset():
    entity = _Stub(points=None)
    project = _project(
        roles=[_Stub(id=19, name="Developer")],
        points=[_Stub(id=100, value=1)],
    )
    assert _format_userstory_points(entity, project) == {}


def test_format_userstory_points_drops_stale_role_id_and_unestimated():
    """A role-id key that's no longer present on the project (deleted
    role) and a point assignment to the "?" point (value=None) must
    both be silently dropped — consumers see only role-name keys with
    concrete numeric values."""
    entity = _Stub(points={"19": 103, "999": 100, "20": 351})
    project = _project(
        roles=[_Stub(id=19, name="Developer"), _Stub(id=20, name="UX")],
        points=[
            _Stub(id=100, value=1),
            _Stub(id=103, value=5),
            _Stub(id=351, value=None),  # the "?" unestimated point
        ],
    )
    out = _format_userstory_points(entity, project)
    # Only Developer survives:
    #   - role 999 not in project.list_roles() -> dropped
    #   - UX (role 20) points to "?" (value None)        -> dropped
    assert out == {"Developer": 5}


# ---------------------------------------------------------------------------
# Tag shape in the response (v2.14.0)
# ---------------------------------------------------------------------------


class _TagEntity(_Stub):
    """Stand-in whose ``tags`` carry Taiga's real read shape."""

    def to_dict(self):
        return {"tags": self.tags, "subject": self.subject}

    def list_tasks(self):
        return getattr(self, "_tasks", [])


def _tag_env(monkeypatch, entity):
    from langchain_taiga.tools import taiga_tools

    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")
    monkeypatch.setattr(
        taiga_tools, "get_project", lambda slug: _Stub(name="P", slug=slug)
    )
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)
    monkeypatch.setattr(taiga_tools, "get_status", lambda *a, **kw: {"name": "New"})
    monkeypatch.setattr(
        taiga_tools, "get_formatted_custom_attributes", lambda *a, **kw: []
    )
    monkeypatch.setattr(taiga_tools, "fetch_history", lambda *a, **kw: [])
    monkeypatch.setattr(taiga_tools, "list_milestones", lambda slug: [])
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: {"id": uid})
    monkeypatch.setattr(taiga_tools, "_format_userstory_points", lambda *a, **kw: {})


def _us(**kwargs):
    base = dict(
        ref=1,
        subject="s",
        description="d",
        status=1,
        milestone=None,
        assigned_to=None,
        watchers=[],
        tags=[],
    )
    base.update(kwargs)
    return _TagEntity(**base)


def test_tags_are_returned_as_flat_names(monkeypatch):
    """Taiga reads tags back as ``[name, color]`` pairs. The response must
    carry flat names so it feeds straight into manage_tags_by_ref_tool."""
    import json

    entity = _us(tags=[["jobs_manager", None], ["voice", "#845EF7"]])
    _tag_env(monkeypatch, entity)
    payload = json.loads(
        get_entity_by_ref_tool.invoke(
            {"project_slug": "p", "entity_ref": 1, "entity_type": "userstory"}
        )
    )
    assert payload["tags"] == ["jobs_manager", "voice"]


def test_related_task_tags_use_the_same_shape(monkeypatch):
    """Regression: ``related.tasks`` splats ``task.to_dict()``, which hands
    back python-taiga's raw field — without normalizing it the same response
    carries two different shapes under the same key."""
    import json

    task = _TagEntity(ref=9, subject="t", status=1, tags=[["voice", "#845EF7"]])
    entity = _us(tags=[["voice", "#845EF7"]], _tasks=[task])
    _tag_env(monkeypatch, entity)
    payload = json.loads(
        get_entity_by_ref_tool.invoke(
            {"project_slug": "p", "entity_ref": 1, "entity_type": "userstory"}
        )
    )
    assert payload["tags"] == ["voice"]
    assert payload["related"]["tasks"][0]["tags"] == ["voice"]


# ---------------------------------------------------------------------------
# owner (creator) — v2.15.0
# ---------------------------------------------------------------------------


def test_owner_is_returned_from_the_embedded_blob(monkeypatch):
    """Taiga ships ``owner_extra_info`` on every detail payload, so the
    creator resolves without a second request."""
    import json

    entity = _us(
        owner=5,
        owner_extra_info={"id": 5, "username": "Wahed", "full_name_display": "Dr. Wahed Hemati"},
    )
    _tag_env(monkeypatch, entity)
    payload = json.loads(
        get_entity_by_ref_tool.invoke(
            {"project_slug": "p", "entity_ref": 1, "entity_type": "userstory"}
        )
    )
    assert payload["owner"] == {"id": 5, "username": "Wahed", "full_name": "Dr. Wahed Hemati"}


def test_owner_is_none_when_the_entity_has_no_creator(monkeypatch):
    """Possible on very old tickets and on entities whose author's account
    was deleted."""
    import json

    entity = _us(owner=None)
    _tag_env(monkeypatch, entity)
    payload = json.loads(
        get_entity_by_ref_tool.invoke(
            {"project_slug": "p", "entity_ref": 1, "entity_type": "userstory"}
        )
    )
    assert payload["owner"] is None


# ---------------------------------------------------------------------------
# 2.16.0 — include_history
# ---------------------------------------------------------------------------


def test_history_is_included_by_default(monkeypatch):
    import json

    from langchain_taiga.tools import taiga_tools
    from langchain_taiga.tools.taiga_tools import get_entity_by_ref_tool

    entity = _us()
    _tag_env(monkeypatch, entity)
    monkeypatch.setattr(
        taiga_tools, "fetch_history", lambda *a, **kw: [{"id": "h1", "comment": "hi"}]
    )

    out = json.loads(
        get_entity_by_ref_tool.invoke(
            {"project_slug": "p", "entity_ref": 1, "entity_type": "issue"}
        )
    )

    assert out["history"] == [{"id": "h1", "comment": "hi"}]


def test_include_history_false_omits_the_key_and_skips_the_fetch(monkeypatch):
    """History is 84-97% of this payload on real tickets (up to ~190 KB for
    one busy story), so a caller that only needs current state should not
    pay for it — nor for the extra round-trip.

    The key is omitted rather than set to ``[]`` on purpose: an empty
    history is a real answer here, because Taiga writes no history entry
    for creation and a never-edited ticket genuinely has none. Returning
    ``[]`` would let a caller read "not fetched" as "nothing happened"."""
    import json

    from langchain_taiga.tools import taiga_tools
    from langchain_taiga.tools.taiga_tools import get_entity_by_ref_tool

    entity = _us()
    _tag_env(monkeypatch, entity)

    def _never(*a, **kw):
        raise AssertionError("fetch_history must not run when include_history=False")

    monkeypatch.setattr(taiga_tools, "fetch_history", _never)

    out = json.loads(
        get_entity_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 1,
                "entity_type": "issue",
                "include_history": False,
            }
        )
    )

    assert "history" not in out
    # The rest of the entity is unaffected.
    assert out["ref"] == 1 and out["subject"] == "s"
