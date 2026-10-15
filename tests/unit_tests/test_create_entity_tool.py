import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools.taiga_tools import create_entity_tool
from langchain_tests.unit_tests import ToolsUnitTests

@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """
    Automatically apply a fake OPENAI_API_KEY environment variable
    for each test function. That way, login() won't raise ValueError.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestCreateEntitiyUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return create_entity_tool

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
        return {"project_slug": "slug",
                       "entity_type": "us",
                       "subject": "subject",
                       "status": "new",
                       "description": "desc",
                       "parent_ref": 5,
                       "assign_to": "user",
                       "due_date": "2022-01-01",
                       "tags": ["tag1", "tag2"]}


# ---------------------------------------------------------------------------
# Tag normalization on create (v2.14.0)
# ---------------------------------------------------------------------------


def test_create_normalizes_tags_before_writing(monkeypatch):
    """Taiga creates unknown tags implicitly, so a raw pass-through mints
    '  voice ' and 'voice' as two permanent project tags. Same invariant as
    manage_tags_by_ref_tool: strip, drop blanks, de-duplicate."""
    import json

    from langchain_taiga.tools import taiga_tools
    from langchain_taiga.tools.taiga_tools import create_entity_tool

    captured = {}

    class _Created:
        ref = 42
        subject = "s"
        id = 1

    class _Project:
        def add_user_story(self, **kwargs):
            captured.update(kwargs)
            return _Created()

    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda **kw: [1])
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")

    json.loads(
        create_entity_tool.invoke(
            {
                "project_slug": "p",
                "entity_type": "us",
                "subject": "s",
                "status": "New",
                "description": "d",
                "tags": ["  voice  ", "voice", "", "k8s"],
            }
        )
    )
    assert captured["tags"] == ["voice", "k8s"]


def test_create_with_tags_invalidates_the_cached_registry(monkeypatch):
    """Creation is the other write path that can register a project tag.
    Left cached, the next manage_tags call re-reports the tag as new and
    writes a differently-cased duplicate for want of the canonical spelling."""
    import json

    from langchain_taiga.tools import taiga_tools
    from langchain_taiga.tools.taiga_tools import (
        _user_scoped_key,
        create_entity_tool,
        list_all_tags_cache,
    )

    class _Created:
        ref = 42
        subject = "s"
        id = 1

    class _Project:
        def add_user_story(self, **kwargs):
            return _Created()

    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda **kw: [1])
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.org")

    key = _user_scoped_key("p")

    def _create(tags):
        list_all_tags_cache[key] = ["voice"]
        json.loads(
            create_entity_tool.invoke(
                {
                    "project_slug": "p",
                    "entity_type": "us",
                    "subject": "s",
                    "status": "New",
                    "tags": tags,
                }
            )
        )
        return key in list_all_tags_cache

    assert _create(["k8s"]) is False
    # No tags written -> no registry change -> no reason to pay for a refetch.
    assert _create([]) is True
    list_all_tags_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Custom attributes on create (v2.23.0)
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402

from langchain_taiga.tools import taiga_tools as _tt  # noqa: E402
from tests.unit_tests import test_custom_attributes_by_name as _attrs  # noqa: E402


class _CreatedStory:
    endpoint = "userstories"
    id = 900
    ref = 42
    subject = "Export button live, by 30.11."

    def __init__(self, fail=None):
        self.requester = _attrs._Requester(fail)

    def get_attributes(self):
        return {"attributes_values": {}, "version": 1}


@pytest.fixture
def create_env(monkeypatch):
    project = _attrs._Project()
    state = {"created": [], "story": _CreatedStory()}

    def add_user_story(**kwargs):
        state["created"].append(kwargs)
        return state["story"]

    project.add_user_story = add_user_story
    _tt.custom_attr_definitions_cache.clear()
    monkeypatch.setattr(_tt, "get_project", lambda slug: project)
    monkeypatch.setattr(_tt, "find_status_ids", lambda **kw: [1])
    monkeypatch.setattr(_tt, "TAIGA_URL", "https://taiga.test")
    yield state
    _tt.custom_attr_definitions_cache.clear()


def _create(**extra):
    args = {
        "project_slug": "team-backend",
        "entity_type": "userstory",
        "subject": "Export button live, by 30.11.",
        "status": "Later",
        **extra,
    }
    return _json.loads(create_entity_tool.invoke(args))


def test_custom_attributes_are_checked_before_the_create(create_env):
    out = _create(custom_attributes={"Next team": "Frontend"})
    assert out["code"] == 400 and "Backend" in out["error"]
    assert create_env["created"] == []


def test_custom_attributes_are_written_after_the_create(create_env):
    out = _create(custom_attributes={"next team": "backend"})
    assert out["created"] is True and out["ref"] == 42
    assert out["custom_attributes"] == {"Next team": "Backend"}
    assert create_env["story"].requester.patches == [
        ("/userstories/custom-attributes-values/900", {"attributes_values": {"91": "Backend"}, "version": 1})
    ]


def test_a_failed_attribute_write_still_reports_the_created_entity(create_env):
    from taiga.exceptions import TaigaRestException

    create_env["story"] = _CreatedStory(fail=TaigaRestException("u", 500, "", "PATCH"))
    out = _create(custom_attributes={"Escalation": "Needs a review slot"})
    assert out["created"] is True and out["ref"] == 42
    assert "custom_attributes_error" in out
    assert "custom_attributes" not in out


def test_control_without_custom_attributes_nothing_extra_is_read(create_env):
    out = _create()
    assert out["created"] is True
    assert "custom_attributes" not in out
    assert create_env["story"].requester.patches == []
