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
