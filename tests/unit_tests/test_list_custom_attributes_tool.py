"""Unit tests for ``list_custom_attributes_tool``.

- ``TestListCustomAttributesUnit`` is the LangChain scaffold.
- Behavioural tests cover the new ``extra`` + ``choices`` fields added
  in 2.8.0: dropdown choice parsing (CRLF normalization, whitespace
  trimming, empty-line drop), non-dropdown null handling, empty
  dropdown configuration, and entity-type dispatch.
"""

import json

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import list_custom_attributes_tool


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.fixture(autouse=True)
def fake_taiga_url(monkeypatch):
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")


class TestListCustomAttributesUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return list_custom_attributes_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"project_slug": "slug", "entity_type": "userstory"}


# ---------------------------------------------------------------------------
# Behavioural test infrastructure.
# ---------------------------------------------------------------------------


class _FakeAttr:
    """Mimics python-taiga's CustomAttribute model for the read path.

    ``extra`` is hydrated at runtime by ``InstanceResource.__init__``
    via ``setattr`` even though it's not in ``allowed_params`` — see
    the implementation comment in ``list_custom_attributes_tool``.
    """

    def __init__(self, aid, name, attr_type, extra=None, description="", order=0):
        self.id = aid
        self.name = name
        self.type = attr_type
        self.extra = extra
        self.description = description
        self.order = order


class _FakeProject:
    """Returns the same list for whichever entity-type call we set."""

    name = "Shikenso Development"

    def __init__(self, attrs_by_type):
        # attrs_by_type: dict like {"userstory": [...], "task": [...]}.
        # Calls touching a key NOT in the dict get [] — used by the
        # dispatch test to confirm only the requested method ran.
        self._attrs_by_type = attrs_by_type

    def list_user_story_attributes(self):
        return list(self._attrs_by_type.get("userstory", []))

    def list_task_attributes(self):
        return list(self._attrs_by_type.get("task", []))

    def list_issue_attributes(self):
        return list(self._attrs_by_type.get("issue", []))

    def list_epic_attributes(self):
        return list(self._attrs_by_type.get("epic", []))


@pytest.fixture
def fake_env(monkeypatch):
    """Install a project whose per-entity-type attribute lists are
    keyed by the entity name. Returns the project so tests can inspect
    call records if needed."""

    def _install(attrs_by_type=None):
        project = _FakeProject(attrs_by_type or {})
        monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
        return project

    return _install


# ---------------------------------------------------------------------------
# Behavioural tests.
# ---------------------------------------------------------------------------


def test_dropdown_attribute_exposes_parsed_choices_and_raw_extra(fake_env):
    fake_env(
        {
            "userstory": [
                _FakeAttr(
                    18,
                    "Task difficulty",
                    "dropdown",
                    extra="easy\nmedium\nhard",
                    description="How long this will take",
                )
            ]
        }
    )
    raw = list_custom_attributes_tool.invoke(
        {"project_slug": "shikenso-development", "entity_type": "userstory"}
    )
    payload = json.loads(raw)
    attrs = payload["custom_attributes"]
    assert len(attrs) == 1
    entry = attrs[0]
    assert entry["id"] == 18
    assert entry["name"] == "Task difficulty"
    assert entry["type"] == "dropdown"
    assert entry["extra"] == "easy\nmedium\nhard"
    assert entry["choices"] == ["easy", "medium", "hard"]


def test_crlf_normalized_and_trailing_newline_stripped(fake_env):
    fake_env(
        {"userstory": [_FakeAttr(1, "X", "dropdown", extra="a\r\nb\n")]}
    )
    raw = list_custom_attributes_tool.invoke(
        {"project_slug": "slug", "entity_type": "userstory"}
    )
    entry = json.loads(raw)["custom_attributes"][0]
    # No phantom empty string from the trailing newline; CRLF collapsed.
    assert entry["choices"] == ["a", "b"]


def test_per_line_whitespace_stripped_empty_lines_dropped(fake_env):
    fake_env(
        {
            "userstory": [
                _FakeAttr(
                    1,
                    "X",
                    "dropdown",
                    extra="  easy (1 day)  \n\nmedium (2-3 days)\n",
                )
            ]
        }
    )
    raw = list_custom_attributes_tool.invoke(
        {"project_slug": "slug", "entity_type": "userstory"}
    )
    entry = json.loads(raw)["custom_attributes"][0]
    assert entry["choices"] == ["easy (1 day)", "medium (2-3 days)"]


def test_non_dropdown_type_has_empty_choices_and_null_extra(fake_env):
    fake_env(
        {"userstory": [_FakeAttr(2, "Notes", "multiline", extra=None)]}
    )
    raw = list_custom_attributes_tool.invoke(
        {"project_slug": "slug", "entity_type": "userstory"}
    )
    entry = json.loads(raw)["custom_attributes"][0]
    assert entry["type"] == "multiline"
    assert entry["extra"] is None
    assert entry["choices"] == []


def test_empty_string_extra_on_dropdown_yields_empty_choices(fake_env):
    """User has the dropdown type set but hasn't configured any
    options yet — extra is the empty string, NOT null."""
    fake_env(
        {"userstory": [_FakeAttr(3, "Priority", "dropdown", extra="")]}
    )
    raw = list_custom_attributes_tool.invoke(
        {"project_slug": "slug", "entity_type": "userstory"}
    )
    entry = json.loads(raw)["custom_attributes"][0]
    assert entry["type"] == "dropdown"
    assert entry["extra"] == ""
    assert entry["choices"] == []


def test_entity_type_dispatch_routes_to_task_attributes(fake_env):
    """Invoking with entity_type='task' must hit
    ``list_task_attributes()``, NOT ``list_user_story_attributes()``."""
    fake_env(
        {
            "userstory": [
                _FakeAttr(99, "Should-not-appear", "dropdown", extra="us-only")
            ],
            "task": [
                _FakeAttr(
                    10,
                    "Task-only attr",
                    "dropdown",
                    extra="t1\nt2",
                )
            ],
        }
    )
    raw = list_custom_attributes_tool.invoke(
        {"project_slug": "slug", "entity_type": "task"}
    )
    payload = json.loads(raw)
    attrs = payload["custom_attributes"]
    assert len(attrs) == 1
    assert attrs[0]["id"] == 10
    assert attrs[0]["choices"] == ["t1", "t2"]
