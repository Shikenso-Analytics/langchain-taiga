"""Unit tests for ``list_attachments_by_ref_tool``.

- ``TestListAttachmentsUnit`` is the standard ``ToolsUnitTests`` scaffold
  that every other tool in this package has — schema parses, tool is
  callable, etc.
- The behavioural tests below monkey-patch ``get_project`` and
  ``fetch_entity`` to return hand-rolled fakes; ``entity.list_attachments()``
  returns ``_FakeAttachment`` instances with the contract that the tool
  reads (id, name, size, url, ...).
"""

import json

import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import list_attachments_by_ref_tool
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """``OPENAI_API_KEY`` is read at module import time by the small_llm
    helper inside ``taiga_tools.py`` — same pattern as every other tool
    test in this directory."""
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.fixture(autouse=True)
def fake_taiga_url(monkeypatch):
    """``TAIGA_URL`` is captured at import time as a module attribute, so
    ``monkeypatch.setenv`` is a no-op. Set the attribute directly — see
    AGENTS.md ``Test`` section."""
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")


class TestListAttachmentsUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return list_attachments_by_ref_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "project_slug": "slug",
            "entity_ref": 7398,
            "entity_type": "issue",
        }


# ---------------------------------------------------------------------------
# Behavioural tests with hand-rolled stubs. We avoid python-taiga's Resource
# hierarchy because all the tool cares about is: project.list_attachments()
# returns a list of objects with id/name/size/url attributes.
# ---------------------------------------------------------------------------


class _FakeAttachment:
    def __init__(self, aid, name, size, url, content_type=None, description="",
                 owner=None, created_date="2026-04-24T08:15:00+00:00",
                 modified_date="2026-04-24T08:15:00+00:00"):
        self.id = aid
        self.name = name
        self.size = size
        self.url = url
        self.content_type = content_type
        self.description = description
        self.owner = owner
        self.created_date = created_date
        self.modified_date = modified_date


class _FakeEntity:
    def __init__(self, attachments):
        self._attachments = attachments

    def list_attachments(self):
        return list(self._attachments)


class _FakeProject:
    name = "Volleyball World"


@pytest.fixture
def fake_env_two_attachments(monkeypatch):
    entity = _FakeEntity([
        _FakeAttachment(
            10334, "tv_viewership_20260424.xlsx", 23847,
            "https://taiga.example.test/media/attachments/abc?token=fresh1",
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        _FakeAttachment(
            10335, "screenshot.png", 4096,
            "https://taiga.example.test/media/attachments/def?token=fresh2",
            content_type="image/png",
            description="dashboard view",
        ),
    ])
    project = _FakeProject()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    monkeypatch.setattr(taiga_tools, "fetch_entity",
                        lambda proj, norm_type, ref: entity)
    return entity


def test_returns_attachments_with_fresh_urls(fake_env_two_attachments):
    raw = list_attachments_by_ref_tool.invoke({
        "project_slug": "volleyball-world-11-25",
        "entity_ref": 7398,
        "entity_type": "issue",
    })
    payload = json.loads(raw)
    assert payload["project"] == "Volleyball World"
    assert payload["type"] == "issue"
    assert payload["ref"] == 7398
    assert payload["count"] == 2
    assert payload["url"] == (
        "https://taiga.example.test/project/volleyball-world-11-25/issue/7398"
    )
    names = [a["name"] for a in payload["attachments"]]
    assert names == ["tv_viewership_20260424.xlsx", "screenshot.png"]
    first = payload["attachments"][0]
    assert first["id"] == 10334
    assert first["size"] == 23847
    assert first["download_url"].endswith("?token=fresh1")
    assert first["content_type"].startswith("application/vnd.openxml")
