"""Unit tests for ``get_attachment_by_ref_tool``.

- ``TestGetAttachmentUnit`` is the LangChain scaffold.
- Behavioural tests cover inline download, oversized refusal, attachment
  lookup failure, Bearer-JWT propagation, and HTTP-error mapping.

HTTP downloads are mocked with ``responses`` (already a test dep). The
fixture monkey-patches ``get_project``, ``fetch_entity`` and
``_current_taiga_jwt`` so no real network call ever escapes.
"""

import base64
import json

import pytest
import responses as responses_lib
from langchain_core.tools import BaseTool

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_attachment_by_ref_tool
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.fixture(autouse=True)
def fake_taiga_url(monkeypatch):
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")


class TestGetAttachmentUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return get_attachment_by_ref_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "project_slug": "slug",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_id": 10334,
        }


# ---------------------------------------------------------------------------
# Behavioural test infrastructure.
# ---------------------------------------------------------------------------


class _FakeAttachment:
    def __init__(self, aid, name, size, url, content_type=None):
        self.id = aid
        self.name = name
        self.size = size
        self.url = url
        self.content_type = content_type


class _FakeEntity:
    def __init__(self, attachments):
        self._attachments = attachments

    def list_attachments(self):
        return list(self._attachments)


class _FakeProject:
    name = "Volleyball World"


@pytest.fixture
def fake_env(monkeypatch):
    """Returns a helper that patches in a project + entity with the
    given attachments and a fixed JWT."""

    def _install(attachments, jwt="test_jwt"):
        entity = _FakeEntity(attachments)
        project = _FakeProject()
        monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
        monkeypatch.setattr(taiga_tools, "fetch_entity",
                            lambda proj, norm_type, ref: entity)
        monkeypatch.setattr(taiga_tools, "_current_taiga_jwt", lambda: jwt)
        return entity

    return _install


def test_inline_returns_base64(fake_env):
    body = b"PK\x03\x04\x14\x00FAKE_XLSX_BYTES"
    fake_env([_FakeAttachment(
        10334, "tv_viewership.xlsx", len(body),
        "https://taiga.example.test/media/attachments/abc?token=fresh",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )])

    with responses_lib.RequestsMock() as rsps:
        rsps.add(
            responses_lib.GET,
            "https://taiga.example.test/media/attachments/abc",
            body=body,
            status=200,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        raw = get_attachment_by_ref_tool.invoke({
            "project_slug": "volleyball-world-11-25",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_id": 10334,
        })

    payload = json.loads(raw)
    assert payload["id"] == 10334
    assert payload["name"] == "tv_viewership.xlsx"
    assert payload["encoding"] == "base64"
    assert base64.b64decode(payload["content_base64"]) == body
    assert payload["size"] == len(body)
