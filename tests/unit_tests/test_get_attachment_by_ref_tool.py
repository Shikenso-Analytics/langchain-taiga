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
        monkeypatch.setattr(taiga_tools, "fetch_entity", lambda proj, norm_type, ref: entity)
        monkeypatch.setattr(taiga_tools, "_current_taiga_jwt", lambda: jwt)
        return entity

    return _install


def test_inline_returns_base64(fake_env):
    body = b"PK\x03\x04\x14\x00FAKE_XLSX_BYTES"
    fake_env(
        [
            _FakeAttachment(
                10334,
                "tv_viewership.xlsx",
                len(body),
                "https://taiga.example.test/media/attachments/abc?token=fresh",
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        ]
    )

    with responses_lib.RequestsMock() as rsps:
        rsps.add(
            responses_lib.GET,
            "https://taiga.example.test/media/attachments/abc",
            body=body,
            status=200,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        raw = get_attachment_by_ref_tool.invoke(
            {
                "project_slug": "volleyball-world-11-25",
                "entity_ref": 7398,
                "entity_type": "issue",
                "attachment_id": 10334,
            }
        )

    payload = json.loads(raw)
    assert payload["id"] == 10334
    assert payload["name"] == "tv_viewership.xlsx"
    assert payload["encoding"] == "base64"
    assert base64.b64decode(payload["content_base64"]) == body
    assert payload["size"] == len(body)


def test_oversized_pre_check_returns_413_without_http_call(fake_env, monkeypatch):
    # Lie about size — pre-check sees 20 MB, refuses, no HTTP call made.
    fake_env(
        [
            _FakeAttachment(
                10334,
                "huge.bin",
                20 * 1024 * 1024,
                "https://taiga.example.test/media/attachments/big?token=fresh",
            )
        ]
    )

    # If the tool tries any HTTP, ``responses`` with no registered mock
    # raises a ConnectionError — which we want to NOT see.
    with responses_lib.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        raw = get_attachment_by_ref_tool.invoke(
            {
                "project_slug": "any",
                "entity_ref": 7398,
                "entity_type": "issue",
                "attachment_id": 10334,
            }
        )
        # responses mock must have seen ZERO calls.
        assert len(rsps.calls) == 0

    payload = json.loads(raw)
    assert payload["code"] == 413
    assert payload["size"] == 20 * 1024 * 1024
    assert payload["max_bytes"] == 10 * 1024 * 1024
    assert "content_base64" not in payload


def test_oversized_streaming_aborts(fake_env, monkeypatch):
    # Tool sees size=10 (will pass pre-check), but the actual body is
    # 11 MB — mid-stream cap must trigger 413.
    fake_env(
        [
            _FakeAttachment(
                10334,
                "lies_about_size.bin",
                10,
                "https://taiga.example.test/media/attachments/lies?token=fresh",
            )
        ]
    )
    huge_body = b"\x00" * (11 * 1024 * 1024)

    with responses_lib.RequestsMock() as rsps:
        rsps.add(
            responses_lib.GET,
            "https://taiga.example.test/media/attachments/lies",
            body=huge_body,
            status=200,
        )
        raw = get_attachment_by_ref_tool.invoke(
            {
                "project_slug": "any",
                "entity_ref": 7398,
                "entity_type": "issue",
                "attachment_id": 10334,
            }
        )

    payload = json.loads(raw)
    assert payload["code"] == 413
    assert payload["max_bytes"] == 10 * 1024 * 1024
    assert "content_base64" not in payload


def test_attachment_id_not_found_returns_404(fake_env):
    fake_env(
        [
            _FakeAttachment(
                10334,
                "exists.txt",
                5,
                "https://taiga.example.test/media/attachments/x?token=fresh",
            )
        ]
    )
    raw = get_attachment_by_ref_tool.invoke(
        {
            "project_slug": "any",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_id": 99999,  # not present
        }
    )
    payload = json.loads(raw)
    assert payload["code"] == 404
    assert "99999" in payload["error"]


def test_bearer_jwt_attached_to_download(fake_env):
    fake_env(
        [
            _FakeAttachment(
                10334,
                "tiny.bin",
                3,
                "https://taiga.example.test/media/attachments/tiny?token=fresh",
            )
        ],
        jwt="alice_jwt_xyz",
    )

    with responses_lib.RequestsMock() as rsps:
        rsps.add(
            responses_lib.GET,
            "https://taiga.example.test/media/attachments/tiny",
            body=b"abc",
            status=200,
        )
        raw = get_attachment_by_ref_tool.invoke(
            {
                "project_slug": "any",
                "entity_ref": 7398,
                "entity_type": "issue",
                "attachment_id": 10334,
            }
        )
        # Inspect the outbound call's Authorization header.
        assert len(rsps.calls) == 1
        sent_auth = rsps.calls[0].request.headers.get("Authorization")
        assert sent_auth == "Bearer alice_jwt_xyz"

    payload = json.loads(raw)
    assert base64.b64decode(payload["content_base64"]) == b"abc"


def test_http_error_returns_502(fake_env):
    fake_env(
        [
            _FakeAttachment(
                10334,
                "forbidden.bin",
                10,
                "https://taiga.example.test/media/attachments/forbidden?token=stale",
            )
        ]
    )

    with responses_lib.RequestsMock() as rsps:
        rsps.add(
            responses_lib.GET,
            "https://taiga.example.test/media/attachments/forbidden",
            body="forbidden",
            status=403,
        )
        raw = get_attachment_by_ref_tool.invoke(
            {
                "project_slug": "any",
                "entity_ref": 7398,
                "entity_type": "issue",
                "attachment_id": 10334,
            }
        )

    payload = json.loads(raw)
    assert payload["code"] == 502
    assert "403" in payload["error"]
