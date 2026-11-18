"""``create_attachment_upload_by_ref_tool(content=...)``: the file's text rides in
the tool call, for a client that cannot reach ``upload_url`` (a cloud sandbox
whose network blocks this host) and has no chat drop zone (an unattended run).

Taiga is the shared fake seam, as in the other attach tests.
"""

from __future__ import annotations

import json

import pytest

from langchain_taiga import upload_tickets
from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import create_attachment_upload_by_ref_tool

from .test_attachment_upload_ticket import _FakeEntity, _FakeProject


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")
    # Inline content needs no upload URL, so it works without the HTTP server.
    monkeypatch.delenv("TAIGA_MCP_BASE_URL", raising=False)
    monkeypatch.setattr(upload_tickets, "ATTACH_WAIT_SECONDS", 0.01)


@pytest.fixture
def entity(monkeypatch):
    e = _FakeEntity()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _FakeProject())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda proj, t, ref: e)
    return e


def _attach(**args):
    base = {"project_slug": "p", "entity_ref": 7, "entity_type": "task"}
    return json.loads(create_attachment_upload_by_ref_tool.invoke({**base, **args}))


def test_content_is_attached_as_utf8_under_the_filenames_basename(entity):
    text = "# Pré-check\n\n| # | ✓/✗ |\n| --- | --- |\n"
    out = _attach(filename="out/pre-check.md", content=text, description="d")
    assert out["added"] is True and out["attachments"]["name"] == "pre-check.md"
    assert entity.attach_calls[0][:3] == ("pre-check.md", text.encode("utf-8"), "d")


@pytest.mark.parametrize(
    "extra",
    [{"uploaded_file": "a.md"}, {"url": "https://files.example.test/a.md"}],
    ids=["with-uploaded_file", "with-url"],
)
def test_content_needs_filename_and_no_other_source(entity, extra):
    out = _attach(content="x", **extra)
    assert out["code"] == 400 and "filename" in out["error"]
    assert entity.attach_calls == []


def test_an_unusable_filename_is_a_400_for_content_too(entity):
    out = _attach(filename="..", content="x")
    assert out["code"] == 400
    assert entity.attach_calls == []


def test_text_that_utf8_cannot_encode_is_a_400(entity):
    # A lone surrogate is legal JSON ("\ud800") but no UTF-8 text.
    out = _attach(filename="a.md", content="\ud800")
    assert out["code"] == 400 and "UTF-8" in out["error"]
    assert entity.attach_calls == []


def test_content_over_the_upload_cap_is_a_413_counted_in_utf8_bytes(
    entity, monkeypatch
):
    # stdio and direct LangChain calls have no 4 MiB transport limit.
    monkeypatch.setattr(upload_tickets, "MAX_UPLOAD_BYTES", 4)
    out = _attach(filename="a.md", content="äää")  # 3 characters, 6 bytes
    assert out["code"] == 413 and out["max_bytes"] == 4
    assert entity.attach_calls == []
    # Control: exactly at the cap is attached.
    assert _attach(filename="a.md", content="ää")["added"] is True


def test_a_busy_attach_slot_is_a_503_for_content_too(entity):
    upload_tickets.ATTACH_SLOTS.acquire()
    try:
        assert _attach(filename="a.md", content="x")["code"] == 503
    finally:
        upload_tickets.ATTACH_SLOTS.release()
    assert entity.attach_calls == []


def test_a_failed_attach_reports_and_frees_the_slot(entity):
    entity.attach_should_raise = RuntimeError("boom")
    out = _attach(filename="a.md", content="x")
    assert out["code"] == 500 and "boom" in out["error"]
    # Control: a held slot would make this a 503.
    entity.attach_should_raise = None
    assert _attach(filename="a.md", content="x")["added"] is True
