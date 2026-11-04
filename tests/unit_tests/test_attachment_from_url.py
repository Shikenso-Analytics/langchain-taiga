"""``create_attachment_upload_by_ref_tool(url=...)``: the server downloads a
URL and attaches it (the former ``add_attachment_by_ref_tool``).

Outbound HTTP is mocked with ``responses``; Taiga is the shared fake seam.
"""

from __future__ import annotations

import json

import pytest
import responses as responses_lib

from langchain_taiga import upload_tickets
from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import create_attachment_upload_by_ref_tool

from .test_attachment_upload_ticket import _FakeEntity, _FakeProject


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")


@pytest.fixture
def entity(monkeypatch):
    e = _FakeEntity()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _FakeProject())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda proj, t, ref: e)
    return e


def _attach(**args):
    base = {"project_slug": "p", "entity_ref": 7, "entity_type": "issue"}
    return json.loads(create_attachment_upload_by_ref_tool.invoke({**base, **args}))


@pytest.fixture
def served():
    """Serve one URL (or more) for the test; every one must be fetched."""
    with responses_lib.RequestsMock() as rsps:

        def _add(
            url, body=b"%PDF-1.7 data", status=200, content_type="application/pdf"
        ):
            rsps.add(
                responses_lib.GET,
                url,
                body=body,
                status=status,
                content_type=content_type,
            )
            return rsps

        yield _add


def test_attaches_the_download_under_the_urls_file_name(entity, served):
    served("https://files.example.test/docs/report%20q3.pdf")
    out = _attach(
        url="https://files.example.test/docs/report%20q3.pdf?sig=abc", description="d"
    )
    assert out["added"] is True and out["attachments"]["name"] == "report q3.pdf"
    (name, content, description, _tid) = entity.attach_calls[0]
    assert (name, content, description) == ("report q3.pdf", b"%PDF-1.7 data", "d")


def test_a_name_without_extension_gets_one_from_the_content_type(entity, served):
    served("https://files.example.test/download")
    _attach(url="https://files.example.test/download?id=5")
    assert entity.attach_calls[0][0] == "download.pdf"


def test_a_url_without_a_file_name_is_called_attachment(entity, served):
    served("https://files.example.test/", content_type="application/x-unknown-kind")
    _attach(url="https://files.example.test/")
    assert entity.attach_calls[0][0] == "attachment"


def test_path_components_in_the_url_name_are_stripped(entity, served):
    served(
        "https://files.example.test/a/..%2F..%2Fetc%2Fpasswd", content_type="x/unknown"
    )
    _attach(url="https://files.example.test/a/..%2F..%2Fetc%2Fpasswd")
    assert entity.attach_calls[0][0] == "passwd"


def test_a_download_over_the_cap_is_cut_off_and_not_attached(
    entity, served, monkeypatch
):
    monkeypatch.setattr(upload_tickets, "MAX_UPLOAD_BYTES", 10)
    served("https://files.example.test/big.bin", body=b"x" * 11)
    out = _attach(url="https://files.example.test/big.bin")
    assert out["code"] == 413 and out["max_bytes"] == 10
    assert entity.attach_calls == []


def test_a_failed_download_is_a_502_and_attaches_nothing(entity, served):
    served("https://files.example.test/gone.pdf", status=404)
    out = _attach(url="https://files.example.test/gone.pdf")
    assert out["code"] == 502 and "404" in out["error"]
    assert entity.attach_calls == []


@pytest.mark.parametrize(
    "bad", ["file:///etc/passwd", "ftp://files.example.test/a.pdf", "a.pdf"]
)
def test_only_http_urls_are_fetched(entity, bad):
    out = _attach(url=bad)
    assert out["code"] == 400
    assert entity.attach_calls == []


@pytest.mark.parametrize(
    "extra",
    [{"filename": "./a.pdf"}, {"uploaded_file": "a.pdf"}],
    ids=["with-filename", "with-uploaded_file"],
)
def test_url_is_exclusive_with_the_other_sources(entity, extra):
    out = _attach(url="https://files.example.test/a.pdf", **extra)
    assert out["code"] == 400 and "exactly one" in out["error"]
    assert entity.attach_calls == []


def test_a_declared_size_over_the_cap_is_refused_before_reading(
    entity, served, monkeypatch
):
    monkeypatch.setattr(upload_tickets, "MAX_UPLOAD_BYTES", 10)
    rsps = served("https://files.example.test/big.bin", body=b"x" * 5)
    rsps.replace(
        responses_lib.GET,
        "https://files.example.test/big.bin",
        body=b"x" * 5,
        headers={"Content-Length": "999"},
        auto_calculate_content_length=False,
    )
    out = _attach(url="https://files.example.test/big.bin")
    assert out["code"] == 413
    assert entity.attach_calls == []


def test_a_taiga_attachment_link_is_fetched_in_cluster(entity, served, monkeypatch):
    monkeypatch.setenv("TAIGA_MEDIA_URL", "http://gateway.internal")
    rsps = served("http://gateway.internal/media/attachments/a/b/shot.png")
    _attach(url="https://taiga.example.test/media/attachments/a/b/shot.png?token=t")
    assert rsps.calls[0].request.url.startswith("http://gateway.internal/media/")
    assert entity.attach_calls[0][0] == "shot.png"


# --- limits shared with the other attachment paths (5.0.1) ------------------


def test_a_malformed_url_is_a_400_not_an_exception(entity):
    out = _attach(url="https://[::1/report.pdf")
    assert out["code"] == 400
    assert entity.attach_calls == []


def test_url_attaches_run_one_at_a_time(entity, served):
    import threading
    import time

    served("https://files.example.test/a.pdf")
    served("https://files.example.test/b.pdf")
    running, peak, lock = [0], [0], threading.Lock()
    original = entity.attach

    def slow_attach(file_path, description=""):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return original(file_path, description=description)

    entity.attach = slow_attach
    threads = [
        threading.Thread(target=lambda u=u: _attach(url=u))
        for u in ("https://files.example.test/a.pdf", "https://files.example.test/b.pdf")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(entity.attach_calls) == 2 and peak[0] == 1


def test_a_busy_attach_slot_is_a_503_for_urls_too(entity, served, monkeypatch):
    monkeypatch.setattr(upload_tickets, "ATTACH_WAIT_SECONDS", 0.01)
    served("https://files.example.test/a.pdf")
    upload_tickets.ATTACH_SLOTS.acquire()
    try:
        assert _attach(url="https://files.example.test/a.pdf")["code"] == 503
    finally:
        upload_tickets.ATTACH_SLOTS.release()
    assert entity.attach_calls == []
