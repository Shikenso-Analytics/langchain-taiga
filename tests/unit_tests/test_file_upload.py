"""In-chat file uploads: ``langchain_taiga.file_upload`` + the ``uploaded_file``
branch of ``create_attachment_upload_by_ref_tool``.

The drop zone (fastmcp ``FileUpload``, MCP Apps) posts the bytes to the
app-only ``store_files`` tool; the model only ever sees previews and passes the
upload's NAME to the attach tool, which reads the bytes server-side.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastmcp import FastMCP
from fastmcp.server.providers.addressing import hash_tool

from langchain_taiga import file_upload, upload_tickets
from langchain_taiga.mcp import make_mcp
from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import create_attachment_upload_by_ref_tool

from .test_attachment_upload_ticket import _FakeEntity, _FakeProject

MODEL_FACING = {"file_manager", "list_files", "read_file"}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _upload(name: str, data: bytes, type_: str = "application/octet-stream"):
    return {"name": name, "size": len(data), "type": type_, "data": _b64(data)}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")
    file_upload.PROVIDER._store.clear()
    yield
    file_upload.PROVIDER._store.clear()


@pytest.fixture
def as_user(monkeypatch):
    """Switch the verified caller: ``as_user(7)`` acts as Taiga user 7."""

    def _set(user_id):
        tok = MagicMock()
        tok.claims = {"user_id": user_id}
        monkeypatch.setattr(taiga_tools, "get_access_token", lambda: tok)

    _set(1)
    return _set


@pytest.fixture
def fake_taiga(monkeypatch):
    entity = _FakeEntity()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _FakeProject())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda proj, t, ref: entity)
    return entity


def _attach(uploaded_file, **overrides):
    args = {
        "project_slug": "my-project",
        "entity_ref": 1234,
        "entity_type": "issue",
        "uploaded_file": uploaded_file,
    }
    args.update(overrides)
    return json.loads(create_attachment_upload_by_ref_tool.invoke(args))


# --- provider ---------------------------------------------------------------


def test_caps_match_the_memory_budget():
    p = file_upload.PROVIDER
    assert p._max_file_size == upload_tickets.MAX_UPLOAD_BYTES == 25 * 1024 * 1024
    assert p._max_total_size == 50 * 1024 * 1024
    assert p._file_ttl_seconds == 900


def test_total_cap_rejects_a_third_full_size_file(as_user):
    p = file_upload.PROVIDER
    big = b"x" * p._max_file_size
    p.on_store([_upload("a.bin", big), _upload("b.bin", big)], None)
    with pytest.raises(ValueError, match="total storage size"):
        p.on_store([_upload("c.bin", b"y")], None)


def test_uploads_are_scoped_per_taiga_user(as_user):
    p = file_upload.PROVIDER
    p.on_store([_upload("secret.zip", b"PK..")], None)
    as_user(2)
    assert p.on_list(None) == []
    with pytest.raises(ValueError, match="not found"):
        p.on_read("secret.zip", None)


def test_a_token_without_user_id_is_refused(monkeypatch):
    tok = MagicMock()
    tok.claims = {}
    monkeypatch.setattr(taiga_tools, "get_access_token", lambda: tok)
    with pytest.raises(PermissionError):
        file_upload.PROVIDER.on_store([_upload("a.txt", b"hi")], None)


@pytest.mark.parametrize(
    "name,type_", [("notes.txt", "text/plain"), ("video.mp4", "video/mp4")]
)
def test_read_file_returns_a_preview_never_the_whole_file(as_user, name, type_):
    p = file_upload.PROVIDER
    p.on_store([_upload(name, b"a" * (10 * 1024 * 1024), type_)], None)
    out = p.on_read(name, None)
    preview = out.get("content", out.get("content_base64"))
    assert len(preview) <= 203 and preview.endswith("...")
    assert out["size"] == 10 * 1024 * 1024


def test_read_file_shows_a_small_text_file_whole(as_user):
    p = file_upload.PROVIDER
    p.on_store([_upload("hi.md", b"# hello", "text/markdown")], None)
    assert p.on_read("hi.md", None)["content"] == "# hello"


def test_mounted_provider_lists_only_the_model_facing_tools():
    """fastmcp 4 lists the app-only receiver; exact equality so a new upstream
    tool fails here instead of reaching the model unreviewed."""
    mcp = FastMCP("t")
    file_upload.mount(mcp)
    assert {t.name for t in _run(mcp.list_tools())} == MODEL_FACING


def test_make_mcp_mounts_the_provider_next_to_the_taiga_tools():
    plain = FastMCP("t")
    taiga_tools._register_mcp_tools(plain)
    taiga_names = {t.name for t in _run(plain.list_tools())}
    names = {t.name for t in _run(make_mcp().list_tools())}
    assert names == taiga_names | MODEL_FACING


def test_make_mcp_works_without_the_apps_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_taiga.file_upload", None)
    names = {t.name for t in _run(make_mcp().list_tools())}
    assert not names & MODEL_FACING
    assert "create_attachment_upload_by_ref_tool" in names


def test_the_drop_zone_reaches_the_unlisted_receiver_by_its_hashed_name(as_user):
    """The UI calls ``<hash>_store_files``. Unlisted is not unreachable: the
    hash is public, so the caps are what bound a caller, not the listing."""
    mcp = make_mcp()
    files = [_upload("n.txt", b"hi", "text/plain")]
    assert "store_files" not in {t.name for t in _run(mcp.list_tools())}
    _run(
        mcp.call_tool(
            f"{hash_tool('Files', 'store_files')}_store_files", {"files": files}
        )
    )
    result = _run(mcp.call_tool("read_file", {"name": "n.txt"}))
    assert result.structured_content["content"] == "hi"


# --- create_attachment_upload_by_ref_tool(uploaded_file=...) ----------------


def test_uploaded_file_is_attached_and_then_dropped(as_user, fake_taiga, monkeypatch):
    monkeypatch.delenv("TAIGA_MCP_BASE_URL", raising=False)  # no curl server needed
    file_upload.PROVIDER.on_store([_upload("clip.mp4", b"\x00\x01video")], None)
    out = _attach("clip.mp4", description="from chat")
    assert out["added"] is True and out["attachments"]["name"] == "clip.mp4"
    (basename, content, description, _tid) = fake_taiga.attach_calls[0]
    assert (basename, content, description) == (
        "clip.mp4",
        b"\x00\x01video",
        "from chat",
    )
    assert file_upload.PROVIDER.on_list(None) == []


def test_a_failed_attach_keeps_the_upload_for_a_retry(as_user, fake_taiga):
    fake_taiga.attach_should_raise = RuntimeError("taiga down")
    file_upload.PROVIDER.on_store([_upload("a.zip", b"PK")], None)
    out = _attach("a.zip")
    assert out["code"] == 500 and "taiga down" in out["error"]
    assert [f["name"] for f in file_upload.PROVIDER.on_list(None)] == ["a.zip"]


def test_unknown_upload_is_a_404_naming_the_available_ones(as_user, fake_taiga):
    file_upload.PROVIDER.on_store([_upload("a.zip", b"PK")], None)
    out = _attach("b.zip")
    assert out["code"] == 404 and "a.zip" in out["error"]
    assert fake_taiga.attach_calls == []


def test_another_users_upload_cannot_be_attached(as_user, fake_taiga):
    file_upload.PROVIDER.on_store([_upload("a.zip", b"PK")], None)
    as_user(2)
    assert _attach("a.zip")["code"] == 404
    assert fake_taiga.attach_calls == []


def test_path_components_in_the_upload_name_are_stripped(as_user, fake_taiga):
    file_upload.PROVIDER.on_store([_upload("..\\..\\evil.txt", b"x")], None)
    _attach("..\\..\\evil.txt")
    assert fake_taiga.attach_calls[0][0] == "evil.txt"


@pytest.mark.parametrize(
    "extra", [{"filename": "./a.txt"}, {"uploaded_file": ""}], ids=["both", "neither"]
)
def test_filename_and_uploaded_file_are_exclusive(as_user, fake_taiga, extra):
    file_upload.PROVIDER.on_store([_upload("a.txt", b"x")], None)
    out = _attach(extra.pop("uploaded_file", "a.txt"), **extra)
    assert out["code"] == 400 and "exactly one" in out["error"]
    assert fake_taiga.attach_calls == []


def test_without_the_apps_extra_uploaded_file_is_refused(
    as_user, fake_taiga, monkeypatch
):
    monkeypatch.setitem(sys.modules, "langchain_taiga.file_upload", None)
    out = _attach("a.txt")
    assert out["code"] == 501
    assert fake_taiga.attach_calls == []


def test_attaches_from_the_store_run_one_at_a_time(as_user, fake_taiga):
    """Each attach peaks at ~2x the file in RSS (requests' multipart encode);
    on top of the curl path's own budget only one may run at a time."""
    p = file_upload.PROVIDER
    p.on_store([_upload("a.bin", b"a"), _upload("b.bin", b"b")], None)
    running, peak, lock = [0], [0], threading.Lock()
    original = fake_taiga.attach

    def slow_attach(file_path, description=""):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return original(file_path, description=description)

    fake_taiga.attach = slow_attach
    threads = [
        threading.Thread(target=lambda n=n: _attach(n)) for n in ("a.bin", "b.bin")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(fake_taiga.attach_calls) == 2
    assert peak[0] == 1
