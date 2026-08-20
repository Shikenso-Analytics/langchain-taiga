"""Unit tests for the out-of-band attachment upload path.

Three layers, tested separately:

- ``langchain_taiga.upload_tickets`` — issue / consume / prune semantics.
- ``create_attachment_upload_by_ref_tool`` — validates the target and mints
  the ticket.
- ``POST /mcp/upload/{token}`` — streams the body and hands it to Taiga.

The route is exercised through a real Starlette app built the same way
``remote_server`` builds production's, so the test covers the actual
registration and path params rather than calling a handler directly.

No real HTTP: the only outbound I/O is python-taiga's ``Entity.attach``,
replaced on the ``taiga_tools`` seam like the sibling attachment tests do.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time

import httpx
import pytest
from starlette.testclient import TestClient

from langchain_taiga import remote_server, upload_tickets
from langchain_taiga.mcp import make_mcp
from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    attach_file_for_ticket,
    create_attachment_upload_by_ref_tool,
)

BASE_URL = "https://taiga.example.test/mcp"


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.fixture(autouse=True)
def clean_tickets():
    upload_tickets.clear()
    yield
    upload_tickets.clear()


@pytest.fixture(autouse=True)
def fake_taiga_url(monkeypatch):
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")
    monkeypatch.setenv("TAIGA_MCP_BASE_URL", BASE_URL)


# ---------------------------------------------------------------------------
# Fakes (same seam as test_add_attachment_inline_by_ref_tool).
# ---------------------------------------------------------------------------


class _FakeAttachment:
    def __init__(self, aid, name, size, description, url):
        self.id, self.name, self.size = aid, name, size
        self.description, self.url = description, url

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "description": self.description,
            "url": self.url,
        }


class _FakeEntity:
    """Records (basename, bytes, description, thread_ident) per attach."""

    def __init__(self):
        self.attach_calls = []
        self.attach_should_raise = None

    def attach(self, file_path, description=""):
        if self.attach_should_raise is not None:
            raise self.attach_should_raise
        basename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            content = f.read()
        self.attach_calls.append(
            (basename, content, description, threading.get_ident())
        )
        return _FakeAttachment(
            aid=42,
            name=basename,
            size=len(content),
            description=description,
            url=f"https://taiga.example.test/media/attachments/{basename}",
        )


class _FakeProject:
    name = "Shikenso Development"


class _FakeApi:
    def __init__(self, project):
        self.projects = self
        self._project = project

    def get_by_slug(self, slug):
        return self._project


@pytest.fixture
def fake_taiga(monkeypatch):
    """Install a project + entity on both lookup paths.

    The tool goes through ``get_project``; ``attach_file_for_ticket``
    deliberately bypasses it (no ambient request context) and builds a
    client via ``get_taiga_api``, so both seams need faking.
    """

    def _install(entity_present=True, attach_raises=None):
        entity = _FakeEntity() if entity_present else None
        if entity is not None:
            entity.attach_should_raise = attach_raises
        project = _FakeProject()
        monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
        monkeypatch.setattr(
            taiga_tools, "fetch_entity", lambda proj, norm_type, ref: entity
        )
        monkeypatch.setattr(
            taiga_tools, "get_taiga_api", lambda token=None: _FakeApi(project)
        )
        monkeypatch.setattr(taiga_tools, "_current_taiga_jwt", lambda: "jwt-abc")
        return entity

    return _install


@pytest.fixture
def client():
    """A real app carrying the production custom routes."""
    mcp = make_mcp()
    remote_server._attach_custom_routes(
        mcp,
        provider=object(),  # only touched by the /oauth/login handlers
        taiga_url="https://taiga.example.test",
        base_url=BASE_URL,
    )
    with TestClient(mcp.http_app(path="/mcp")) as c:
        yield c


def _issue(**overrides):
    kwargs = dict(
        taiga_jwt="jwt-abc",
        project_slug="shikenso-development",
        entity_type="issue",
        entity_ref=7398,
        filename="report.csv",
        description="",
    )
    kwargs.update(overrides)
    return upload_tickets.issue(**kwargs)


# ---------------------------------------------------------------------------
# Ticket store.
# ---------------------------------------------------------------------------


def test_consume_returns_the_ticket_once_and_then_nothing():
    ticket = _issue()
    assert upload_tickets.consume(ticket.token) is ticket
    # Single-use: the second presentation finds nothing, so a replayed or
    # retried upload cannot re-run against the same target.
    assert upload_tickets.consume(ticket.token) is None


def test_consume_rejects_an_expired_ticket():
    live = _issue()
    expired = _issue(filename="dead.txt", ttl_seconds=-1)
    assert upload_tickets.consume(expired.token) is None
    assert upload_tickets.consume(live.token) is live


def test_consume_rejects_an_unknown_token():
    assert upload_tickets.consume("never-issued") is None


def test_prune_drops_only_expired_tickets():
    live = _issue()
    _issue(filename="dead.txt", ttl_seconds=-1)
    assert upload_tickets.prune() == 1
    assert upload_tickets.consume(live.token) is live


def test_ttl_default_is_read_at_call_time(monkeypatch):
    monkeypatch.setattr(upload_tickets, "UPLOAD_TICKET_TTL_SECONDS", 5.0)
    ticket = _issue()
    assert ticket.expires_at - time.time() == pytest.approx(5.0, abs=1.0)


# ---------------------------------------------------------------------------
# The tool.
# ---------------------------------------------------------------------------


def test_tool_returns_upload_url_and_curl(fake_taiga):
    fake_taiga()
    out = json.loads(
        create_attachment_upload_by_ref_tool.invoke(
            {
                "project_slug": "shikenso-development",
                "entity_ref": 7398,
                "entity_type": "issue",
                "filename": "./out/report.csv",
            }
        )
    )
    assert out["upload_url"].startswith(f"{BASE_URL}/upload/")
    # The display name is the basename, but the curl source path keeps what
    # the caller passed so the command is runnable as-is.
    assert out["filename"] == "report.csv"
    assert "--data-binary @./out/report.csv" in out["curl"]
    assert out["upload_url"] in out["curl"]
    assert out["ref"] == 7398 and out["type"] == "issue"
    assert out["max_bytes"] == upload_tickets.MAX_UPLOAD_BYTES


def test_tool_rejects_invalid_entity_type(fake_taiga):
    fake_taiga()
    out = json.loads(
        create_attachment_upload_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 1,
                "entity_type": "sprint",
                "filename": "a.txt",
            }
        )
    )
    assert out["code"] == 400


@pytest.mark.parametrize("bad", ["", "..", ".", "foo/..", "/", "\\", "a\x00b", "\x00"])
def test_tool_rejects_unusable_filenames(fake_taiga, bad):
    fake_taiga()
    out = json.loads(
        create_attachment_upload_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 1,
                "entity_type": "issue",
                "filename": bad,
            }
        )
    )
    assert out["code"] == 400


def test_tool_strips_path_components_posix_and_windows(fake_taiga):
    fake_taiga()
    for given, expected in [
        ("../../etc/passwd", "passwd"),
        ("C:\\tmp\\evil.txt", "evil.txt"),
    ]:
        out = json.loads(
            create_attachment_upload_by_ref_tool.invoke(
                {
                    "project_slug": "p",
                    "entity_ref": 1,
                    "entity_type": "issue",
                    "filename": given,
                }
            )
        )
        assert out["filename"] == expected


def test_tool_404s_on_missing_entity_and_mints_no_ticket(fake_taiga):
    fake_taiga(entity_present=False)
    out = json.loads(
        create_attachment_upload_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 999,
                "entity_type": "issue",
                "filename": "a.txt",
            }
        )
    )
    assert out["code"] == 404
    # A ticket for a non-existent entity would only fail later, after the
    # bytes are already on the wire.
    assert upload_tickets.prune(now=time.time() + 10**6) == 0


def test_tool_404s_on_missing_project(fake_taiga, monkeypatch):
    fake_taiga()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: None)
    out = json.loads(
        create_attachment_upload_by_ref_tool.invoke(
            {
                "project_slug": "nope",
                "entity_ref": 1,
                "entity_type": "issue",
                "filename": "a.txt",
            }
        )
    )
    assert out["code"] == 404


def test_tool_refuses_without_base_url(fake_taiga, monkeypatch):
    fake_taiga()
    monkeypatch.delenv("TAIGA_MCP_BASE_URL", raising=False)
    out = json.loads(
        create_attachment_upload_by_ref_tool.invoke(
            {
                "project_slug": "p",
                "entity_ref": 1,
                "entity_type": "issue",
                "filename": "a.txt",
            }
        )
    )
    assert out["code"] == 500
    assert "TAIGA_MCP_BASE_URL" in out["error"]


# ---------------------------------------------------------------------------
# The upload route.
# ---------------------------------------------------------------------------


def test_upload_attaches_the_body_under_the_ticket_filename(client, fake_taiga):
    entity = fake_taiga()
    ticket = _issue(description="RCA dump")
    body = b"col_a,col_b\n1,2\n"

    resp = client.post(f"/mcp/upload/{ticket.token}", content=body)

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["added"] is True
    assert payload["ref"] == 7398
    assert payload["url"].endswith("/project/shikenso-development/issue/7398")
    basename, content, description, _ = entity.attach_calls[0]
    assert (basename, content, description) == ("report.csv", body, "RCA dump")


def test_upload_runs_taiga_io_off_the_event_loop(client, fake_taiga):
    """The blocking attach must not execute on the loop thread.

    Running it inline stalls FastMCP's event loop, ``/mcp/health`` stops
    answering and kubelet SIGKILLs the pod. ``_TOOLS_NEEDING_ASYNC_OFFLOAD``
    covers registered tools but not a custom route, so this route offloads
    by hand and that has to stay true.
    """
    fake_taiga()
    ticket = _issue()
    observed = {}

    def _record(ticket_arg, file_path):
        # Inside the coroutine there IS a running loop in this thread; on a
        # worker thread there is not. Comparing thread idents against the
        # test's own would pass either way, because TestClient already runs
        # the app on a separate portal thread.
        try:
            asyncio.get_running_loop()
            observed["on_loop"] = True
        except RuntimeError:
            observed["on_loop"] = False
        return {"added": True, "project": "p", "type": "issue", "ref": 1, "url": "u"}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(remote_server, "attach_file_for_ticket", _record)
    try:
        resp = client.post(f"/mcp/upload/{ticket.token}", content=b"x")
    finally:
        monkeypatch.undo()

    assert resp.status_code == 200
    assert observed["on_loop"] is False


def test_upload_404s_on_unknown_token(client, fake_taiga):
    fake_taiga()
    resp = client.post("/mcp/upload/never-issued", content=b"x")
    assert resp.status_code == 404


def test_upload_404s_on_expired_token(client, fake_taiga):
    fake_taiga()
    ticket = _issue(ttl_seconds=-1)
    resp = client.post(f"/mcp/upload/{ticket.token}", content=b"x")
    assert resp.status_code == 404


def test_upload_is_single_use(client, fake_taiga):
    entity = fake_taiga()
    ticket = _issue()
    assert client.post(f"/mcp/upload/{ticket.token}", content=b"x").status_code == 200
    assert client.post(f"/mcp/upload/{ticket.token}", content=b"y").status_code == 404
    assert len(entity.attach_calls) == 1


def test_upload_rejects_an_empty_body(client, fake_taiga):
    entity = fake_taiga()
    ticket = _issue()
    resp = client.post(f"/mcp/upload/{ticket.token}", content=b"")
    assert resp.status_code == 400
    assert entity.attach_calls == []


def test_upload_refuses_a_body_over_the_cap(client, fake_taiga, monkeypatch):
    """The size guard must stop the upload, not merely report on it."""
    entity = fake_taiga()
    monkeypatch.setattr(upload_tickets, "MAX_UPLOAD_BYTES", 16)
    ticket = _issue()

    resp = client.post(f"/mcp/upload/{ticket.token}", content=b"z" * 4096)

    assert resp.status_code == 413
    assert resp.json()["max_bytes"] == 16
    # Nothing reached Taiga — an oversized upload is refused, not truncated
    # and attached.
    assert entity.attach_calls == []


def test_upload_accepts_a_body_exactly_at_the_cap(client, fake_taiga, monkeypatch):
    entity = fake_taiga()
    monkeypatch.setattr(upload_tickets, "MAX_UPLOAD_BYTES", 16)
    ticket = _issue()
    resp = client.post(f"/mcp/upload/{ticket.token}", content=b"z" * 16)
    assert resp.status_code == 200
    assert entity.attach_calls[0][1] == b"z" * 16


def test_upload_maps_a_taiga_failure_to_502(client, fake_taiga):
    fake_taiga(attach_raises=RuntimeError("taiga said no"))
    ticket = _issue()
    resp = client.post(f"/mcp/upload/{ticket.token}", content=b"x")
    assert resp.status_code == 502
    assert "taiga said no" in resp.json()["error"]


def test_upload_target_comes_only_from_the_ticket(client, fake_taiga):
    """A guessed token cannot be steered at a different entity.

    The route takes no parameters beyond the token, so query strings and
    headers naming another project or ref have to be inert.
    """
    entity = fake_taiga()
    ticket = _issue()
    resp = client.post(
        f"/mcp/upload/{ticket.token}?project_slug=other&entity_ref=1&filename=evil.sh",
        content=b"x",
        headers={"X-Filename": "evil.sh"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ref"] == 7398
    assert payload["url"].endswith("/project/shikenso-development/issue/7398")
    assert entity.attach_calls[0][0] == "report.csv"


# ---------------------------------------------------------------------------
# attach_file_for_ticket directly.
# ---------------------------------------------------------------------------


def test_attach_file_for_ticket_raises_when_the_entity_vanished(fake_taiga, tmp_path):
    fake_taiga(entity_present=False)
    ticket = _issue()
    path = tmp_path / "report.csv"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="not found"):
        attach_file_for_ticket(ticket, str(path))


def test_attach_file_for_ticket_uses_the_ticket_jwt(fake_taiga, monkeypatch, tmp_path):
    """The upload runs as the ticket's user, never on server ENV creds."""
    project = _FakeProject()
    entity = _FakeEntity()
    seen = {}

    def _api(token=None):
        seen["token"] = token
        return _FakeApi(project)

    monkeypatch.setattr(taiga_tools, "get_taiga_api", _api)
    monkeypatch.setattr(
        taiga_tools, "fetch_entity", lambda proj, norm_type, ref: entity
    )
    path = tmp_path / "report.csv"
    path.write_bytes(b"x")

    attach_file_for_ticket(_issue(taiga_jwt="jwt-of-the-caller"), str(path))

    assert seen["token"] == "jwt-of-the-caller"


@pytest.mark.asyncio
async def test_upload_bounds_concurrent_taiga_attaches(fake_taiga, monkeypatch):
    """The per-request size cap is meaningless in aggregate without this.

    python-taiga reads the whole file back into memory to build the multipart
    body, so N simultaneous attaches cost N x ~2x the file size in RSS. The
    default ``asyncio.to_thread`` executor would run ~20 of them, and minting
    a ticket costs one cheap tool call — so an authenticated user uploading
    in parallel could OOM a 1Gi pod for every tenant through entirely
    legitimate use.

    Driven with ``httpx.ASGITransport`` + ``asyncio.gather`` rather than
    ``TestClient``, so the requests are genuinely in flight together.
    """
    monkeypatch.setattr(upload_tickets, "UPLOAD_CONCURRENCY", 2)
    fake_taiga()

    mcp = make_mcp()
    remote_server._attach_custom_routes(
        mcp,
        provider=object(),
        taiga_url="https://taiga.example.test",
        base_url=BASE_URL,
    )

    live = 0
    peak = 0
    counter_lock = threading.Lock()

    def _slow_attach(_ticket, _file_path):
        nonlocal live, peak
        with counter_lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)  # runs on a worker thread, never on the loop
        with counter_lock:
            live -= 1
        return {"added": True, "project": "p", "type": "issue", "ref": 1, "url": "u"}

    monkeypatch.setattr(remote_server, "attach_file_for_ticket", _slow_attach)

    tokens = [_issue(filename=f"f{i}.txt").token for i in range(6)]
    app = mcp.http_app(path="/mcp")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        responses = await asyncio.gather(
            *(c.post(f"/mcp/upload/{t}", content=b"x") for t in tokens)
        )

    assert [r.status_code for r in responses] == [200] * 6
    # Upper bound: the limiter holds.
    assert peak <= 2
    # Lower bound: the harness really did overlap requests, so `peak <= 2`
    # above is evidence of the limiter and not of accidental serialisation.
    assert peak == 2


# ---------------------------------------------------------------------------
# python-taiga HTTP timeouts.
# ---------------------------------------------------------------------------


def test_python_taiga_verbs_default_to_a_timeout():
    """python-taiga ships no timeouts; without these a wedged Taiga backend
    holds a worker thread — and an upload semaphore slot — forever."""
    import taiga.requestmaker as requestmaker

    assert isinstance(
        requestmaker.requests, taiga_tools._TimeoutDefaultingRequests
    ), "the timeout proxy must be installed at import of taiga_tools"

    seen = {}

    class _Fake:
        not_a_verb = "passthrough"

        def __getattr__(self, name):
            def _call(*args, **kwargs):
                seen[name] = kwargs.get("timeout")
                return "ok"

            return _call

    proxy = taiga_tools._TimeoutDefaultingRequests(_Fake(), (10, 300))
    for verb in ("get", "post", "put", "patch", "delete"):
        assert getattr(proxy, verb)("http://taiga.example.test") == "ok"
        assert seen[verb] == (10, 300), f"{verb} lost its timeout"

    # Non-verb attributes must pass straight through — requestmaker also
    # reaches for requests.packages and requests.exceptions.
    assert proxy.not_a_verb == "passthrough"


def test_explicit_timeout_wins_over_the_default():
    """setdefault, not override — a future python-taiga that passes its own
    timeout must not be silently overruled."""
    seen = {}

    class _Fake:
        def post(self, *args, **kwargs):
            seen.update(kwargs)
            return "ok"

    proxy = taiga_tools._TimeoutDefaultingRequests(_Fake(), (10, 300))
    proxy.post("http://taiga.example.test", timeout=1)
    assert seen["timeout"] == 1


def test_installing_the_timeout_proxy_is_idempotent():
    """A double install would nest proxies and re-wrap on every reload."""
    import taiga.requestmaker as requestmaker

    before = requestmaker.requests
    taiga_tools._install_taiga_http_timeouts()
    assert requestmaker.requests is before


@pytest.mark.asyncio
async def test_a_disconnect_does_not_free_the_slot_under_a_running_attach():
    """``asyncio.to_thread`` cannot cancel its worker.

    On a client disconnect the request task is cancelled, but the thread
    keeps running and keeps holding the multipart body in memory. Unshielded,
    unwinding would release the semaphore slot and delete the temp file out
    from under that worker — so repeated disconnects would push real
    concurrency past UPLOAD_CONCURRENCY and defeat the OOM guard the
    semaphore exists for.
    """
    mcp = make_mcp()
    remote_server._attach_custom_routes(
        mcp,
        provider=object(),
        taiga_url="https://taiga.example.test",
        base_url=BASE_URL,
    )

    observed = {}
    finished = threading.Event()

    def _slow_attach(_ticket, file_path):
        time.sleep(0.15)
        # If the request's unwinding had deleted the tempdir, this is where
        # the worker would find its input gone.
        observed["file_present_during_attach"] = os.path.exists(file_path)
        observed["dir_present_during_attach"] = os.path.isdir(
            os.path.dirname(file_path)
        )
        finished.set()
        return {"added": True, "project": "p", "type": "issue", "ref": 1, "url": "u"}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(remote_server, "attach_file_for_ticket", _slow_attach)
    try:
        ticket = _issue()
        app = mcp.http_app(path="/mcp")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            task = asyncio.create_task(
                c.post(f"/mcp/upload/{ticket.token}", content=b"payload")
            )
            await asyncio.sleep(0.02)  # let it reach the attach
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        # The worker must still run to completion, with its input intact.
        assert finished.wait(timeout=5.0), "the attach worker never finished"
        assert observed["file_present_during_attach"] is True
        assert observed["dir_present_during_attach"] is True
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_a_queued_upload_cancelled_before_its_slot_still_cleans_up(monkeypatch):
    """Cleanup must wrap the semaphore WAIT, not just the attach.

    A shutdown while every slot is busy cancels the queued shielded tasks
    before they ever enter the ``async with`` body. With the cleanup inside
    that body, each queued upload would leave its already-written file behind
    — up to MAX_UPLOAD_BYTES per interrupted upload.
    """
    monkeypatch.setattr(upload_tickets, "UPLOAD_CONCURRENCY", 1)

    created = []
    real_mkdtemp = tempfile.mkdtemp

    def _spy_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(remote_server.tempfile, "mkdtemp", _spy_mkdtemp)

    release = threading.Event()

    def _blocking_attach(_ticket, _file_path):
        release.wait(timeout=5.0)
        return {"added": True, "project": "p", "type": "issue", "ref": 1, "url": "u"}

    monkeypatch.setattr(remote_server, "attach_file_for_ticket", _blocking_attach)

    mcp = make_mcp()
    remote_server._attach_custom_routes(
        mcp,
        provider=object(),
        taiga_url="https://taiga.example.test",
        base_url=BASE_URL,
    )
    holder, queued = _issue(filename="a.txt"), _issue(filename="b.txt")

    app = mcp.http_app(path="/mcp")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        first = asyncio.create_task(
            c.post(f"/mcp/upload/{holder.token}", content=b"aaaa")
        )
        second = asyncio.create_task(
            c.post(f"/mcp/upload/{queued.token}", content=b"bbbb")
        )
        # Let the first take the only slot and the second queue behind it.
        await asyncio.sleep(0.05)

        # Shutdown: cancel the request tasks, then everything still pending —
        # which is what the loop does to the shielded uploads on teardown.
        for task in (first, second):
            task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        release.set()
        current = asyncio.current_task()
        pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert len(created) == 2, "both uploads should have spooled to disk"
    leaked = [d for d in created if os.path.exists(d)]
    assert leaked == [], f"temp directories left behind: {leaked}"


# ---------------------------------------------------------------------------
# fetch_entity's 404 translation.
# ---------------------------------------------------------------------------


def test_fetch_entity_returns_none_when_taiga_says_404():
    """Every ``*_by_ref`` tool documents 404 for a missing entity and reaches
    that branch by testing the return value — but python-taiga RAISES on a
    404, so the raise landed in their generic except and became a 500.

    Caught by driving the real server against a real python-taiga; the unit
    fakes returned ``None`` where the library raises, so nothing here saw it.
    """
    from taiga.exceptions import TaigaRestException

    class _Project:
        def get_issue_by_ref(self, ref):
            raise TaigaRestException("http://taiga/issues/by_ref", 404, "not found")

    assert taiga_tools.fetch_entity(_Project(), "issue", 7398) is None


def test_fetch_entity_still_raises_on_a_real_taiga_fault():
    """Only 404 is translated — a 500 from Taiga must not be disguised as a
    missing entity, or an outage reads as 'your ticket does not exist'."""
    from taiga.exceptions import TaigaRestException

    class _Project:
        def get_issue_by_ref(self, ref):
            raise TaigaRestException("http://taiga/issues/by_ref", 500, "boom")

    with pytest.raises(TaigaRestException):
        taiga_tools.fetch_entity(_Project(), "issue", 7398)
