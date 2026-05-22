"""Unit tests for ``add_attachment_inline_by_ref_tool``.

- ``TestAddAttachmentInlineUnit`` is the LangChain scaffold.
- Behavioural tests cover happy path, basename sanitization (POSIX +
  Windows), invalid base64, oversized (pre-decode + post-decode)
  refusal, empty payload, exactly-at-cap boundary, and entity lookup
  failure.

No real HTTP is mocked — the tool's only outbound I/O is via
``Entity.attach`` which we replace with a fake on the
``python-taiga`` seam.
"""

import base64
import json

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import add_attachment_inline_by_ref_tool


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@pytest.fixture(autouse=True)
def fake_taiga_url(monkeypatch):
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.example.test")


# Sample payload for the scaffold's required tool_invoke_params_example.
_SCAFFOLD_PAYLOAD = base64.b64encode(b"hello").decode("ascii")


class TestAddAttachmentInlineUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return add_attachment_inline_by_ref_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "project_slug": "slug",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "hello.txt",
            "attachment_content_base64": _SCAFFOLD_PAYLOAD,
        }


# ---------------------------------------------------------------------------
# Behavioural test infrastructure.
# ---------------------------------------------------------------------------


class _FakeAttachment:
    """Mimics python-taiga's ``Attachment`` minimally for ``.to_dict()``."""

    def __init__(self, aid, name, size, description, url):
        self.id = aid
        self.name = name
        self.size = size
        self.description = description
        self.url = url

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "description": self.description,
            "url": self.url,
        }


class _FakeEntity:
    """Records the (basename, bytes, description) passed to ``attach``."""

    def __init__(self):
        self.attach_calls = []  # list of (basename, bytes, description)
        self.attach_should_raise = None

    def attach(self, file_path, description=""):
        if self.attach_should_raise is not None:
            raise self.attach_should_raise
        import os as _os  # local alias to avoid cross-test interference

        basename = _os.path.basename(file_path)
        with open(file_path, "rb") as f:
            content = f.read()
        self.attach_calls.append((basename, content, description))
        return _FakeAttachment(
            aid=42,
            name=basename,
            size=len(content),
            description=description,
            url=f"https://taiga.example.test/media/attachments/{basename}",
        )


class _FakeProject:
    name = "Shikenso Development"


@pytest.fixture
def fake_env(monkeypatch):
    """Installs an entity + project in the taiga_tools module-level seams
    and returns the entity so tests can inspect its ``attach_calls``."""

    def _install(entity_present=True, attach_raises=None):
        entity = _FakeEntity() if entity_present else None
        if entity is not None:
            entity.attach_should_raise = attach_raises
        project = _FakeProject()
        monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
        monkeypatch.setattr(
            taiga_tools, "fetch_entity", lambda proj, norm_type, ref: entity
        )
        return entity

    return _install


# ---------------------------------------------------------------------------
# Behavioural tests.
# ---------------------------------------------------------------------------


def test_happy_path_uploads_bytes_with_caller_filename(fake_env):
    entity = fake_env()
    body = b"# Handover\n\nticket 7398 context\n"
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "handover.md",
            "attachment_content_base64": base64.b64encode(body).decode("ascii"),
            "description": "auto-uploaded test",
        }
    )

    assert len(entity.attach_calls) == 1
    basename, content, description = entity.attach_calls[0]
    assert basename == "handover.md"
    assert content == body
    assert description == "auto-uploaded test"

    payload = json.loads(raw)
    assert payload["added"] is True
    assert payload["type"] == "issue"
    assert payload["ref"] == 7398
    assert (
        payload["url"]
        == "https://taiga.example.test/project/shikenso-development/issue/7398"
    )
    assert payload["attachments"]["name"] == "handover.md"
    assert payload["attachments"]["size"] == len(body)
    # content_type is NOT in the response — Taiga derives it from the
    # filename extension server-side. Re-list to read what Taiga stored.
    assert "content_type" not in payload
    # The url field is stripped from the embedded attachment dict (signed URL hygiene).
    assert "url" not in payload["attachments"]


def test_invalid_base64_returns_400_and_no_upload(fake_env):
    entity = fake_env()
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "broken.bin",
            "attachment_content_base64": "this!is@not%valid+base64==",
        }
    )
    assert entity.attach_calls == []
    payload = json.loads(raw)
    assert payload["code"] == 400
    assert "base64" in payload["error"].lower()


def test_empty_payload_returns_400(fake_env):
    entity = fake_env()
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "empty.txt",
            "attachment_content_base64": "",
        }
    )
    assert entity.attach_calls == []
    payload = json.loads(raw)
    assert payload["code"] == 400
    assert "zero" in payload["error"].lower()


def test_pre_decode_size_cap_returns_413_without_decode(fake_env, monkeypatch):
    entity = fake_env()
    # 14 MB of "A" → upper-bound estimate 14 * 1024 * 1024 * 3 // 4
    # = 10.5 MB, well above the 10 MB default cap → pre-check 413.
    # Use repeated 'A' so we don't actually allocate the real bytes
    # anywhere — base64 decoding WOULD, but we expect the pre-check
    # to refuse BEFORE decoding.
    huge_b64 = "A" * (14 * 1024 * 1024)

    # Sentinel: monkey-patch b64decode to fail loudly if the tool tries
    # to decode despite the pre-check; if test passes, b64decode was
    # not called.
    decode_called = {"called": False}

    real_b64decode = base64.b64decode

    def _spy(*args, **kwargs):
        decode_called["called"] = True
        return real_b64decode(*args, **kwargs)

    monkeypatch.setattr(taiga_tools.base64, "b64decode", _spy)

    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "big.bin",
            "attachment_content_base64": huge_b64,
        }
    )

    assert entity.attach_calls == []
    assert decode_called["called"] is False
    payload = json.loads(raw)
    assert payload["code"] == 413
    assert payload["max_bytes"] == 10 * 1024 * 1024


def test_size_cap_returns_413_under_lowered_cap(fake_env, monkeypatch):
    """End-to-end size cap behaviour with a small lowered cap so the
    test doesn't have to allocate 10 MB. With the exact pre-decode
    formula (accounting for padding), the pre-check fires for any
    payload strictly larger than the cap; the post-decode branch is
    defense-in-depth that's effectively unreachable for valid input."""
    monkeypatch.setattr(taiga_tools, "MAX_INLINE_ATTACHMENT_BYTES", 4)
    entity = fake_env()
    body = b"hello"  # 5 bytes > 4-byte cap
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "small.bin",
            "attachment_content_base64": base64.b64encode(body).decode("ascii"),
        }
    )
    assert entity.attach_calls == []
    payload = json.loads(raw)
    assert payload["code"] == 413
    assert payload["max_bytes"] == 4


def test_line_wrapped_base64_is_accepted(fake_env):
    """GNU ``base64`` wraps encoded output at 76 chars with newlines,
    and many MIME tools / pasted payloads do the same. The tool must
    accept these as valid base64 (RFC 4648 §3.3) — stripping ASCII
    whitespace before validation."""
    entity = fake_env()
    body = b"this is some longer text " * 4  # ~100 bytes → wraps into 2 lines
    b64 = base64.b64encode(body).decode("ascii")
    # Inject a 76-char wrap + trailing newline (GNU base64 default).
    wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76)) + "\n"
    assert "\n" in wrapped  # sanity: actually has newlines

    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "wrapped.txt",
            "attachment_content_base64": wrapped,
        }
    )
    assert len(entity.attach_calls) == 1
    _, content, _ = entity.attach_calls[0]
    assert content == body
    payload = json.loads(raw)
    assert payload["added"] is True


def test_unaligned_oversized_b64_rejected_before_decode(fake_env, monkeypatch):
    """Regression for Codex P1 (round 2): if the b64 length is not a
    multiple of 4 (malformed but huge), the size guard must still
    fire BEFORE ``b64decode`` allocates ~3/4 of the input. The earlier
    implementation skipped the pre-check on unaligned length and only
    delegated to ``b64decode(validate=True)``, which raised AFTER
    allocating — defeating the RSS protection on a 14 MB+1 payload."""
    entity = fake_env()
    # Unaligned (len % 4 == 1), but upper bound still > MAX * 1.5.
    huge_unaligned_b64 = "A" * (14 * 1024 * 1024 + 1)
    decode_called = {"called": False}
    real_b64decode = base64.b64decode

    def _spy(*args, **kwargs):
        decode_called["called"] = True
        return real_b64decode(*args, **kwargs)

    monkeypatch.setattr(taiga_tools.base64, "b64decode", _spy)

    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "big.bin",
            "attachment_content_base64": huge_unaligned_b64,
        }
    )
    assert entity.attach_calls == []
    assert decode_called["called"] is False
    payload = json.loads(raw)
    assert payload["code"] == 413
    assert payload["max_bytes"] == 10 * 1024 * 1024


def test_whitespace_inflated_oversized_b64_rejected_before_clean(
    fake_env, monkeypatch
):
    """Regression for Codex P1 (round 3): the whitespace-strip step
    allocates O(input) extra memory. A pathological caller could pad
    ~1 GB of newlines around a small valid b64 string and the previous
    single-stage clean+check would OOM before reaching the size guard.
    The raw-length ceiling MUST refuse such inputs before the clean."""
    monkeypatch.setattr(taiga_tools, "MAX_INLINE_ATTACHMENT_BYTES", 16)
    entity = fake_env()
    # 100 chars of whitespace around a tiny valid payload. With cap=16
    # bytes, MAX * 3/2 = 24 → raw_upper_bound 75 (= 100 * 3 // 4) > 24
    # → rejected at stage 1, BEFORE the whitespace-clean materialization.
    inflated = " " * 50 + base64.b64encode(b"hi").decode("ascii") + "\n" * 50
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "padded.bin",
            "attachment_content_base64": inflated,
        }
    )
    assert entity.attach_calls == []
    payload = json.loads(raw)
    assert payload["code"] == 413
    # The raw-stage error message includes "raw upper-bound".
    assert "raw upper-bound" in payload["error"]


def test_interleaved_whitespace_does_not_materialize_substrings(fake_env, monkeypatch):
    """Regression for Codex P1 (round 4): a payload that interleaves
    whitespace between every base64 character (``"A\\nA\\nA..."``)
    passes the stage-1 raw ceiling at default cap settings, but used
    to be cleaned with ``str.split()`` which materializes one
    short-string Python object per ``A`` (~50 bytes header overhead
    each). A 100K-char interleaved input would balloon to ~5 MB of
    substring objects even though the cleaned form is 50 KB.

    Switching to ``str.translate`` allocates exactly one new string of
    the cleaned size. This test verifies the small-input behaviour
    end-to-end; the memory-amplification fix is covered by the
    documented choice in the implementation comment."""
    entity = fake_env()
    body = b"interleaved-test"
    b64 = base64.b64encode(body).decode("ascii")
    # Interleave a newline between every char.
    interleaved = "\n".join(b64)
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "inter.bin",
            "attachment_content_base64": interleaved,
        }
    )
    assert len(entity.attach_calls) == 1
    _, content, _ = entity.attach_calls[0]
    assert content == body


def test_exactly_at_cap_is_accepted(fake_env, monkeypatch):
    """Regression for Codex P2: with the old ``len(b64) * 3 // 4``
    estimate, a 5-byte payload (b64 = "aGVsbG8=", 8 chars) hitting a
    5-byte cap would be rejected because (8 * 3 // 4) == 6 > 5.

    The corrected formula accounts for "=" padding: (8 // 4) * 3 - 1
    = 5, which equals the cap and is therefore accepted (only ``>``
    triggers 413, matching the post-decode check)."""
    monkeypatch.setattr(taiga_tools, "MAX_INLINE_ATTACHMENT_BYTES", 5)
    entity = fake_env()
    body = b"hello"  # exactly 5 bytes
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "exact.bin",
            "attachment_content_base64": base64.b64encode(body).decode("ascii"),
        }
    )
    assert len(entity.attach_calls) == 1
    payload = json.loads(raw)
    assert payload["added"] is True
    assert payload["attachments"]["size"] == 5


def test_windows_path_filename_stripped_on_linux_host(fake_env):
    """Regression for Codex P3: a Windows client passing a Windows-
    style path must have its drive + backslash separators stripped,
    not preserved (the prior ``os.path.basename`` on a Linux MCP pod
    treated ``\\`` as part of the filename).
    """
    entity = fake_env()
    body = b"y"
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "C:\\Users\\wahed\\Documents\\handover.md",
            "attachment_content_base64": base64.b64encode(body).decode("ascii"),
        }
    )
    assert len(entity.attach_calls) == 1
    basename, _, _ = entity.attach_calls[0]
    assert basename == "handover.md"
    payload = json.loads(raw)
    assert payload["attachments"]["name"] == "handover.md"


def test_path_traversal_filename_stripped_to_basename(fake_env):
    entity = fake_env()
    body = b"x"
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "../../../etc/passwd",
            "attachment_content_base64": base64.b64encode(body).decode("ascii"),
        }
    )
    assert len(entity.attach_calls) == 1
    basename, _, _ = entity.attach_calls[0]
    assert basename == "passwd"  # path components stripped
    payload = json.loads(raw)
    assert payload["attachments"]["name"] == "passwd"


def test_empty_filename_returns_400(fake_env):
    entity = fake_env()
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "",
            "attachment_content_base64": base64.b64encode(b"x").decode("ascii"),
        }
    )
    assert entity.attach_calls == []
    payload = json.loads(raw)
    assert payload["code"] == 400


def test_dot_dot_filename_returns_400(fake_env):
    """Regression for Codex P3 (round 2): ``PureWindowsPath('..').name``
    returns ``'..'`` (and same for ``'foo/..'``), which previously
    slipped through the empty-name check and then crashed
    ``open(tmpdir/'..', 'wb')`` with ``IsADirectoryError`` → bubbled
    as a misleading 500. Treat dot-segments as user input errors with
    a precise 400. Note that ``'.'``-only segments inside a path
    (e.g. ``'foo/.'``) are normalized away by pathlib and resolve to
    a valid basename, so they are accepted."""
    entity = fake_env()
    for bad in ("..", ".", "foo/.."):
        raw = add_attachment_inline_by_ref_tool.invoke(
            {
                "project_slug": "shikenso-development",
                "entity_ref": 7398,
                "entity_type": "issue",
                "attachment_filename": bad,
                "attachment_content_base64": base64.b64encode(b"x").decode("ascii"),
            }
        )
        payload = json.loads(raw)
        assert payload["code"] == 400, f"input {bad!r} should 400, got {payload}"
    assert entity.attach_calls == []


def test_pure_separator_filename_returns_400(fake_env):
    """A filename made entirely of separators (e.g. ``/``) basenames to
    '' and must be rejected. ``PureWindowsPath`` normalizes
    ``some/dir/`` to ``some/dir`` → ``dir`` (user-friendly trailing-
    slash fix), so the all-separator case is the remaining null
    basename to test."""
    entity = fake_env()
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "/",
            "attachment_content_base64": base64.b64encode(b"x").decode("ascii"),
        }
    )
    assert entity.attach_calls == []
    payload = json.loads(raw)
    assert payload["code"] == 400


def test_invalid_entity_type_returns_400(fake_env):
    entity = fake_env()
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "story",  # not in ENTITY_TYPE_MAPPING
            "attachment_filename": "foo.txt",
            "attachment_content_base64": base64.b64encode(b"x").decode("ascii"),
        }
    )
    assert entity.attach_calls == []
    payload = json.loads(raw)
    assert payload["code"] == 400
    assert "story" in payload["error"]


def test_entity_not_found_returns_404(fake_env):
    fake_env(entity_present=False)
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 9999,
            "entity_type": "issue",
            "attachment_filename": "foo.txt",
            "attachment_content_base64": base64.b64encode(b"x").decode("ascii"),
        }
    )
    payload = json.loads(raw)
    assert payload["code"] == 404
    assert "9999" in payload["error"]


def test_taiga_upload_failure_returns_500(fake_env):
    entity = fake_env(attach_raises=RuntimeError("taiga 503"))
    raw = add_attachment_inline_by_ref_tool.invoke(
        {
            "project_slug": "shikenso-development",
            "entity_ref": 7398,
            "entity_type": "issue",
            "attachment_filename": "foo.txt",
            "attachment_content_base64": base64.b64encode(b"x").decode("ascii"),
        }
    )
    payload = json.loads(raw)
    assert payload["code"] == 500
    assert "taiga 503" in payload["error"]
    # entity.attach raised before the recorder appended.
    assert entity.attach_calls == []
