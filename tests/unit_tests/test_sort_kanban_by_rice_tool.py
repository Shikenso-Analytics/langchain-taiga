"""Regression tests for the ``sort_kanban_by_rice_tool`` async pipeline.

History (the contract these tests lock):
- 2.3.0: effort moved off hard-coded ``role_id="19"`` to the sum across
  every role's points.
- 2.3.2: per-US custom-attribute fetch parallelised via
  ThreadPoolExecutor; total-failure guard + ``attribute_fetch_errors``.
- 2.3.4 (this file): fetcher migrated from ThreadPoolExecutor + python-
  taiga ``us.get_attributes()`` to ``httpx.AsyncClient`` + ``asyncio.gather``,
  and project/epic discovery cached in module-level TTL caches. Tests
  now mock httpx via ``respx_mock`` instead of stubbing the per-instance
  ``get_attributes`` method, which is no longer called.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import sort_kanban_by_rice_tool


@pytest.fixture(autouse=True)
def fake_env_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")
    # Both ``TAIGA_URL`` and ``TAIGA_API_URL`` are captured at import
    # time via ``os.getenv``, so ``monkeypatch.setenv`` would be a
    # no-op here. Patch the module attributes directly so
    # ``_resolve_taiga_api_base_url`` (the 2.3.4 host-selector that
    # picks API > UI) returns the respx-mocked test host. In Shikenso's
    # local conda env both vars are set to the prod URL via .env, so
    # leaving ``TAIGA_API_URL`` unpatched would route the async fetchers
    # at the real Taiga and fail every per-US fetch.
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "TAIGA_API_URL", "https://taiga.test")


@pytest.fixture(autouse=True)
def clear_sort_caches():
    """Reset the sort-tool TTL caches between tests.

    Both ``sort_attr_def_cache`` (5-min TTL, project-level RICE attr
    discovery) and ``list_all_statuses_cache`` (5-min TTL, project
    status definitions) are keyed by ``(user_scope, project_slug)``.
    Without this fixture, the second test using the same project_slug
    would hit a stale cached entry from the previous test's
    monkeypatched ``get_project`` and silently use the wrong attribute
    or status IDs. Per-epic Multiplicator values are intentionally
    *not* cached (see the module-level comment in taiga_tools.py).
    """
    taiga_tools.sort_attr_def_cache.clear()
    taiga_tools.list_all_statuses_cache.clear()
    yield


class _FakeAttr:
    def __init__(self, aid, name):
        self.id = aid
        self.name = name


class _FakePoint:
    def __init__(self, pid, value):
        self.id = pid
        self.value = value


class _FakeStatus:
    """Mirrors the python-taiga UserStoryStatus fields the tool reads.

    ``list_all_statuses(slug, "us")`` calls ``project.list_user_story_statuses()``
    and projects each status via ``{**status.to_dict(), "id": status.id}``.
    Matching the same shape lets the production helper consume our fakes
    without modification.
    """

    def __init__(self, sid, name, is_closed=False):
        self.id = sid
        self.name = name
        self.is_closed = is_closed

    def to_dict(self):
        return {"id": self.id, "name": self.name, "is_closed": self.is_closed}


class _FakeUS:
    """Mirrors the python-taiga UserStory fields the tool reads.

    Since 2.3.4 the tool fetches custom attributes via httpx, so
    ``_attr_values`` is consumed by :func:`_register_us_attr_routes` to
    build the respx mocks (no per-instance ``get_attributes()``).
    """

    def __init__(self, ref, points, attr_values, status=100, total_points=None):
        self.ref = ref
        self.id = ref * 1000
        self.subject = f"US {ref}"
        self.points = points
        # Taiga's list endpoint inlines ``total_points`` (sum across
        # computable roles); the tool reads it directly since 2.3.2.
        self.total_points = total_points
        self._attr_values = attr_values
        self.status = status
        self.epics = None
        self.due_date = None
        self.is_closed = False


class _FakeProject:
    name = "Test"
    id = 7

    def __init__(
        self,
        stories,
        roles,
        points,
        us_attrs,
        epic_attrs=None,
        us_statuses=None,
    ):
        self._stories = stories
        self._roles = roles
        self._points = points
        self._us_attrs = us_attrs
        self._epic_attrs = epic_attrs or []
        # Default: a single open status matching the existing
        # _FakeUS(status=100) default. Tests that need closed columns
        # or custom names override via ``us_statuses=...``.
        self._us_statuses = (
            us_statuses
            if us_statuses is not None
            else [_FakeStatus(100, "New", is_closed=False)]
        )
        # Counters so cache-hit tests can assert no second fetch.
        self.list_user_story_attributes_calls = 0
        self.list_epic_attributes_calls = 0
        self.list_epics_calls = 0
        self.list_user_story_statuses_calls = 0

    def list_user_stories(self):
        return self._stories

    def list_roles(self):
        return self._roles

    def list_points(self):
        return self._points

    def list_user_story_attributes(self):
        self.list_user_story_attributes_calls += 1
        return self._us_attrs

    def list_epic_attributes(self):
        self.list_epic_attributes_calls += 1
        return self._epic_attrs

    def list_epics(self):
        self.list_epics_calls += 1
        return []

    def list_user_story_statuses(self):
        self.list_user_story_statuses_calls += 1
        return self._us_statuses


def _register_us_attr_routes(respx_mock, stories, *, raise_refs=()):
    """Register a respx GET route per story for the new async fetcher.

    The fetcher hits
    ``GET https://taiga.test/api/v1/userstories/custom-attributes-values/<us.id>``
    once per story. Each route returns the story's ``_attr_values`` as
    ``{"attributes_values": {...}, "version": 1}``. Refs listed in
    ``raise_refs`` instead return HTTP 500 — used to drive the
    partial-/total-failure paths (replaces the pre-2.3.4
    ``us._attrs_raises`` flag).
    """
    raise_set = set(raise_refs)
    for us in stories:
        url = f"https://taiga.test/api/v1/userstories/custom-attributes-values/{us.id}"
        if us.ref in raise_set:
            response = httpx.Response(500, json={"detail": "boom"})
        else:
            response = httpx.Response(
                200,
                json={"attributes_values": us._attr_values, "version": 1},
            )
        respx_mock.get(url).mock(return_value=response)


@pytest.fixture
def patched_http(monkeypatch):
    """The bulk-update POST still goes through ``requests.post`` (the
    column count is bounded so async there isn't worth the test churn).
    Stub it to a 200 no-op + stub ``get_taiga_api`` so no real auth
    handshake happens."""
    fake_response = SimpleNamespace(status_code=200, json=lambda: {})
    monkeypatch.setattr(
        taiga_tools.requests, "post", lambda *a, **kw: fake_response
    )
    fake_api = SimpleNamespace(token="fake-token")
    monkeypatch.setattr(taiga_tools, "get_taiga_api", lambda token=None: fake_api)


def _baseline_attrs():
    """Custom attribute definitions: 1=reach, 2=impact, 3=confidence."""
    return [
        _FakeAttr(1, "Reach"),
        _FakeAttr(2, "Impact"),
        _FakeAttr(3, "Confidence"),
    ]


def _baseline_points():
    return [
        _FakePoint(100, 1),
        _FakePoint(101, 2),
        _FakePoint(102, 3),
        _FakePoint(103, 5),
        _FakePoint(104, 8),
    ]


def test_effort_takes_us_total_points(monkeypatch, patched_http, respx_mock):
    """Effort = ``us.total_points`` (Taiga-computed sum across all
    computable roles' point assignments). Pre-2.3.2 we computed this
    locally from ``us.points`` against a separate ``list_points()``
    lookup; since 2.3.2 we trust Taiga's own pre-computed field, which
    is inlined on every list-userstories response."""
    story = _FakeUS(
        ref=34,
        points={"19": 103, "20": 101},  # Developer=5, UX=2 (legacy field)
        total_points=7.0,                # what Taiga computed
        attr_values={"1": 4, "2": 3, "3": 1},  # reach, impact, confidence
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer"), _FakeAttr(20, "UX")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [story])

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    cols = payload["columns_updated"]
    assert len(cols) == 1
    order = cols[0]["order"]
    assert len(order) == 1
    assert order[0]["effort"] == 7.0


def test_works_regardless_of_role_id(monkeypatch, patched_http, respx_mock):
    """Pre-2.3.0 the tool hard-coded ``developer_role_id = "19"`` and
    silently zeroed effort on projects where Developer was a different
    role-id. Pre-2.3.2 it summed ``us.points.values()`` itself which
    needed a separate role/points lookup. Now it reads ``total_points``
    directly — completely role-id-agnostic, no project-wide lookup,
    works on any project regardless of how Taiga numbers its roles."""
    story = _FakeUS(
        ref=10,
        points={"42": 103},  # Developer at role-id 42 (legacy field)
        total_points=5.0,
        attr_values={"1": 1, "2": 1, "3": 1},
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(42, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [story])

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    order = payload["columns_updated"][0]["order"]
    assert order[0]["effort"] == 5.0


def test_effort_zero_when_no_points_assigned(
    monkeypatch, patched_http, respx_mock
):
    """Stories without any role-points assigned (``total_points=None``
    from Taiga) get effort=0 and consequently rice_score=0 — they sort
    to the bottom of their column instead of crashing the tool. Edge
    case for newly-created stories that haven't been estimated yet."""
    story = _FakeUS(
        ref=99,
        points={},
        total_points=None,
        attr_values={"1": 4, "2": 3, "3": 1},
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [story])

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    order = payload["columns_updated"][0]["order"]
    assert order[0]["effort"] == 0
    assert order[0]["rice"] == 0


def test_partial_attr_fetch_failure_is_surfaced(
    monkeypatch, patched_http, respx_mock
):
    """A single failing per-US fetch must NOT poison the whole sort.
    The new 2.3.4 contract: failures are RECORDED in
    ``attribute_fetch_errors`` of the success payload, the rest of the
    sort still runs, and the LLM/caller can decide whether to retry.
    Pre-2.3.2 a single failure aborted with a 500; post-2.3.2 they were
    silently swallowed (Codex/Copilot review on PR #13 caught this);
    post-2.3.4 (this test) they're surfaced via the same shape."""
    good = _FakeUS(
        ref=1, points={}, total_points=2.0,
        attr_values={"1": 5, "2": 5, "3": 5},
    )
    bad = _FakeUS(
        ref=2, points={}, total_points=2.0, attr_values={},
    )
    project = _FakeProject(
        stories=[good, bad],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [good, bad], raise_refs={2})

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    assert payload.get("sorted") is True
    errors = payload["attribute_fetch_errors"]
    assert isinstance(errors, list)
    assert len(errors) == 1
    assert errors[0]["ref"] == 2
    # httpx.HTTPStatusError carries "500" in its repr — the exact class
    # name varies between httpx versions, so just spot-check the status.
    assert "500" in errors[0]["error"]
    # Both stories still present in the column ordering (the failure
    # didn't drop the story, just flagged it).
    refs = {s["ref"] for s in payload["columns_updated"][0]["order"]}
    assert refs == {1, 2}


def test_total_attr_fetch_failure_returns_500(
    monkeypatch, patched_http, respx_mock
):
    """If EVERY story's attribute fetch fails, RICE scores are
    uniformly garbage and reordering the Kanban from defaults would be
    actively harmful. The tool must abort with 500 + the error list,
    not silently sort by default-1 RICE."""
    s1 = _FakeUS(ref=1, points={}, total_points=2.0, attr_values={})
    s2 = _FakeUS(ref=2, points={}, total_points=2.0, attr_values={})
    project = _FakeProject(
        stories=[s1, s2],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [s1, s2], raise_refs={1, 2})

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    assert payload["code"] == 500
    assert "All per-story custom-attribute fetches failed" in payload["error"]
    assert len(payload["attribute_fetch_errors"]) == 2


def test_outer_try_returns_json_on_unexpected_failure(monkeypatch):
    """Pre-2.3.2 the tool had inner try/excepts only — uncaught failures
    bubbled up to the FastMCP harness as the generic 'Error occurred
    during tool execution' with no diagnostic. The 2.3.2 outer
    try/except (now in ``_sort_kanban_async_impl``) catches every
    uncaught exception and returns a JSON 500 with ``trace_tail`` so
    the LLM (and the next debugger) sees what actually broke."""
    class _Boom:
        # Looks project-shaped enough for the early-validation gate to
        # pass, but explodes when ``_discover_sort_attr_ids`` calls
        # ``list_user_story_attributes``.
        name = "wahed"
        slug = "wahed"
        id = 1

        def list_user_story_attributes(self):
            raise RuntimeError("simulated network failure mid-fetch")

    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Boom())

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    assert payload["code"] == 500
    assert "RuntimeError" in payload["error"]
    assert "simulated network failure mid-fetch" in payload["error"]
    # Trace tail must be a list of strings (last 5 lines of the
    # traceback) — NOT a single newline-joined blob, so the harness's
    # JSON pretty-printer keeps each line readable.
    assert isinstance(payload["trace_tail"], list)
    assert len(payload["trace_tail"]) <= 5


def test_api_url_takes_precedence_over_ui_url(
    monkeypatch, patched_http, respx_mock
):
    """Codex P1 regression guard for 2.3.4: when ``TAIGA_API_URL`` and
    ``TAIGA_URL`` differ (the documented split-host setup —
    ``tree.taiga.io`` UI / ``api.taiga.io`` API, or the cluster-internal
    API in remote-MCP mode), the new async fetchers MUST hit the API
    host. Pre-fix, both fetchers used ``TAIGA_URL`` and would either
    404 against the UI host or land in a CDN that doesn't speak v1 API
    — silently turning every story into ``attribute_fetch_errors`` and
    aborting the sort with ``All per-story custom-attribute fetches
    failed``. python-taiga's own client (``get_taiga_api``) routes via
    ``TAIGA_API_URL`` → so the async refactor must do the same.
    """
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga-ui.test")
    monkeypatch.setattr(taiga_tools, "TAIGA_API_URL", "https://taiga-api.test")

    story = _FakeUS(
        ref=5, points={}, total_points=2.0,
        attr_values={"1": 2, "2": 2, "3": 2},
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    # Register the route ONLY on the API host. If the fetcher hits the
    # UI host instead, respx returns ConnectionError → the assertion
    # below catches it as a failed fetch, not a successful sort.
    respx_mock.get(
        f"https://taiga-api.test/api/v1/userstories/"
        f"custom-attributes-values/{story.id}"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"attributes_values": story._attr_values, "version": 1},
        )
    )

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    assert payload.get("sorted") is True, (
        f"async fetcher hit the wrong host (UI instead of API). "
        f"payload={payload}"
    )
    assert payload.get("attribute_fetch_errors") is None


def test_epic_multiplicator_is_always_fresh(
    monkeypatch, patched_http, respx_mock
):
    """Codex P2 regression guard for 2.3.4: per-epic Multiplicator
    values must NOT be TTL-cached. The user flow ``sort → edit a
    Multiplicator → re-sort`` needs to reflect the edit immediately.
    A draft 2.3.4 cached the dict for 60 s; this test would have caught
    the staleness by re-running with a different mocked response and
    asserting the second call sees the new value.
    """
    class _FakeEpic:
        def __init__(self, eid):
            self.id = eid

    epic = _FakeEpic(eid=42)
    story = _FakeUS(
        ref=5, points={}, total_points=2.0,
        attr_values={"1": 2, "2": 2, "3": 2},
    )
    story.epics = [{"id": epic.id, "ref": 100}]

    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
        # Adding a "Multiplicator" attribute (id=99) to the epic schema
        # is what flips ``multiplicator_attr_id`` non-None and forces
        # ``_fetch_epic_multiplicators_async`` to run.
        epic_attrs=[_FakeAttr(99, "Multiplicator")],
    )
    project.list_epics = lambda: [epic]
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [story])

    epic_route = respx_mock.get(
        f"https://taiga.test/api/v1/epics/custom-attributes-values/{epic.id}"
    )

    epic_route.mock(
        return_value=httpx.Response(
            200, json={"attributes_values": {"99": 2.0}, "version": 1}
        )
    )
    raw1 = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload1 = json.loads(raw1)
    assert (
        payload1["columns_updated"][0]["order"][0]["epic_mult"] == 2.0
    )

    epic_route.mock(
        return_value=httpx.Response(
            200, json={"attributes_values": {"99": 5.0}, "version": 2}
        )
    )
    raw2 = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload2 = json.loads(raw2)
    assert payload2["columns_updated"][0]["order"][0]["epic_mult"] == 5.0, (
        "Per-epic Multiplicator was served from a stale cache. The "
        "new value (5.0) was NOT reflected in the second sort, which "
        "means the Codex P2 regression slipped back in. See the "
        "module-level comment in taiga_tools.py for why this isn't "
        "cached."
    )


def test_missing_taiga_url_config_returns_clear_error(
    monkeypatch, patched_http
):
    """Copilot follow-up regression guard for 2.3.4: when neither
    ``TAIGA_API_URL`` nor ``TAIGA_URL`` is set,
    ``_resolve_taiga_api_base_url`` must raise a ``ValueError``
    naming the missing env vars rather than letting
    ``None.rstrip('/')`` bubble up as ``AttributeError`` and surface
    as a confusing ``'NoneType' object has no attribute 'rstrip'``
    500 in the JSON outer-try payload.

    The outer try/except catches the ValueError and turns it into a
    JSON 500 — assert the error string mentions the env vars so the
    LLM/operator sees the actual fix instead of a NoneType trace.
    """
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", None)
    monkeypatch.setattr(taiga_tools, "TAIGA_API_URL", None)

    story = _FakeUS(
        ref=1, points={}, total_points=2.0,
        attr_values={"1": 1, "2": 1, "3": 1},
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)
    assert payload["code"] == 500
    assert "TAIGA_API_URL" in payload["error"]
    assert "TAIGA_URL" in payload["error"]


def test_skips_closed_status_columns(monkeypatch, respx_mock):
    """Closed status columns (Done, Cancelled, ...) MUST be filtered out
    of the sort. Re-ranking already-completed work has no business
    value and produces redundant ``bulk_update_kanban_order`` POSTs
    against Taiga. New default in 2.4.0; not opt-out-able.
    """
    open_story = _FakeUS(
        ref=1, points={}, total_points=2.0,
        attr_values={"1": 5, "2": 5, "3": 5},
        status=100,  # open status id
    )
    closed_story = _FakeUS(
        ref=2, points={}, total_points=2.0,
        attr_values={"1": 5, "2": 5, "3": 5},
        status=200,  # closed status id
    )
    project = _FakeProject(
        stories=[open_story, closed_story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
        us_statuses=[
            _FakeStatus(100, "New", is_closed=False),
            _FakeStatus(200, "Done", is_closed=True),
        ],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [open_story, closed_story])

    # Count how many times the bulk-update POST is hit. Replaces the
    # ``patched_http`` fixture's lambda so we can assert call count
    # without re-running its setup.
    bulk_calls = []
    fake_response = SimpleNamespace(status_code=200, json=lambda: {})

    def _record_post(*args, **kwargs):
        bulk_calls.append((args, kwargs))
        return fake_response

    monkeypatch.setattr(taiga_tools.requests, "post", _record_post)
    monkeypatch.setattr(
        taiga_tools, "get_taiga_api",
        lambda token=None: SimpleNamespace(token="fake-token"),
    )

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)

    assert payload.get("sorted") is True
    cols = payload["columns_updated"]
    # Only the open column was sorted.
    assert len(cols) == 1
    assert cols[0]["status_id"] == 100
    # Bulk-update POST hit exactly once (no redundant call for the
    # closed column).
    assert len(bulk_calls) == 1


def test_response_includes_status_name(monkeypatch, patched_http, respx_mock):
    """Per-column entry in ``columns_updated`` must include a
    ``status_name`` string sourced from the same ``list_all_statuses``
    call as the closed-skip filter. Pre-2.4.0 only ``status_id`` was
    returned — consumers had to round-trip through Taiga to translate.
    """
    story = _FakeUS(
        ref=1, points={}, total_points=2.0,
        attr_values={"1": 5, "2": 5, "3": 5},
        status=100,
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
        us_statuses=[_FakeStatus(100, "New", is_closed=False)],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [story])

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)

    cols = payload["columns_updated"]
    assert len(cols) == 1
    assert cols[0]["status_id"] == 100
    assert cols[0]["status_name"] == "New"


def test_orphan_status_id_is_not_dropped(
    monkeypatch, patched_http, respx_mock
):
    """A story whose ``status`` references an id absent from
    ``us_statuses`` (orphan — possible after an admin renames or
    deletes a status mid-cache-window) MUST be sorted normally, not
    dropped. Skip-closed semantics fail OPEN: treat the unknown
    status as not-closed rather than silently losing work. The
    matching ``status_name`` falls back to None to be honest about
    the lookup miss.
    """
    story = _FakeUS(
        ref=1, points={}, total_points=2.0,
        attr_values={"1": 5, "2": 5, "3": 5},
        status=999,  # orphan: not present in us_statuses
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
        us_statuses=[_FakeStatus(100, "New", is_closed=False)],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [story])

    raw = sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    payload = json.loads(raw)

    cols = payload["columns_updated"]
    # Orphan column was sorted, not dropped.
    assert len(cols) == 1
    assert cols[0]["status_id"] == 999
    # status_name is honest about the lookup miss.
    assert cols[0]["status_name"] is None
    # The story is in the order array.
    assert {row["ref"] for row in cols[0]["order"]} == {1}


def test_attr_def_cache_skips_second_discovery(
    monkeypatch, patched_http, respx_mock
):
    """Regression guard for 2.3.4: ``_discover_sort_attr_ids`` is
    cached for 5 min, so back-to-back invocations of the tool against
    the same project must NOT re-call ``list_user_story_attributes``
    (or ``list_epic_attributes``). Skipping these saves one round-trip
    each per repeat call — ~30% of the project-level overhead before
    the per-US fetch even starts."""
    story = _FakeUS(
        ref=7, points={}, total_points=2.0,
        attr_values={"1": 1, "2": 1, "3": 1},
    )
    project = _FakeProject(
        stories=[story],
        roles=[_FakeAttr(19, "Developer")],
        points=_baseline_points(),
        us_attrs=_baseline_attrs(),
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    _register_us_attr_routes(respx_mock, [story])

    sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    first_us_attr_calls = project.list_user_story_attributes_calls
    first_epic_attr_calls = project.list_epic_attributes_calls
    assert first_us_attr_calls == 1
    assert first_epic_attr_calls == 1

    sort_kanban_by_rice_tool.invoke({"project_slug": "wahed"})
    # Same project_slug, within 5-min TTL → discovery skipped.
    assert project.list_user_story_attributes_calls == first_us_attr_calls
    assert project.list_epic_attributes_calls == first_epic_attr_calls
