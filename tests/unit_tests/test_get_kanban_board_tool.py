"""Unit tests for ``get_kanban_board_tool``.

The tool renders the user-story Kanban board the way the Taiga UI does:
status columns in ``order``, each holding its user stories sorted by
``kanban_order``. It reuses the same building blocks the rest of the
module already relies on — ``get_project`` (monkeypatched here),
``project.list_user_story_statuses()`` /
``project.list_user_stories()`` (fakes), and the cached ``get_user``
for assignee-name resolution (monkeypatched so no network call and so
call-count can be asserted).

Contract locked by these tests:
- columns ordered by status ``order``; empty columns still present.
- cards carry ref/subject/assigned_to/kanban_order, sorted by
  kanban_order (None sinks to the bottom).
- ``include_closed=False`` drops closed-status columns.
- unknown project → JSON ``{"error", "code": 404}``.
- a story whose status id matches no column (orphan — only possible in
  a cache-race the fresh single-call fetch avoids, but guarded anyway)
  is surfaced under ``orphan_cards``, never silently dropped.
"""

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_kanban_board_tool


class _FakeStatus:
    """Mirrors the python-taiga UserStoryStatus attributes the tool reads.

    ``allowed_params = ["color", "is_closed", "name", "order",
    "project", "wip_limit"]`` (+ ``id`` from the base resource), read as
    plain attributes.
    """

    def __init__(self, sid, name, order, is_closed=False, wip_limit=None):
        self.id = sid
        self.name = name
        self.order = order
        self.is_closed = is_closed
        self.wip_limit = wip_limit


class _FakeUS:
    """Mirrors the python-taiga UserStory attributes the tool reads."""

    def __init__(self, ref, status, kanban_order=0, assigned_to=None):
        self.ref = ref
        self.subject = f"US {ref}"
        self.status = status
        self.kanban_order = kanban_order
        self.assigned_to = assigned_to


class _FakeUser:
    """Mirrors the python-taiga User attributes read off ``project.members``."""

    def __init__(self, uid, username):
        self.id = uid
        self.username = username


class _FakeProject:
    name = "Sourcing"

    def __init__(self, statuses, stories, members=None):
        self._statuses = statuses
        self._stories = stories
        # ``project.members`` is parser-hydrated on the real Project — the
        # tool reads it to resolve assignee usernames without an API call.
        self.members = members or []

    def list_user_story_statuses(self):
        return self._statuses

    def list_user_stories(self):
        return self._stories


@pytest.fixture(autouse=True)
def stub_get_user(monkeypatch):
    """Resolve any assignee id to ``user<id>`` and count lookups so tests
    can assert the tool skips the call for unassigned cards."""
    calls = []

    def _fake(user_id):
        calls.append(user_id)
        return {"username": f"user{user_id}"}

    monkeypatch.setattr(taiga_tools, "get_user", _fake)
    return calls


def _invoke(slug="sourcing", **kw):
    return json.loads(get_kanban_board_tool.invoke({"project_slug": slug, **kw}))


def test_groups_stories_into_ordered_status_columns(monkeypatch):
    project = _FakeProject(
        statuses=[
            _FakeStatus(2, "In progress", order=2),
            _FakeStatus(1, "New", order=1),
        ],
        stories=[
            _FakeUS(ref=10, status=1),
            _FakeUS(ref=11, status=2),
            _FakeUS(ref=12, status=1),
        ],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    payload = _invoke()

    assert payload["project"] == "Sourcing"
    cols = payload["columns"]
    # Columns follow status ``order``, not insertion order.
    assert [c["status"] for c in cols] == ["New", "In progress"]
    assert [c["status_id"] for c in cols] == [1, 2]
    assert {card["ref"] for card in cols[0]["cards"]} == {10, 12}
    assert {card["ref"] for card in cols[1]["cards"]} == {11}


def test_cards_sorted_by_kanban_order(monkeypatch):
    project = _FakeProject(
        statuses=[_FakeStatus(1, "New", order=1)],
        stories=[
            _FakeUS(ref=1, status=1, kanban_order=30),
            _FakeUS(ref=2, status=1, kanban_order=10),
            _FakeUS(ref=3, status=1, kanban_order=None),
            _FakeUS(ref=4, status=1, kanban_order=20),
        ],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    cards = _invoke()["columns"][0]["cards"]
    # Ascending kanban_order; the unordered (None) card sinks last.
    assert [c["ref"] for c in cards] == [2, 4, 1, 3]


def test_empty_column_is_included(monkeypatch):
    project = _FakeProject(
        statuses=[
            _FakeStatus(1, "New", order=1),
            _FakeStatus(2, "Ready", order=2),
        ],
        stories=[_FakeUS(ref=1, status=1)],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    cols = _invoke()["columns"]
    assert len(cols) == 2
    assert cols[1]["status"] == "Ready"
    assert cols[1]["cards"] == []


def test_wip_limit_is_passed_through(monkeypatch):
    project = _FakeProject(
        statuses=[_FakeStatus(1, "New", order=1, wip_limit=5)],
        stories=[],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    col = _invoke()["columns"][0]
    assert col["wip_limit"] == 5
    assert col["is_closed"] is False


def test_include_closed_false_drops_closed_columns(monkeypatch):
    project = _FakeProject(
        statuses=[
            _FakeStatus(1, "New", order=1),
            _FakeStatus(2, "Done", order=2, is_closed=True),
        ],
        stories=[
            _FakeUS(ref=1, status=1),
            _FakeUS(ref=2, status=2),
        ],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    # Default includes the closed column.
    assert [c["status"] for c in _invoke()["columns"]] == ["New", "Done"]

    # Filtered out on request; the closed card is not resurfaced as an
    # orphan (its column was intentionally hidden, not missing).
    filtered = _invoke(include_closed=False)
    assert [c["status"] for c in filtered["columns"]] == ["New"]
    assert "orphan_cards" not in filtered


def test_member_assignee_resolved_without_api_call(monkeypatch, stub_get_user):
    project = _FakeProject(
        statuses=[_FakeStatus(1, "New", order=1)],
        stories=[
            _FakeUS(ref=1, status=1, assigned_to=None),
            _FakeUS(ref=2, status=1, assigned_to=42),
        ],
        members=[_FakeUser(42, "alice")],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    cards = {c["ref"]: c for c in _invoke()["columns"][0]["cards"]}
    assert cards[1]["assigned_to"] is None
    assert cards[2]["assigned_to"] == "alice"
    # Resolved from the in-memory member list — no per-user API call at all.
    assert stub_get_user == []


def test_ex_member_assignee_falls_back_to_get_user(monkeypatch, stub_get_user):
    project = _FakeProject(
        statuses=[_FakeStatus(1, "New", order=1)],
        # 7 is assigned but no longer a project member (still stamped on the
        # old story) → must fall back to the cached get_user lookup.
        stories=[_FakeUS(ref=1, status=1, assigned_to=7)],
        members=[_FakeUser(42, "alice")],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    card = _invoke()["columns"][0]["cards"][0]
    assert card["assigned_to"] == "user7"
    assert stub_get_user == [7]


def test_unknown_project_returns_404(monkeypatch):
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: None)

    payload = _invoke(slug="nope")
    assert payload["code"] == 404
    assert "nope" in payload["error"]


def test_unexpected_error_returns_500(monkeypatch):
    class _Boom:
        name = "x"

        def list_user_story_statuses(self):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Boom())

    payload = _invoke()
    assert payload["code"] == 500
    assert "kaboom" in payload["error"]


def test_orphan_status_card_is_surfaced_not_dropped(monkeypatch):
    project = _FakeProject(
        statuses=[_FakeStatus(1, "New", order=1)],
        stories=[
            _FakeUS(ref=1, status=1),
            _FakeUS(ref=99, status=777),  # status id with no column
        ],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)

    payload = _invoke()
    # The valid card is placed; the orphan is not lost.
    assert {c["ref"] for c in payload["columns"][0]["cards"]} == {1}
    assert len(payload["orphan_cards"]) == 1
    orphan = payload["orphan_cards"][0]
    assert orphan["ref"] == 99
    assert orphan["status_id"] == 777
