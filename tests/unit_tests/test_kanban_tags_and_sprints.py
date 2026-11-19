"""``get_kanban_board_tool``: the ``tags`` filter and a card's sprint (5.6.0).

A routine goes through the stories tagged for one person every day. Reading a whole column for that
(~900 stories, ~400 KB, to find 16) is the cost ``tags`` removes: the tags are asked of Taiga like
``statuses``. Taiga matches a story carrying ANY of the given tags (measured on a project with ~900
stories in one column: one tag 16, another 5, both together 20, one story carrying both).
The card's sprint (``id``, ``milestone``, ``milestone_name``, ``sprint_finish``) lets the caller
keep the stories of the sprints that run now.
"""

import json

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import get_kanban_board_tool
from tests.unit_tests.test_get_kanban_board_tool import (  # noqa: F401 - stub_get_user is an autouse fixture
    _FakeProject,
    _FakeStatus,
    _FakeUS,
    stub_get_user,
)

MILESTONES = [
    {"id": 262, "name": "Sprint 12", "closed": False, "estimated_start": "2026-11-17", "estimated_finish": "2026-12-15"},
    {"id": 254, "name": "Sprint 11", "closed": False, "estimated_start": "2026-10-16", "estimated_finish": "2026-11-16"},
]


def _board():
    project = _FakeProject(
        statuses=[_FakeStatus(1, "Sprint", order=1), _FakeStatus(2, "Done", order=2, is_closed=True)],
        stories=[_FakeUS(ref=501, status=1), _FakeUS(ref=502, status=1)],
    )
    for story, (story_id, milestone, name) in zip(
        project._stories, [(9001, 262, "Sprint 12"), (9002, 254, "Sprint 11")]
    ):
        story.id, story.milestone, story.milestone_name = story_id, milestone, name
        story.tags = [["alice", None]]
    return project


@pytest.fixture
def board(monkeypatch):
    project = _board()
    reads = []

    def milestones(slug):
        reads.append(slug)
        return MILESTONES

    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    monkeypatch.setattr(taiga_tools, "list_milestones", milestones)
    project.milestone_reads = reads
    return project


def _invoke(**kw):
    return json.loads(get_kanban_board_tool.invoke({"project_slug": "my-project", **kw}))


@pytest.mark.parametrize(
    "kw, sent",
    [
        ({"tags": ["alice"]}, {"tags": "alice"}),
        ({"tags": ["alice", " q4 "]}, {"tags": "alice,q4"}),
        ({"tags": ["alice"], "statuses": ["Sprint"], "include_closed": False}, {"status": "1", "tags": "alice"}),
        ({}, {}),  # control: without tags nothing is sent
    ],
)
def test_tags_are_asked_of_taiga(board, kw, sent):
    _invoke(**kw)
    assert board.queries == [sent]


def test_blank_tags_are_a_400_before_any_read(board):
    out = _invoke(tags=["  "])
    assert out["code"] == 400
    assert getattr(board, "queries", []) == []


def test_cards_carry_their_sprint(board):
    out = _invoke(
        tags=["alice"],
        fields=["columns.cards.ref", "columns.cards.id", "columns.cards.milestone", "columns.cards.milestone_name",
                "columns.cards.sprint_finish"],
    )
    assert out["columns"][0]["cards"] == [
        {"ref": 501, "id": 9001, "milestone": 262, "milestone_name": "Sprint 12", "sprint_finish": "2026-12-15"},
        {"ref": 502, "id": 9002, "milestone": 254, "milestone_name": "Sprint 11", "sprint_finish": "2026-11-16"},
    ]
    assert out["query"]["tags"] == ["alice"]


def test_sprints_read_only_when_sprint_finish_is_asked(board):
    _invoke(tags=["alice"], fields=["columns.cards.ref", "columns.cards.milestone_name"])
    assert board.milestone_reads == []
    _invoke(tags=["alice"], fields=["columns.cards.sprint_finish"])
    assert board.milestone_reads == ["my-project"]


def test_backlog_card_has_no_sprint(board):
    board._stories[1].milestone, board._stories[1].milestone_name = None, None
    out = _invoke(fields=["columns.cards.milestone", "columns.cards.sprint_finish"])
    assert out["columns"][0]["cards"][1] == {"milestone": None, "sprint_finish": None}


def test_tags_reach_every_board(monkeypatch):
    projects = {slug: _board() for slug in ("a", "b")}
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: projects[slug])
    out = json.loads(get_kanban_board_tool.invoke({"project_slugs": ["a", "b"], "tags": ["alice"]}))
    assert [p.queries for p in projects.values()] == [[{"tags": "alice"}], [{"tags": "alice"}]]
    assert out["query"]["tags"] == ["alice"]


def test_an_ancestor_path_never_reads_the_sprints(board):
    out = _invoke(tags=["alice"], fields=["columns"])
    assert board.milestone_reads == []
    card = out["columns"][0]["cards"][0]
    assert card["milestone_name"] == "Sprint 12" and "sprint_finish" not in card
