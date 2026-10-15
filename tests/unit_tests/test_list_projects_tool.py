"""``list_projects_tool``: which projects you can see, and what you may do in each (2.23.0).

Slugs are not always derivable from names (a project renamed to "Team · Data" keeps its slug
``team-data-platform-legacy``), and before this no tool listed projects at all.
"""

import json
from types import SimpleNamespace

import pytest
from taiga.exceptions import TaigaRestException

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import list_projects_tool


def _project(pid, slug, name, admin=False, perms=("view_us",)):
    return SimpleNamespace(
        id=pid,
        slug=slug,
        name=name,
        is_private=True,
        i_am_admin=admin,
        i_am_member=True,
        my_permissions=list(perms),
    )


PROJECTS = [
    _project(40, "team-mobile", "Team · Mobile"),
    _project(3, "my-project", "My Project", admin=True),
    _project(
        38,
        "Team-Data-Platform-Legacy",
        "Team · Data",
        perms=("view_us", "modify_us", "add_task"),
    ),
]


@pytest.fixture
def env(monkeypatch):
    api = SimpleNamespace(projects=SimpleNamespace(list=lambda: list(PROJECTS)))
    monkeypatch.setattr(taiga_tools, "get_taiga_api", lambda token=None: api)
    return api


def _list(**kw):
    return json.loads(list_projects_tool.invoke({"compact": True, **kw}))


def test_prefix_filters_any_case_and_sorts_by_slug(env):
    out = _list(slug_prefix="TEAM-")
    assert [p["slug"] for p in out] == ["Team-Data-Platform-Legacy", "team-mobile"]
    assert out[0] == {
        "id": 38,
        "slug": "Team-Data-Platform-Legacy",
        "name": "Team · Data",
        "is_private": True,
        "i_am_admin": False,
        "i_am_member": True,
        "my_permissions": ["add_task", "modify_us", "view_us"],
    }


def test_without_a_prefix_every_visible_project(env):
    assert len(_list()) == 3


def test_fields_cut_the_entries(env):
    assert _list(slug_prefix="my-", fields=["slug", "i_am_admin"]) == [
        {"slug": "my-project", "i_am_admin": True}
    ]


def test_an_unknown_field_is_a_400(env):
    assert _list(fields=["owner"])["code"] == 400


def test_a_refused_listing_keeps_taigas_code(monkeypatch):
    def refuse():
        raise TaigaRestException("u", 401, '{"_error_message": "Invalid token"}', "GET")

    monkeypatch.setattr(
        taiga_tools, "get_taiga_api", lambda token=None: SimpleNamespace(projects=SimpleNamespace(list=refuse))
    )
    assert _list()["code"] == 401
