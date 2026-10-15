"""Custom attributes by name, with dropdown values checked before anything is written (2.23.0).

Attribute ids differ per project, and Taiga stores any string in a dropdown attribute. A value
spelled differently from the dropdown ("Data Platform (Legacy)" for "Data Platform") is stored
without complaint and matches nothing that reads it.
"""

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    _AttributeInputError,
    _resolve_custom_attributes,
)

TEAMS = "Marketing\nBackend\r\nData Platform\n"


class _Attr:
    def __init__(self, attr_id, name, attr_type="text", extra=None):
        self.id = attr_id
        self.name = name
        self.type = attr_type
        self.extra = extra
        self.description = ""


class _Project:
    id = 39
    name = "Team · Backend"

    def __init__(self):
        self.task_reads = 0
        self.task_attrs = [
            _Attr(23, "Supplier team", "dropdown", TEAMS),
            _Attr(24, "Cadence", "dropdown", "one-off\nrecurring"),
            _Attr(26, "Supplier status"),
            _Attr(27, "2026"),
        ]

    def list_task_attributes(self):
        self.task_reads += 1
        return list(self.task_attrs)

    def list_user_story_attributes(self):
        return [_Attr(91, "Next team", "dropdown", TEAMS), _Attr(87, "Escalation")]


@pytest.fixture(autouse=True)
def fresh_cache():
    taiga_tools.custom_attr_definitions_cache.clear()
    yield
    taiga_tools.custom_attr_definitions_cache.clear()


def test_names_in_any_case_and_ids_resolve():
    by_id, by_name = _resolve_custom_attributes(_Project(), "task", {"supplier TEAM": "backend", "24": "Recurring"})
    assert by_id == {"23": "Backend", "24": "recurring"}
    assert by_name == {"Supplier team": "Backend", "Cadence": "recurring"}


def test_whitespace_and_case_store_the_exact_choice():
    by_id, _ = _resolve_custom_attributes(_Project(), "task", {"Supplier team": "  data platform "})
    assert by_id == {"23": "Data Platform"}


def test_a_numeric_name_resolves_by_name_and_a_real_id_wins():
    by_id, _ = _resolve_custom_attributes(_Project(), "task", {"2026": "x", "26": "y"})
    assert by_id == {"27": "x", "26": "y"}


def test_unknown_attribute_lists_the_valid_names():
    with pytest.raises(_AttributeInputError) as caught:
        _resolve_custom_attributes(_Project(), "task", {"Suplier team": "Backend"})
    assert caught.value.code == 400
    assert "Suplier team" in str(caught.value)
    assert "Supplier team" in str(caught.value) and "Cadence" in str(caught.value)


def test_a_value_outside_the_choices_is_refused_with_the_choices():
    with pytest.raises(_AttributeInputError) as caught:
        _resolve_custom_attributes(_Project(), "task", {"Supplier team": "Data Platform (Legacy)"})
    assert caught.value.code == 400
    assert "Data Platform" in str(caught.value)


def test_none_and_empty_clear_without_a_choice_check():
    by_id, _ = _resolve_custom_attributes(_Project(), "task", {"Supplier team": None, "Cadence": ""})
    assert by_id == {"23": None, "24": ""}


def test_free_text_is_not_checked():
    by_id, _ = _resolve_custom_attributes(_Project(), "task", {"Supplier status": "anything"})
    assert by_id == {"26": "anything"}


def test_an_attribute_added_since_the_last_write_resolves_at_once():
    project = _Project()
    _resolve_custom_attributes(project, "task", {"Cadence": "one-off"})
    assert project.task_reads == 1
    project.task_attrs.append(_Attr(25, "Linked item"))
    by_id, _ = _resolve_custom_attributes(project, "task", {"Linked item": "#4"})
    assert by_id == {"25": "#4"}
    assert project.task_reads == 2


def test_every_write_checks_against_fresh_definitions():
    """A choice added or renamed within the cache TTL must count at once, in both directions."""
    project = _Project()
    _resolve_custom_attributes(project, "task", {"Cadence": "one-off"})
    project.task_attrs[1] = _Attr(24, "Cadence", "dropdown", "one-off\nweekly")
    by_id, _ = _resolve_custom_attributes(project, "task", {"Cadence": "weekly"})
    assert by_id == {"24": "weekly"}
    with pytest.raises(_AttributeInputError):
        _resolve_custom_attributes(project, "task", {"Cadence": "recurring"})


def test_choices_given_as_a_list_are_checked_too():
    """Taiga hands dropdown options back as a JSON list for attributes created through the API
    (measured live), not as the newline string the admin UI stores."""
    from langchain_taiga.tools.taiga_tools import _dropdown_choices

    assert _dropdown_choices("dropdown", [" Frontend ", "Backend", ""]) == ["Frontend", "Backend"]
    project = _Project()
    project.task_attrs[0] = _Attr(23, "Supplier team", "dropdown", ["Frontend", "Backend"])
    with pytest.raises(_AttributeInputError) as caught:
        _resolve_custom_attributes(project, "task", {"Supplier team": "nonsense"})
    assert caught.value.code == 400 and "Backend" in str(caught.value)
    by_id, _ = _resolve_custom_attributes(project, "task", {"Supplier team": " backend"})
    assert by_id == {"23": "Backend"}


def test_unreadable_definitions_are_a_502_not_unknown():
    project = _Project()

    def broken():
        raise RuntimeError("connection reset")

    project.list_task_attributes = broken
    with pytest.raises(_AttributeInputError) as caught:
        _resolve_custom_attributes(project, "task", {"Cadence": "one-off"})
    assert caught.value.code == 502


def test_listing_still_parses_choices_the_same_way():
    from langchain_taiga.tools.taiga_tools import _dropdown_choices

    assert _dropdown_choices("dropdown", TEAMS) == ["Marketing", "Backend", "Data Platform"]
    assert _dropdown_choices("text", TEAMS) == []


# -- set_custom_attributes_tool: names or ids, checked first, one PATCH ----------------------------

import json  # noqa: E402

from taiga.exceptions import TaigaRestException  # noqa: E402

from langchain_taiga.tools.taiga_tools import set_custom_attributes_tool  # noqa: E402


class _Requester:
    def __init__(self, fail=None):
        self.patches = []
        self.fail = fail

    def patch(self, uri, payload=None, query=None, **params):
        if self.fail:
            raise self.fail
        self.patches.append((uri.format(**params), payload))

        class _Response:
            @staticmethod
            def json():
                return {"attributes_values": payload["attributes_values"], "version": payload["version"] + 1}

        return _Response()


class _Task:
    endpoint = "tasks"
    id = 7001
    ref = 12

    def __init__(self, fail=None):
        self.requester = _Requester(fail)

    def get_attributes(self):
        return {"attributes_values": {"26": "Backend · Next · since 2026-01-14"}, "version": 4}


@pytest.fixture
def task_env(monkeypatch):
    state = {"project": _Project(), "task": _Task()}
    monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://taiga.test")
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: state["project"])
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: state["task"])
    return state


def _set(attributes):
    return json.loads(
        set_custom_attributes_tool.invoke(
            {"project_slug": "team-backend", "entity_ref": 12, "entity_type": "task", "attributes": attributes}
        )
    )


def test_several_attributes_are_one_patch_with_the_version(task_env):
    out = _set({"Supplier team": "backend", "24": "one-off"})
    assert task_env["task"].requester.patches == [
        (
            "/tasks/custom-attributes-values/7001",
            {
                "attributes_values": {
                    "26": "Backend · Next · since 2026-01-14",
                    "23": "Backend",
                    "24": "one-off",
                },
                "version": 4,
            },
        )
    ]
    assert out["updated"] is True
    assert out["attributes_resolved"] == {"Supplier team": "Backend", "Cadence": "one-off"}
    assert out["attributes_set"] == {"Supplier team": "backend", "24": "one-off"}


def test_a_bad_value_writes_nothing(task_env):
    out = _set({"Cadence": "one-off", "Supplier team": "Data Platform (Legacy)"})
    assert out["code"] == 400
    assert task_env["task"].requester.patches == []


def test_empty_attributes_are_refused(task_env):
    out = _set({})
    assert out["code"] == 400


def test_a_refused_patch_keeps_taigas_403(task_env):
    task_env["task"] = _Task(fail=TaigaRestException("u", 403, '{"_error_message": "No permission"}', "PATCH"))
    out = _set({"Cadence": "one-off"})
    assert out["code"] == 403
