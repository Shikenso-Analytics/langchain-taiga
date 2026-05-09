import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools.taiga_tools import (
    _format_userstory_points,
    get_entity_by_ref_tool,
)
from langchain_tests.unit_tests import ToolsUnitTests

@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """
    Automatically apply a fake OPENAI_API_KEY environment variable
    for each test function. That way, login() won't raise ValueError.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestGetEntityUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return get_entity_by_ref_tool

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"project_slug": "slug", "entity_ref": 555, "entity_type": "us"}


# ---------------------------------------------------------------------------
# Behavioural tests for ``_format_userstory_points`` — the helper that
# produces the ``points`` field added to the ``get_entity_by_ref_tool``
# response for user stories. Lightweight fakes keep the test focused on
# the {role_id: point_id} -> {role_name: point_value} translation; the
# tool's other heavy machinery (history, tasks, custom attributes) is
# tested elsewhere via integration paths.
# ---------------------------------------------------------------------------


class _Stub:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _project(roles, points):
    return _Stub(list_roles=lambda: roles, list_points=lambda: points)


def test_format_userstory_points_returns_role_name_to_value(monkeypatch):
    entity = _Stub(points={"19": 103, "20": 101})
    project = _project(
        roles=[_Stub(id=19, name="Developer"), _Stub(id=20, name="UX")],
        points=[
            _Stub(id=100, value=1),
            _Stub(id=101, value=2),
            _Stub(id=103, value=5),
        ],
    )
    assert _format_userstory_points(entity, project) == {
        "Developer": 5,
        "UX": 2,
    }


def test_format_userstory_points_empty_when_unset():
    entity = _Stub(points=None)
    project = _project(
        roles=[_Stub(id=19, name="Developer")],
        points=[_Stub(id=100, value=1)],
    )
    assert _format_userstory_points(entity, project) == {}


def test_format_userstory_points_drops_stale_role_id_and_unestimated():
    """A role-id key that's no longer present on the project (deleted
    role) and a point assignment to the "?" point (value=None) must
    both be silently dropped — consumers see only role-name keys with
    concrete numeric values."""
    entity = _Stub(points={"19": 103, "999": 100, "20": 351})
    project = _project(
        roles=[_Stub(id=19, name="Developer"), _Stub(id=20, name="UX")],
        points=[
            _Stub(id=100, value=1),
            _Stub(id=103, value=5),
            _Stub(id=351, value=None),  # the "?" unestimated point
        ],
    )
    out = _format_userstory_points(entity, project)
    # Only Developer survives:
    #   - role 999 not in project.list_roles() -> dropped
    #   - UX (role 20) points to "?" (value None)        -> dropped
    assert out == {"Developer": 5}
