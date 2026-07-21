import json

import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import update_entity_by_ref_tool
from langchain_tests.unit_tests import ToolsUnitTests

@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """
    Automatically apply a fake OPENAI_API_KEY environment variable
    for each test function. That way, login() won't raise ValueError.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestUpdateEntityUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return update_entity_by_ref_tool

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
        return {"project_slug": "slug", "entity_ref": 555, "entity_type": "us",
                "description": "description", "status": "status", "due_data": "due_date"}


# ---------------------------------------------------------------------------
# Behavioural test for the write path. update_entity_by_ref_tool must issue a
# scoped PATCH of only the changed fields (+ the OCC ``version``) — NOT a full
# PUT via update() — so a regression that drops ``version``, omits a requested
# field, or reverts to update() is caught. Lightweight fakes keep it focused
# on the wire call.
# ---------------------------------------------------------------------------


class _Entity:
    def __init__(self):
        self.id = 1
        self.version = 3
        self.subject = "old"
        self.patch_fields = None
        self.patch_kwargs = None
        self.put_called = False

    def patch(self, fields, **kwargs):
        self.patch_fields = fields
        self.patch_kwargs = kwargs
        return self

    def update(self, **kwargs):  # must NOT be called
        self.put_called = True
        return self


def test_update_entity_issues_scoped_patch_not_full_put(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")
    entity = _Entity()
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: object())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda slug, etype, q: [42])
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q: [{"id": 7}])

    out = json.loads(
        update_entity_by_ref_tool.invoke(
            {
                "project_slug": "s",
                "entity_ref": 5,
                "entity_type": "us",
                "subject": "New subject",
                "description": "New description",
                "status": "Done",
                "assign_to": "bob",
                "due_date": "2026-01-01",
            }
        )
    )

    # Scoped PATCH carrying the OCC version, never a full PUT.
    assert entity.put_called is False
    assert entity.patch_fields == ["version"]
    assert entity.patch_kwargs == {
        "subject": "New subject",
        "description": "New description",
        "status": 42,           # resolved via find_status_ids
        "assigned_to": 7,       # resolved via find_users
        "due_date": "2026-01-01",
    }
    assert "updated successfully" in out["message"]
