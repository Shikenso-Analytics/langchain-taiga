import pytest
from langchain_core.tools import BaseTool

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
