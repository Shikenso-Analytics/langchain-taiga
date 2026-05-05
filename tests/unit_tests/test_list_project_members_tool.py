import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools.taiga_tools import list_project_members_tool
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestListProjectMembersUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return list_project_members_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"project_slug": "shikenso-development"}
