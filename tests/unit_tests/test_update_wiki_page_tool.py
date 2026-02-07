import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools.taiga_tools import update_wiki_page_tool
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """
    Automatically apply a fake OPENAI_API_KEY environment variable
    for each test function. That way, login() won't raise ValueError.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestUpdateWikiPageUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return update_wiki_page_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {
            "project_slug": "slug",
            "wiki_slug": "home",
            "content": "# Updated Content\n\nNew content here.",
            "version": 1,
        }
