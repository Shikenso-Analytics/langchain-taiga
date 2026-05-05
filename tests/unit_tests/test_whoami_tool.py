import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools.taiga_tools import whoami_tool
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """OPENAI_API_KEY needed at module-import time for the small_llm
    helper inside taiga_tools.py — same pattern as the other tool tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestWhoamiUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return whoami_tool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        # whoami_tool takes no arguments
        return {}
