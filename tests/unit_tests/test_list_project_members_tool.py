import json

import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools import taiga_tools
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


# ---------------------------------------------------------------------------
# Privacy-default behavioural test.
#
# Why this test is heavier than the scaffold above: ``include_email`` is
# the only knob in this tool that has security consequences (other users'
# email addresses being shipped to the LLM/MCP client). The default of
# ``False`` is the contract we ship; if a future refactor flips that
# default or quietly leaks email through another field, this test fails
# loudly.
# ---------------------------------------------------------------------------


class _FakeUser:
    def __init__(self, uid, username, full_name, email):
        self.id = uid
        self.username = username
        self.full_name = full_name
        self.email = email


class _FakeMembership:
    def __init__(self, user_id, role_name, is_admin=False):
        self.user = user_id
        self.role_name = role_name
        self.is_admin = is_admin


class _FakeProject:
    def __init__(self, members, memberships):
        self.members = members
        self._memberships = memberships

    def list_memberships(self):
        return self._memberships


@pytest.fixture
def fake_project(monkeypatch):
    project = _FakeProject(
        members=[
            _FakeUser(1, "alice", "Alice A", "alice@example.com"),
            _FakeUser(2, "bob", "Bob B", "bob@example.com"),
        ],
        memberships=[
            _FakeMembership(1, "Product Owner", is_admin=True),
            _FakeMembership(2, "Developer"),
        ],
    )
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: project)
    return project


def test_include_email_defaults_to_false_no_email_field(fake_project):
    """Privacy-by-default: emails are NOT in the response unless asked for."""
    raw = list_project_members_tool.invoke({"project_slug": "p"})
    members = json.loads(raw)
    assert isinstance(members, list) and len(members) == 2
    for m in members:
        assert "email" not in m, (
            f"email leaked into response with default args: {m!r}"
        )
    # Sanity: the join still works — role_name reaches the user-level entry.
    by_user = {m["username"]: m for m in members}
    assert by_user["alice"]["role"] == "Product Owner"
    assert by_user["alice"]["is_admin"] is True
    assert by_user["bob"]["is_admin"] is False


def test_include_email_true_adds_email_from_user_object(fake_project):
    """Opt-in: emails are sourced from the User object, not Membership."""
    raw = list_project_members_tool.invoke(
        {"project_slug": "p", "include_email": True}
    )
    members = json.loads(raw)
    by_user = {m["username"]: m for m in members}
    assert by_user["alice"]["email"] == "alice@example.com"
    assert by_user["bob"]["email"] == "bob@example.com"


def test_project_not_accessible_returns_404_with_honest_wording(monkeypatch):
    """get_project swallows auth/permission/outage errors as None.
    The 404 message has to acknowledge the underlying ambiguity."""
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: None)
    raw = list_project_members_tool.invoke({"project_slug": "missing"})
    payload = json.loads(raw)
    assert payload["code"] == 404
    # Wording must NOT pretend it's only "not found" — it could be
    # auth/permission/outage from the swallowed exception.
    assert "not accessible" in payload["error"].lower()
