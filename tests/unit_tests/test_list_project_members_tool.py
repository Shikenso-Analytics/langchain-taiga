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
        return {"project_slug": "my-project"}


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
    def __init__(self, uid, username, full_name, email, gravatar_id=None):
        self.id = uid
        self.username = username
        self.full_name = full_name
        self.email = email
        self.gravatar_id = gravatar_id


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


# md5 of the lower-cased address, which is what Taiga sends as gravatar_id.
ALICE_GRAVATAR = "c160f8cc69a4f0bf2b0362752353d060"  # alice@example.com
BOB_GRAVATAR = "4b9bb80620f03eb3719e0a061c14283d"  # bob@example.com


@pytest.fixture
def fake_project(monkeypatch):
    project = _FakeProject(
        members=[
            _FakeUser(1, "alice", "Alice A", "alice@example.com", ALICE_GRAVATAR),
            _FakeUser(2, "bob", "Bob B", "bob@example.com", BOB_GRAVATAR),
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


# ---------------------------------------------------------------------------
# Lookup by e-mail. Taiga hides other members' e-mails from non-admins but
# shows every member's gravatar id, the MD5 of the lower-cased address, so
# ``email`` matches on that hash and must never put an address in the output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "address, usernames",
    [("  Bob@Example.COM ", ["bob"]), ("carol@example.com", [])],
)
def test_email_returns_only_the_member_with_that_address(
    fake_project, address, usernames
):
    members = json.loads(
        list_project_members_tool.invoke({"project_slug": "p", "email": address})
    )
    assert [m["username"] for m in members] == usernames
    assert all("email" not in m for m in members)


def test_email_matches_on_the_gravatar_id_not_on_a_visible_email(fake_project):
    # What a non-admin caller gets from Taiga: no e-mails, only gravatar ids.
    for user in fake_project.members:
        user.email = None
    raw = list_project_members_tool.invoke(
        {"project_slug": "p", "email": "alice@example.com"}
    )
    assert [m["username"] for m in json.loads(raw)] == ["alice"]


def test_email_skips_members_without_a_gravatar_id(fake_project):
    for user in fake_project.members:
        user.gravatar_id = None
    raw = list_project_members_tool.invoke(
        {"project_slug": "p", "email": "alice@example.com"}
    )
    assert json.loads(raw) == []


def test_email_without_a_match_skips_the_memberships_request(fake_project):
    def _no_call():
        raise AssertionError("memberships fetched although no member matched")

    fake_project.list_memberships = _no_call
    raw = list_project_members_tool.invoke(
        {"project_slug": "p", "email": "carol@example.com"}
    )
    assert json.loads(raw) == []


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "alice",
        "alice @example.com",
        "a@b.com, c@d.com",
        "alice@example.com,bob@example.com",
        "alice@example.com;bob@example.com",
        "@example.com",
        "alice@",
        "a@b@example.com",
    ],
)
def test_email_that_is_not_one_address_is_a_400(fake_project, bad):
    payload = json.loads(
        list_project_members_tool.invoke({"project_slug": "p", "email": bad})
    )
    assert payload["code"] == 400


def test_email_and_include_email_still_return_the_address_only_when_asked(fake_project):
    raw = list_project_members_tool.invoke(
        {"project_slug": "p", "email": "alice@example.com", "include_email": True}
    )
    assert json.loads(raw) == [
        {
            "user_id": 1,
            "username": "alice",
            "full_name": "Alice A",
            "role": "Product Owner",
            "is_admin": True,
            "email": "alice@example.com",
        }
    ]
