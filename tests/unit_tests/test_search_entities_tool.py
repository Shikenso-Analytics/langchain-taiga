import json
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import search_entities_tool
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """
    Automatically apply a fake OPENAI_API_KEY environment variable
    for each test function. That way, login() won't raise ValueError.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestSearchEntityUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return search_entities_tool

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
        return {"project_slug": "slug", "query": "query", "entity_type": "task"}


# ---------------------------------------------------------------------------
# Behavioural tests for v2.2 search_entities_tool fixes:
#   1. created_after / closed_before are tz-aware → no TypeError when
#      comparing against python-taiga's tz-aware ``entity.created_date``.
#   2. ``max_results`` is parameterised + the response carries a
#      ``truncated`` flag.
#   3. ``include_custom_attributes=False`` skips the per-match
#      ``fetch_entity`` N+1 round-trip and ``description_max_chars``
#      caps long bodies.
# ---------------------------------------------------------------------------


class _FakeEntity:
    def __init__(self, ref, subject="Test", description="", created=None,
                 finished=None, status=100, assigned_to=None, tags=None,
                 milestone=None):
        self.ref = ref
        self.subject = subject
        self.description = description
        self.status = status
        self.assigned_to = assigned_to
        self.tags = tags or []
        self.milestone = milestone
        self.created_date = created or datetime(
            2026, 4, 1, 12, 0, tzinfo=timezone.utc
        )
        self.finished_date = finished


class _FakeProject:
    id = 1
    members: list = []

    def __init__(self, entities):
        self._entities = entities

    def list_issues(self):
        return self._entities

    def list_user_stories(self, **kwargs):
        return self._entities

    def list_epics(self):
        return self._entities


@pytest.fixture
def fake_search_env(monkeypatch):
    """Stub out every helper search_entities_tool calls so the test
    is hermetic. Returns a small FakeProject preloaded with three
    issues; tests can then drive the LLM mock to inject filters."""
    entities = [
        _FakeEntity(
            ref=1,
            subject="alpha",
            created=datetime(2026, 2, 1, tzinfo=timezone.utc),
        ),
        _FakeEntity(
            ref=2,
            subject="beta",
            created=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
        _FakeEntity(
            ref=3,
            subject="gamma",
            description="x" * 1500,  # long, exercises truncation
            created=datetime(2026, 5, 1, tzinfo=timezone.utc),
        ),
    ]
    project = _FakeProject(entities)

    monkeypatch.setattr(taiga_tools, "get_project", lambda s: project)
    monkeypatch.setattr(
        taiga_tools, "list_all_statuses",
        lambda *a, **kw: {"issue_statuses": [], "us_statuses": [],
                          "task_statuses": [], "epic_statuses": []},
    )
    monkeypatch.setattr(taiga_tools, "list_all_tags", lambda s: [])
    monkeypatch.setattr(taiga_tools, "list_milestones", lambda s: [])
    monkeypatch.setattr(taiga_tools, "get_current_milestone", lambda s: None)
    monkeypatch.setattr(
        taiga_tools, "get_status", lambda *a, **kw: {"name": "Open"}
    )
    monkeypatch.setattr(
        taiga_tools, "get_user", lambda uid: {"username": f"user{uid}"}
    )
    monkeypatch.setattr(
        taiga_tools, "find_milestone_id", lambda *a, **kw: None
    )
    return entities


class _StubLLM:
    """Replaces the module-level ``small_llm`` (a Pydantic ChatOpenAI,
    whose attributes can't be monkey-patched directly) with a tiny
    object that returns a canned AIMessage on ``.invoke()``."""

    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages):
        return AIMessage(content=self._content)


def _patch_llm(monkeypatch, search_params: dict):
    """Force the LLM-parsed search_params to a known dict so tests
    deterministically exercise filter logic."""
    monkeypatch.setattr(
        taiga_tools, "small_llm", _StubLLM(json.dumps(search_params))
    )


def test_created_after_filter_no_typeerror_on_tz_aware_entities(
    fake_search_env, monkeypatch
):
    """Regression: tz-aware filter date vs tz-aware ``entity.created_date``
    used to raise TypeError ('can't compare offset-naive and offset-aware
    datetimes') and silently truncate results mid-loop."""
    _patch_llm(monkeypatch, {"created_after": "2026-03-01"})
    raw = search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "issues since March",
            "entity_type": "issue",
        }
    )
    payload = json.loads(raw)
    # February entity (ref=1) is filtered out, April + May pass.
    refs = [m["ref"] for m in payload["matches"]]
    assert refs == [2, 3]
    assert payload["count"] == 2
    assert payload["truncated"] is False


def test_max_results_caps_and_sets_truncated_flag(
    fake_search_env, monkeypatch
):
    """``max_results`` is now caller-controlled; the response surfaces
    a ``truncated`` bool so claude.ai can detect when more matches
    exist beyond the cap."""
    _patch_llm(monkeypatch, {})  # empty filters → all entities match
    raw = search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "all",
            "entity_type": "issue",
            "max_results": 2,
        }
    )
    payload = json.loads(raw)
    assert payload["count"] == 2
    assert payload["max_results"] == 2
    assert payload["truncated"] is True


def test_include_custom_attributes_default_false_skips_fetch_entity(
    fake_search_env, monkeypatch
):
    """N+1 guard: by default the tool MUST NOT call fetch_entity per
    match. A 200-match search would otherwise issue 200 extra HTTP
    round-trips against Taiga."""
    fetch_calls = []

    def _spy(*args, **kwargs):
        fetch_calls.append(args)
        return None

    monkeypatch.setattr(taiga_tools, "fetch_entity", _spy)
    monkeypatch.setattr(
        taiga_tools, "get_formatted_custom_attributes", lambda *a, **kw: []
    )
    _patch_llm(monkeypatch, {})

    search_entities_tool.invoke(
        {"project_slug": "p", "query": "all", "entity_type": "issue"}
    )
    assert fetch_calls == [], (
        f"fetch_entity called {len(fetch_calls)} times despite "
        "include_custom_attributes=False"
    )


def test_include_custom_attributes_true_does_fetch_entity(
    fake_search_env, monkeypatch
):
    """Opt-in path still works: callers who explicitly want custom
    attributes accept the per-match round-trip."""
    fetch_calls = []
    monkeypatch.setattr(
        taiga_tools, "fetch_entity",
        lambda *a, **kw: fetch_calls.append(a) or None,
    )
    monkeypatch.setattr(
        taiga_tools, "get_formatted_custom_attributes", lambda *a, **kw: []
    )
    _patch_llm(monkeypatch, {})

    search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "all",
            "entity_type": "issue",
            "include_custom_attributes": True,
        }
    )
    assert len(fetch_calls) == 3  # one per match


def test_description_truncated_to_description_max_chars(
    fake_search_env, monkeypatch
):
    """Long descriptions get truncated with a sentinel suffix so
    claude.ai can detect the truncation and re-fetch via
    get_entity_by_ref_tool when needed."""
    _patch_llm(monkeypatch, {})
    raw = search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "all",
            "entity_type": "issue",
            "description_max_chars": 100,
        }
    )
    payload = json.loads(raw)
    by_ref = {m["ref"]: m for m in payload["matches"]}
    # ref=3 had a 1500-char description; should now be capped + suffixed.
    assert by_ref[3]["description"].endswith("… [truncated]")
    assert len(by_ref[3]["description"]) == 100 + len("… [truncated]")
    # ref=1, ref=2 have short descriptions and remain untouched.
    assert "[truncated]" not in by_ref[1]["description"]


def test_description_max_chars_zero_disables_truncation(
    fake_search_env, monkeypatch
):
    """Escape hatch: ``description_max_chars=0`` returns full bodies."""
    _patch_llm(monkeypatch, {})
    raw = search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "all",
            "entity_type": "issue",
            "description_max_chars": 0,
        }
    )
    payload = json.loads(raw)
    by_ref = {m["ref"]: m for m in payload["matches"]}
    assert len(by_ref[3]["description"]) == 1500
    assert "[truncated]" not in by_ref[3]["description"]
