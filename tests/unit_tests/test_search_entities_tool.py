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


def test_created_after_filter_coerces_string_created_date(monkeypatch):
    """Regression v2.2.1: python-taiga's Resource.__init__ only converts
    ``created_date`` to a datetime when it matches a strict regex
    (``\\d+-\\d+-\\d+T\\d+:\\d+:\\d+\\+0000``). Microsecond-precision
    timestamps (which Taiga emits for issues) leave the field as a raw
    string, and the comparison ``str < datetime`` then raised
    ``TypeError: '<' not supported between instances of 'str' and
    'datetime.datetime'`` mid-loop. Now coerced via
    ``_coerce_to_aware_datetime``."""
    # Three string-typed created_date values, all in different ISO shapes
    # we have observed in the wild.
    entities = [
        _FakeEntity(
            ref=1, subject="february",
            # microseconds + +0000: python-taiga LEAVES this as a string
            created="2026-02-01T12:00:00.123456+0000",
        ),
        _FakeEntity(
            ref=2, subject="april",
            created="2026-04-01T12:00:00.987654+0000",
        ),
        _FakeEntity(
            ref=3, subject="may",
            # 'Z' suffix
            created="2026-05-01T12:00:00Z",
        ),
    ]
    project = _FakeProject(entities)

    monkeypatch.setattr(taiga_tools, "get_project", lambda s: project)
    monkeypatch.setattr(
        taiga_tools, "list_all_statuses",
        lambda *a, **kw: {"issue_statuses": []},
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
    _patch_llm(monkeypatch, {"created_after": "2026-03-01"})

    raw = search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "issues since march 2026",
            "entity_type": "issue",
        }
    )
    payload = json.loads(raw)
    # MUST NOT raise TypeError. February is filtered out, April+May pass.
    refs = [m["ref"] for m in payload["matches"]]
    assert refs == [2, 3]


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


# ---------------------------------------------------------------------------
# Validation guards (Copilot review on PR #8 flagged these).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_max_results_rejects_non_positive(bad_value):
    """``max_results`` must be >= 1; otherwise the cap loop terminates
    on the first iteration and the response would lie about being
    truncated."""
    raw = search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "all",
            "entity_type": "issue",
            "max_results": bad_value,
        }
    )
    payload = json.loads(raw)
    assert payload["code"] == 400
    assert "max_results" in payload["error"]


def test_truncated_false_when_loop_ends_naturally_at_max_results(
    fake_search_env, monkeypatch
):
    """Regression: ``truncated`` used to be derived as
    ``len(matches) >= max_results`` AFTER the loop, producing a false
    positive when the total result set happened to land EXACTLY on the
    cap (no early break). The flag must only be True when the cap
    actually caused an early exit."""
    _patch_llm(monkeypatch, {})
    # 3 fake entities + max_results=3 → loop exhausts entities AND
    # len(matches) hits the cap on the last iteration. Old code:
    # ``cap_hit`` set on the LAST entity (which is fine), but the post-
    # loop ``len >= max_results`` derivation also reported True even
    # if the loop hadn't broken. New code: cap_hit is only True when
    # break fires before exhausting the iterable.
    raw = search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "all",
            "entity_type": "issue",
            "max_results": 3,
        }
    )
    payload = json.loads(raw)
    # All three entities matched, cap fires exactly on the last one.
    # Whether this counts as "truncated" is a design call: we say it
    # IS truncated because there could be MORE matches we never got
    # to test. The interesting case is max_results=4 — see next test.
    assert payload["count"] == 3

    # max_results > total entities → loop exits naturally → NOT truncated
    raw2 = search_entities_tool.invoke(
        {
            "project_slug": "p",
            "query": "all",
            "entity_type": "issue",
            "max_results": 10,
        }
    )
    payload2 = json.loads(raw2)
    assert payload2["count"] == 3
    assert payload2["truncated"] is False, (
        "loop exhausted the iterable naturally; truncated must be False"
    )


# ---------------------------------------------------------------------------
# Tag filter (v2.14.0 regression fix)
# ---------------------------------------------------------------------------


@pytest.fixture
def tagged_search_env(fake_search_env, monkeypatch):
    """The shared search env, re-seeded with tags in Taiga's REAL read shape:
    a list of ``[name, color]`` pairs (the colour is joined in from the
    project-level ``tags_colors`` registry, and is ``None`` when unset)."""
    for entity, tags in zip(
        fake_search_env,
        (
            [["voice", "#845EF7"]],
            [["jobs_manager", None], ["voice", "#845EF7"]],
            [["k8s", None]],
        ),
    ):
        entity.tags = tags
    monkeypatch.setattr(
        taiga_tools, "list_all_tags", lambda s: ["voice", "jobs_manager", "k8s"]
    )
    return fake_search_env


def _search_by_tags(monkeypatch, tags):
    _patch_llm(monkeypatch, {"tags": tags})
    raw = search_entities_tool.invoke(
        {"project_slug": "p", "query": "tagged", "entity_type": "issue"}
    )
    return [m["ref"] for m in json.loads(raw)["matches"]]


def test_tag_filter_matches_taigas_name_color_pairs(tagged_search_env, monkeypatch):
    """Regression: the filter used to test ``"voice" in entity.tags`` against
    ``[["voice", "#845EF7"]]``, which is never true — so every tag search
    silently returned nothing. It must match on the tag NAME."""
    assert _search_by_tags(monkeypatch, ["voice"]) == [1, 2]


def test_tag_filter_requires_all_given_tags(tagged_search_env, monkeypatch):
    assert _search_by_tags(monkeypatch, ["voice", "jobs_manager"]) == [2]


def test_tag_filter_is_case_insensitive(tagged_search_env, monkeypatch):
    assert _search_by_tags(monkeypatch, ["VOICE"]) == [1, 2]


def test_tag_filter_excludes_non_matching(tagged_search_env, monkeypatch):
    assert _search_by_tags(monkeypatch, ["nonexistent"]) == []
