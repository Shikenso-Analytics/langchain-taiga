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
                 milestone=None, owner=None, owner_extra_info=None,
                 status_extra_info=None, milestone_name=None,
                 assigned_to_extra_info=None):
        self.owner = owner
        if owner_extra_info is not None:
            self.owner_extra_info = owner_extra_info
        # Set only when supplied, so a test can exercise the "Taiga did not
        # send this key" branch — which is the real shape for epics
        # (no milestone at all) and for issues (id but no inline name).
        if status_extra_info is not None:
            self.status_extra_info = status_extra_info
        if milestone_name is not None:
            self.milestone_name = milestone_name
        if assigned_to_extra_info is not None:
            self.assigned_to_extra_info = assigned_to_extra_info
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
    # ``_list_project_entities`` constructs the resource managers with this.
    requester = object()

    def __init__(self, entities):
        self._entities = entities
        # Records what got pushed server-side, so a test can assert on the
        # query params rather than only on the filtered result.
        self.list_user_stories_kwargs: list = []
        # Same, for the issue/epic managers.
        self.list_endpoint_kwargs: list = []

    def list_user_stories(self, **kwargs):
        self.list_user_stories_kwargs.append(kwargs)
        return self._entities

    # Deliberately fail loudly instead of returning entities. python-taiga
    # declares both of these ``(self)`` — they accept no queryparams — so
    # calling them means every filter silently degrades to a full-project
    # scan (139 sequential pages / 9.4 MB on shikenso-development) and is
    # then applied client-side. That is the regression this guards.
    def list_issues(self):
        raise AssertionError(
            "search_entities_tool must not call project.list_issues(): it takes no "
            "queryparams, so filters degrade to a full-project scan. Route through "
            "_list_project_entities()."
        )

    def list_epics(self):
        raise AssertionError(
            "search_entities_tool must not call project.list_epics(): it takes no "
            "queryparams, so filters degrade to a full-project scan. Route through "
            "_list_project_entities()."
        )


class _FakeListEndpoint:
    """Stand-in for python-taiga's ``Issues`` / ``Epics`` resource managers.

    ``_list_project_entities`` does ``Issues(project.requester).list(
    project=..., **filters)``, so instances are both the class being
    constructed and the manager being called. Recorded kwargs are exactly
    what would have gone to the REST API as query params.

    It ENFORCES the params it is given, not just records them. A fake that
    recorded and ignored them could not tell a correct pushdown from one
    that returns a different row set than the client predicate that follows
    it — the test would pass either way. ``server_is_closed`` models what
    the server knows about a row's status, deliberately separate from the
    ``status_extra_info`` blob the row carries, so a row can be open to the
    server while its closedness is unknowable to the client.
    """

    def __init__(self, entities, recorder):
        self._entities = entities
        self._recorder = recorder

    def __call__(self, _requester):
        return self

    def list(self, **kwargs):
        self._recorder.append(kwargs)
        rows = self._entities
        if kwargs.get("status__is_closed") == "false":
            rows = [e for e in rows if getattr(e, "server_is_closed", False) is not True]
        if "owner" in kwargs:
            rows = [e for e in rows if e.owner == kwargs["owner"]]
        if "assigned_to" in kwargs:
            rows = [e for e in rows if e.assigned_to == kwargs["assigned_to"]]
        return rows


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
    endpoint = _FakeListEndpoint(entities, project.list_endpoint_kwargs)

    monkeypatch.setattr(taiga_tools, "get_project", lambda s: project)
    monkeypatch.setattr(taiga_tools, "Issues", endpoint)
    monkeypatch.setattr(taiga_tools, "Epics", endpoint)
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
    endpoint = _FakeListEndpoint(entities, project.list_endpoint_kwargs)

    monkeypatch.setattr(taiga_tools, "get_project", lambda s: project)
    monkeypatch.setattr(taiga_tools, "Issues", endpoint)
    monkeypatch.setattr(taiga_tools, "Epics", endpoint)
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


# ---------------------------------------------------------------------------
# Owner (creator) filter + output field (v2.15.0)
# ---------------------------------------------------------------------------


def _extra(uid, username, full_name):
    """Taiga's embedded ``owner_extra_info`` blob, trimmed to the keys the
    code reads. The real payload also carries photo/gravatar/is_active."""
    return {"id": uid, "username": username, "full_name_display": full_name}


@pytest.fixture
def owner_search_env(fake_search_env, monkeypatch):
    """The shared search env, re-seeded with owners. Two people, and ref=3
    is owned by one but assigned to the other — the case the whole feature
    exists for."""
    owners = [
        (5, _extra(5, "Wahed", "Dr. Wahed Hemati"), None),
        (51, _extra(51, "Whemati", "Walid Hemati"), None),
        (5, _extra(5, "Wahed", "Dr. Wahed Hemati"), 51),
    ]
    for entity, (owner, extra, assigned) in zip(fake_search_env, owners):
        entity.owner = owner
        entity.owner_extra_info = extra
        entity.assigned_to = assigned

    monkeypatch.setattr(
        taiga_tools,
        "find_users",
        lambda slug, q=None: [{"id": 5, "username": "Wahed"}],
    )
    return fake_search_env


def _search(monkeypatch, params, entity_type="issue"):
    raw = search_entities_tool.invoke(
        {"project_slug": "p", "query": "q", "entity_type": entity_type}
    )
    return json.loads(raw)


def test_search_reports_the_owner_of_every_match(owner_search_env, monkeypatch):
    _patch_llm(monkeypatch, {})
    out = _search(monkeypatch, {})
    assert [m["owner"] for m in out["matches"]] == ["Wahed", "Whemati", "Wahed"]


def test_owner_needs_no_extra_api_call(owner_search_env, monkeypatch):
    """``owner_extra_info`` already carries the username, so resolving it
    must not add a lookup — unlike ``assigned_to``, which still needs one
    per distinct user."""
    calls = []
    monkeypatch.setattr(
        taiga_tools, "get_user", lambda uid: calls.append(uid) or {"username": f"user{uid}"}
    )
    _patch_llm(monkeypatch, {})

    out = _search(monkeypatch, {})

    assert [m["owner"] for m in out["matches"]] == ["Wahed", "Whemati", "Wahed"]
    assert calls == [51]  # only ref=3's assignee, nothing for any owner


def test_owner_filter_narrows_to_that_creator(owner_search_env, monkeypatch):
    _patch_llm(monkeypatch, {"owner": "Wahed"})
    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [1, 3]


def test_owner_filter_matching_nobody_returns_nothing(owner_search_env, monkeypatch):
    """An unresolvable name must not degrade to 'no filter' — that would
    hand back the whole project as if it all matched. Same silent-no-op
    class of bug that left the tag filter dead until 2.14.0."""
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: [])
    _patch_llm(monkeypatch, {"owner": "ghost"})

    assert _search(monkeypatch, {})["matches"] == []


def test_owner_and_assigned_to_are_independent_filters(owner_search_env, monkeypatch):
    """'I filed it, someone else owns it now' has to be expressible. ref=2
    gives this teeth: it is also owned by 51, so an implementation that
    ignored ``owner`` and leaned on ``assigned_to`` alone would differ."""
    monkeypatch.setattr(
        taiga_tools,
        "find_users",
        lambda slug, q=None: (
            [{"id": 5, "username": "Wahed"}] if q == "Wahed" else [{"id": 51, "username": "Whemati"}]
        ),
    )
    _patch_llm(monkeypatch, {"owner": "Wahed", "assigned_to": "Whemati"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [3]


def test_owner_filter_is_pushed_server_side_for_userstories(owner_search_env, monkeypatch):
    """``list_user_stories`` forwards **queryparams to the REST API, which
    accepts ``owner=<id>``. On the production project this is a 1005 -> 286
    cut before anything is scanned client-side."""
    _patch_llm(monkeypatch, {"owner": "Wahed"})
    project = taiga_tools.get_project("p")

    _search(monkeypatch, {}, entity_type="userstory")

    assert project.list_user_stories_kwargs == [{"owner": 5}]


def test_owner_filter_is_never_pushed_down_when_searching_tasks(owner_search_env, monkeypatch):
    """Tasks are reached by walking user stories, so pushing ``owner`` onto
    that walk would select on the *story's* creator and silently drop every
    task filed by that person under somebody else's story."""
    for entity in owner_search_env:
        entity.list_tasks = lambda: []
        entity.is_closed = False
    _patch_llm(monkeypatch, {"owner": "Wahed"})
    project = taiga_tools.get_project("p")

    _search(monkeypatch, {}, entity_type="task")

    assert project.list_user_stories_kwargs == [{}]


def test_owner_filter_still_finds_a_departed_members_tickets(owner_search_env, monkeypatch):
    """``find_users`` only searches ``project.members``, so someone who has
    left resolves to nobody — while their tickets keep carrying the
    original owner blob. Reporting "they filed nothing" is a silent wrong
    answer, and chasing a leaver's tickets is a real reason to search."""
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: [])
    _patch_llm(monkeypatch, {"owner": "Whemati"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [2]


def test_owner_filter_matches_a_departed_member_by_display_name(owner_search_env, monkeypatch):
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: [])
    _patch_llm(monkeypatch, {"owner": "walid hemati"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [2]


def test_owner_filter_accepts_a_bare_user_id_without_any_lookup(owner_search_env, monkeypatch):
    """A numeric query is already the exact answer. Routing it through
    ``find_users`` would be wrong, not just wasteful: that prompt matches
    ids by CONTAINMENT, so "51" also resolves 151 — returning a second
    person's tickets and, with two ids, losing the pushdown."""
    calls = []
    monkeypatch.setattr(
        taiga_tools,
        "find_users",
        lambda slug, q=None: calls.append(q) or [{"id": 51}, {"id": 151}],
    )
    _patch_llm(monkeypatch, {"owner": "51"})
    project = taiga_tools.get_project("p")

    out = _search(monkeypatch, {}, entity_type="userstory")

    assert calls == []  # no fuzzy lookup at all
    assert [m["ref"] for m in out["matches"]] == [2]
    assert project.list_user_stories_kwargs == [{"owner": 51}]  # pushdown kept


def test_owner_filter_reports_a_broken_user_lookup_instead_of_crashing(
    owner_search_env, monkeypatch
):
    """``find_users`` is annotated ``-> List[Dict]`` but returns a plain
    STRING on both LLM failure paths. Iterating it yields characters and
    raises TypeError on ``u["id"]``, killing the whole tool call."""
    monkeypatch.setattr(
        taiga_tools, "find_users", lambda slug, q=None: "Error decoding LLM response: boom"
    )
    _patch_llm(monkeypatch, {"owner": "Wahed"})

    out = _search(monkeypatch, {})

    assert out["code"] == 500
    assert "Owner lookup failed" in out["error"]


def test_owner_lookup_provider_error_is_reported_not_raised(owner_search_env, monkeypatch):
    """``find_users`` invokes the LLM outside its own try, so a provider
    timeout propagates — and this block sits outside the entity-listing
    try, so it would kill the whole tool call."""
    def _boom(slug, q=None):
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(taiga_tools, "find_users", _boom)
    _patch_llm(monkeypatch, {"owner": "Wahed"})

    out = _search(monkeypatch, {})

    assert out["code"] == 500
    assert "Owner lookup failed" in out["error"]


@pytest.mark.parametrize("junk", [[{}], ["user"], [None], [{"id": True}]])
def test_owner_lookup_survives_a_well_formed_list_of_junk(owner_search_env, monkeypatch, junk):
    """``find_users`` returns whatever JSON list the model produced without
    checking the elements, so ``[{}]`` raised KeyError and ``["user"]``
    TypeError — both outside the tool's error handling."""
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: junk)
    _patch_llm(monkeypatch, {"owner": "Wahed"})

    out = _search(monkeypatch, {})

    # Degrades to the name fallback rather than crashing; "Wahed" is a
    # real owner here, so it still resolves.
    assert [m["ref"] for m in out["matches"]] == [1, 3]


def test_owner_lookup_coerces_a_stringified_id(owner_search_env, monkeypatch):
    """The quiet one: a model that echoes ``"id": "5"`` survives every type
    check and then never equals the integer ``entity.owner``, matching
    nothing at all.

    The query is deliberately a nickname that matches no username or
    display name, so the departed-member name fallback cannot rescue this
    and the assertion tests the coercion itself."""
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: [{"id": "5"}])
    _patch_llm(monkeypatch, {"owner": "the boss"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [1, 3]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Walid", [2]),
        ("whemat", [2]),
        ("Walid Hemati", [2]),
        # Both users are Hematis, so a shared surname legitimately matches
        # everyone — containment is ambiguous by design, exactly as it is
        # on the current-member path.
        ("hemati", [1, 2, 3]),
        ("ghost", []),
    ],
)
def test_departed_owner_matches_partial_names_like_the_member_path(
    owner_search_env, monkeypatch, query, expected
):
    """``find_users``' prompt matches names by containment. Requiring an
    exact match in the fallback would make a first name find a current
    colleague but not a departed one — the very case it exists for."""
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: [])
    _patch_llm(monkeypatch, {"owner": query})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == expected


# ---------------------------------------------------------------------------
# 2.16.0 — server-side pushdown for issues/epics, open_only, tri-stated
# assignee/status filters, and milestone/is_closed in the match payload.
# ---------------------------------------------------------------------------


def _endpoint_kwargs(monkeypatch):
    """What ``_list_project_entities`` pushed to the REST layer."""
    return taiga_tools.get_project("p").list_endpoint_kwargs


def test_issue_search_pushes_the_owner_filter_server_side(owner_search_env, monkeypatch):
    """``Project.list_issues`` is declared ``(self)`` and takes no
    queryparams, so the pre-2.16 code paged the entire project down and
    filtered client-side — 139 sequential requests and 9.4 MB on
    shikenso-development to return 7 rows. The manager underneath accepts
    the same ``owner`` param ``/userstories`` does."""
    _patch_llm(monkeypatch, {"owner": "Wahed"})

    _search(monkeypatch, {}, entity_type="issue")

    assert _endpoint_kwargs(monkeypatch) == [{"project": 1, "owner": 5}]


def test_epic_search_pushes_the_owner_filter_server_side(owner_search_env, monkeypatch):
    _patch_llm(monkeypatch, {"owner": "Wahed"})

    _search(monkeypatch, {}, entity_type="epic")

    assert _endpoint_kwargs(monkeypatch) == [{"project": 1, "owner": 5}]


def test_assignee_filter_is_pushed_server_side(owner_search_env, monkeypatch):
    monkeypatch.setattr(
        taiga_tools, "find_users", lambda slug, q=None: [{"id": 51, "username": "Whemati"}]
    )
    _patch_llm(monkeypatch, {"assigned_to": "Whemati"})

    _search(monkeypatch, {}, entity_type="issue")

    assert _endpoint_kwargs(monkeypatch) == [{"project": 1, "assigned_to": 51}]


def test_ambiguous_owner_is_not_pushed_down(owner_search_env, monkeypatch):
    """The REST param takes a single id. Two candidates must fall back to
    the client-side pass rather than silently narrowing to the first."""
    monkeypatch.setattr(
        taiga_tools,
        "find_users",
        lambda slug, q=None: [{"id": 5, "username": "Wahed"}, {"id": 51, "username": "Whemati"}],
    )
    _patch_llm(monkeypatch, {"owner": "hemati"})

    _search(monkeypatch, {}, entity_type="issue")

    assert _endpoint_kwargs(monkeypatch) == [{"project": 1}]


def test_milestone_is_never_pushed_down_for_epics(owner_search_env, monkeypatch):
    """Epics carry no milestone field at all — the key is absent from the
    payload, not null — so the param is meaningless there."""
    monkeypatch.setattr(taiga_tools, "find_milestone_id", lambda *a, **kw: 77)
    _patch_llm(monkeypatch, {"milestone": "Sprint 87"})

    _search(monkeypatch, {}, entity_type="epic")

    assert _endpoint_kwargs(monkeypatch) == [{"project": 1}]


def test_milestone_is_pushed_down_for_issues(owner_search_env, monkeypatch):
    monkeypatch.setattr(taiga_tools, "find_milestone_id", lambda *a, **kw: 77)
    _patch_llm(monkeypatch, {"milestone": "Sprint 87"})

    _search(monkeypatch, {}, entity_type="issue")

    assert _endpoint_kwargs(monkeypatch) == [{"project": 1, "milestone": 77}]


# --- open_only -------------------------------------------------------------


def _closed(flag):
    return {"name": "Done" if flag else "Ready", "is_closed": flag}


def test_open_only_drops_closed_entities(fake_search_env, monkeypatch):
    fake_search_env[0].status_extra_info = _closed(True)
    fake_search_env[1].status_extra_info = _closed(False)
    fake_search_env[2].status_extra_info = _closed(True)
    _patch_llm(monkeypatch, {})

    raw = search_entities_tool.invoke(
        {"project_slug": "p", "query": "q", "entity_type": "issue", "open_only": True}
    )

    assert [m["ref"] for m in json.loads(raw)["matches"]] == [2]


def test_open_only_is_pushed_server_side(fake_search_env, monkeypatch):
    _patch_llm(monkeypatch, {})

    search_entities_tool.invoke(
        {"project_slug": "p", "query": "q", "entity_type": "issue", "open_only": True}
    )

    assert taiga_tools.get_project("p").list_endpoint_kwargs == [
        {"project": 1, "status__is_closed": "false"}
    ]


def test_open_only_keeps_entities_whose_closedness_is_unknown(fake_search_env, monkeypatch):
    """A missing ``status_extra_info`` is not evidence the work is finished.
    Dropping those would silently hide tickets."""
    _patch_llm(monkeypatch, {})

    raw = search_entities_tool.invoke(
        {"project_slug": "p", "query": "q", "entity_type": "issue", "open_only": True}
    )

    assert [m["ref"] for m in json.loads(raw)["matches"]] == [1, 2, 3]


def test_open_only_is_never_pushed_down_for_tasks(fake_search_env, monkeypatch):
    """Tasks are reached by walking user stories, so the param would filter
    the *stories* and drop open tasks living under a finished story. It is
    still applied client-side against each task's own status."""
    for entity in fake_search_env:
        entity.is_closed = False
        entity.list_tasks = lambda: []
    _patch_llm(monkeypatch, {})

    search_entities_tool.invoke(
        {"project_slug": "p", "query": "q", "entity_type": "task", "open_only": True}
    )

    assert taiga_tools.get_project("p").list_user_stories_kwargs == [{}]


def test_parse_prompt_marks_closed_statuses(fake_search_env, monkeypatch):
    """Without ``is_closed`` in the prompt the model can only reason
    lexically about a negated query, which is how "nicht geschlossen und
    nicht archiviert" kept ``Done`` in the filter on 30/30 live parses."""
    seen = {}

    class _Recorder:
        def invoke(self, messages):
            seen["prompt"] = messages[0].content
            return AIMessage(content="{}")

    monkeypatch.setattr(
        taiga_tools,
        "list_all_statuses",
        lambda *a, **kw: {
            "issue_statuses": [
                {"id": 1, "name": "New", "is_closed": False},
                {"id": 2, "name": "Done", "is_closed": True},
            ]
        },
    )
    monkeypatch.setattr(taiga_tools, "small_llm", _Recorder())

    search_entities_tool.invoke(
        {"project_slug": "p", "query": "open ones", "entity_type": "issue"}
    )

    assert "New, Done [CLOSED]" in seen["prompt"]


# --- tri-stated assignee / status filters ----------------------------------


def test_unresolvable_assignee_matches_nothing_not_the_whole_project(
    fake_search_env, monkeypatch
):
    """The bug this replaces: ``assigned_to_ids`` was a plain list tested
    with ``if resolved_filters.get(...)``, so an empty list was falsy and
    the filter was skipped entirely. Live on shikenso-development an
    unknown assignee returned all 14 epics and 200 (capped) user stories
    and issues — labelled as that person's work. An id list that resolved
    to nobody is a real filter, not the absence of one."""
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: [])
    _patch_llm(monkeypatch, {"assigned_to": "Zzzunknownperson"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == []


def test_unresolvable_status_matches_nothing_not_the_whole_project(
    fake_search_env, monkeypatch
):
    """Same falsy-empty-list trap on the status leg. A renamed or
    misspelled status silently dropped the filter — and since that filter
    is usually what excludes finished work, the fallback returned exactly
    the closed items the caller was trying to avoid."""
    monkeypatch.setattr(taiga_tools, "find_status_ids", lambda *a, **kw: [])
    _patch_llm(monkeypatch, {"status_names": ["Zzznosuchstatus"]})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == []


def test_assignee_lookup_returning_a_string_is_reported_not_raised(
    fake_search_env, monkeypatch
):
    """``find_users`` is annotated ``-> List[Dict]`` but returns a plain
    string on both parse-failure paths. The old ``[u["id"] for u in users]``
    iterated its characters and died with a TypeError that escaped the
    tool entirely."""
    monkeypatch.setattr(
        taiga_tools, "find_users", lambda slug, q=None: "Error decoding LLM response: x"
    )
    _patch_llm(monkeypatch, {"assigned_to": "Wahed"})

    payload = _search(monkeypatch, {})

    assert payload["code"] == 500
    assert "Assignee lookup failed" in payload["error"]


def test_assignee_lookup_propagating_an_exception_is_reported_not_raised(
    fake_search_env, monkeypatch
):
    """``find_users`` invokes the LLM outside its own try, so a provider
    timeout propagates out of it."""
    def _boom(slug, q=None):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(taiga_tools, "find_users", _boom)
    _patch_llm(monkeypatch, {"assigned_to": "Wahed"})

    payload = _search(monkeypatch, {})

    assert payload["code"] == 500
    assert "429" in payload["error"]


def test_assignee_lookup_coerces_a_stringified_id(fake_search_env, monkeypatch):
    """A model that stringifies the id used to produce ``count=0`` with no
    error — the quietest failure of the lot, and cached for a day."""
    fake_search_env[1].assigned_to = 51
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: [{"id": "51"}])
    _patch_llm(monkeypatch, {"assigned_to": "the boss"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [2]


def test_assignee_filter_still_finds_a_departed_members_tickets(
    fake_search_env, monkeypatch
):
    """``find_users`` only searches *current* members. Without the blob
    fallback, "what is assigned to <departed colleague>?" answers
    "nothing" — a safer wrong answer than the old whole-project dump, but
    still a wrong one. Entities keep carrying ``assigned_to_extra_info``."""
    fake_search_env[2].assigned_to = 95
    fake_search_env[2].assigned_to_extra_info = {
        "id": 95, "username": "mbos617", "full_name_display": "Marko Bosnjak",
    }
    monkeypatch.setattr(taiga_tools, "find_users", lambda slug, q=None: [])
    _patch_llm(monkeypatch, {"assigned_to": "Marko Bosnjak"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [3]


def test_assignee_filter_accepts_a_bare_user_id_without_a_lookup(
    fake_search_env, monkeypatch
):
    """Numeric queries need no resolution, and routing them through
    ``find_users`` would match ids by containment ("51" also finding 151)."""
    fake_search_env[0].assigned_to = 51

    def _never(slug, q=None):
        raise AssertionError("find_users must not be called for a numeric query")

    monkeypatch.setattr(taiga_tools, "find_users", _never)
    _patch_llm(monkeypatch, {"assigned_to": "51"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [1]


# --- milestone / is_closed in the match payload ----------------------------


def test_match_carries_is_closed_and_milestone(fake_search_env, monkeypatch):
    """Both ride along in every list row. Surfacing them is what removes
    the get_entity_by_ref_tool call per match that a "group by sprint" or
    "drop finished work" report otherwise needs."""
    fake_search_env[0].status_extra_info = {"name": "Ready", "is_closed": False}
    fake_search_env[0].milestone = 184
    fake_search_env[0].milestone_name = "Sprint 87"
    _patch_llm(monkeypatch, {})

    match = _search(monkeypatch, {})["matches"][0]

    assert match["is_closed"] is False
    assert match["milestone"] == 184
    assert match["milestone_name"] == "Sprint 87"


def test_milestone_name_resolved_for_issues_that_carry_only_the_id(
    fake_search_env, monkeypatch
):
    """User stories ship ``milestone_name`` inline; issues carry only the
    id, so it is resolved against the TTL-cached milestone list rather
    than costing a request per match."""
    fake_search_env[0].milestone = 201
    monkeypatch.setattr(
        taiga_tools, "list_milestones",
        lambda s: [
            {"id": 201, "name": "Sprint 90", "closed": False},
            {"id": 7, "name": "Sprint 1", "closed": True},
        ],
    )
    _patch_llm(monkeypatch, {})

    match = _search(monkeypatch, {})["matches"][0]

    assert match["milestone_name"] == "Sprint 90"


def test_backlog_entity_reports_null_milestone(fake_search_env, monkeypatch):
    _patch_llm(monkeypatch, {})

    match = _search(monkeypatch, {})["matches"][0]

    assert match["milestone"] is None
    assert match["milestone_name"] is None


# --- cleanup-pass follow-ups ------------------------------------------------


def test_unresolvable_milestone_matches_nothing_not_the_whole_project(
    fake_search_env, monkeypatch
):
    """The fourth instance of the same fail-open class. ``find_milestone_id``
    returns None both for "no sprint asked for" and for "sprint asked for but
    unknown", so keying the filter off its result let a typo'd or renamed
    sprint name return the entire project."""
    monkeypatch.setattr(taiga_tools, "find_milestone_id", lambda *a, **kw: None)
    _patch_llm(monkeypatch, {"milestone": "Sprint 99"})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == []


def test_assignee_username_comes_from_the_blob_without_a_lookup(
    fake_search_env, monkeypatch
):
    """``assigned_to_extra_info`` is on every list row, so the display name
    must not cost a ``get_user`` round-trip per distinct assignee — the last
    per-match network call in this loop."""
    fake_search_env[0].assigned_to = 17
    fake_search_env[0].assigned_to_extra_info = {
        "id": 17, "username": "Tobi", "full_name_display": "Tobias Gau",
    }
    calls = []
    monkeypatch.setattr(
        taiga_tools, "get_user",
        lambda uid: calls.append(uid) or {"username": f"user{uid}"},
    )
    _patch_llm(monkeypatch, {})

    out = _search(monkeypatch, {})

    assert out["matches"][0]["assigned_to"] == "Tobi"
    assert calls == []


def test_assignee_username_falls_back_to_get_user_without_a_blob(
    fake_search_env, monkeypatch
):
    fake_search_env[0].assigned_to = 17
    _patch_llm(monkeypatch, {})

    out = _search(monkeypatch, {})

    assert out["matches"][0]["assigned_to"] == "user17"


def test_status_name_and_flag_come_from_the_blob_without_a_lookup(
    fake_search_env, monkeypatch
):
    """``status_extra_info`` carries both name and is_closed. The status
    registry is only a 5-minute cache, so reading it per match re-pays a
    request per distinct status on nearly every search."""
    for entity in fake_search_env:
        entity.status_extra_info = {"name": "Needs Info", "is_closed": False}
    calls = []
    monkeypatch.setattr(
        taiga_tools, "get_status",
        lambda *a, **kw: calls.append(a) or {"name": "Open"},
    )
    _patch_llm(monkeypatch, {})

    out = _search(monkeypatch, {})

    assert [m["status"] for m in out["matches"]] == ["Needs Info"] * 3
    assert calls == []


def test_milestone_list_is_fetched_once_not_per_match(fake_search_env, monkeypatch):
    """``list_milestones`` is a 5-minute cache over a multi-page fetch, so
    calling it inside the loop lets a long enough search outrun the TTL and
    silently re-pay mid-iteration."""
    calls = []
    monkeypatch.setattr(
        taiga_tools, "list_milestones", lambda s: calls.append(s) or []
    )
    _patch_llm(monkeypatch, {})

    _search(monkeypatch, {})

    assert len(calls) == 1


def test_list_project_entities_raises_for_an_unroutable_type():
    """Returning [] would turn a programming error into a plausible-looking
    "no matches" — the silent-wrong-answer shape this tool exists to avoid."""
    with pytest.raises(ValueError, match="no project-level list endpoint"):
        taiga_tools._list_project_entities(object(), "task")


# --- codex adversarial-review follow-ups -----------------------------------


def test_explicitly_empty_status_list_matches_nothing(fake_search_env, monkeypatch):
    """``"status_names": []`` is a third case, distinct from absent and from
    null: the parser WAS asked for a status filter and produced no names.
    This tool's own prompt can emit exactly that for a negated query on a
    project whose statuses are all terminal — and truthiness read it as
    "no filter", handing back the whole project."""
    _patch_llm(monkeypatch, {"status_names": []})

    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == []


def test_absent_and_null_status_names_still_mean_no_filter(
    fake_search_env, monkeypatch
):
    """The other side of the same distinction — this must NOT become a
    fail-closed filter, or every unfiltered search returns nothing."""
    _patch_llm(monkeypatch, {"status_names": None})
    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [1, 2, 3]

    _patch_llm(monkeypatch, {})
    assert [m["ref"] for m in _search(monkeypatch, {})["matches"]] == [1, 2, 3]


def test_open_only_server_filter_and_client_predicate_compose(
    fake_search_env, monkeypatch
):
    """The server-side ``status__is_closed=false`` and the client-side
    "keep unknown closedness" rule answer different questions, and the fake
    endpoint now enforces the param so the composition is actually exercised.

    ref=1 is closed as far as the server is concerned -> dropped there.
    ref=2 is open server-side but carries no ``status_extra_info``, so the
    client cannot tell -> kept, because a missing flag is not evidence the
    work is finished.
    ref=3 is open and says so -> kept.
    """
    fake_search_env[0].server_is_closed = True
    fake_search_env[0].status_extra_info = {"name": "Done", "is_closed": True}
    fake_search_env[1].server_is_closed = False  # no status_extra_info
    fake_search_env[2].server_is_closed = False
    fake_search_env[2].status_extra_info = {"name": "Ready", "is_closed": False}
    _patch_llm(monkeypatch, {})

    raw = search_entities_tool.invoke(
        {"project_slug": "p", "query": "q", "entity_type": "issue", "open_only": True}
    )
    payload = json.loads(raw)

    assert [m["ref"] for m in payload["matches"]] == [2, 3]
    assert [m["is_closed"] for m in payload["matches"]] == [None, False]
