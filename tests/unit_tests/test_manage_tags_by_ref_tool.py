import json

import pytest
from langchain_core.tools import BaseTool

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import (
    _normalize_tag_names,
    manage_tags_by_ref_tool,
)
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """
    Automatically apply a fake OPENAI_API_KEY environment variable
    for each test function. That way, login() won't raise ValueError.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


class TestManageTagsUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> BaseTool:
        return manage_tags_by_ref_tool

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
            "entity_ref": 555,
            "entity_type": "us",
            "tags": ["voice"],
            "mode": "add",
        }


# ---------------------------------------------------------------------------
# Lightweight fakes so behavioural tests stay focused on the tool's tag
# set-math and normalization, without touching the Taiga API.
# ---------------------------------------------------------------------------


class _Entity:
    """Minimal stand-in for a python-taiga entity: records the scoped
    patch() call so tests can assert both the tag payload (kwargs) and
    that the OCC version field is included.

    ``tags`` is seeded in Taiga's READ shape (``[name, color]`` pairs)
    because that is what the real API hands back."""

    def __init__(self, tags=None):
        self.tags = list(tags) if tags is not None else None
        self.version = 7
        self.patch_fields = None
        self.updated_with = None

    def patch(self, fields, **kwargs):
        self.patch_fields = fields
        self.updated_with = kwargs
        if "tags" in kwargs:
            self.tags = kwargs["tags"]
        return self


class _Project:
    """Only has to be truthy — the tool hands it straight to fetch_entity."""


def _patch_common(monkeypatch, entity, project_tags=("jobs_manager", "voice")):
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)
    monkeypatch.setattr(taiga_tools, "list_all_tags", lambda slug: list(project_tags))


def _invoke(**kwargs):
    return json.loads(manage_tags_by_ref_tool.invoke(kwargs))


# --- _normalize_tag_names -------------------------------------------------


def test_normalize_flattens_taigas_name_color_pairs():
    """Regression: Taiga reads tags back as ``[name, color]`` pairs, so a
    bare ``"voice" in entity.tags`` never matches."""
    assert _normalize_tag_names(
        [["jobs_manager", None], ["voice", "#845EF7"]]
    ) == ["jobs_manager", "voice"]


def test_normalize_accepts_plain_strings_and_mixed_payloads():
    assert _normalize_tag_names(["voice", ["k8s", "#fff"]]) == ["voice", "k8s"]


def test_normalize_handles_none_empty_and_blank_entries():
    assert _normalize_tag_names(None) == []
    assert _normalize_tag_names([]) == []
    assert _normalize_tag_names([[], "  ", None, ["voice", "#fff"]]) == ["voice"]


def test_normalize_strips_whitespace_and_dedupes():
    assert _normalize_tag_names(["  voice  ", "voice"]) == ["voice"]


# --- manage_tags_by_ref_tool: set math ------------------------------------


def test_add_merges_into_existing(monkeypatch):
    entity = _Entity(tags=[["jobs_manager", None]])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["voice"], mode="add")
    # Written back FLAT — Taiga accepts names on write and joins the
    # project-level colour back in on read.
    assert entity.updated_with == {"tags": ["jobs_manager", "voice"]}
    # Scoped PATCH must carry the OCC version, not a full PUT.
    assert entity.patch_fields == ["version"]
    assert out["tags"] == ["jobs_manager", "voice"]
    assert "updated" in out["message"]


def test_add_does_not_drop_the_other_tags(monkeypatch):
    """The whole reason this is a separate tool: adding one tag must not
    silently wipe the ones the caller never mentioned."""
    entity = _Entity(tags=[["jobs_manager", None], ["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["k8s"], mode="add")
    assert entity.updated_with == {"tags": ["jobs_manager", "voice", "k8s"]}


def test_add_existing_tag_is_noop(monkeypatch):
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["voice"], mode="add")
    assert entity.updated_with is None  # no PATCH issued
    assert "No tag change" in out["message"]
    assert out["tags"] == ["voice"]


def test_add_matches_case_insensitively_and_keeps_existing_spelling(monkeypatch):
    """``Voice`` must not become a second tag alongside ``voice``. This one
    lands in the no-op branch; the case-insensitivity has teeth only when a
    write actually goes through — see the next test."""
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["Voice"], mode="add")
    assert entity.updated_with is None
    assert out["tags"] == ["voice"]


def test_add_case_variant_alongside_a_new_tag_does_not_duplicate(monkeypatch):
    """The teeth of the case-insensitive add: when the call also carries a
    genuinely new tag the write goes through, so a case-sensitive
    comparison would ship ['voice', 'Voice', 'k8s'] — a duplicate tag that
    the no-op guard can no longer absorb."""
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    _invoke(
        project_slug="s", entity_ref=1, entity_type="us", tags=["Voice", "k8s"], mode="add"
    )
    assert entity.updated_with == {"tags": ["voice", "k8s"]}


def test_replace_sets_exact_list(monkeypatch):
    entity = _Entity(tags=[["jobs_manager", None], ["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["k8s"], mode="replace")
    assert entity.updated_with == {"tags": ["k8s"]}


def test_replace_empty_clears_all(monkeypatch):
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=[], mode="replace")
    assert entity.updated_with == {"tags": []}


def test_replace_reuses_existing_spelling(monkeypatch):
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    _invoke(
        project_slug="s", entity_ref=1, entity_type="us", tags=["VOICE", "k8s"], mode="replace"
    )
    assert entity.updated_with == {"tags": ["voice", "k8s"]}


def test_remove_drops_given(monkeypatch):
    entity = _Entity(tags=[["jobs_manager", None], ["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["voice"], mode="remove")
    assert entity.updated_with == {"tags": ["jobs_manager"]}


def test_remove_is_case_insensitive(monkeypatch):
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["VOICE"], mode="remove")
    assert entity.updated_with == {"tags": []}


def test_remove_absent_is_noop(monkeypatch):
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["k8s"], mode="remove")
    assert entity.updated_with is None
    assert "No tag change" in out["message"]


def test_entity_with_no_tags_yet(monkeypatch):
    entity = _Entity(tags=None)
    _patch_common(monkeypatch, entity)
    _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["voice"], mode="add")
    assert entity.updated_with == {"tags": ["voice"]}


def test_blank_tags_are_dropped_from_input(monkeypatch):
    entity = _Entity(tags=[])
    _patch_common(monkeypatch, entity)
    _invoke(
        project_slug="s", entity_ref=1, entity_type="us", tags=["  voice  ", "  "], mode="add"
    )
    assert entity.updated_with == {"tags": ["voice"]}


# --- created_tags reporting ------------------------------------------------


def test_reports_tags_that_are_new_to_the_project(monkeypatch):
    """Taiga creates an unknown tag implicitly on write, so a typo becomes a
    permanent project tag. Surface it rather than swallowing it."""
    entity = _Entity(tags=[])
    _patch_common(monkeypatch, entity, project_tags=("jobs_manager", "voice"))
    out = _invoke(
        project_slug="s", entity_ref=1, entity_type="us", tags=["voice", "jobs_manger"], mode="add"
    )
    assert out["created_tags"] == ["jobs_manger"]


def test_created_tags_matches_the_registry_case_insensitively(monkeypatch):
    entity = _Entity(tags=[])
    _patch_common(monkeypatch, entity, project_tags=("Voice",))
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["voice"], mode="add")
    assert out["created_tags"] == []


def test_created_tags_ignores_tags_the_entity_already_had(monkeypatch):
    """A tag already on the entity is not being created, even if the
    project registry lookup doesn't know about it."""
    entity = _Entity(tags=[["legacy", None]])
    _patch_common(monkeypatch, entity, project_tags=("voice",))
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["voice"], mode="add")
    assert out["created_tags"] == []


def test_registry_lookup_failure_does_not_break_the_write(monkeypatch):
    """``created_tags`` is informational; losing it must not cost the edit."""
    entity = _Entity(tags=[])
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)

    def _boom(slug):
        raise RuntimeError("tags_colors endpoint down")

    monkeypatch.setattr(taiga_tools, "list_all_tags", _boom)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["voice"], mode="add")
    assert entity.updated_with == {"tags": ["voice"]}
    # null, not absent and not [] — "could not verify" must be tellable
    # apart from "nothing new was created".
    assert out["created_tags"] is None


# --- validation ------------------------------------------------------------


def test_add_empty_is_rejected(monkeypatch):
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=[], mode="add")
    assert out["code"] == 400
    assert entity.updated_with is None


def test_invalid_mode_is_rejected(monkeypatch):
    entity = _Entity(tags=[["voice", "#845EF7"]])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["k8s"], mode="toggle")
    assert out["code"] == 400
    assert entity.updated_with is None


def test_unsupported_entity_type_is_rejected(monkeypatch):
    entity = _Entity(tags=[])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="banana", tags=["k8s"], mode="add")
    assert out["code"] == 400
    assert entity.updated_with is None


def test_project_not_found_is_rejected(monkeypatch):
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: None)
    out = _invoke(project_slug="nope", entity_ref=1, entity_type="us", tags=["k8s"], mode="add")
    assert out["code"] == 404


def test_entity_not_found_is_rejected(monkeypatch):
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: None)
    out = _invoke(project_slug="s", entity_ref=9, entity_type="us", tags=["k8s"], mode="add")
    assert out["code"] == 404


def test_patch_failure_is_surfaced_as_500(monkeypatch):
    class _Boom(_Entity):
        def patch(self, fields, **kwargs):
            raise RuntimeError("taiga said no")

    entity = _Boom(tags=[])
    _patch_common(monkeypatch, entity)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["k8s"], mode="add")
    assert out["code"] == 500
    assert "taiga said no" in out["error"]


def test_registry_is_read_before_the_write(monkeypatch):
    """Taiga creates an unknown tag implicitly as part of the same save, so a
    registry read issued *after* the patch always finds the tag already there
    and reports nothing. Ordering is the whole feature."""
    entity = _Entity(tags=[])
    calls = []

    class _Recording(_Entity):
        def patch(self, fields, **kwargs):
            calls.append("patch")
            return super().patch(fields, **kwargs)

    entity = _Recording(tags=[])
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)

    def _registry(slug):
        calls.append("registry")
        return ["voice"]

    monkeypatch.setattr(taiga_tools, "list_all_tags", _registry)
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["typoo"], mode="add")
    assert calls == ["registry", "patch"]
    assert out["created_tags"] == ["typoo"]


def test_add_reuses_a_spelling_the_project_knows(monkeypatch):
    """The entity carries no tags, but the project registry already has
    'voice'. Adding 'Voice' must reuse it instead of minting a second
    project-level tag that differs only in case."""
    entity = _Entity(tags=[])
    _patch_common(monkeypatch, entity, project_tags=("voice",))
    out = _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["Voice"], mode="add")
    assert entity.updated_with == {"tags": ["voice"]}
    assert out["created_tags"] == []


def test_remove_does_not_read_the_registry(monkeypatch):
    """'remove' can only shrink the list, so it needs neither canonical
    spellings nor a created-tags check — and must not pay for the call."""
    entity = _Entity(tags=[["voice", "#845EF7"], ["k8s", None]])
    monkeypatch.setattr(taiga_tools, "get_project", lambda slug: _Project())
    monkeypatch.setattr(taiga_tools, "fetch_entity", lambda project, norm, ref: entity)

    # Counted, not raised: the tool swallows Exception around this call, and
    # AssertionError is an Exception — a raising spy would be silently eaten
    # and the test would pass no matter what.
    calls = []
    monkeypatch.setattr(taiga_tools, "list_all_tags", lambda slug: calls.append(slug) or [])

    _invoke(project_slug="s", entity_ref=1, entity_type="us", tags=["voice"], mode="remove")
    assert entity.updated_with == {"tags": ["k8s"]}
    assert calls == []
