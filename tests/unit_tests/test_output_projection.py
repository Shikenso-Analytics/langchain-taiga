"""Field projection and compact JSON for the read tools (2.19.0).

A caller that only needs three fields of a story should not pay for the other thirty: history
alone is 84-97 % of a ``get_entity_by_ref_tool`` payload, and every byte of it lands in the
model's context. ``fields`` selects dotted paths (lists are walked), ``compact`` drops the
indentation. The rules below are the ones a caller relies on to tell "not asked for" from
"asked for and empty".
"""

import json

import pytest

from langchain_taiga.tools import output


STORY = {
    "ref": 7,
    "subject": "Stage 3",
    "assigned_to": None,
    "watchers": [{"id": 1, "username": "anna", "email": "a@x"}, {"id": 2, "username": "ben", "email": "b@x"}],
    "related": {"tasks": [{"ref": 70, "subject": "t1", "status": "New"}, {"ref": 71, "subject": "t2"}]},
    "history": [{"id": "h1", "comment": "ok", "user": {"pk": 5, "username": "alice"}}],
}


def paths(*items):
    return output.parse_fields(list(items))


def test_top_level_paths_keep_only_those_keys():
    assert output.project(STORY, paths("ref", "subject")) == {"ref": 7, "subject": "Stage 3"}


def test_a_path_walks_every_element_of_a_list():
    assert output.project(STORY, paths("related.tasks.ref")) == {"related": {"tasks": [{"ref": 70}, {"ref": 71}]}}


def test_several_paths_into_the_same_list_merge():
    projected = output.project(STORY, paths("watchers.id", "watchers.username"))
    assert projected == {"watchers": [{"id": 1, "username": "anna"}, {"id": 2, "username": "ben"}]}


def test_a_whole_key_wins_over_a_deeper_path_into_it():
    projected = output.project(STORY, paths("history", "history.comment"))
    assert projected == {"history": STORY["history"]}


def test_a_requested_key_an_element_lacks_is_null_not_absent():
    """``related.tasks.status`` asked for: the second task carries none, and the caller must see
    that as null, not as a missing key it would read as "not asked for"."""
    projected = output.project(STORY, paths("related.tasks.status"))
    assert projected == {"related": {"tasks": [{"status": "New"}, {"status": None}]}}


def test_a_path_through_null_stays_null():
    assert output.project(STORY, paths("assigned_to.username")) == {"assigned_to": None}


def test_no_fields_means_the_value_is_untouched():
    assert output.project(STORY, None) is STORY


@pytest.mark.parametrize("bad", [[""], ["  "], ["ref", "a..b"], ["history."], [".ref"]])
def test_blank_path_segments_are_refused(bad):
    with pytest.raises(ValueError):
        output.parse_fields(bad)


def test_an_empty_field_list_is_refused():
    """``[]`` would project everything away; a caller who wants nothing has no reason to call."""
    with pytest.raises(ValueError):
        output.parse_fields([])


def test_unknown_top_level_keys_are_named_together_with_the_valid_ones():
    message = output.unknown_fields(paths("ref", "subjekt", "histroy.comment"), {"ref", "subject", "history"})
    assert message == "Unknown field(s) histroy, subjekt. Valid top-level fields: history, ref, subject."


def test_known_top_level_keys_raise_nothing():
    assert output.unknown_fields(paths("ref", "history.comment"), {"ref", "history"}) is None


@pytest.mark.parametrize(
    ("requested", "key", "expected"),
    [
        (None, "history", True),
        (["history.comment"], "history", True),
        (["history"], "history", True),
        (["ref"], "history", False),
    ],
)
def test_wants_tells_a_tool_whether_to_fetch_a_part(requested, key, expected):
    assert output.wants(output.parse_fields(requested) if requested else None, key) is expected


def test_wants_follows_a_nested_prefix():
    assert output.wants(paths("related.tasks.ref"), "related", "tasks") is True
    assert output.wants(paths("related"), "related", "tasks") is True
    assert output.wants(paths("related.user_stories"), "related", "tasks") is False


def test_compact_json_has_no_padding_and_keeps_umlauts():
    text = output.dumps({"subject": "Übertragung", "ref": 1}, compact=True, projected=False)
    assert text == '{"subject":"Übertragung","ref":1}'


def test_compact_without_fields_drops_nulls_at_every_depth():
    text = output.dumps({"a": None, "b": {"c": None, "d": 1}, "e": [{"f": None}], "g": []}, compact=True, projected=False)
    assert json.loads(text) == {"b": {"d": 1}, "e": [{}], "g": []}


def test_compact_with_fields_keeps_a_requested_null():
    """Unassigned and not-asked-for must stay two different answers."""
    text = output.dumps({"assigned_to": None}, compact=True, projected=True)
    assert json.loads(text) == {"assigned_to": None}


def test_the_default_output_is_the_indented_json_every_caller_parses_today():
    value = {"ref": 1, "when": object()}
    text = output.dumps(value, compact=False, projected=False)
    assert text.startswith('{\n  "ref": 1')
    assert '"when": "<object object' in text


# -- nested schemas --------------------------------------------------------------------------------

BOARD = {"project": None, "columns": {"status": None, "cards": {"ref", "tags"}}}


def test_an_unknown_nested_key_is_named_with_its_full_path_and_siblings():
    message = output.unknown_fields([["columns", "cards", "tag"]], BOARD)
    assert message == "Unknown field columns.cards.tag. Valid fields under columns.cards: ref, tags."


def test_control_a_known_nested_key_passes():
    assert output.unknown_fields([["columns", "cards", "tags"], ["columns", "status"]], BOARD) is None


def test_a_key_mapped_to_none_is_not_checked_below():
    assert output.unknown_fields([["project", "anything", "deeper"]], BOARD) is None


def test_a_set_level_leaves_the_levels_below_it_unchecked():
    assert output.unknown_fields([["columns", "cards", "ref", "x"]], BOARD) is None


def test_checked_fields_returns_the_paths_or_a_ready_400():
    assert output.checked_fields(["project"], BOARD, compact=False) == ([["project"]], None)
    paths, invalid = output.checked_fields(["columns.bogus"], BOARD, compact=True)
    assert paths is None and json.loads(invalid)["code"] == 400 and "\n" not in invalid
    paths, invalid = output.checked_fields([], BOARD, compact=False)
    assert paths is None and json.loads(invalid)["code"] == 400


def test_project_keeping_carries_the_meta_keys_after_the_projected_ones():
    result = {"matches": [{"ref": 1, "subject": "x"}], "count": 1, "truncated": False}
    answer = output.project_keeping(result, [["matches", "ref"]], ("count", "truncated", "absent"))
    assert list(answer.items()) == [("matches", [{"ref": 1}]), ("count", 1), ("truncated", False)]
    assert result["matches"][0] == {"ref": 1, "subject": "x"}
