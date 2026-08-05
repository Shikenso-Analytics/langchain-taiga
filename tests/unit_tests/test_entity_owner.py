"""Unit tests for ``_owner_summary`` and the read-only-ness of ``owner``.

The tool-level wiring is tested where the tools live — search-side in
``test_search_entities_tool.py``, detail-side in
``test_get_entity_by_ref_tool.py``. What is left here needs no fixtures at
all: the helper's three branches, and an SDK canary.
"""

import pytest

from langchain_taiga.tools import taiga_tools
from langchain_taiga.tools.taiga_tools import _owner_summary


class _Entity:
    """python-taiga ``setattr``s every key the API returns, so an entity
    either carries ``owner_extra_info`` or does not — it is never ``None``
    on a real object. Mirror that rather than defaulting the attribute."""

    def __init__(self, owner=None, owner_extra_info=None):
        self.owner = owner
        if owner_extra_info is not None:
            self.owner_extra_info = owner_extra_info


def test_prefers_the_embedded_blob():
    entity = _Entity(
        owner=5,
        owner_extra_info={"id": 5, "username": "Wahed", "full_name_display": "Dr. Wahed Hemati"},
    )
    assert _owner_summary(entity) == {
        "id": 5,
        "username": "Wahed",
        "full_name": "Dr. Wahed Hemati",
    }


def test_falls_back_to_get_user_without_the_blob(monkeypatch):
    """Older payloads can omit the blob. The id is still there, and
    ``get_user`` is TTL-cached for a day, so this costs one call per
    distinct user rather than one per entity."""
    monkeypatch.setattr(
        taiga_tools,
        "get_user",
        lambda uid: {"id": uid, "username": f"user{uid}", "full_name": f"User {uid}"},
    )
    assert _owner_summary(_Entity(owner=5))["username"] == "user5"


def test_failed_lookup_still_reports_the_id(monkeypatch):
    """``get_user`` returns a bare {"error", "code"} dict on failure. The
    owner id is known regardless and must not be swallowed."""
    monkeypatch.setattr(taiga_tools, "get_user", lambda uid: {"error": "boom", "code": 500})
    assert _owner_summary(_Entity(owner=5)) == {"id": 5, "username": None, "full_name": None}


def test_is_none_without_an_owner():
    assert _owner_summary(_Entity()) is None


@pytest.mark.parametrize("model_name", ["UserStory", "Task", "Issue", "Epic"])
def test_owner_is_not_a_writable_field(model_name):
    """python-taiga sends ``to_dict()`` filtered by ``allowed_params`` on
    update/patch. ``owner`` being absent there is what guarantees
    ``update_entity_by_ref_tool`` cannot silently reassign authorship of a
    ticket. If an SDK bump ever adds it, this fails loudly rather than
    letting a write path grow a new capability unnoticed."""
    from taiga.models import models

    assert "owner" not in getattr(models, model_name).allowed_params
