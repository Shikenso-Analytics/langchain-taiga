"""``custom_attr_definitions_cache`` is a ``TTLCache``, which is not thread-safe,
and fastmcp 4 runs every sync tool in a worker thread. Every access must hold
``_cache_lock`` and must be a single call, never ``in`` followed by ``[]``."""

from types import SimpleNamespace

import pytest
from cachetools import TTLCache

from langchain_taiga.tools import taiga_tools


class _AuditedCache(TTLCache):
    """Records whether the module lock was held on each access, and expires
    between ``in`` and ``[]`` like a real entry can under concurrency."""

    def __init__(self):
        super().__init__(maxsize=10, ttl=3600)
        self.unlocked_access = []
        self.stored = {}

    def _check(self, op):
        if not taiga_tools._cache_lock._is_owned():
            self.unlocked_access.append(op)

    def __contains__(self, key):
        self._check("in")
        return True  # claims the entry is there ...

    def __getitem__(self, key):
        self._check("getitem")
        raise KeyError(key)  # ... and it has expired by the time it is read

    def get(self, key, default=None):
        self._check("get")
        return self.stored.get(key, default)

    def __setitem__(self, key, value):
        self._check("setitem")
        self.stored[key] = value


def _project():
    attr = SimpleNamespace(id=7, name="Area", description="", type="text", extra=None)
    return SimpleNamespace(id=1, list_user_story_attributes=lambda: [attr])


def test_attribute_definitions_cache_is_only_touched_under_the_lock(monkeypatch):
    cache = _AuditedCache()
    monkeypatch.setattr(taiga_tools, "custom_attr_definitions_cache", cache)

    first = taiga_tools._load_attribute_definitions(_project(), "us")
    second = taiga_tools._load_attribute_definitions(_project(), "us")  # cache hit

    assert first == second and first["7"]["name"] == "Area"
    assert cache.unlocked_access == []


def test_an_entry_expiring_between_check_and_read_does_not_raise(monkeypatch):
    # Control: the double-lookup pattern really raises on this cache.
    cache = _AuditedCache()
    with pytest.raises(KeyError):
        if "k" in cache:
            cache["k"]

    monkeypatch.setattr(taiga_tools, "custom_attr_definitions_cache", cache)
    assert taiga_tools._load_attribute_definitions(_project(), "us")["7"]["name"] == "Area"
