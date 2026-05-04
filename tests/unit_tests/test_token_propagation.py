"""Tests for per-request Taiga-JWT propagation via FastMCP AccessToken.claims.

Verifies two contracts:
1. Stdio mode (no HTTP request context) keeps working unchanged with ENV auth.
2. HTTP mode reads the verified Taiga JWT from the FastMCP AccessToken claims
   and never leaks cached data between users.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setenv("TAIGA_URL", "https://taiga.example.test")
    monkeypatch.setenv("TAIGA_API_URL", "https://taiga.example.test")
    monkeypatch.setenv("TAIGA_USERNAME", "test_user")
    monkeypatch.setenv("TAIGA_PASSWORD", "test_pass")
    monkeypatch.setenv("OPENAI_API_KEY", "FAKE_TOKEN_FOR_TESTS")


@contextmanager
def patched_access_token(monkeypatch, *, claims=None):
    """Stub fastmcp.server.dependencies.get_access_token.

    claims=None  → simulate "no active auth context" (stdio / unauthenticated)
    claims=dict  → simulate a verified incoming request with these claims
    """
    from langchain_taiga.tools import taiga_tools

    if claims is None:
        def _fake():
            raise LookupError("No access token in context")
    else:
        fake_token = MagicMock()
        fake_token.claims = claims
        def _fake():
            return fake_token

    monkeypatch.setattr(taiga_tools, "get_access_token", _fake)
    yield
