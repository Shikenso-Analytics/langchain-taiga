"""Unit tests for ``langchain_taiga.auth.login_page.render_login_page``.

Verifies the form is rendered, errors are shown, and HTML is escaped.
"""

from __future__ import annotations


def test_login_page_renders_form_with_state():
    from langchain_taiga.auth.login_page import render_login_page

    html = render_login_page(
        state="csrf_xyz",
        error=None,
        taiga_url="https://taiga.shikenso.org",
    )
    assert "<form" in html
    assert 'name="state" value="csrf_xyz"' in html
    assert 'name="username"' in html
    assert 'name="password"' in html
    assert "taiga.shikenso.org" in html


def test_login_page_displays_error():
    from langchain_taiga.auth.login_page import render_login_page

    html = render_login_page(
        state="csrf_xyz",
        error="Invalid username or password",
        taiga_url="https://taiga.shikenso.org",
    )
    assert "Invalid username or password" in html


def test_login_page_escapes_html_in_state():
    from langchain_taiga.auth.login_page import render_login_page

    html = render_login_page(
        state="<script>alert(1)</script>",
        error="<img src=x onerror=alert(2)>",
        taiga_url="https://taiga.shikenso.org",
    )
    # Raw payloads must not appear unescaped
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
