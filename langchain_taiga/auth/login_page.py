"""Render the HTML form where end-users provide their Taiga credentials.

Used by remote_server.py's ``/oauth/login`` GET handler. Autoescape is
turned on so ``state`` and ``error`` values from untrusted sources are
HTML-escaped automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)


def render_login_page(
    *, state: str, error: Optional[str], taiga_url: str
) -> str:
    return _env.get_template("login.html").render(
        state=state, error=error, taiga_url=taiga_url
    )
