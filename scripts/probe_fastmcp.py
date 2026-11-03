"""Phase 1 -- fastmcp 4 bump: probe the installed fastmcp 4.x / mcp 2.x surface.

Prints one ``label: value`` line per check so the answers can be pasted into
the PR as ``phase1-probe.txt``. Library signatures only: no paths, no hosts.

Run: ``python scripts/probe_fastmcp.py > phase1-probe.txt``
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect


def params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def out(label: str, value) -> None:
    print(f"{label}: {value}")


from fastmcp import FastMCP  # noqa: E402

# 1. versions
out("fastmcp version", md.version("fastmcp"))
out("mcp version", md.version("mcp"))

# 2. constructor
out("FastMCP.__init__ params", params(FastMCP.__init__))
try:
    FastMCP("x", host="h")  # type: ignore[call-arg]
    out("FastMCP(host=...) raises TypeError", False)
except TypeError as exc:
    out("FastMCP(host=...) raises TypeError", True)
    out("  message", exc)

# 3. transport entry points
m = FastMCP("probe")
out("run_async params", params(m.run_async))
out("run_http_async params", params(m.run_http_async))
out("http_app params", params(m.http_app))
out("stateless_http in http_app", "stateless_http" in params(m.http_app))
out("stateless_http in run_http_async", "stateless_http" in params(m.run_http_async))
out("stateless_http in run_async", "stateless_http" in params(m.run_async))


# 4. registration
def _probe_tool() -> str:
    """Probe."""
    return "x"


m4 = FastMCP("probe4")
out("tool() returns fn", m4.tool()(_probe_tool) is _probe_tool)


def _probe_tool_2() -> str:
    """Probe two."""
    return "y"


out("add_tool returns", type(m4.add_tool(_probe_tool_2)).__name__)

# 5. tool listing API
for name in ("get_tools", "list_tools", "get_tool"):
    out(f"hasattr {name}", hasattr(m, name))

# 6. FunctionTool.run_in_thread default
from fastmcp.tools import FunctionTool  # noqa: E402

try:
    ft = FunctionTool.from_function(_probe_tool)
    out("run_in_thread default", getattr(ft, "run_in_thread", "<no attribute>"))
except Exception as exc:  # pylint: disable=broad-except
    out("run_in_thread default", f"probe failed: {type(exc).__name__}: {exc}")

# 7. is fastmcp's build_metadata the SDK's (i.e. would an SDK patch reach it)?
from mcp.server.auth import routes as sdk_routes  # noqa: E402

try:
    import fastmcp.server.auth.auth as fm_auth

    out(
        "fastmcp build_metadata is sdk build_metadata",
        fm_auth.build_metadata is sdk_routes.build_metadata,
    )
except Exception as exc:  # pylint: disable=broad-except
    out("build_metadata identity", f"probe failed: {type(exc).__name__}: {exc}")

# 8. OAuthProvider surface
from fastmcp.server.auth import OAuthProvider  # noqa: E402

out(
    "OAuthProvider methods",
    sorted(
        n
        for n in dir(OAuthProvider)
        if not n.startswith("_") and callable(getattr(OAuthProvider, n))
    ),
)
out("OAuthProvider.__init__ params", params(OAuthProvider.__init__))

# 9. the SDK imports provider.py relies on
for mod, names in [
    ("fastmcp.server.auth", ["AccessToken", "OAuthProvider"]),
    (
        "mcp.server.auth.provider",
        [
            "AuthorizationCode",
            "AuthorizationParams",
            "RefreshToken",
            "TokenError",
            "construct_redirect_uri",
        ],
    ),
    ("mcp.server.auth.settings", ["ClientRegistrationOptions"]),
    ("mcp.shared.auth", ["OAuthClientInformationFull", "OAuthToken"]),
]:
    try:
        module = importlib.import_module(mod)
        missing = [n for n in names if not hasattr(module, n)]
        out(f"import {mod}", "OK" if not missing else f"MISSING {missing}")
    except ImportError as exc:
        out(f"import {mod}", f"ImportError: {exc}")

# 10. optional apps extra
try:
    importlib.import_module("fastmcp.apps.file_upload")
    out("fastmcp.apps.file_upload", "imports")
except ImportError as exc:
    out("fastmcp.apps.file_upload", f"ImportError: {exc}")
