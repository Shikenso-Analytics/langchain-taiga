"""FastMCP API surface probe — Phase 0 of the multi-tenant OAuth bridge plan.

Captures the actually-installed FastMCP and MCP-SDK signatures so PR 2/PR 3
code can be written against verified shapes instead of guessed ones.

Output: paste into the PR 1 description as `phase0-probe.txt`.
"""
from __future__ import annotations

import inspect

import fastmcp
from fastmcp import FastMCP


print("fastmcp version:", fastmcp.__version__)
print()

# 1. Constructor accepts auth= and lifespan=?
init_params = list(inspect.signature(FastMCP.__init__).parameters)
print("FastMCP.__init__ params:", init_params)
print("  → auth supported at construction:", "auth" in init_params)
print("  → lifespan supported at construction:", "lifespan" in init_params)
print()

# 2. mcp.run() params
m = FastMCP("probe")
run_params = list(inspect.signature(m.run).parameters)
print("mcp.run params:", run_params)
print("  → lifespan in mcp.run:", "lifespan" in run_params)
print("  → run_async exists:", hasattr(m, "run_async"))
print()

# 3. Custom-route decorator
print("has mcp.custom_route:", hasattr(m, "custom_route"))
print()

# 4. ASGI-app accessor
print("has streamable_http_app:", hasattr(m, "streamable_http_app"))
print("has http_app:", hasattr(m, "http_app"))
print()

# 5. OAuthProvider abstract surface + constructor signature
from fastmcp.server.auth import OAuthProvider
# ClientRegistrationOptions location varies — try canonical paths
try:
    from mcp.server.auth.settings import ClientRegistrationOptions
    print("ClientRegistrationOptions location: mcp.server.auth.settings")
except ImportError:
    try:
        from fastmcp.server.auth import ClientRegistrationOptions  # noqa
        print("ClientRegistrationOptions location: fastmcp.server.auth")
    except ImportError as e:
        print("ClientRegistrationOptions location: NOT FOUND —", e)

abc_methods = [
    n for n in dir(OAuthProvider)
    if not n.startswith("_") and callable(getattr(OAuthProvider, n))
]
print("OAuthProvider methods:", abc_methods)
print(
    "OAuthProvider.__init__ params:",
    list(inspect.signature(OAuthProvider.__init__).parameters),
)
print()

# 6. run_async signature (transport name, path arg, etc.)
if hasattr(m, "run_async"):
    print(
        "run_async params:",
        list(inspect.signature(m.run_async).parameters),
    )
else:
    print("run_async: MISSING — escalate, do not deploy")
print()

# 7. MCP-SDK type imports — try the canonical paths
try:
    from mcp.shared.auth import (
        OAuthClientMetadata, OAuthClientInformationFull, OAuthToken,
    )
    print("mcp.shared.auth imports:", "OK")
except ImportError as e:
    print("mcp.shared.auth imports:", "FAILED —", e)

try:
    from mcp.server.auth.provider import (
        AccessToken, AuthorizationCode, AuthorizationParams,
    )
    print("mcp.server.auth.provider imports:", "OK")
except ImportError as e:
    print("mcp.server.auth.provider imports:", "FAILED —", e)

# 8. Pydantic field shapes — required vs optional
print()
for cls_path, cls_name in [
    ("mcp.shared.auth", "OAuthClientInformationFull"),
    ("mcp.shared.auth", "OAuthClientMetadata"),
    ("mcp.shared.auth", "OAuthToken"),
    ("mcp.server.auth.provider", "AccessToken"),
    ("mcp.server.auth.provider", "AuthorizationCode"),
    ("mcp.server.auth.provider", "AuthorizationParams"),
]:
    try:
        mod = __import__(cls_path, fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        if hasattr(cls, "model_fields"):
            fields = {
                name: (
                    "required" if f.is_required() else f"default={f.default!r}"
                )
                for name, f in cls.model_fields.items()
            }
        else:
            fields = list(getattr(cls, "__dataclass_fields__", {}).keys())
        print(f"{cls_name}:", fields)
    except (ImportError, AttributeError) as e:
        print(f"{cls_name}: FAILED — {e}")
