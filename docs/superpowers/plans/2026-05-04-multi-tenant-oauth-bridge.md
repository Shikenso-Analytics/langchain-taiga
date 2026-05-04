# langchain-taiga Multi-Tenant OAuth Bridge — Implementation Plan (v3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a remote HTTP+OAuth 2.1 mode to `langchain-taiga`'s MCP server so claude.ai can use it as a Custom Connector with per-user Taiga authentication, while preserving the existing stdio mode used by Claude Code, VSCode and the LangChain Toolkit.

**Architecture:** OAuth Bridge pattern — the MCP server is simultaneously an OAuth Authorization Server (talking to claude.ai via Discovery + DCR + PKCE) and an OAuth client (talking to Taiga's native auth API). Implemented as a `fastmcp.server.auth.OAuthProvider` subclass; FastMCP auto-mounts the OAuth + discovery routes. Per-user Taiga JWTs are stored in Postgres (Fernet-encrypted at rest), looked up on every request via FastMCP's auth-context middleware, and surfaced to tool bodies through the verified `AccessToken.claims["taiga_jwt"]`.

**Tech Stack:** Python 3.10+, FastMCP `>=2.14.0,<3.0.0`, `python-taiga`, `cachetools`, `asyncpg`, `cryptography` (Fernet), Postgres 15 (Bitnami subchart), Helm 3, Jenkins, OVH Managed Kubernetes, cert-manager, nginx-ingress.

**Hosting decision:** MCP runs as path-suffix under existing Taiga host: `https://taiga.shikenso.org/mcp`. Reuses existing TLS secret `taiga-tls`, no new certificate. Deployed in the `taiga` namespace alongside Taiga.

**Repo split:**
- Code changes (PRs 1–3) → `langchain-taiga` repo (https://github.com/Shikenso-Analytics/langchain-taiga)
- Helm chart (PR 4) → `taiga` repo (https://github.com/Shikenso-Analytics/taiga) under `deployment/helm/taiga-mcp/`
- Jenkins pipeline (PR 5) → `taiga` repo, separate `Jenkinsfile.taiga-mcp`

---

## What changed vs. the v2 plan (architecture review pivots)

This document replaces the v2 plan archived at `2026-05-04-multi-tenant-oauth-bridge.v2-archive.md`. An external architecture review of v2 surfaced four classes of over- and under-engineering — all integrated below:

| v2 approach | v3 approach | Reason |
|---|---|---|
| Custom `BearerToContextVarMiddleware` ASGI wrapper around FastMCP | **Deleted.** Tools call `fastmcp.server.dependencies.get_access_token()` | FastMCP 2.14+'s built-in auth-context middleware does this when an `OAuthProvider` is attached. |
| Explicit Starlette routes for `/oauth/{authorize,token,register}` and `/.well-known/*` | **Deleted.** `FastMCP(auth=provider)` auto-mounts them per RFC 8414/7591 | Reimplementing what the framework provides. |
| Custom `_taiga_token_context` ContextVar | **Deleted.** Verified Taiga JWT is carried in `AccessToken.claims["taiga_jwt"]` | The framework's `AccessToken` already plays this role. |
| 15 tools + 11 helpers refactored to take keyword-only `token` arg | **Deleted.** Helpers read the JWT inline via `_current_taiga_jwt()` (a 5-line wrapper around `get_access_token()`). Helper/tool signatures stay unchanged. | The previous refactor was load-bearing for our custom middleware design; once the middleware goes away, the kwarg-threading goes with it. |
| Plaintext Taiga `auth_token` / `refresh_token` columns | **Fernet-encrypted** with key from `TAIGA_MCP_FERNET_KEY` | DB compromise alone leaks all active Taiga sessions. Mirrors what FastMCP's own `OAuthProxy` does. |
| Allowlist `https://claude.ai/`, `http://localhost:` | Add `https://claude.com/api/mcp/auth_callback` | Anthropic's connector docs require both `claude.ai` and `claude.com` callbacks. |
| Token endpoint advertises `client_secret_basic`, `client_secret_post` | Also advertise `none` | claude.ai is a public OAuth client — registers with `token_endpoint_auth_method=none`. |
| Discovery metadata at path-aware location only | **Mirror at root path** as defensive fallback for non-spec-compliant clients (MCP TS SDK has an open RFC-8414 issue) | Costs nothing; immunizes against client-side path-handling bugs. |
| Bridge methods `start_authorize` / `complete_login` / `exchange_code` | Renamed to MCP-SDK `OAuthProvider` ABC: `authorize`, `exchange_authorization_code`, `load_access_token`, etc. | Required for FastMCP auto-mount to find them. |
| In-memory `_authorize_states` flagged as v2 tech-debt | **Kept as v1 design with `replicaCount: 1` constraint** | Reviewer confirms: PKCE state lives <10 min, single replica is fine; promoting to Postgres is YAGNI. |
| HTTP 400 / generic error on unknown DCR client | HTTP `400 invalid_client` per RFC 6749 | Anthropic-required so claude.ai re-registers. |

**Net code reduction: ~40% of v2's custom code is gone** — replaced by FastMCP idiomatic surface. v3 is roughly 1200–1500 LOC versus v2's ~2000.

---

## Decisions Captured

| Topic | Decision | Rationale |
|---|---|---|
| Plan location | `langchain-taiga/docs/superpowers/plans/` | Code-co-located, easier to find during execution |
| Plan structure | Single mega-plan, 5 PR sections | PR 1+2 are inseparable; phases build linearly |
| Helm chart repo | `taiga` repo at `deployment/helm/taiga-mcp/` | Same namespace, lifecycle co-located with Taiga deploy |
| Sentry | Out of scope, deferred | `langchain-taiga` is PyPI-public; `kenso_utils` integration is a separate decision |
| Postgres tests | `testcontainers` real Postgres in `tests/integration_tests/` | `FOR UPDATE` row-locking and SQL constraints are part of correctness |
| Hostname | `https://taiga.shikenso.org/mcp` (path-suffix) | Reuse existing cert; no DNS or cert-manager work |
| Auth-server discovery | Path-aware per RFC 8414 §3.1 + root-mirror fallback | Spec-conformant; survives non-conforming clients |
| FastMCP integration | `OAuthProvider` subclass, `FastMCP(auth=...)` | Framework-idiomatic; framework auto-mounts OAuth + well-known routes |
| Per-request Taiga JWT carrier | `AccessToken.claims["taiga_jwt"]` returned by `load_access_token` | Avoids custom middleware + custom ContextVar |
| Replicas | 1 (documented constraint in Helm chart) | In-memory authorize-state lives <10 min; horizontal scaling is YAGNI for v1 |
| Taiga JWT at-rest | Fernet (AES-128-CBC + HMAC-SHA256) | Defense in depth — DB compromise alone insufficient |
| Authlib for AS engine | Deferred (out of scope for v1) | Saves ~400 LOC, but adds a dependency and a non-trivial Starlette adapter; v2 reconsideration |

---

## Phase 0: FastMCP API Surface Verification (BLOCKING — run before PR 1)

The plan makes assumptions about FastMCP 2.14+'s public surface that need to be locked down to a specific patch version **before** any of the code in PRs 1–3 is written. Different patch versions ship slightly different signatures for `FastMCP.__init__`, `FastMCP.run()`, the OAuth-attachment mechanism, and the ASGI-app accessor. Mismatches between plan code and runtime API would only surface at deploy time.

**Run this once, paste the output into the PR description of PR 1:**

```python
# scripts/probe_fastmcp.py
import inspect, fastmcp
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
from fastmcp.server.auth import OAuthProvider, ClientRegistrationOptions
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
```

```bash
poetry run python scripts/probe_fastmcp.py
```

**Three outcomes that change the plan code:**

| Probe finding | Implication | Plan section to adjust |
|---|---|---|
| `FastMCP.__init__` accepts `auth=` | Use the constructor — pass `auth=provider` at FastMCP creation | Task 3.1 main() — set provider before factory call |
| `FastMCP.__init__` does NOT accept `auth=` | Fallback: set `mcp.auth = provider` BEFORE calling `streamable_http_app()` | Adds one line; same lifespan story |
| `FastMCP.__init__` accepts `lifespan=` | Pass `lifespan=_lifespan` to constructor | Cleanest path |
| `FastMCP.__init__` does NOT accept `lifespan=` but `mcp.run` does | Pass via `mcp.run(lifespan=...)` | OK |
| Neither accepts `lifespan=` | Wrap `streamable_http_app()` in `Starlette(lifespan=...)` and run uvicorn ourselves | More plumbing; documented in Task 3.1 |
| `streamable_http_app` missing, `http_app` present | Use `http_app()` instead | One-line swap in Task 3.1 |
| MCP-SDK imports fail | Find the actual locations via `python -c "import mcp; print(dir(mcp))"` | Adjust imports in Task 2.5 |

**The plan code in PR 2/PR 3 is written for the most common shape (`auth=` and `lifespan=` both at construction).** The post-probe step at the start of PR 3 is mandatory: read the probe output, adjust if needed, *then* write the code.

---

## Amendment v3.4 — In-Memory Storage (PR 2 + PR 4 reduced)

**Decision (2026-05-04):** Drop Postgres entirely. ~30 users on a single-replica deployment do not need SaaS-grade persistence. Postgres' purpose was to survive pod restart without forcing re-OAuth — at this scale, asking 30 users to re-click "Connect" in claude.ai a few times per quarter is cheaper than the dependency footprint.

### What changes vs. v3.3

| Concern | v3.3 (Postgres) | v3.4 (in-memory) |
|---|---|---|
| Token + DCR storage | Postgres tables, HMAC-hashed tokens, Fernet-encrypted Taiga JWTs | Plain dataclasses in async-safe `dict`s; nothing at rest because nothing's persisted |
| Concurrent refresh serialization | `SELECT … FOR UPDATE` row lock | `asyncio.Lock` per token (in `dict`) |
| Postgres subchart in Helm | Bitnami `postgresql-13.2.24` + 1Gi PVC | None — drop the subchart and PVC |
| Pod restart UX | Tokens persist across restart | All sessions wiped; users re-OAuth (one click + Taiga password). claude.ai re-DCRs automatically on `invalid_client` per RFC 6749. |
| Dependencies | `asyncpg`, `cryptography`, `testcontainers` | Only `httpx`, `jinja2`, `respx`, `responses` |
| Test infra | `tests/integration_tests/` with Docker-spawned Postgres | Plain `tests/unit_tests/` — no Docker required |
| Code volume in PR 2 | ~600 LOC (TokenStore + tests) | ~250 LOC (InMemoryStore + tests) |

### What stays the same

- All Phase 0 deltas (import paths, AccessToken from `fastmcp.server.auth`, AuthorizationCode requires `redirect_uri_provided_explicitly`, etc.) still apply — they're FastMCP API surface, independent of where state is stored.
- `TaigaOAuthProvider` (Task 2.5) interface is unchanged: same MCP-SDK ABC methods (`register_client`, `get_client`, `authorize`, `load_access_token`, etc.). Only the underlying `_store` field type changes.
- Login page, redirect_uri allowlist (claude.ai + claude.com + localhost), `none` token-endpoint-auth-method support, root-path metadata mirror — all unchanged.
- `replicaCount: 1` constraint in Helm — still hard-pinned (in-memory state is per-pod).
- E2E token-flow security test (Task 2.7) — same dual-mock-stack (`respx` + `responses`), runs in unit-test land now.

### Concrete `InMemoryStore` design

Single class, ~80 LOC, replaces `TokenStore` from v3.3:

```python
# langchain_taiga/auth/store.py
from __future__ import annotations
import asyncio
import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, List, Optional


@dataclass
class AccessTokenRecord:
    token: str
    taiga_auth_token: str
    taiga_refresh_token: str
    taiga_user_id: int
    taiga_username: str
    client_id: str
    scopes: List[str]
    expires_at: datetime


@dataclass
class AuthCodeRecord:
    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    taiga_auth_token: str
    taiga_refresh_token: str
    taiga_user_id: int
    taiga_username: str
    scopes: List[str]
    expires_at: datetime


@dataclass
class ClientRecord:
    client_id: str
    client_secret: Optional[str]  # None on lookup; only set right after register_client
    redirect_uris: List[str]
    client_name: str
    token_endpoint_auth_method: str


class InMemoryStore:
    """Process-local OAuth state. Pod restart wipes everything; users re-OAuth.

    Acceptable for single-replica deployments at ~30-user scale. For
    multi-replica or persistence requirements, swap to a Postgres-backed
    implementation with the same interface.
    """

    def __init__(self) -> None:
        self._access_tokens: dict[str, AccessTokenRecord] = {}
        self._auth_codes: dict[str, AuthCodeRecord] = {}
        self._clients: dict[str, ClientRecord] = {}
        # Per-token asyncio.Lock for serializing refresh attempts. Replaces
        # SELECT ... FOR UPDATE. Only meaningful within one event loop / one pod.
        self._refresh_locks: dict[str, asyncio.Lock] = {}

    @classmethod
    async def from_env(cls) -> "InMemoryStore":
        """Match the from_env() shape of the Postgres design for swap-out symmetry."""
        return cls()

    # --- Access tokens ----------------------------------------------------

    async def store_access_token(
        self, *, token: str, taiga_auth_token: str, taiga_refresh_token: str,
        taiga_user_id: int, taiga_username: str, client_id: str,
        scopes: List[str], expires_at: datetime,
    ) -> None:
        self._access_tokens[token] = AccessTokenRecord(
            token=token,
            taiga_auth_token=taiga_auth_token,
            taiga_refresh_token=taiga_refresh_token,
            taiga_user_id=taiga_user_id,
            taiga_username=taiga_username,
            client_id=client_id,
            scopes=list(scopes),
            expires_at=expires_at,
        )

    async def lookup_access_token(self, token: str) -> Optional[AccessTokenRecord]:
        record = self._access_tokens.get(token)
        if record is None:
            return None
        # Defensive: don't return expired records (cleanup may not have run yet)
        if record.expires_at < datetime.now(timezone.utc):
            return None
        return record

    async def update_taiga_token(
        self, *, token: str, taiga_auth_token: str,
        taiga_refresh_token: str, expires_at: datetime,
    ) -> None:
        record = self._access_tokens.get(token)
        if record is None:
            return
        record.taiga_auth_token = taiga_auth_token
        record.taiga_refresh_token = taiga_refresh_token
        record.expires_at = expires_at

    @asynccontextmanager
    async def refresh_lock(
        self, token: str
    ) -> AsyncIterator[Optional[AccessTokenRecord]]:
        """Per-token lock for atomic refresh. Replaces SELECT … FOR UPDATE."""
        lock = self._refresh_locks.setdefault(token, asyncio.Lock())
        async with lock:
            yield self._access_tokens.get(token)

    # --- Auth codes -------------------------------------------------------

    async def store_authorization_code(
        self, *, code: str, client_id: str, redirect_uri: str,
        code_challenge: str, code_challenge_method: str,
        taiga_auth_token: str, taiga_refresh_token: str,
        taiga_user_id: int, taiga_username: str,
        scopes: List[str], expires_at: datetime,
    ) -> None:
        self._auth_codes[code] = AuthCodeRecord(
            code=code, client_id=client_id, redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            taiga_auth_token=taiga_auth_token,
            taiga_refresh_token=taiga_refresh_token,
            taiga_user_id=taiga_user_id, taiga_username=taiga_username,
            scopes=list(scopes), expires_at=expires_at,
        )

    async def consume_authorization_code(self, code: str) -> Optional[AuthCodeRecord]:
        """Single-use atomic pop. Returns None if missing or expired."""
        record = self._auth_codes.pop(code, None)
        if record is None:
            return None
        if record.expires_at < datetime.now(timezone.utc):
            return None
        return record

    # --- DCR --------------------------------------------------------------

    async def register_client(
        self, *, client_id: str, client_secret: str,
        redirect_uris: List[str], client_name: str,
        token_endpoint_auth_method: str = "client_secret_basic",
    ) -> None:
        self._clients[client_id] = ClientRecord(
            client_id=client_id,
            client_secret=client_secret,  # plaintext in-memory; not at rest
            redirect_uris=list(redirect_uris),
            client_name=client_name,
            token_endpoint_auth_method=token_endpoint_auth_method,
        )

    async def lookup_client(self, client_id: str) -> Optional[ClientRecord]:
        """Returns metadata only. Use ``verify_client_secret`` for /token validation."""
        record = self._clients.get(client_id)
        if record is None:
            return None
        return ClientRecord(
            client_id=record.client_id,
            client_secret=None,  # never expose plaintext on lookup
            redirect_uris=record.redirect_uris,
            client_name=record.client_name,
            token_endpoint_auth_method=record.token_endpoint_auth_method,
        )

    async def verify_client_secret(self, client_id: str, presented: str) -> bool:
        record = self._clients.get(client_id)
        if record is None or record.client_secret is None:
            return False
        return hmac.compare_digest(record.client_secret, presented)

    # --- Cleanup ----------------------------------------------------------

    async def cleanup_expired(self) -> int:
        """Sweep expired access tokens + auth codes. Returns total purged."""
        now = datetime.now(timezone.utc)
        purged_tokens = [k for k, v in self._access_tokens.items() if v.expires_at < now]
        for k in purged_tokens:
            self._access_tokens.pop(k, None)
            self._refresh_locks.pop(k, None)
        purged_codes = [k for k, v in self._auth_codes.items() if v.expires_at < now]
        for k in purged_codes:
            self._auth_codes.pop(k, None)
        return len(purged_tokens) + len(purged_codes)

    async def close(self) -> None:
        """No-op — kept for API symmetry with future persistent backends."""
        return
```

### Test changes (Task 2.2)

- File moves to `tests/unit_tests/test_store.py` (no longer integration_tests).
- No `fresh_pool` / `fresh_store` testcontainers fixtures. Just `pytest` with `pytest-asyncio` and a fresh `InMemoryStore()` per test.
- Drop `test_taiga_jwt_is_encrypted_at_rest` (no rest, nothing to encrypt).
- Keep `test_concurrent_refresh_serialized` — `asyncio.Lock` semantics. Test exercises two coroutines hitting `refresh_lock` simultaneously and verifies serialized order.
- Keep `test_authorization_code_single_use` — pop semantics still hold.
- New `test_lookup_expired_returns_none` — ensures defensive expiry check works without DB-side TTL.

### What changes in PR 4 (Helm)

Drop entirely:
- `Chart.yaml` `dependencies` — no Postgres subchart.
- `charts/postgresql-13.2.24.tgz`.
- `values.yaml` `postgresql:` block.
- `templates/secret.yaml` keys: `TAIGA_MCP_TOKEN_SECRET`, `TAIGA_MCP_FERNET_KEY` (HMAC + Fernet keys are dead — nothing's at rest). Only `OPENAI_API_KEY` remains in the secret.
- `templates/deployment.yaml` `POSTGRES_PASSWORD` env, `TAIGA_MCP_DB_URL` env-var construction.
- `templates/networkpolicy.yaml` egress rule to Postgres pod.

Keep:
- `replicaCount: 1` hard-pinned.
- `strategy: Recreate` (no rolling).
- Path-suffix ingress with shared `taiga-tls`, root + path-aware `/.well-known/oauth-*` routes.
- NetworkPolicy ingress from claude.ai CIDR + nginx.

### What changes in PR 5 (Jenkinsfile)

Drop the Bitnami-PG password passthrough block (`PG_PASS`, `PG_ADMIN_PASS`, the `--set postgresql.auth.*` flags). The `helm upgrade --install` becomes a plain invocation. Pre-flight PyPI version check stays.

### Risks introduced by this amendment

| Risk | Mitigation |
|---|---|
| Pod restart drops all sessions → 30 users re-OAuth simultaneously | Acceptable at scale. Communicate via Slack before planned restarts. |
| `asyncio.Lock` is event-loop-scoped, not process-scoped | `replicaCount: 1` ensures one event loop per cluster. Multi-replica would break this — already a hard constraint in v3.3. |
| Memory leak: `_refresh_locks` grows unbounded if cleanup doesn't run | `cleanup_expired` removes the lock alongside the token. Periodic cleanup loop covers it. Worst case at 30 users × 1 token each: ~30 small dict entries — negligible. |
| Lost audit trail (no DB to query historical state) | If audit becomes a real need: add structured logging with request-correlation-IDs to stdout, ship to Loki/Sentry. Not in v1 scope. |

---

## Code-Audit Corrections to Original Spec

These were verified against `langchain_taiga/tools/taiga_tools.py` and corrected vs. the user's original draft:

1. **Direct `requests.post` calls:** Only **one** tool (`sort_kanban_by_rice_tool` line 2374) uses raw `requests.post` with a manual `Authorization: Bearer {api.token}` header. The other two the draft listed (`promote_issue_to_userstory_tool`, `update_entity_by_ref_tool`) use `api.raw_request.post(...)` — `python-taiga`'s built-in wrapper that pulls the token from the `TaigaAPI` instance automatically. **Net effect:** instantiating a fresh `TaigaAPI(host=..., token=user_token)` per request makes those two tools work transparently; only `sort_kanban_by_rice_tool` needs explicit token injection in the header (and that's a one-line change inside the tool body that reads `_current_taiga_jwt()`).

2. **Cached helpers needing user-scoping:** Inventory from `taiga_tools.py`:
   - `get_taiga_api()` — keep ENV-cached for stdio; HTTP path bypasses cache
   - `get_project()`, `get_user()`, `find_users()`, `get_status()`, `find_issue_type_ids()`, `find_severity_ids()`, `find_priority_ids()`, `find_status_ids()`, `list_milestones()`, `list_all_statuses()`, `list_all_tags()` — all `@cached(cache=...)`, all need user-scoped cache keys
   - `get_custom_attribute_definitions()` — uses dict-style cache (`custom_attr_definitions_cache[key] = ...`); needs user-scoping in the cache key tuple

3. **Tool count:** 15 `@tool`-decorated tools, all registered via `_register_mcp_tools()` at end of `taiga_tools.py`. **In v3 their signatures do not change** — they call helpers as before, and helpers read the per-request JWT via the FastMCP dependency.

4. **Test infra:** Existing tests in `tests/unit_tests/` use `--disable-socket`. New OAuth/Postgres tests live in `tests/integration_tests/` (directory does not yet exist; create it).

---

## File Structure

### `langchain-taiga` repo (PRs 1, 2, 3)

```
langchain_taiga/
├── __init__.py                 # unchanged
├── mcp.py                      # unchanged
├── mcp_server.py               # unchanged (stdio entry)
├── remote_server.py            # NEW — HTTP entry point (PR 3); ~80 LOC
├── toolkits.py                 # unchanged (LangChain Toolkit BC contract)
├── tools/
│   ├── __init__.py             # unchanged
│   └── taiga_tools.py          # MODIFIED (PR 1) — 3 small additions, no signature changes
└── auth/                       # NEW (PR 2)
    ├── __init__.py
    ├── token_store.py          # Postgres + HMAC + Fernet
    ├── taiga_client.py         # Wrapper for /api/v1/auth + /refresh
    ├── provider.py             # FastMCP OAuthProvider subclass (was oauth_bridge.py in v2)
    └── login_page.py           # HTML form rendering
tests/
├── unit_tests/
│   └── test_token_propagation.py    # NEW (PR 1)
└── integration_tests/          # NEW (PR 2)
    ├── __init__.py
    ├── conftest.py
    ├── test_token_store.py
    ├── test_taiga_client.py
    ├── test_provider.py        # OAuthProvider methods
    ├── test_login_routes.py    # /oauth/login GET+POST smoke tests
    └── test_e2e_token_flow.py  # End-to-end Multi-Tenant security check
docs/
└── superpowers/plans/2026-05-04-multi-tenant-oauth-bridge.md  # this file
pyproject.toml                  # MODIFIED (PR 2 deps + PR 3 scripts entry)
```

### `taiga` repo (PRs 4, 5)

```
deployment/helm/taiga-mcp/
├── Chart.yaml
├── values.yaml                 # replicaCount: 1, Fernet key reference
├── README.md
├── Dockerfile                  # NEW (PR 5) — pip install langchain-taiga
└── templates/
    ├── _helpers.tpl
    ├── NOTES.txt
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── configmap.yaml
    ├── secret.yaml
    ├── serviceaccount.yaml
    ├── networkpolicy.yaml
    └── pdb.yaml
Jenkinsfile.taiga-mcp           # NEW (PR 5)
```

---

# PR 1: Per-Request Token via FastMCP `AccessToken` Claims

**Goal of this PR:** Plumb the verified Taiga JWT from FastMCP's `get_access_token()` dependency through to the cached helpers, with user-scoped cache keys. **Tool and helper signatures stay unchanged.** This is a small PR — three new helpers and a decorator change on each `@cached` function.

**Branch:** `feat/per-request-taiga-jwt`

**What this PR is NOT:** It is not a refactor that threads `token` kwargs through the API. The v2 plan did that; v3 doesn't, because FastMCP's auth-context middleware + `AccessToken.claims` makes the whole pattern unnecessary.

## Task 1.0: Confirm FastMCP version pin and run Phase 0 probe

PR 1 imports `from fastmcp.server.dependencies import get_access_token` — that symbol exists from FastMCP 2.14+. Before any code changes, confirm the existing `pyproject.toml` floor is at least 2.14, and run the Phase 0 probe to capture the actual surface of the installed patch version.

- [ ] **Step 1: Inspect current FastMCP pin**

```bash
cd /home/wahed/workspace/langchain-taiga
grep -E '^fastmcp\s*=' pyproject.toml
```

If the floor is below `2.14.0` — bump it to `>=2.14.0,<3.0.0`:

```toml
fastmcp = ">=2.14.0,<3.0.0"
```

- [ ] **Step 2: Run the Phase 0 probe (also documented in plan §"Phase 0")**

```bash
poetry install
mkdir -p scripts
# Paste the probe script from plan §Phase 0 into scripts/probe_fastmcp.py
poetry run python scripts/probe_fastmcp.py | tee phase0-probe.txt
```

Save `phase0-probe.txt` — its contents go into the PR 1 description so reviewers (and future you) understand which FastMCP shape this branch is coded against.

- [ ] **Step 3: Confirm probe output meets minimum requirements**

The plan's PR 2/PR 3 code requires:
- `FastMCP.__init__` accepts `auth=` AND `lifespan=`
- `mcp.run_async` exists with `transport`, `host`, `port`, `path` parameters
- `from fastmcp.server.dependencies import get_access_token` succeeds
- `from fastmcp.server.auth import OAuthProvider, ClientRegistrationOptions` succeeds
- `from mcp.shared.auth import OAuthClientMetadata, OAuthClientInformationFull, OAuthToken` succeeds
- `from mcp.server.auth.provider import AccessToken, AuthorizationCode, AuthorizationParams` succeeds

If any of these fails, **stop**: the plan needs a localized update to point at the actual symbols/parameters in your installed version. Do not paper over with try/except wrappers — make the import clear and the constraint explicit in the version pin.

- [ ] **Step 4: Commit pin bump (if any)**

```bash
git checkout -b feat/per-request-taiga-jwt
git add pyproject.toml poetry.lock scripts/probe_fastmcp.py phase0-probe.txt
git commit -m "chore: bump fastmcp floor to 2.14 + record Phase 0 probe output"
```

## Task 1.1: Test scaffolding

**Files:**
- Create: `tests/unit_tests/test_token_propagation.py`

- [ ] **Step 1: Confirm existing tests pass**

```bash
make test
```
Expected: existing tests pass. Record baseline.

- [ ] **Step 2: Create empty test file with shared fixtures**

```python
# tests/unit_tests/test_token_propagation.py
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
```

- [ ] **Step 3: Confirm collection works**

```bash
make test TEST_FILE=tests/unit_tests/test_token_propagation.py
```
Expected: 0 tests collected, exit 0/5.

- [ ] **Step 4: Commit baseline**

```bash
git add tests/unit_tests/test_token_propagation.py
git commit -m "test: scaffold token propagation test file"
```

## Task 1.2: `get_taiga_api()` accepts optional token

**Files:**
- Modify: `langchain_taiga/tools/taiga_tools.py:163-174`
- Modify: `tests/unit_tests/test_token_propagation.py`

- [ ] **Step 1: Failing tests**

```python
def test_get_taiga_api_no_arg_uses_env_cache():
    from langchain_taiga.tools.taiga_tools import get_taiga_api, taiga_api_cache
    taiga_api_cache.clear()
    from unittest.mock import patch
    with patch("langchain_taiga.tools.taiga_tools.TaigaAPI") as MockAPI:
        instance = MagicMock()
        MockAPI.return_value = instance
        first = get_taiga_api()
        second = get_taiga_api()
        assert first is second
        MockAPI.assert_called_once()


def test_get_taiga_api_with_token_returns_fresh_client():
    from langchain_taiga.tools.taiga_tools import get_taiga_api
    from unittest.mock import patch
    with patch("langchain_taiga.tools.taiga_tools.TaigaAPI") as MockAPI:
        MockAPI.side_effect = [MagicMock(), MagicMock()]
        a = get_taiga_api(token="user_a")
        b = get_taiga_api(token="user_b")
        assert a is not b
        assert MockAPI.call_count == 2
        assert MockAPI.call_args_list[0].kwargs.get("token") == "user_a"


def test_get_taiga_api_per_request_not_cached():
    from langchain_taiga.tools.taiga_tools import get_taiga_api
    from unittest.mock import patch
    with patch("langchain_taiga.tools.taiga_tools.TaigaAPI") as MockAPI:
        MockAPI.side_effect = [MagicMock(), MagicMock()]
        a = get_taiga_api(token="same")
        b = get_taiga_api(token="same")
        assert a is not b
        assert MockAPI.call_count == 2
```

- [ ] **Step 2: Run to confirm failure**

```bash
make test TEST_FILE=tests/unit_tests/test_token_propagation.py
```
Expected: FAIL — `get_taiga_api` doesn't accept `token`.

- [ ] **Step 3: Refactor `get_taiga_api()`**

Replace lines 163–174 of `langchain_taiga/tools/taiga_tools.py`:

```python
@cached(cache=taiga_api_cache)
def _get_taiga_api_from_env() -> TaigaAPI:
    """ENV-credentialed client, cached. Used by stdio mode."""
    if TAIGA_USERNAME and TAIGA_PASSWORD:
        taiga_api = TaigaAPI(host=TAIGA_API_URL)
        taiga_api.auth(TAIGA_USERNAME, TAIGA_PASSWORD)
    elif TAIGA_TOKEN:
        taiga_api = TaigaAPI(host=TAIGA_API_URL, token=TAIGA_TOKEN)
    else:
        raise ValueError("Taiga credentials not provided.")
    return taiga_api


def get_taiga_api(token: Optional[str] = None) -> TaigaAPI:
    """Get a Taiga API client.

    - No ``token`` → ENV-cached singleton (stdio path).
    - With ``token`` → fresh per-request ``TaigaAPI(host=..., token=token)``,
      uncached. Multi-tenant HTTP path.
    """
    if token is not None:
        return TaigaAPI(host=TAIGA_API_URL, token=token)
    return _get_taiga_api_from_env()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
make test TEST_FILE=tests/unit_tests/test_token_propagation.py
make test  # full suite — must remain green
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_taiga/tools/taiga_tools.py tests/unit_tests/test_token_propagation.py
git commit -m "feat(taiga): get_taiga_api accepts optional per-request token"
```

## Task 1.3: User-scoped cache key + `_current_taiga_jwt()` helper

**Files:**
- Modify: `langchain_taiga/tools/taiga_tools.py` (add helpers near top)
- Modify: `tests/unit_tests/test_token_propagation.py`

This task adds two small helpers that read the per-request user identity from FastMCP's auth-context middleware:
- `_current_taiga_jwt()` — returns the per-user Taiga JWT or `None` outside a request.
- `_user_scoped_key()` — cachetools key function that prepends a `user_id`-derived scope.

Both wrap `fastmcp.server.dependencies.get_access_token()` and degrade gracefully to "default scope / ENV auth" outside of an active HTTP request. **No new ContextVar, no custom middleware.**

- [ ] **Step 1: Failing tests**

Append to `tests/unit_tests/test_token_propagation.py`:

```python
def test_current_taiga_jwt_none_outside_request(monkeypatch):
    from langchain_taiga.tools.taiga_tools import _current_taiga_jwt
    with patched_access_token(monkeypatch, claims=None):
        assert _current_taiga_jwt() is None


def test_current_taiga_jwt_reads_from_claims(monkeypatch):
    from langchain_taiga.tools.taiga_tools import _current_taiga_jwt
    with patched_access_token(
        monkeypatch, claims={"taiga_jwt": "user_jwt_xyz", "user_id": 7}
    ):
        assert _current_taiga_jwt() == "user_jwt_xyz"


def test_current_taiga_jwt_handles_missing_claim(monkeypatch):
    """If somehow the claim is absent on a verified token, return None — never crash."""
    from langchain_taiga.tools.taiga_tools import _current_taiga_jwt
    with patched_access_token(monkeypatch, claims={"user_id": 7}):
        assert _current_taiga_jwt() is None


def test_user_scoped_key_default_scope_outside_request(monkeypatch):
    from langchain_taiga.tools.taiga_tools import _user_scoped_key
    with patched_access_token(monkeypatch, claims=None):
        key = _user_scoped_key("slug")
    assert key[0] == "default"


def test_user_scoped_key_distinct_per_user(monkeypatch):
    from langchain_taiga.tools.taiga_tools import _user_scoped_key
    with patched_access_token(monkeypatch, claims={"user_id": 1}):
        a = _user_scoped_key("slug")
    with patched_access_token(monkeypatch, claims={"user_id": 2}):
        b = _user_scoped_key("slug")
    with patched_access_token(monkeypatch, claims=None):
        d = _user_scoped_key("slug")
    assert a != b != d != a


def test_user_scoped_key_64_bit_scope(monkeypatch):
    """16 hex chars (64 bits) keeps collision risk negligible."""
    from langchain_taiga.tools.taiga_tools import _user_scoped_key
    with patched_access_token(monkeypatch, claims={"user_id": 42}):
        key = _user_scoped_key("slug")
    assert len(key[0]) == 16
```

- [ ] **Step 2: Run to confirm failure**

```bash
make test TEST_FILE=tests/unit_tests/test_token_propagation.py
```
Expected: FAIL — helpers not defined.

- [ ] **Step 3: Implement the helpers**

Add to `langchain_taiga/tools/taiga_tools.py` immediately after the cache-instantiation block (after line 53, before `ENTITY_TYPE_MAPPING`):

```python
import hashlib
from typing import Any

# FastMCP 2.14+ exposes the verified AccessToken for the current HTTP request
# via this dependency. In stdio mode (no active request context), it raises
# LookupError. Both helpers below catch that and fall through to defaults.
from fastmcp.server.dependencies import get_access_token


def _current_taiga_jwt() -> Optional[str]:
    """Return the per-request Taiga JWT, or None outside an authenticated request.

    The OAuth provider (PR 2) populates ``AccessToken.claims["taiga_jwt"]`` in
    its ``load_access_token()`` method — that's the single producer. Helpers
    consume via this function. Stdio path: no auth context → None → ENV fallback.
    """
    try:
        tok = get_access_token()
    except (LookupError, RuntimeError):
        return None
    return tok.claims.get("taiga_jwt") if tok else None


def _user_scoped_key(*args: Any, **kwargs: Any) -> tuple:
    """cachetools key function that prepends a user-derived scope.

    Scope is sha256(user_id)[:16] when the request carries a verified
    AccessToken with ``user_id`` claim, else ``"default"``. 64 bits is enough
    to keep birthday-collision probability under 1e-9 at realistic scale, while
    not bloating the cache key.

    Why hash a non-secret? Consistent key shape (always 16 hex chars), and
    leaves the door open to switching the scope source without changing the
    cache layout.
    """
    user_scope = "default"
    try:
        tok = get_access_token()
        if tok is not None:
            uid = tok.claims.get("user_id")
            if uid is not None:
                user_scope = hashlib.sha256(str(uid).encode()).hexdigest()[:16]
    except (LookupError, RuntimeError):
        pass
    return (user_scope, *args, *sorted(kwargs.items()))
```

Add `import hashlib` to the top imports if not present.

- [ ] **Step 4: Run tests**

```bash
make test TEST_FILE=tests/unit_tests/test_token_propagation.py
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add langchain_taiga/tools/taiga_tools.py tests/unit_tests/test_token_propagation.py
git commit -m "feat(taiga): _current_taiga_jwt + _user_scoped_key via FastMCP AccessToken"
```

## Task 1.4: Wire helpers + tools to the per-request JWT

**Files:**
- Modify: `langchain_taiga/tools/taiga_tools.py` — every `@cached` decorator + every helper/tool body

This is the small, mechanical wiring step. **No signature changes** — only:
1. Add `key=_user_scoped_key` to every `@cached(cache=...)` decorator.
2. Inside every helper that calls `get_taiga_api()`, change to `get_taiga_api(token=_current_taiga_jwt())`.
3. Inside `sort_kanban_by_rice_tool` (the only tool with a hand-rolled `Authorization: Bearer {api.token}` header), the same one-line change applies — `api = get_taiga_api(token=_current_taiga_jwt())` makes `api.token` per-request.

The `python-taiga` `api.raw_request.post(...)` calls in `promote_issue_to_userstory_tool` and `update_entity_by_ref_tool` already pull the token from the `TaigaAPI` instance — they get the right behaviour automatically once their `get_taiga_api()` call is per-request.

- [ ] **Step 1: Cache-isolation test**

Append to `tests/unit_tests/test_token_propagation.py`:

```python
def test_cache_isolation_get_project(monkeypatch):
    """Two different users looking up the same slug → two cache entries."""
    from langchain_taiga.tools import taiga_tools
    taiga_tools.project_cache.clear()

    api_clients_by_user = {}

    def fake_get_taiga_api(token=None):
        if token not in api_clients_by_user:
            api = MagicMock()
            proj = MagicMock(); proj.id = 999; proj._token_used = token
            api.projects.get_by_slug.return_value = proj
            api_clients_by_user[token] = api
        return api_clients_by_user[token]

    monkeypatch.setattr(taiga_tools, "get_taiga_api", fake_get_taiga_api)

    with patched_access_token(
        monkeypatch, claims={"user_id": 1, "taiga_jwt": "alice_jwt"}
    ):
        p_a = taiga_tools.get_project("shikenso-development")
    with patched_access_token(
        monkeypatch, claims={"user_id": 2, "taiga_jwt": "bob_jwt"}
    ):
        p_b = taiga_tools.get_project("shikenso-development")

    assert p_a is not p_b
    assert p_a._token_used == "alice_jwt"
    assert p_b._token_used == "bob_jwt"
    assert len(taiga_tools.project_cache) == 2


def test_default_scope_caches_when_no_request(monkeypatch):
    """Stdio path keeps single shared cache (today's behaviour preserved)."""
    from langchain_taiga.tools import taiga_tools
    taiga_tools.project_cache.clear()

    calls = {"n": 0}

    def fake_get_taiga_api(token=None):
        calls["n"] += 1
        api = MagicMock(); api.projects.get_by_slug.return_value = MagicMock(id=1)
        return api

    monkeypatch.setattr(taiga_tools, "get_taiga_api", fake_get_taiga_api)
    with patched_access_token(monkeypatch, claims=None):
        taiga_tools.get_project("shikenso-development")
        taiga_tools.get_project("shikenso-development")
    assert calls["n"] == 1, "stdio path should cache hit on second call"
```

- [ ] **Step 2: Run to confirm failure**

```bash
make test TEST_FILE=tests/unit_tests/test_token_propagation.py::test_cache_isolation_get_project
```
Expected: FAIL — `get_project` doesn't yet user-scope, both lookups land on the same cache entry.

- [ ] **Step 3: Apply the wiring change to every cached helper**

For each cached helper in `taiga_tools.py`, two minimal edits:

**(a) Decorator:** add `key=_user_scoped_key` to the `@cached` call.

Affected (line numbers approximate from the existing tree):

| Helper | Line |
|---|---|
| `get_project` | 178 |
| `get_user` | 196 |
| `find_users` | 218 |
| `get_status` | 272 |
| `find_issue_type_ids` | 351 |
| `find_severity_ids` | 360 |
| `find_priority_ids` | 369 |
| `find_status_ids` | 384 |
| `list_milestones` | 406 |
| `list_all_statuses` | 508 |
| `list_all_tags` | 617 |

Also `get_custom_attribute_definitions` (line 74) which uses a manual dict-style cache: prepend a `user_scope` element to its `cache_key` tuple via the same logic.

**(b) Body:** inside each helper that calls `get_taiga_api()`, change to `get_taiga_api(token=_current_taiga_jwt())`. The line that says `get_taiga_api()` becomes `get_taiga_api(token=_current_taiga_jwt())`. That's it. **No signature change.**

Concrete diff for `get_project` (the canonical example):

```diff
-@cached(cache=project_cache)
+@cached(cache=project_cache, key=_user_scoped_key)
 def get_project(slug: str) -> Optional[Project]:
     """Get project by slug with auto-refreshing 5-minute, user-scoped cache."""
     if "/project/" in slug:
         match = re.search(r"/project/([^/]+)", slug)
         if match:
             slug = match.group(1)
     try:
-        project = get_taiga_api().projects.get_by_slug(slug)
+        project = get_taiga_api(token=_current_taiga_jwt()).projects.get_by_slug(slug)
         return project
     except Exception as e:
         print(f"Error fetching project {slug}: {e}")
         return None
```

For `sort_kanban_by_rice_tool` (the only tool with a manual Bearer header), at line ~2352:

```diff
 base_url = TAIGA_URL.rstrip("/")
-api = get_taiga_api()
+api = get_taiga_api(token=_current_taiga_jwt())
 headers = {
     "Authorization": f"Bearer {api.token}",
     ...
 }
```

For `get_custom_attribute_definitions` (manual dict cache, line ~82):

```diff
 def get_custom_attribute_definitions(project: Project, norm_type: str) -> Dict[str, Dict]:
-    cache_key = (project.id, norm_type)
+    user_scope = _user_scoped_key()[0]  # extract just the scope element
+    cache_key = (user_scope, project.id, norm_type)
     if cache_key in custom_attr_definitions_cache:
```

- [ ] **Step 4: Run all tests — every helper isolated**

```bash
make test TEST_FILE=tests/unit_tests/test_token_propagation.py
make test  # full unit suite
```
Expected: PASS. Existing tests that do not patch `get_access_token` will see `LookupError` raised by the (yet-to-be-stubbed) dependency, but our `_current_taiga_jwt()` swallows it → returns `None` → ENV path. Same observable behaviour as before, plus per-user cache scoping when there's an active request.

- [ ] **Step 5: Run LangChain Toolkit smoke test**

```bash
poetry run python -c "
from langchain_taiga.toolkits import TaigaToolkit
print(len(TaigaToolkit().get_tools()))
"
```
Expected: prints `10`.

- [ ] **Step 6: Commit**

```bash
git add langchain_taiga/tools/taiga_tools.py tests/unit_tests/test_token_propagation.py
git commit -m "feat(taiga): wire cached helpers to per-request Taiga JWT"
```

## Task 1.5: Final verification + PR

- [ ] **Step 1: Stdio smoke test against real Taiga**

```bash
TAIGA_URL=https://taiga.shikenso.org \
TAIGA_API_URL=https://taiga.shikenso.org \
TAIGA_USERNAME=$YOUR_USERNAME \
TAIGA_PASSWORD=$YOUR_PASSWORD \
OPENAI_API_KEY=$OPENAI_API_KEY \
poetry run python -c "
from langchain_taiga.tools.taiga_tools import search_entities_tool
print(search_entities_tool.invoke({'project_slug': 'shikenso-development', 'query': 'test', 'entity_type': 'task'})[:200])
"
```
Expected: returns JSON list, no exceptions. Stdio path is unaffected.

- [ ] **Step 2: Open PR**

```bash
git push -u origin feat/per-request-taiga-jwt
gh pr create --title "feat: per-request Taiga JWT via FastMCP AccessToken claims" \
  --body "$(cat <<'EOF'
## Summary
- `get_taiga_api()` accepts an optional `token` arg; with one, returns a fresh per-request `TaigaAPI` instance instead of the cached ENV singleton.
- New helpers `_current_taiga_jwt()` and `_user_scoped_key()` read identity from `fastmcp.server.dependencies.get_access_token()`. Outside an HTTP request they degrade to `None` / `"default"` scope so the stdio path is unchanged.
- All `@cached` helpers gain `key=_user_scoped_key` and call `get_taiga_api(token=_current_taiga_jwt())` internally. **Helper signatures do not change.**
- `sort_kanban_by_rice_tool` (the only tool with a manual Bearer header) now constructs that header from a per-request client.
- LangChain Toolkit imports cleanly; existing unit tests stay green.

## What this unlocks
The OAuth bridge in PR 2 populates `AccessToken.claims["taiga_jwt"]` and `["user_id"]` in its `load_access_token()`. From PR 2 forward, every tool call automatically uses the requesting user's Taiga session.

## Test plan
- [ ] `make test` passes — existing + new `test_token_propagation.py`
- [ ] Stdio smoke test against `taiga.shikenso.org` returns expected data
- [ ] `TaigaToolkit` reports 10 tools

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 2: Auth Module — Postgres TokenStore + Fernet Encryption + FastMCP `OAuthProvider`

**Goal of this PR:** Ship the `auth/` package containing `TokenStore`, `TaigaClient`, `LoginPage`, and `TaigaOAuthProvider` (FastMCP `OAuthProvider` subclass). Includes integration tests with real Postgres via testcontainers. Not yet wired into the HTTP server (PR 3 does that).

**Branch:** `feat/auth-module`

**New runtime dependencies:**
- `asyncpg = "^0.30.0"` — async Postgres
- `httpx = "^0.27.0"` — async HTTP for Taiga `/auth` endpoints
- `cryptography = "^43.0"` — Fernet (AES-128-CBC + HMAC-SHA256) for at-rest Taiga JWT encryption
- `jinja2 = "^3.1"` — login page template

**New test deps:**
- `testcontainers = {version = "^4.8", extras = ["postgres"]}`
- `respx = "^0.21"` — mocks `httpx` outbound (TaigaClient — which uses httpx)
- `responses = "^0.25"` — mocks `requests` outbound (python-taiga uses `requests`, not httpx; respx does NOT intercept it)

## Task 2.1: Dependencies + integration test infra

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/integration_tests/__init__.py`
- Create: `tests/integration_tests/conftest.py`
- Create: `tests/integration_tests/test_postgres_smoke.py`

- [ ] **Step 1: Update `pyproject.toml`**

Add to `[tool.poetry.dependencies]`:
```toml
asyncpg = "^0.30.0"
httpx = "^0.27.0"
cryptography = "^43.0"
jinja2 = "^3.1"
```

Add to `[tool.poetry.group.test_integration.dependencies]`:
```toml
testcontainers = {version = "^4.8", extras = ["postgres"]}
respx = "^0.21"          # mocks httpx (TaigaClient outbound)
responses = "^0.25"      # mocks requests (python-taiga outbound — different lib!)
asyncpg = "^0.30.0"
httpx = "^0.27.0"
```

- [ ] **Step 2: Install + verify**

```bash
git checkout -b feat/auth-module
poetry lock --no-update
poetry install --with test,test_integration
poetry run python -c "import asyncpg, httpx, testcontainers, cryptography.fernet; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Create test fixtures**

`tests/integration_tests/__init__.py`: empty file.

`tests/integration_tests/conftest.py`:
```python
"""Integration test fixtures backed by real Postgres via testcontainers."""
from __future__ import annotations

from typing import AsyncIterator

import asyncpg
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from testcontainers.postgres import PostgresContainer


# Deterministic test secrets — never use these in production.
TEST_HMAC_SECRET = bytes.fromhex(
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)
TEST_FERNET_KEY = Fernet.generate_key()  # bytes


@pytest.fixture(scope="session")
def postgres_container() -> PostgresContainer:
    with PostgresContainer("postgres:15-alpine") as pg:
        yield pg


@pytest.fixture(scope="session")
def postgres_dsn(postgres_container: PostgresContainer) -> str:
    raw = postgres_container.get_connection_url()
    return raw.replace("postgresql+psycopg2://", "postgresql://")


@pytest_asyncio.fixture
async def fresh_pool(postgres_dsn: str) -> AsyncIterator[asyncpg.Pool]:
    """Per-test pool with all auth tables created and torn down."""
    from langchain_taiga.auth.token_store import TokenStore

    pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=4)
    try:
        store = TokenStore(
            pool=pool,
            token_hmac_secret=TEST_HMAC_SECRET,
            fernet_key=TEST_FERNET_KEY,
        )
        await store.create_schema()
        yield pool
        await store.drop_schema()
    finally:
        await pool.close()


@pytest_asyncio.fixture
async def fresh_store(fresh_pool: asyncpg.Pool):
    """Convenience — pre-built TokenStore."""
    from langchain_taiga.auth.token_store import TokenStore
    yield TokenStore(
        pool=fresh_pool,
        token_hmac_secret=TEST_HMAC_SECRET,
        fernet_key=TEST_FERNET_KEY,
    )
```

`tests/integration_tests/test_postgres_smoke.py`:
```python
import pytest


@pytest.mark.asyncio
async def test_postgres_smoke(postgres_dsn):
    import asyncpg
    conn = await asyncpg.connect(postgres_dsn)
    val = await conn.fetchval("SELECT 1")
    await conn.close()
    assert val == 1
```

- [ ] **Step 4: Run smoke test**

```bash
make integration_test
```
Expected: PASS, ~10s for container startup.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock tests/integration_tests/
git commit -m "test: integration test infra with testcontainers Postgres + Fernet fixture"
```

## Task 2.2: `TokenStore` — schema + HMAC + Fernet

**Files:**
- Create: `langchain_taiga/auth/__init__.py`
- Create: `langchain_taiga/auth/token_store.py`
- Create: `tests/integration_tests/test_token_store.py`

**Design:**
- MCP-issued opaque tokens + DCR client secrets → HMAC-SHA256 hashed, lookups via constant-time compare. DB compromise alone cannot mint or impersonate.
- Taiga `auth_token` and `refresh_token` → Fernet-encrypted (AES-128-CBC + HMAC-SHA256, key from `TAIGA_MCP_FERNET_KEY` env). DB compromise alone leaks active sessions only with access to the K8s Secret too.
- Single source of truth for the raw values: never persisted, never logged.

- [ ] **Step 1: Failing tests**

`tests/integration_tests/test_token_store.py`:
```python
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pytest


@pytest.mark.asyncio
async def test_store_and_lookup_access_token(fresh_store):
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    await fresh_store.store_access_token(
        token="mcp_token_abc",
        taiga_auth_token="taiga_jwt_xyz",
        taiga_refresh_token="taiga_refresh_xyz",
        taiga_user_id=42,
        taiga_username="alice",
        expires_at=expires_at,
        client_id="client_xyz",
        scopes=["taiga"],
    )
    record = await fresh_store.lookup_access_token("mcp_token_abc")
    assert record.taiga_auth_token == "taiga_jwt_xyz"
    assert record.taiga_user_id == 42
    assert record.scopes == ["taiga"]


@pytest.mark.asyncio
async def test_lookup_unknown_token_returns_none(fresh_store):
    assert await fresh_store.lookup_access_token("never_existed") is None


@pytest.mark.asyncio
async def test_taiga_jwt_is_encrypted_at_rest(fresh_store, fresh_pool):
    """The plaintext Taiga JWT must NEVER appear in the DB column."""
    await fresh_store.store_access_token(
        token="mcp", taiga_auth_token="PLAINTEXT_SECRET",
        taiga_refresh_token="PLAINTEXT_REFRESH",
        taiga_user_id=1, taiga_username="u",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        client_id="c", scopes=["taiga"],
    )
    async with fresh_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT taiga_auth_token_enc, taiga_refresh_token_enc FROM mcp_access_tokens LIMIT 1"
        )
    assert "PLAINTEXT_SECRET" not in row["taiga_auth_token_enc"]
    assert "PLAINTEXT_REFRESH" not in row["taiga_refresh_token_enc"]


@pytest.mark.asyncio
async def test_update_taiga_token_for_refresh(fresh_store):
    await fresh_store.store_access_token(
        token="mcp", taiga_auth_token="old_jwt", taiga_refresh_token="old_refresh",
        taiga_user_id=42, taiga_username="alice",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        client_id="c", scopes=["taiga"],
    )
    new_expiry = datetime.now(timezone.utc) + timedelta(hours=2)
    await fresh_store.update_taiga_token(
        token="mcp", taiga_auth_token="new_jwt", taiga_refresh_token="new_refresh",
        expires_at=new_expiry,
    )
    rec = await fresh_store.lookup_access_token("mcp")
    assert rec.taiga_auth_token == "new_jwt"


@pytest.mark.asyncio
async def test_authorization_code_single_use(fresh_store):
    await fresh_store.store_authorization_code(
        code="auth_code_123", client_id="c", redirect_uri="r",
        code_challenge="cc", code_challenge_method="S256",
        taiga_auth_token="t", taiga_refresh_token="r",
        taiga_user_id=42, taiga_username="alice",
        scopes=["taiga"],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    consumed = await fresh_store.consume_authorization_code("auth_code_123")
    assert consumed.taiga_auth_token == "t"
    assert await fresh_store.consume_authorization_code("auth_code_123") is None


@pytest.mark.asyncio
async def test_dynamic_client_registration(fresh_store):
    await fresh_store.register_client(
        client_id="claude_xyz", client_secret="secret_value",
        redirect_uris=["https://claude.ai/cb"], client_name="Claude",
        token_endpoint_auth_method="none",
    )
    client = await fresh_store.lookup_client("claude_xyz")
    assert client.client_id == "claude_xyz"
    assert client.token_endpoint_auth_method == "none"
    assert client.client_secret is None  # plaintext not retrievable
    assert await fresh_store.verify_client_secret("claude_xyz", "secret_value")
    assert not await fresh_store.verify_client_secret("claude_xyz", "wrong")


@pytest.mark.asyncio
async def test_lookup_unknown_client_for_invalid_client_error(fresh_store):
    """Anthropic requires HTTP 400 invalid_client when DCR client missing."""
    assert await fresh_store.lookup_client("never_registered") is None


@pytest.mark.asyncio
async def test_concurrent_refresh_serialized(fresh_store):
    import asyncio
    from datetime import datetime, timedelta, timezone
    await fresh_store.store_access_token(
        token="race", taiga_auth_token="initial", taiga_refresh_token="ir",
        taiga_user_id=1, taiga_username="u",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        client_id="c", scopes=["taiga"],
    )

    async def refresh_attempt(jwt, ref):
        async with fresh_store.refresh_lock("race") as locked:
            assert locked is not None
            await asyncio.sleep(0.05)
            await fresh_store.update_taiga_token(
                token="race", taiga_auth_token=jwt, taiga_refresh_token=ref,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )

    await asyncio.gather(refresh_attempt("a", "ar"), refresh_attempt("b", "br"))
    final = await fresh_store.lookup_access_token("race")
    assert final.taiga_auth_token in ("a", "b")
```

- [ ] **Step 2: Run to confirm failure**

```bash
make integration_test TEST_FILE=tests/integration_tests/test_token_store.py
```
Expected: FAIL — TokenStore doesn't exist.

- [ ] **Step 3: Implement TokenStore**

`langchain_taiga/auth/__init__.py`:
```python
"""OAuth bridge: per-user Taiga authentication for the remote MCP server."""
```

`langchain_taiga/auth/token_store.py`:
```python
"""Postgres-backed OAuth state with HMAC-hashed MCP secrets and Fernet-
encrypted Taiga JWTs.

At-rest model:
- mcp_access_tokens.token_hash         → HMAC-SHA256 of MCP token
- mcp_access_tokens.taiga_auth_token_enc → Fernet(plaintext Taiga JWT)
- mcp_access_tokens.taiga_refresh_token_enc → Fernet(plaintext refresh)
- dynamic_clients.client_secret_hash    → HMAC-SHA256 of plaintext secret

A DB compromise alone cannot:
  * mint MCP tokens (would need TAIGA_MCP_TOKEN_SECRET)
  * impersonate a registered DCR client (would need original client_secret)
  * decrypt active Taiga sessions (would need TAIGA_MCP_FERNET_KEY)

Refresh-on-near-expiry uses ``SELECT ... FOR UPDATE`` to serialize concurrent
refresh attempts on the same token row.
"""
from __future__ import annotations

import hmac
import hashlib
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, List, Optional

import asyncpg
from cryptography.fernet import Fernet


def _hmac_hex(secret: bytes, value: str) -> str:
    return hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass
class AccessTokenRecord:
    token_hash: str
    taiga_auth_token: str   # decrypted on read
    taiga_refresh_token: str  # decrypted on read
    taiga_user_id: int
    taiga_username: str
    client_id: str
    scopes: List[str]
    expires_at: datetime


@dataclass
class AuthCodeRecord:
    code: str
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    taiga_auth_token: str
    taiga_refresh_token: str
    taiga_user_id: int
    taiga_username: str
    scopes: List[str]
    expires_at: datetime


@dataclass
class ClientRecord:
    client_id: str
    client_secret: Optional[str]  # only populated immediately after register_client
    redirect_uris: List[str]
    client_name: str
    token_endpoint_auth_method: str  # "none" | "client_secret_basic" | "client_secret_post"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mcp_access_tokens (
    token_hash TEXT PRIMARY KEY,
    taiga_auth_token_enc TEXT NOT NULL,
    taiga_refresh_token_enc TEXT NOT NULL,
    taiga_user_id INTEGER NOT NULL,
    taiga_username TEXT NOT NULL,
    client_id TEXT NOT NULL,
    scopes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS oauth_authorization_codes (
    code TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    code_challenge TEXT NOT NULL,
    code_challenge_method TEXT NOT NULL,
    taiga_auth_token_enc TEXT NOT NULL,
    taiga_refresh_token_enc TEXT NOT NULL,
    taiga_user_id INTEGER NOT NULL,
    taiga_username TEXT NOT NULL,
    scopes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dynamic_clients (
    client_id TEXT PRIMARY KEY,
    client_secret_hash TEXT NOT NULL,
    redirect_uris TEXT[] NOT NULL,
    client_name TEXT NOT NULL,
    token_endpoint_auth_method TEXT NOT NULL DEFAULT 'client_secret_basic',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_access_tokens_expires_at ON mcp_access_tokens(expires_at);
CREATE INDEX IF NOT EXISTS idx_auth_codes_expires_at ON oauth_authorization_codes(expires_at);
"""

DROP_SQL = """
DROP TABLE IF EXISTS mcp_access_tokens CASCADE;
DROP TABLE IF EXISTS oauth_authorization_codes CASCADE;
DROP TABLE IF EXISTS dynamic_clients CASCADE;
"""


class TokenStore:
    def __init__(self, *, pool: asyncpg.Pool, token_hmac_secret: bytes, fernet_key: bytes):
        if not token_hmac_secret or len(token_hmac_secret) < 32:
            raise ValueError("token_hmac_secret must be ≥32 bytes")
        self._pool = pool
        self._hmac_secret = token_hmac_secret
        self._fernet = Fernet(fernet_key)

    @classmethod
    async def from_env(cls) -> "TokenStore":
        dsn = os.environ["TAIGA_MCP_DB_URL"]
        hmac_secret = bytes.fromhex(os.environ["TAIGA_MCP_TOKEN_SECRET"])
        fernet_key = os.environ["TAIGA_MCP_FERNET_KEY"].encode()
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
        return cls(pool=pool, token_hmac_secret=hmac_secret, fernet_key=fernet_key)

    def hash(self, value: str) -> str:
        return _hmac_hex(self._hmac_secret, value)

    def _encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode()

    def _decrypt(self, ciphertext: str) -> str:
        return self._fernet.decrypt(ciphertext.encode()).decode()

    async def create_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    async def drop_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(DROP_SQL)

    async def close(self) -> None:
        await self._pool.close()

    # --- Access tokens ----------------------------------------------------

    async def store_access_token(
        self, *, token: str, taiga_auth_token: str, taiga_refresh_token: str,
        taiga_user_id: int, taiga_username: str, expires_at: datetime,
        client_id: str, scopes: List[str],
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO mcp_access_tokens
                  (token_hash, taiga_auth_token_enc, taiga_refresh_token_enc,
                   taiga_user_id, taiga_username, client_id, scopes, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (token_hash) DO UPDATE SET
                  taiga_auth_token_enc = EXCLUDED.taiga_auth_token_enc,
                  taiga_refresh_token_enc = EXCLUDED.taiga_refresh_token_enc,
                  expires_at = EXCLUDED.expires_at
                """,
                self.hash(token),
                self._encrypt(taiga_auth_token),
                self._encrypt(taiga_refresh_token),
                taiga_user_id, taiga_username, client_id, scopes, expires_at,
            )

    async def lookup_access_token(self, token: str) -> Optional[AccessTokenRecord]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT token_hash, taiga_auth_token_enc, taiga_refresh_token_enc,
                       taiga_user_id, taiga_username, client_id, scopes, expires_at
                FROM mcp_access_tokens
                WHERE token_hash = $1
                """,
                self.hash(token),
            )
        if row is None:
            return None
        return AccessTokenRecord(
            token_hash=row["token_hash"],
            taiga_auth_token=self._decrypt(row["taiga_auth_token_enc"]),
            taiga_refresh_token=self._decrypt(row["taiga_refresh_token_enc"]),
            taiga_user_id=row["taiga_user_id"],
            taiga_username=row["taiga_username"],
            client_id=row["client_id"],
            scopes=list(row["scopes"]),
            expires_at=row["expires_at"],
        )

    async def update_taiga_token(
        self, *, token: str, taiga_auth_token: str, taiga_refresh_token: str,
        expires_at: datetime,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE mcp_access_tokens
                SET taiga_auth_token_enc = $2,
                    taiga_refresh_token_enc = $3,
                    expires_at = $4
                WHERE token_hash = $1
                """,
                self.hash(token),
                self._encrypt(taiga_auth_token),
                self._encrypt(taiga_refresh_token),
                expires_at,
            )

    @asynccontextmanager
    async def refresh_lock(self, token: str) -> AsyncIterator[Optional[AccessTokenRecord]]:
        token_hash = self.hash(token)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT token_hash, taiga_auth_token_enc, taiga_refresh_token_enc,
                           taiga_user_id, taiga_username, client_id, scopes, expires_at
                    FROM mcp_access_tokens
                    WHERE token_hash = $1
                    FOR UPDATE
                    """,
                    token_hash,
                )
                if row is None:
                    yield None
                else:
                    yield AccessTokenRecord(
                        token_hash=row["token_hash"],
                        taiga_auth_token=self._decrypt(row["taiga_auth_token_enc"]),
                        taiga_refresh_token=self._decrypt(row["taiga_refresh_token_enc"]),
                        taiga_user_id=row["taiga_user_id"],
                        taiga_username=row["taiga_username"],
                        client_id=row["client_id"],
                        scopes=list(row["scopes"]),
                        expires_at=row["expires_at"],
                    )

    # --- Authorization codes ---------------------------------------------

    async def store_authorization_code(
        self, *, code: str, client_id: str, redirect_uri: str,
        code_challenge: str, code_challenge_method: str,
        taiga_auth_token: str, taiga_refresh_token: str,
        taiga_user_id: int, taiga_username: str,
        scopes: List[str], expires_at: datetime,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO oauth_authorization_codes
                  (code, client_id, redirect_uri, code_challenge,
                   code_challenge_method, taiga_auth_token_enc, taiga_refresh_token_enc,
                   taiga_user_id, taiga_username, scopes, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                code, client_id, redirect_uri, code_challenge, code_challenge_method,
                self._encrypt(taiga_auth_token), self._encrypt(taiga_refresh_token),
                taiga_user_id, taiga_username, scopes, expires_at,
            )

    async def consume_authorization_code(self, code: str) -> Optional[AuthCodeRecord]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                DELETE FROM oauth_authorization_codes
                WHERE code = $1 AND expires_at > NOW()
                RETURNING code, client_id, redirect_uri, code_challenge,
                          code_challenge_method, taiga_auth_token_enc,
                          taiga_refresh_token_enc, taiga_user_id, taiga_username,
                          scopes, expires_at
                """,
                code,
            )
        if row is None:
            return None
        return AuthCodeRecord(
            code=row["code"], client_id=row["client_id"], redirect_uri=row["redirect_uri"],
            code_challenge=row["code_challenge"],
            code_challenge_method=row["code_challenge_method"],
            taiga_auth_token=self._decrypt(row["taiga_auth_token_enc"]),
            taiga_refresh_token=self._decrypt(row["taiga_refresh_token_enc"]),
            taiga_user_id=row["taiga_user_id"], taiga_username=row["taiga_username"],
            scopes=list(row["scopes"]), expires_at=row["expires_at"],
        )

    # --- Dynamic Client Registration -------------------------------------

    async def register_client(
        self, *, client_id: str, client_secret: str,
        redirect_uris: List[str], client_name: str,
        token_endpoint_auth_method: str = "client_secret_basic",
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO dynamic_clients
                  (client_id, client_secret_hash, redirect_uris, client_name,
                   token_endpoint_auth_method)
                VALUES ($1, $2, $3, $4, $5)
                """,
                client_id, self.hash(client_secret), redirect_uris,
                client_name, token_endpoint_auth_method,
            )

    async def lookup_client(self, client_id: str) -> Optional[ClientRecord]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT client_id, redirect_uris, client_name, token_endpoint_auth_method
                FROM dynamic_clients WHERE client_id = $1
                """,
                client_id,
            )
        if row is None:
            return None
        return ClientRecord(
            client_id=row["client_id"], client_secret=None,
            redirect_uris=list(row["redirect_uris"]),
            client_name=row["client_name"],
            token_endpoint_auth_method=row["token_endpoint_auth_method"],
        )

    async def verify_client_secret(self, client_id: str, presented: str) -> bool:
        async with self._pool.acquire() as conn:
            stored = await conn.fetchval(
                "SELECT client_secret_hash FROM dynamic_clients WHERE client_id = $1",
                client_id,
            )
        if stored is None:
            return False
        return hmac.compare_digest(stored, self.hash(presented))

    # --- Cleanup ---------------------------------------------------------

    async def cleanup_expired(self) -> int:
        async with self._pool.acquire() as conn:
            r1 = await conn.execute(
                "DELETE FROM mcp_access_tokens WHERE expires_at < NOW()"
            )
            r2 = await conn.execute(
                "DELETE FROM oauth_authorization_codes WHERE expires_at < NOW()"
            )
        return int(r1.split()[-1]) + int(r2.split()[-1])
```

- [ ] **Step 4: Run integration tests**

```bash
make integration_test TEST_FILE=tests/integration_tests/test_token_store.py
```
Expected: PASS, including `test_taiga_jwt_is_encrypted_at_rest` proving the column never contains the plaintext.

- [ ] **Step 5: Commit**

```bash
git add langchain_taiga/auth/__init__.py langchain_taiga/auth/token_store.py tests/integration_tests/test_token_store.py
git commit -m "feat(auth): TokenStore with HMAC-hashed MCP tokens and Fernet-encrypted Taiga JWTs"
```

## Task 2.3: `TaigaClient` — credential exchange + refresh

**Files:**
- Create: `langchain_taiga/auth/taiga_client.py`
- Create: `tests/integration_tests/test_taiga_client.py`

- [ ] **Step 1: Failing tests**

`tests/integration_tests/test_taiga_client.py`:
```python
from __future__ import annotations
import pytest
import respx
from httpx import Response


@pytest.mark.asyncio
@respx.mock
async def test_authenticate_user_returns_credentials():
    from langchain_taiga.auth.taiga_client import TaigaClient
    respx.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(200, json={
            "auth_token": "jwt", "refresh": "ref", "id": 42, "username": "alice",
        })
    )
    client = TaigaClient(api_url="https://taiga.example.test")
    creds = await client.authenticate_user("alice", "wonderland")
    assert creds.auth_token == "jwt"
    assert creds.refresh == "ref"
    assert creds.user_id == 42


@pytest.mark.asyncio
@respx.mock
async def test_authenticate_user_invalid_raises():
    from langchain_taiga.auth.taiga_client import TaigaClient, TaigaAuthenticationError
    respx.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(400, json={"_error_message": "Invalid"})
    )
    client = TaigaClient(api_url="https://taiga.example.test")
    with pytest.raises(TaigaAuthenticationError):
        await client.authenticate_user("alice", "wrong")


@pytest.mark.asyncio
@respx.mock
async def test_refresh_token():
    from langchain_taiga.auth.taiga_client import TaigaClient
    respx.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(200, json={"auth_token": "new", "refresh": "newref"})
    )
    client = TaigaClient(api_url="https://taiga.example.test")
    refreshed = await client.refresh_taiga_token("old_refresh")
    assert refreshed.auth_token == "new"


@pytest.mark.asyncio
@respx.mock
async def test_refresh_failure_raises():
    from langchain_taiga.auth.taiga_client import TaigaClient, TaigaRefreshError
    respx.post("https://taiga.example.test/api/v1/auth/refresh").mock(
        return_value=Response(401, json={"_error_message": "Expired"})
    )
    client = TaigaClient(api_url="https://taiga.example.test")
    with pytest.raises(TaigaRefreshError):
        await client.refresh_taiga_token("dead")
```

- [ ] **Step 2: Run to confirm failure**

```bash
make integration_test TEST_FILE=tests/integration_tests/test_taiga_client.py
```

- [ ] **Step 3: Implement TaigaClient**

`langchain_taiga/auth/taiga_client.py`:
```python
"""HTTP client for Taiga's native auth endpoints."""
from __future__ import annotations
from dataclasses import dataclass
import httpx


class TaigaAuthenticationError(Exception):
    """User credentials rejected by Taiga."""


class TaigaRefreshError(Exception):
    """Refresh token rejected — claude.ai must restart OAuth."""


@dataclass
class TaigaCredentials:
    auth_token: str
    refresh: str
    user_id: int
    username: str


@dataclass
class RefreshedTokens:
    auth_token: str
    refresh: str


class TaigaClient:
    def __init__(self, api_url: str, timeout: float = 10.0):
        self._api_url = api_url.rstrip("/")
        self._timeout = timeout

    async def authenticate_user(self, username: str, password: str) -> TaigaCredentials:
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            resp = await http.post(
                f"{self._api_url}/api/v1/auth",
                json={"type": "normal", "username": username, "password": password},
            )
        if resp.status_code != 200:
            raise TaigaAuthenticationError(
                f"Taiga auth failed: {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()
        return TaigaCredentials(
            auth_token=body["auth_token"],
            refresh=body["refresh"],
            user_id=int(body["id"]),
            username=body["username"],
        )

    async def refresh_taiga_token(self, refresh_token: str) -> RefreshedTokens:
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            resp = await http.post(
                f"{self._api_url}/api/v1/auth/refresh",
                json={"refresh": refresh_token},
            )
        if resp.status_code != 200:
            raise TaigaRefreshError(
                f"Taiga refresh failed: {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()
        return RefreshedTokens(auth_token=body["auth_token"], refresh=body["refresh"])
```

- [ ] **Step 4: Run tests + commit**

```bash
make integration_test TEST_FILE=tests/integration_tests/test_taiga_client.py
git add langchain_taiga/auth/taiga_client.py tests/integration_tests/test_taiga_client.py
git commit -m "feat(auth): TaigaClient for credential exchange and refresh"
```

## Task 2.4: Login page renderer

**Files:**
- Create: `langchain_taiga/auth/login_page.py`
- Create: `langchain_taiga/auth/templates/login.html`
- Create: `tests/unit_tests/test_login_page.py`

- [ ] **Step 1: Failing tests**

`tests/unit_tests/test_login_page.py`:
```python
def test_login_page_renders_form_with_state():
    from langchain_taiga.auth.login_page import render_login_page
    html = render_login_page(
        state="csrf_xyz", error=None, taiga_url="https://taiga.shikenso.org",
    )
    assert '<form' in html
    assert 'name="state" value="csrf_xyz"' in html
    assert 'name="username"' in html
    assert 'name="password"' in html
    assert "taiga.shikenso.org" in html


def test_login_page_displays_error():
    from langchain_taiga.auth.login_page import render_login_page
    html = render_login_page(
        state="csrf_xyz", error="Invalid username or password",
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
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x" not in html
```

- [ ] **Step 2: Implement template + renderer**

`langchain_taiga/auth/templates/login.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sign in to Taiga</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #f4f7fb; display: flex; align-items: center; justify-content: center;
           min-height: 100vh; margin: 0; }
    .card { background: white; padding: 2rem 2.5rem; border-radius: 8px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08); width: 360px; }
    h1 { font-size: 1.25rem; margin-top: 0; color: #2d2d2d; }
    .subtitle { color: #6e7682; font-size: 0.9rem; margin-bottom: 1.5rem; }
    label { display: block; margin: 0.75rem 0 0.25rem; font-size: 0.85rem; color: #4a5160; }
    input[type=text], input[type=password] {
      width: 100%; padding: 0.6rem; border: 1px solid #d1d5db;
      border-radius: 4px; font-size: 1rem; box-sizing: border-box; }
    button { margin-top: 1.5rem; width: 100%; padding: 0.7rem;
             background: #5a8feb; color: white; border: none; border-radius: 4px;
             font-size: 1rem; cursor: pointer; }
    button:hover { background: #4878d6; }
    .error { background: #fde8e8; color: #c53030; padding: 0.6rem;
             border-radius: 4px; margin-bottom: 1rem; font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Sign in to Taiga</h1>
    <p class="subtitle">Authorize the MCP server to act on your behalf at <strong>{{ taiga_url }}</strong>.</p>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST" action="login">
      <input type="hidden" name="state" value="{{ state }}">
      <label for="username">Username</label>
      <input type="text" id="username" name="username" required autofocus>
      <label for="password">Password</label>
      <input type="password" id="password" name="password" required>
      <button type="submit">Sign in</button>
    </form>
  </div>
</body>
</html>
```

`langchain_taiga/auth/login_page.py`:
```python
"""Render the HTML form where end-users provide their Taiga credentials."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)


def render_login_page(*, state: str, error: Optional[str], taiga_url: str) -> str:
    return _env.get_template("login.html").render(
        state=state, error=error, taiga_url=taiga_url
    )
```

Add to `pyproject.toml`:
```toml
[tool.poetry]
include = ["langchain_taiga/auth/templates/*.html"]
```

- [ ] **Step 3: Run + commit**

```bash
make test TEST_FILE=tests/unit_tests/test_login_page.py
git add langchain_taiga/auth/login_page.py langchain_taiga/auth/templates/login.html tests/unit_tests/test_login_page.py pyproject.toml
git commit -m "feat(auth): Jinja2 login form template with HTML-escaping"
```

## Task 2.5: `TaigaOAuthProvider` — FastMCP `OAuthProvider` subclass

**Files:**
- Create: `langchain_taiga/auth/provider.py`
- Create: `tests/integration_tests/test_provider.py`

This is the centerpiece of PR 2. We subclass `fastmcp.server.auth.OAuthProvider` (which inherits from `mcp.server.auth.provider.OAuthAuthorizationServerProvider`) and implement the abstract methods. **FastMCP auto-mounts the OAuth + discovery routes when this is passed as `FastMCP(auth=provider)`** — we do not register Starlette routes for them ourselves.

**Method mapping** (MCP-SDK ABC → our impl):

| ABC method | What we do |
|---|---|
| `register_client(client_info)` | Validate `redirect_uris` against allowlist; persist via `TokenStore.register_client` |
| `get_client(client_id)` | Return `OAuthClientInformationFull` from `TokenStore.lookup_client`, or **None** so FastMCP returns HTTP 400 `invalid_client` |
| `authorize(client, params)` | Stash `(client_id, redirect_uri, code_challenge, code_challenge_method, claude_state)` in `_authorize_states` (in-memory dict), return redirect URL to `/oauth/login?internal_state=...` |
| `load_authorization_code(client, code)` | Look up by code, no consume |
| `exchange_authorization_code(client, code)` | Verify PKCE, mint MCP access token, store with `taiga_jwt`/`user_id` claims |
| `load_refresh_token(client, refresh)` | Not implemented in v1 — return None (claude.ai re-OAuths on token expiry) |
| `exchange_refresh_token(client, refresh, scopes)` | Same — raise NotImplementedError; v2 enables refresh |
| `load_access_token(token)` | Look up via `TokenStore.lookup_access_token`, return `AccessToken` with `claims = {"taiga_jwt": ..., "user_id": ..., "username": ...}` — **this is the load-bearing step that makes per-request JWT propagation work** |
| `revoke_token(token)` | Optional; v1 leaves expiry-based cleanup |

The custom HTML login form is **not** part of this provider — it's two `mcp.custom_route` handlers in `remote_server.py` (PR 3).

- [ ] **Step 1: Verify FastMCP `OAuthProvider` API surface**

Before writing the subclass, confirm the actual class names and method signatures in the installed FastMCP:

```bash
poetry run python -c "
from fastmcp.server.auth import OAuthProvider
import inspect
print('Bases:', [b.__name__ for b in OAuthProvider.__mro__])
print()
print(inspect.getsource(OAuthProvider))
" | head -80
```

Note the exact method signatures and types (`OAuthClientInformationFull`, `AuthorizationParams`, `AuthorizationCode`, `RefreshToken`, `AccessToken`, `OAuthToken`). The pseudocode below assumes the documented MCP-SDK shapes; **adapt exact types where the installed version differs.**

- [ ] **Step 2: Failing tests**

`tests/integration_tests/test_provider.py`:
```python
from __future__ import annotations
import base64
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
import respx
from httpx import Response


@pytest.mark.asyncio
async def test_register_client_rejects_unallowed_redirect(fresh_store):
    """Open-redirect protection."""
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.taiga_client import TaigaClient
    provider = TaigaOAuthProvider(
        store=fresh_store,
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )
    from mcp.shared.auth import OAuthClientMetadata  # SDK type
    with pytest.raises(ValueError, match="Redirect URI not allowed"):
        await provider.register_client(OAuthClientMetadata(
            redirect_uris=["https://attacker.example.com/steal"],
            client_name="Attacker",
        ))


@pytest.mark.asyncio
async def test_register_client_accepts_claude_ai_and_claude_com(fresh_store):
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.taiga_client import TaigaClient
    from mcp.shared.auth import OAuthClientMetadata
    provider = TaigaOAuthProvider(
        store=fresh_store,
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )
    info_a = await provider.register_client(OAuthClientMetadata(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_name="Claude (claude.ai)",
        token_endpoint_auth_method="none",
    ))
    info_b = await provider.register_client(OAuthClientMetadata(
        redirect_uris=["https://claude.com/api/mcp/auth_callback"],
        client_name="Claude (claude.com)",
        token_endpoint_auth_method="none",
    ))
    assert info_a.client_id != info_b.client_id


@pytest.mark.asyncio
async def test_get_client_returns_none_for_unknown(fresh_store):
    """Anthropic requires HTTP 400 invalid_client — provider returns None and FastMCP renders the error."""
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.taiga_client import TaigaClient
    provider = TaigaOAuthProvider(
        store=fresh_store,
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )
    assert await provider.get_client("never_registered") is None


@pytest.mark.asyncio
@respx.mock
async def test_load_access_token_attaches_taiga_jwt_to_claims(fresh_store):
    """The load_access_token contract: returned AccessToken.claims must carry
    the Taiga JWT — this is what tools see via get_access_token().claims."""
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.taiga_client import TaigaClient

    provider = TaigaOAuthProvider(
        store=fresh_store,
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )
    await fresh_store.store_access_token(
        token="mcp_xyz", taiga_auth_token="taiga_jwt_456",
        taiga_refresh_token="ref",
        taiga_user_id=42, taiga_username="alice",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        client_id="c", scopes=["taiga"],
    )
    result = await provider.load_access_token("mcp_xyz")
    assert result is not None
    assert result.claims["taiga_jwt"] == "taiga_jwt_456"
    assert result.claims["user_id"] == 42
    assert result.claims["username"] == "alice"


@pytest.mark.asyncio
async def test_load_access_token_returns_none_for_unknown(fresh_store):
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.taiga_client import TaigaClient
    provider = TaigaOAuthProvider(
        store=fresh_store,
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )
    assert await provider.load_access_token("never_minted") is None


@pytest.mark.asyncio
@respx.mock
async def test_full_auth_flow(fresh_store):
    """Authorize → login (driven by remote_server's /oauth/login) → exchange_authorization_code → AccessToken."""
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.taiga_client import TaigaClient

    respx.post("https://taiga.example.test/api/v1/auth").mock(
        return_value=Response(200, json={
            "auth_token": "alice_jwt", "refresh": "alice_ref",
            "id": 42, "username": "alice",
        })
    )

    provider = TaigaOAuthProvider(
        store=fresh_store,
        taiga_client=TaigaClient(api_url="https://taiga.example.test"),
        issuer_url="https://taiga.shikenso.org/mcp",
    )

    # Register a client
    from mcp.shared.auth import OAuthClientMetadata
    client_info = await provider.register_client(OAuthClientMetadata(
        redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
        client_name="Claude", token_endpoint_auth_method="none",
    ))

    # Start authorize → returns redirect URL to /oauth/login
    from mcp.server.auth.provider import AuthorizationParams
    redirect = await provider.authorize(
        client=client_info,
        params=AuthorizationParams(
            response_type="code",
            scopes=["taiga"],
            code_challenge="cc",
            code_challenge_method="S256",
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            state="claude_csrf",
        ),
    )
    assert redirect.startswith("https://taiga.shikenso.org/mcp/oauth/login?internal_state=")

    # The /oauth/login POST handler in remote_server.py would call this:
    internal_state = redirect.split("internal_state=", 1)[1]
    code, redirect_url = await provider.complete_login(
        internal_state=internal_state, username="alice", password="x",
    )
    assert "code=" in redirect_url
    assert "state=claude_csrf" in redirect_url

    # claude.ai now POSTs /token; provider verifies PKCE, mints MCP token
    verifier = "verifier_for_alice_with_enough_entropy_xx"
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    # Re-do with real PKCE pair:
    redirect2 = await provider.authorize(
        client=client_info,
        params=AuthorizationParams(
            response_type="code", scopes=["taiga"],
            code_challenge=challenge, code_challenge_method="S256",
            redirect_uri="https://claude.ai/api/mcp/auth_callback",
            state="csrf2",
        ),
    )
    is2 = redirect2.split("internal_state=", 1)[1]
    code2, _ = await provider.complete_login(
        internal_state=is2, username="alice", password="x"
    )

    # Look up the AuthorizationCode object FastMCP normally fetches via load_authorization_code
    auth_code_obj = await provider.load_authorization_code(client_info, code2)
    assert auth_code_obj is not None

    # Exchange — claude.ai is a public client (token_endpoint_auth_method=none),
    # so no client_secret check is needed inside the provider; FastMCP enforces
    # client_id presence and our get_client lookup.
    oauth_token = await provider.exchange_authorization_code(
        client=client_info, authorization_code=auth_code_obj,
    )
    assert oauth_token.access_token

    # That MCP token, looked up via load_access_token, must carry the Taiga JWT in claims
    access = await provider.load_access_token(oauth_token.access_token)
    assert access.claims["taiga_jwt"] == "alice_jwt"
    assert access.claims["user_id"] == 42
```

> Note: the test imports `OAuthClientMetadata`, `AuthorizationParams`, `AuthorizationCode`, etc. from `mcp.shared.auth` and `mcp.server.auth.provider`. The exact import paths in the installed `mcp` SDK may differ slightly between releases — verify with `python -c "from mcp.server.auth.provider import *; help(...)"` and adapt.

- [ ] **Step 3: Implement `TaigaOAuthProvider`**

`langchain_taiga/auth/provider.py`:
```python
"""FastMCP OAuthProvider subclass for Taiga username/password backends.

When constructed and passed to ``FastMCP(auth=provider)``, this triggers
auto-mounting of:
  GET /.well-known/oauth-authorization-server  (RFC 8414)
  GET /.well-known/oauth-protected-resource    (RFC 9728)
  POST /register                              (RFC 7591 DCR)
  GET /authorize                              (response_type=code, PKCE)
  POST /token                                 (Authorization Code grant)

The custom HTML login page is NOT auto-mounted — it is registered in
remote_server.py via ``mcp.custom_route("/oauth/login", ...)``.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

from fastmcp.server.auth import OAuthProvider
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from langchain_taiga.auth.taiga_client import (
    TaigaAuthenticationError,
    TaigaClient,
)
from langchain_taiga.auth.token_store import (
    AccessTokenRecord,
    AuthCodeRecord,
    ClientRecord,
    TokenStore,
)


ACCESS_TOKEN_TTL = timedelta(hours=1)
AUTH_CODE_TTL = timedelta(minutes=10)


@dataclass
class _PendingAuthorize:
    client_id: str
    redirect_uri: str
    code_challenge: str
    code_challenge_method: str
    scopes: List[str]
    claude_state: str
    expires_at: datetime


class TaigaOAuthProvider(OAuthProvider):
    """OAuth Authorization Server bound to Taiga as the credential source."""

    DEFAULT_ALLOWED_REDIRECT_PREFIXES = (
        "https://claude.ai/",
        "https://claude.com/",
        "http://localhost:",  # MCP Inspector
    )

    def __init__(
        self,
        *,
        store: TokenStore,
        taiga_client: TaigaClient,
        issuer_url: str,
        allowed_redirect_uri_prefixes: Optional[Tuple[str, ...]] = None,
    ):
        # Pass framework config to OAuthProvider — exact arg names per installed
        # FastMCP version. ClientRegistrationOptions lives at
        # fastmcp.server.auth.ClientRegistrationOptions in 2.14+.
        from fastmcp.server.auth import ClientRegistrationOptions
        super().__init__(
            issuer_url=issuer_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=["taiga"],
                default_scopes=["taiga"],
            ),
            required_scopes=["taiga"],
        )
        self._store = store
        self._taiga = taiga_client
        self._issuer = issuer_url.rstrip("/")
        self._allowed_redirect_prefixes = (
            allowed_redirect_uri_prefixes
            if allowed_redirect_uri_prefixes is not None
            else self.DEFAULT_ALLOWED_REDIRECT_PREFIXES
        )
        self._authorize_states: Dict[str, _PendingAuthorize] = {}

    # ---- Allowlist ------------------------------------------------------

    def _validate_redirect_uri(self, uri: str) -> None:
        if not any(uri.startswith(p) for p in self._allowed_redirect_prefixes):
            raise ValueError(
                f"Redirect URI not allowed: {uri!r}. "
                f"Allowed prefixes: {self._allowed_redirect_prefixes}"
            )

    # ---- DCR ------------------------------------------------------------

    async def register_client(
        self, client_info: OAuthClientMetadata
    ) -> OAuthClientInformationFull:
        for uri in client_info.redirect_uris:
            self._validate_redirect_uri(str(uri))

        client_id = f"mcp_{secrets.token_urlsafe(16)}"
        # Public clients (token_endpoint_auth_method="none") still get a
        # client_secret minted but it's not required at /token.
        client_secret = secrets.token_urlsafe(32)
        method = client_info.token_endpoint_auth_method or "client_secret_basic"

        await self._store.register_client(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=[str(u) for u in client_info.redirect_uris],
            client_name=client_info.client_name or "unnamed",
            token_endpoint_auth_method=method,
        )
        return OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,  # only returned once
            redirect_uris=client_info.redirect_uris,
            client_name=client_info.client_name,
            token_endpoint_auth_method=method,
        )

    async def get_client(
        self, client_id: str
    ) -> Optional[OAuthClientInformationFull]:
        record = await self._store.lookup_client(client_id)
        if record is None:
            # Returning None makes FastMCP render HTTP 400 invalid_client per
            # RFC 6749 — what Anthropic docs require so claude.ai re-registers.
            return None
        return OAuthClientInformationFull(
            client_id=record.client_id,
            client_secret=None,  # plaintext not retrievable
            redirect_uris=record.redirect_uris,
            client_name=record.client_name,
            token_endpoint_auth_method=record.token_endpoint_auth_method,
        )

    # ---- Authorize ------------------------------------------------------

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        """Stash the authorize request and return URL to our HTML login page."""
        if params.code_challenge_method != "S256":
            raise ValueError("Only S256 PKCE is supported")
        redirect_uri = str(params.redirect_uri)
        if redirect_uri not in [str(r) for r in client.redirect_uris]:
            raise ValueError("Redirect URI not registered for this client")
        self._validate_redirect_uri(redirect_uri)

        internal_state = secrets.token_urlsafe(24)
        self._authorize_states[internal_state] = _PendingAuthorize(
            client_id=client.client_id,
            redirect_uri=redirect_uri,
            code_challenge=params.code_challenge,
            code_challenge_method=params.code_challenge_method,
            scopes=list(params.scopes or ["taiga"]),
            claude_state=params.state or "",
            expires_at=datetime.now(timezone.utc) + AUTH_CODE_TTL,
        )
        return f"{self._issuer}/oauth/login?internal_state={internal_state}"

    async def complete_login(
        self, *, internal_state: str, username: str, password: str
    ) -> Tuple[str, str]:
        """Called by remote_server.py's /oauth/login POST handler.

        Authenticates against Taiga, mints an authorization code, returns
        ``(code, redirect_url_with_code_and_claude_state)``. State is preserved
        on TaigaAuthenticationError so the user can retry.
        """
        st = self._authorize_states.get(internal_state)
        if st is None:
            raise ValueError("Invalid or expired internal_state")
        if st.expires_at < datetime.now(timezone.utc):
            self._authorize_states.pop(internal_state, None)
            raise ValueError("Authorize state expired")

        creds = await self._taiga.authenticate_user(username, password)
        # Only consume on success
        self._authorize_states.pop(internal_state, None)

        code = secrets.token_urlsafe(32)
        await self._store.store_authorization_code(
            code=code,
            client_id=st.client_id,
            redirect_uri=st.redirect_uri,
            code_challenge=st.code_challenge,
            code_challenge_method=st.code_challenge_method,
            taiga_auth_token=creds.auth_token,
            taiga_refresh_token=creds.refresh,
            taiga_user_id=creds.user_id,
            taiga_username=creds.username,
            scopes=st.scopes,
            expires_at=datetime.now(timezone.utc) + AUTH_CODE_TTL,
        )
        params = urlencode({"code": code, "state": st.claude_state})
        return code, f"{st.redirect_uri}?{params}"

    # ---- Authorization Code grant ---------------------------------------

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        # We can't peek without consuming under our schema (DELETE ... RETURNING
        # is the only atomic single-use shape). For load+exchange, FastMCP
        # convention is: load returns the AuthorizationCode object, exchange
        # consumes. We model this by deferring the DB delete to exchange and
        # returning a synthesized AuthorizationCode here.
        async with self._store._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT code, client_id, redirect_uri, code_challenge,
                       code_challenge_method, scopes, expires_at
                FROM oauth_authorization_codes
                WHERE code = $1 AND expires_at > NOW()
                """,
                authorization_code,
            )
        if row is None or row["client_id"] != client.client_id:
            return None
        return AuthorizationCode(
            code=row["code"],
            client_id=row["client_id"],
            redirect_uri=row["redirect_uri"],
            code_challenge=row["code_challenge"],
            scopes=list(row["scopes"]),
            expires_at=int(row["expires_at"].timestamp()),
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        # Atomic single-use consume
        consumed = await self._store.consume_authorization_code(
            authorization_code.code
        )
        if consumed is None:
            raise ValueError("Authorization code already used or expired")
        if consumed.client_id != client.client_id:
            raise ValueError("Code was issued to a different client")

        # Mint MCP access token
        mcp_access_token = secrets.token_urlsafe(32)
        await self._store.store_access_token(
            token=mcp_access_token,
            taiga_auth_token=consumed.taiga_auth_token,
            taiga_refresh_token=consumed.taiga_refresh_token,
            taiga_user_id=consumed.taiga_user_id,
            taiga_username=consumed.taiga_username,
            expires_at=datetime.now(timezone.utc) + ACCESS_TOKEN_TTL,
            client_id=consumed.client_id,
            scopes=consumed.scopes,
        )
        return OAuthToken(
            access_token=mcp_access_token,
            token_type="Bearer",
            expires_in=int(ACCESS_TOKEN_TTL.total_seconds()),
            scope=" ".join(consumed.scopes),
        )

    # ---- Refresh tokens (deferred to v2) --------------------------------

    async def load_refresh_token(self, *args, **kwargs):
        return None  # claude.ai falls back to re-OAuth on expiry

    async def exchange_refresh_token(self, *args, **kwargs):
        raise NotImplementedError("Refresh tokens deferred to v2")

    # ---- Access token verification --------------------------------------

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        """Look up the MCP access token; expiry is enforced by the FastMCP layer.

        **No transparent Taiga-token refresh here** — see the design note below.

        Lifetime invariant: the MCP access token expiry (1h) is always
        shorter than Taiga's JWT expiry (1d default). claude.ai will re-OAuth
        on MCP token expiry long before the Taiga JWT itself expires — so a
        refresh-before-handing-out path would be dead code at the v1 TTLs.

        If your Taiga deployment shortens its JWT TTL below MCP's, the right
        fix is to ALSO shorten ``ACCESS_TOKEN_TTL`` to stay below it — not to
        add transparent refresh, because rebumping ``mcp_access_tokens.expires_at``
        would put the server's view out of sync with the ``expires_in`` claude.ai
        was told at /token. Refresh-on-the-fly is a v2 feature that requires
        splitting the schema into ``mcp_expires_at`` + ``taiga_expires_at`` and
        a refresh-token grant flow.
        """
        record = await self._store.lookup_access_token(token)
        if record is None:
            return None
        return AccessToken(
            token=token,
            client_id=record.client_id,
            scopes=record.scopes,
            expires_at=int(record.expires_at.timestamp()),
            claims={
                "taiga_jwt": record.taiga_auth_token,
                "user_id": record.taiga_user_id,
                "username": record.taiga_username,
            },
        )

    # ---- Optional revocation --------------------------------------------

    async def revoke_token(self, token: str) -> None:
        # No-op in v1 — expiry-based cleanup suffices.
        pass


# ---- Cleanup loop (called from remote_server lifespan) ------------------

import asyncio
import logging

_log = logging.getLogger(__name__)


async def run_cleanup_loop(
    store: "TokenStore",
    *,
    period_seconds: float = 300.0,
    stop: Optional[asyncio.Event] = None,
) -> None:
    stop = stop or asyncio.Event()
    while not stop.is_set():
        try:
            deleted = await store.cleanup_expired()
            if deleted:
                _log.info("Cleanup deleted %d expired rows", deleted)
        except Exception:
            _log.exception("Cleanup iteration failed; continuing")
        try:
            await asyncio.wait_for(stop.wait(), timeout=period_seconds)
        except asyncio.TimeoutError:
            pass


__all__ = [
    "TaigaOAuthProvider",
    "TaigaAuthenticationError",
    "run_cleanup_loop",
]
```

> **Note on `OAuthClientMetadata` types:** the constructor / field names (`token_endpoint_auth_method` vs `token_endpoint_auth_method_supported`, `redirect_uris` shape, etc.) come from the installed `mcp` SDK. If your version differs, adjust the field accesses — the LOGIC stays the same.

- [ ] **Step 4: Run integration tests**

```bash
make integration_test TEST_FILE=tests/integration_tests/test_provider.py
```
Expected: PASS, all flows including round-trip with `claims["taiga_jwt"]`.

- [ ] **Step 5: Commit**

```bash
git add langchain_taiga/auth/provider.py tests/integration_tests/test_provider.py
git commit -m "feat(auth): TaigaOAuthProvider with Fernet-decrypted JWT in AccessToken claims"
```

## Task 2.6: End-to-end token-flow security test

The most important assertion of the whole project: **two users' tool calls reach Taiga with their respective Bearer tokens, never the other user's, never the server's ENV token.** Tests the full chain through the FastMCP context.

**Files:**
- Create: `tests/integration_tests/test_e2e_token_flow.py`

- [ ] **Step 1: Write the test**

`tests/integration_tests/test_e2e_token_flow.py`:
```python
"""End-to-end Multi-Tenant security check.

Boots the OAuth provider, drives two users through the OAuth flow, then
simulates a tool call by setting up a fake AccessToken context and asserting
that python-taiga's outbound HTTP carries the right Bearer per user.
"""
from __future__ import annotations
import base64
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
import respx
from httpx import Response


@pytest.mark.asyncio
async def test_tool_call_carries_per_user_taiga_jwt(monkeypatch, fresh_store):
    """Two users, one tool call each. Outbound headers must carry distinct JWTs.

    Two HTTP libraries → two mock layers:
    - TaigaClient uses httpx (async) → respx for /api/v1/auth
    - python-taiga uses requests (sync) → responses for the project lookup that
      search_entities_tool triggers internally
    """
    import respx
    import responses
    from langchain_taiga.auth.provider import TaigaOAuthProvider
    from langchain_taiga.auth.taiga_client import TaigaClient
    from langchain_taiga.tools import taiga_tools

    captured: list[str] = []

    with respx.mock(assert_all_called=False) as respx_router, \
         responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:

        # ---- httpx mocks: TaigaClient credential exchange ----
        import json as _json

        def auth_handler(request):
            payload = _json.loads(request.content.decode())
            if payload["username"] == "alice":
                return Response(200, json={
                    "auth_token": "alice_jwt", "refresh": "alice_ref",
                    "id": 1, "username": "alice",
                })
            if payload["username"] == "bob":
                return Response(200, json={
                    "auth_token": "bob_jwt", "refresh": "bob_ref",
                    "id": 2, "username": "bob",
                })
            return Response(401)

        respx_router.post("https://taiga.example.test/api/v1/auth").mock(
            side_effect=auth_handler
        )

        # ---- requests mocks: python-taiga outbound ----
        def capture_callback(request):
            captured.append(request.headers.get("Authorization", ""))
            return (200, {}, _json.dumps({
                "id": 99, "slug": "shikenso-development", "name": "T", "members": [],
            }))

        rsps.add_callback(
            responses.GET,
            "https://taiga.example.test/api/v1/projects/by_slug/shikenso-development",
            callback=capture_callback,
        )
        # search_entities_tool may also list tasks/userstories/issues — broad allowlist
        for entity_path in ("tasks", "userstories", "issues", "epics"):
            rsps.add(
                responses.GET,
                f"https://taiga.example.test/api/v1/{entity_path}",
                json=[],
                status=200,
            )

        # ---- Drive OAuth flow for two users -----
        provider = TaigaOAuthProvider(
            store=fresh_store,
            taiga_client=TaigaClient(api_url="https://taiga.example.test"),
            issuer_url="https://taiga.shikenso.org/mcp",
        )
        from mcp.shared.auth import OAuthClientMetadata
        from mcp.server.auth.provider import AuthorizationParams

        client_info = await provider.register_client(OAuthClientMetadata(
            redirect_uris=["https://claude.ai/api/mcp/auth_callback"],
            client_name="Claude", token_endpoint_auth_method="none",
        ))

        async def issue(username):
            verifier = f"v_{username}" + "x" * 50
            challenge = base64.urlsafe_b64encode(
                hashlib.sha256(verifier.encode()).digest()
            ).rstrip(b"=").decode()
            redirect = await provider.authorize(
                client=client_info,
                params=AuthorizationParams(
                    response_type="code", scopes=["taiga"],
                    code_challenge=challenge, code_challenge_method="S256",
                    redirect_uri="https://claude.ai/api/mcp/auth_callback",
                    state="csrf",
                ),
            )
            internal = redirect.split("internal_state=", 1)[1]
            code, _ = await provider.complete_login(
                internal_state=internal, username=username, password="x",
            )
            auth_obj = await provider.load_authorization_code(client_info, code)
            oauth_tok = await provider.exchange_authorization_code(
                client_info, auth_obj
            )
            return oauth_tok.access_token

        alice_mcp = await issue("alice")
        bob_mcp = await issue("bob")

        # ---- Configure tool environment to point at our mocked Taiga ----
        monkeypatch.setattr(
            taiga_tools, "TAIGA_API_URL", "https://taiga.example.test"
        )
        taiga_tools.taiga_api_cache.clear()
        taiga_tools.project_cache.clear()

        # Alice invokes search_entities_tool
        alice_access = await provider.load_access_token(alice_mcp)
        monkeypatch.setattr(
            taiga_tools, "get_access_token", lambda: alice_access
        )
        taiga_tools.search_entities_tool.invoke({
            "project_slug": "shikenso-development",
            "query": "x", "entity_type": "task",
        })

        # Bob invokes
        bob_access = await provider.load_access_token(bob_mcp)
        monkeypatch.setattr(
            taiga_tools, "get_access_token", lambda: bob_access
        )
        taiga_tools.search_entities_tool.invoke({
            "project_slug": "shikenso-development",
            "query": "x", "entity_type": "task",
        })

        # ---- Killer assertions ----
        assert any("alice_jwt" in h for h in captured), \
            f"alice_jwt missing from outbound: {captured}"
        assert any("bob_jwt" in h for h in captured), \
            f"bob_jwt missing from outbound: {captured}"
        for h in captured:
            assert not ("alice_jwt" in h and "bob_jwt" in h), \
                "Token leakage — single header contained both tokens!"
```

- [ ] **Step 2: Run + commit + open PR**

```bash
make integration_test TEST_FILE=tests/integration_tests/test_e2e_token_flow.py
git add tests/integration_tests/test_e2e_token_flow.py
git commit -m "test(auth): end-to-end Multi-Tenant token-flow security check"

git push -u origin feat/auth-module
gh pr create --title "feat(auth): TokenStore + TaigaOAuthProvider with Fernet-encrypted JWTs" \
  --body "$(cat <<'EOF'
## Summary
- **TokenStore**: Postgres + HMAC-hashed MCP tokens + DCR client secrets + Fernet-encrypted Taiga JWTs at rest. Row-locked refresh on near-expiry.
- **TaigaClient**: async wrapper for Taiga `/api/v1/auth` and `/auth/refresh`.
- **TaigaOAuthProvider**: subclass of `fastmcp.server.auth.OAuthProvider`. Implements MCP-SDK ABC: `register_client`, `get_client`, `authorize`, `load_authorization_code`, `exchange_authorization_code`, `load_access_token`. Returns `AccessToken` with `claims = {"taiga_jwt", "user_id", "username"}` — this is what tools see via `get_access_token().claims`.
- Open-redirect protection: redirect_uri allowlist includes both `https://claude.ai/` AND `https://claude.com/` (Anthropic-required), plus `http://localhost:` for MCP Inspector.
- Public-client support: `token_endpoint_auth_method="none"` accepted at DCR.
- Login form via Jinja2; HTML-escaped state and error fields.
- End-to-end Multi-Tenant security test proves two users' tool calls carry distinct outbound Bearer headers and never leak.

## Not yet wired up
This PR ships the auth module standalone. PR 3 wires it into `remote_server.py` via `FastMCP(auth=provider)`.

## Test plan
- [ ] `make integration_test` passes
- [ ] `make test` passes (login page rendering)
- [ ] Stdio path unaffected

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 3: HTTP Entry Point — `FastMCP(auth=provider)` + custom `/oauth/login`

**Goal of this PR:** Wire `TaigaOAuthProvider` into a new HTTP entry point. **80% smaller than the v2 plan** — FastMCP auto-mounts OAuth and discovery routes. We add only:
- `mcp.custom_route("/oauth/login", methods=["GET"])` — render Jinja2 form
- `mcp.custom_route("/oauth/login", methods=["POST"])` — validate + redirect-with-code
- `mcp.custom_route("/health", methods=["GET"])` — K8s probes
- (defensive) Mirror `.well-known/oauth-authorization-server` and `oauth-protected-resource` at root path for non-spec-compliant clients

**Critical structural correction from v3 review:**
FastMCP auto-mounts OAuth + discovery routes when the ASGI app is *built*, not at request time. Setting `mcp.auth = provider` *inside* a lifespan startup is too late — the routes were never registered. The fix:
1. Refactor `langchain_taiga/mcp.py` from a module-level singleton to a `make_mcp(auth=None, lifespan=None)` factory.
2. In `remote_server.py`, eagerly bootstrap the provider (and its TokenStore) BEFORE calling `make_mcp(auth=provider, ...)`.
3. The lifespan only owns the cleanup-loop task and store-close on shutdown — it doesn't construct the provider.

This is reflected in Task 3.0 (factory refactor, mandatory first step) and Task 3.1 below.

**Branch:** `feat/remote-server`

## Task 3.0: Refactor `mcp.py` to factory; backwards-compatible singleton for stdio

**Files:**
- Modify: `langchain_taiga/mcp.py`
- Modify: `langchain_taiga/tools/taiga_tools.py` — `_register_mcp_tools` accepts an explicit `mcp` parameter
- Modify: `langchain_taiga/mcp_server.py` (already exists, just confirm import path)

The existing `langchain_taiga/mcp.py` exposes a module-level `mcp = FastMCP(...)` singleton, and `taiga_tools.py` registers all 15 tools against it at import time. That's correct for stdio, but the remote server needs a fresh `FastMCP` constructed with `auth=provider` (and possibly `lifespan=`) — which means tools have to be registered against a different `FastMCP` instance.

The factory pattern: `make_mcp(...)` constructs a fresh instance, registers all tools against it, returns it. The module-level `mcp = make_mcp()` (without auth) keeps stdio + the existing `LangChain Toolkit` import path working.

- [ ] **Step 1: Failing test**

`tests/unit_tests/test_mcp_factory.py`:
```python
"""The factory-style FastMCP construction must (a) keep the stdio singleton
intact and (b) allow a fresh instance with auth attached at construction."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock


def test_module_level_mcp_remains_importable():
    """Stdio path: importing langchain_taiga.mcp gives a usable FastMCP.
    Functional verification (tools really registered) happens in Step 5 via
    the LangChain Toolkit smoke test — that's the load-bearing check.
    """
    from langchain_taiga.mcp import mcp
    assert mcp is not None


def test_make_mcp_returns_fresh_instance():
    from langchain_taiga.mcp import make_mcp
    a = make_mcp()
    b = make_mcp()
    assert a is not b


def test_make_mcp_accepts_auth_kwarg():
    """The whole point of the factory: pass an OAuthProvider at construction."""
    from fastmcp.server.auth import OAuthProvider
    from langchain_taiga.mcp import make_mcp
    # FastMCP isinstance-checks auth=; bare MagicMock would be rejected with
    # TypeError, masking the real assertion. spec= makes the mock pass the type check.
    fake_provider = MagicMock(spec=OAuthProvider)
    fresh = make_mcp(auth=fake_provider)
    attached = getattr(fresh, "auth", None) or getattr(fresh, "_auth", None)
    assert attached is fake_provider
```

- [ ] **Step 2: Run to confirm failure**

```bash
git checkout -b feat/remote-server
make test TEST_FILE=tests/unit_tests/test_mcp_factory.py
```
Expected: FAIL — `make_mcp` not defined.

- [ ] **Step 3: Refactor `mcp.py`**

Replace the contents of `langchain_taiga/mcp.py`:

```python
"""FastMCP factory for langchain-taiga.

Two consumers:
- ``mcp_server.py`` (stdio) imports the module-level ``mcp`` singleton.
- ``remote_server.py`` (HTTP+OAuth) calls ``make_mcp(auth=provider, lifespan=...)``
  to build a fresh instance with the OAuth provider attached at construction
  time — this is **mandatory** because FastMCP auto-mounts OAuth + discovery
  routes when the ASGI app is built. Setting ``.auth`` after construction is
  too late.
"""
from __future__ import annotations

from importlib import metadata
from typing import Any, Optional

from fastmcp import FastMCP

try:
    __version__ = metadata.version("langchain-taiga")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"


def make_mcp(*, auth: Any = None, lifespan: Any = None) -> FastMCP:
    """Construct a fresh FastMCP with all langchain-taiga tools registered.

    Args:
        auth: OAuthProvider, TokenVerifier, or None. Triggers FastMCP's
            auto-mount of OAuth + discovery routes when non-None.
        lifespan: async context manager wired into the underlying ASGI app.
            Used by remote_server.py for cleanup-loop start/stop.

    The contract for ``auth=`` and ``lifespan=`` constructor support is
    enforced by ``pyproject.toml``'s ``fastmcp = ">=2.14.0,<3.0.0"`` pin —
    Phase 0 verifies that the installed patch version honours both. If a
    future patch within the pin range removes either kwarg, the build fails
    loud with a TypeError (which is what we want — better than a silent
    misroute). Do not add try/except wrappers here; the version pin is the
    contract.
    """
    init_kwargs: dict[str, Any] = {
        "name": "langchain-taiga",
        "version": __version__ or "0.0.0",
        "instructions": (
            "MCP server that surfaces Taiga project management tools from the "
            "langchain-taiga package."
        ),
    }
    if auth is not None:
        init_kwargs["auth"] = auth
    if lifespan is not None:
        init_kwargs["lifespan"] = lifespan

    mcp = FastMCP(**init_kwargs)

    # Register tools against this specific instance. _register_mcp_tools is
    # idempotent per-instance (tracks via id()).
    from langchain_taiga.tools.taiga_tools import _register_mcp_tools
    _register_mcp_tools(mcp)
    return mcp


# Module-level singleton — used by stdio entry, by the LangChain Toolkit, and
# by every existing test that does ``from langchain_taiga.mcp import mcp``.
mcp = make_mcp()
```

- [ ] **Step 4a (REQUIRED): Drop the top-level `mcp` import in `taiga_tools.py`**

The existing `taiga_tools.py:18` has `from langchain_taiga.mcp import mcp`. After Task 3.0's factory refactor, this import is **deterministically broken**: when Python loads `langchain_taiga/mcp.py`, the module body calls `make_mcp()` → which calls `_register_mcp_tools(mcp)` → which lives in `taiga_tools.py` → which Python tries to load → which tries to do `from langchain_taiga.mcp import mcp` → but `mcp` doesn't exist yet (it's being assigned by the line that's currently waiting on us). `ImportError` at import time.

Required edit: delete the top-level import. Tools are only attached via the `mcp_instance` argument to `_register_mcp_tools`, so the module-level reference is dead anyway.

```diff
 # langchain_taiga/tools/taiga_tools.py — top of file
 ...
-from langchain_taiga.mcp import mcp
 ...
```

- [ ] **Step 4b: Update `_register_mcp_tools` to accept an instance — preserve existing registration shape**

Modify `langchain_taiga/tools/taiga_tools.py` near the bottom (~line 2666). **The current shape is:**

```python
def _register_mcp_tools() -> None:
    global _MCP_REGISTERED
    if _MCP_REGISTERED:
        return
    for structured_tool in (
        create_entity_tool, search_entities_tool, ...,
    ):
        mcp.tool()(structured_tool.func)   # <-- existing form
    _MCP_REGISTERED = True

_register_mcp_tools()   # <-- existing eager call
```

**The minimum correct change:** swap the global `_MCP_REGISTERED` boolean for an `id(mcp_instance)`-keyed set (so per-instance idempotency holds), and accept `mcp_instance` as a parameter. **Keep the body's `mcp.tool()(structured_tool.func)` shape exactly as-is** unless the existing code uses a different form (e.g., `mcp.add_tool(...)`, `mcp.tool(name=..., description=...)(...)` etc.). Read the existing function before editing — preserve whatever it does, just parametrize the FastMCP target.

Conceptual diff:
```diff
-_MCP_REGISTERED = False
+_MCP_REGISTERED_INSTANCES: set[int] = set()
 
-def _register_mcp_tools() -> None:
-    global _MCP_REGISTERED
-    if _MCP_REGISTERED:
+def _register_mcp_tools(mcp_instance) -> None:
+    if id(mcp_instance) in _MCP_REGISTERED_INSTANCES:
         return
     for structured_tool in (
         create_entity_tool, search_entities_tool, ...,
     ):
-        mcp.tool()(structured_tool.func)
+        mcp_instance.tool()(structured_tool.func)
-    _MCP_REGISTERED = True
+    _MCP_REGISTERED_INSTANCES.add(id(mcp_instance))
 
-_register_mcp_tools()  # eager call at module import — DELETE
+# (no eager call; make_mcp() invokes this with the correct instance)
```

**If the existing `_register_mcp_tools` does anything fancier** (e.g., passes `name=`, `description=`, or wraps `args_schema` for MCP tool-discovery metadata), keep that fancy bit — your only job here is parametrizing the target FastMCP, not redesigning tool registration. The `StructuredTool.args_schema` propagation matters for MCP clients to introspect tool inputs; don't drop it.

- [ ] **Step 4c: Sanity-check the import chain**

```bash
poetry run python -c "import langchain_taiga; print('OK')"
poetry run python -c "from langchain_taiga.mcp import mcp; print('singleton:', mcp)"
poetry run python -c "from langchain_taiga.toolkits import TaigaToolkit; print('toolkit tools:', len(TaigaToolkit().get_tools()))"
```
All three must print without ImportError. If the first one fails, the circular-import fix in Step 4a wasn't applied.

- [ ] **Step 5: Run tests**

```bash
make test
poetry run python -c "from langchain_taiga.mcp import mcp, make_mcp; print('singleton:', type(mcp).__name__); print('factory:', type(make_mcp()).__name__)"
poetry run python -c "from langchain_taiga.toolkits import TaigaToolkit; print('toolkit tools:', len(TaigaToolkit().get_tools()))"
```
Expected: tests PASS, both prints succeed, toolkit reports 10.

- [ ] **Step 6: Commit**

```bash
git add langchain_taiga/mcp.py langchain_taiga/tools/taiga_tools.py tests/unit_tests/test_mcp_factory.py
git commit -m "refactor(mcp): expose make_mcp factory; tools register per-instance"
```

## Task 3.1: Implement `remote_server.py`

**Files:**
- Create: `langchain_taiga/remote_server.py`
- Modify: `pyproject.toml` (scripts entry)
- Modify: `langchain_taiga/mcp_server.py` (add `main()`)

- [ ] **Step 1: Implement `remote_server.py` with eager provider bootstrap**

The architecture forced by the v3 review:
1. `TokenStore.from_env()` is async (asyncpg pool init) — call inside `asyncio.run()` at startup.
2. `TaigaOAuthProvider` is constructed synchronously, holding a reference to the booted `TokenStore`.
3. `make_mcp(auth=provider, lifespan=_lifespan)` builds a FastMCP with both the OAuth provider AND lifespan attached at construction time. Tools are registered against this fresh instance by `make_mcp`.
4. The lifespan only owns: cleanup-loop task start/stop, `store.close()` on shutdown. It does NOT construct the provider.
5. Custom routes (`/oauth/login` GET+POST, `/health`, root well-known mirrors) are attached to the fresh `mcp` via `mcp.custom_route(...)` decorators — applied AFTER `make_mcp` returns so they bind to the right instance.

```python
"""HTTP entry point for the multi-tenant remote MCP server.

Required environment:
    TAIGA_API_URL          — Taiga API URL (cluster-internal in K8s)
    TAIGA_URL              — Taiga UI URL (used in tool-output URL construction)
    TAIGA_MCP_BASE_URL     — Public URL, e.g. https://taiga.shikenso.org/mcp
    TAIGA_MCP_DB_URL       — Postgres DSN
    TAIGA_MCP_TOKEN_SECRET — 64-hex (32-byte) HMAC key for at-rest token hashing
    TAIGA_MCP_FERNET_KEY   — 32-byte URL-safe base64 Fernet key for at-rest JWT encryption
    OPENAI_API_KEY         — for LLM-powered helpers
    TAIGA_MCP_HOST         — bind host (default 0.0.0.0)
    TAIGA_MCP_PORT         — bind port (default 8000)

URL surface (FastMCP auto-mounts the OAuth + discovery routes):
  GET  /.well-known/oauth-authorization-server[/mcp]   ← AS metadata
  GET  /.well-known/oauth-protected-resource[/mcp]     ← RS metadata
  POST /register                                       ← DCR
  GET  /authorize                                      ← redirects to /oauth/login
  POST /token                                          ← code exchange
  GET  /oauth/login                                    ← OUR custom HTML form
  POST /oauth/login                                    ← OUR credential handler
  GET  /health, /mcp/health                            ← OUR K8s probes
  POST /mcp/...                                        ← Bearer-protected tool calls (FastMCP)
"""
from __future__ import annotations
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from starlette.requests import Request
from starlette.responses import (
    JSONResponse, PlainTextResponse, RedirectResponse, Response,
)

from langchain_taiga.auth.login_page import render_login_page
from langchain_taiga.auth.provider import (
    TaigaAuthenticationError,
    TaigaOAuthProvider,
    run_cleanup_loop,
)
from langchain_taiga.auth.taiga_client import TaigaClient
from langchain_taiga.auth.token_store import TokenStore
from langchain_taiga.mcp import make_mcp

_log = logging.getLogger(__name__)


async def _bootstrap_provider() -> tuple[TaigaOAuthProvider, TokenStore]:
    """Eagerly create the TokenStore + Provider before FastMCP is built.

    This is the architectural fix for FastMCP's auto-mount-at-construction
    behaviour: setting ``mcp.auth`` AFTER construction is too late, the OAuth
    and discovery routes are never mounted. The provider must exist before
    ``make_mcp(auth=provider, ...)`` is called.
    """
    store = await TokenStore.from_env()
    await store.create_schema()
    provider = TaigaOAuthProvider(
        store=store,
        taiga_client=TaigaClient(api_url=os.environ["TAIGA_API_URL"]),
        issuer_url=os.environ["TAIGA_MCP_BASE_URL"],
    )
    return provider, store


def _make_lifespan(store: TokenStore):
    """Build a lifespan context manager closed over the booted TokenStore.

    The lifespan owns ONLY the cleanup-loop task and the store-close on shutdown.
    Provider construction happened in _bootstrap_provider() before us.
    """
    @asynccontextmanager
    async def _lifespan(_app):
        stop = asyncio.Event()
        cleanup_task = asyncio.create_task(run_cleanup_loop(store, stop=stop))
        try:
            yield
        finally:
            stop.set()
            try:
                await cleanup_task
            except Exception:
                _log.exception("Cleanup loop teardown error; continuing")
            try:
                await store.close()
            except Exception:
                _log.exception("Store.close() error; continuing")
    return _lifespan


def _attach_custom_routes(mcp, provider: TaigaOAuthProvider) -> None:
    """Bind /oauth/login, /health, and defensive root well-known mirrors.

    Called AFTER make_mcp() returns so the decorators bind to the correct
    FastMCP instance.
    """

    @mcp.custom_route("/health", methods=["GET"])
    async def _health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    @mcp.custom_route("/mcp/health", methods=["GET"])
    async def _mcp_health(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    @mcp.custom_route("/oauth/login", methods=["GET"])
    async def _login_get(request: Request) -> Response:
        internal_state = request.query_params.get("internal_state", "")
        if not internal_state:
            return PlainTextResponse("Missing internal_state", status_code=400)
        html = render_login_page(
            state=internal_state, error=None,
            taiga_url=os.environ["TAIGA_URL"],
        )
        return Response(html, media_type="text/html")

    @mcp.custom_route("/oauth/login", methods=["POST"])
    async def _login_post(request: Request) -> Response:
        form = await request.form()
        internal_state = form.get("state", "")
        username = form.get("username", "")
        password = form.get("password", "")
        if not (internal_state and username and password):
            return PlainTextResponse("Missing field(s)", status_code=400)
        try:
            _, redirect_url = await provider.complete_login(
                internal_state=internal_state, username=username, password=password,
            )
        except TaigaAuthenticationError:
            # Re-render with error; state is preserved (see complete_login)
            html = render_login_page(
                state=internal_state, error="Invalid Taiga username or password.",
                taiga_url=os.environ["TAIGA_URL"],
            )
            return Response(html, media_type="text/html", status_code=401)
        except ValueError as e:
            return PlainTextResponse(str(e), status_code=400)
        return RedirectResponse(redirect_url, status_code=303)

    # Defensive: mirror well-known discovery metadata at root path.
    # MCP clients have an open RFC-8414 conformance issue (TS SDK #822); some
    # look at root /.well-known/ regardless of issuer path. Belt-and-suspenders.
    @mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
    async def _as_metadata_root(_request: Request) -> JSONResponse:
        try:
            return JSONResponse(provider.authorization_server_metadata())
        except AttributeError:
            # Provider doesn't expose the helper — hand-build a minimal doc
            base = os.environ["TAIGA_MCP_BASE_URL"].rstrip("/")
            return JSONResponse({
                "issuer": base,
                "authorization_endpoint": f"{base}/authorize",
                "token_endpoint": f"{base}/token",
                "registration_endpoint": f"{base}/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": [
                    "none", "client_secret_basic", "client_secret_post",
                ],
            })

    @mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def _rs_metadata_root(_request: Request) -> JSONResponse:
        base = os.environ["TAIGA_MCP_BASE_URL"].rstrip("/")
        return JSONResponse({
            "resource": base,
            "authorization_servers": [base],
            "bearer_methods_supported": ["header"],
        })


# ---- Main --------------------------------------------------------------

async def _async_main(host: str, port: int) -> None:
    """Run inside a single event loop so async TokenStore + mcp.run_async share it."""
    provider, store = await _bootstrap_provider()
    lifespan = _make_lifespan(store)
    mcp = make_mcp(auth=provider, lifespan=lifespan)
    _attach_custom_routes(mcp, provider)

    if not hasattr(mcp, "run_async"):
        raise RuntimeError(
            "FastMCP does not expose run_async() — installed version is incompatible. "
            "Re-run the Phase 0 probe and update the plan; do NOT manually mount the "
            "ASGI app under /mcp as a workaround. Doing so would route OAuth and "
            "well-known endpoints under /mcp/... instead of the spec-required root "
            "(RFC 8414 §3.1), silently breaking discovery for claude.ai. This needs "
            "design rework, not a runtime fallback."
        )

    await mcp.run_async(
        transport="streamable-http", host=host, port=port, path="/mcp",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.getenv("TAIGA_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("TAIGA_MCP_PORT", "8000"))
    asyncio.run(_async_main(host, port))


if __name__ == "__main__":
    main()
```

> **Why no Path B fallback:** The naive idea (`Mount("/mcp", app=asgi_app)` under a Starlette parent) routes FastMCP's auto-mounted OAuth + discovery endpoints under `/mcp/authorize`, `/mcp/token`, `/mcp/.well-known/oauth-authorization-server` etc. RFC 8414 §3.1 requires the well-known doc for issuer `https://host/mcp` to live at `https://host/.well-known/oauth-authorization-server/mcp` — root path with the issuer's path appended, not `/mcp/.well-known/...`. A Starlette mount cannot fix that without re-implementing FastMCP's route layout. **If `run_async` is missing, the right answer is to update the plan, not to deploy broken OAuth discovery.** Phase 0 probe identifies this before any code is written.

> **Verification before merging:** Phase 0 must already have confirmed:
> - `mcp.run_async` exists with `transport`, `host`, `port`, `path` parameters
> - `FastMCP.__init__` accepts `auth=` and `lifespan=` (or, if not, the route-layout consequences are explicitly understood and the plan adjusted)
> - `OAuthProvider` `__init__` parameters match what `provider.py` passes (`issuer_url`, `client_registration_options`, `required_scopes`)

- [ ] **Step 2: Add main() to mcp_server.py + script entries**

`langchain_taiga/mcp_server.py`:
```python
"""Model Context Protocol server exposing Taiga tools via fastmcp (stdio mode)."""
from __future__ import annotations
import langchain_taiga.tools.taiga_tools  # noqa: F401
from langchain_taiga.mcp import mcp


def main() -> None:
    mcp.run()


run = mcp.run  # legacy

if __name__ == "__main__":
    main()
```

In `pyproject.toml`:
```toml
[tool.poetry.scripts]
langchain-taiga-mcp = "langchain_taiga.mcp_server:main"
langchain-taiga-mcp-remote = "langchain_taiga.remote_server:main"
```

- [ ] **Step 3: Reinstall**

```bash
git checkout -b feat/remote-server
poetry install
poetry run langchain-taiga-mcp-remote --help 2>&1 | head -3
```

## Task 3.2: Local validation with MCP Inspector

- [ ] **Step 1: Local Postgres**

```bash
docker run --rm -d --name taiga-mcp-pg \
  -e POSTGRES_USER=mcp -e POSTGRES_PASSWORD=mcp -e POSTGRES_DB=mcp \
  -p 55432:5432 postgres:15-alpine
```

- [ ] **Step 2: Generate secrets and start the server**

```bash
TOKEN_SECRET=$(python -c 'import secrets; print(secrets.token_hex(32))')
FERNET_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')

TAIGA_API_URL=https://taiga.shikenso.org \
TAIGA_URL=https://taiga.shikenso.org \
TAIGA_MCP_BASE_URL=http://localhost:8000/mcp \
TAIGA_MCP_DB_URL=postgresql://mcp:mcp@localhost:55432/mcp \
TAIGA_MCP_TOKEN_SECRET=$TOKEN_SECRET \
TAIGA_MCP_FERNET_KEY=$FERNET_KEY \
OPENAI_API_KEY=$OPENAI_API_KEY \
poetry run langchain-taiga-mcp-remote
```

Expected: server starts on `:8000`, logs schema creation. Save `TOKEN_SECRET` and `FERNET_KEY` for restart consistency.

- [ ] **Step 3a: Curl smoke-test the load-bearing FastMCP routing assumption**

Before the Inspector flow, verify that FastMCP serves discovery metadata at the RFC 8414 §3.1 path-aware location. If these 404, the entire ingress story (Task 4.2) is misconfigured and the Inspector flow won't work either:

```bash
curl -sS http://localhost:8000/.well-known/oauth-authorization-server/mcp | jq .issuer
curl -sS http://localhost:8000/.well-known/oauth-protected-resource/mcp | jq .resource
```

Expected: both return JSON with `"https://taiga.shikenso.org/mcp"` (or whatever your `TAIGA_MCP_BASE_URL` is). If either returns 404 or HTML, the path-aware mount is wrong and Task 3.1 needs adjustment before going further.

Also exercise the defensive root-mirror routes added in `_attach_custom_routes`:

```bash
curl -sS http://localhost:8000/.well-known/oauth-authorization-server | jq .issuer
curl -sS http://localhost:8000/.well-known/oauth-protected-resource | jq .resource
```

Expected: same shape. These exist for non-spec-compliant MCP clients per the v3 review.

- [ ] **Step 3b: MCP Inspector**

```bash
npx @modelcontextprotocol/inspector
```

- HTTP transport, URL `http://localhost:8000/mcp`
- Connect → OAuth flow auto-discovers via `/.well-known/oauth-authorization-server`
- Click through DCR → Authorize → login form
- Submit Taiga credentials → redirect → Inspector exchanges code → token
- List tools, invoke `search_entities_tool` with your real project slug — verify your data

- [ ] **Step 4: Multi-user check**

In an incognito window, sign in as a different Taiga user, invoke the same tool, verify different (correctly-scoped) results.

- [ ] **Step 5: PR**

```bash
git add langchain_taiga/remote_server.py langchain_taiga/mcp_server.py pyproject.toml poetry.lock
git commit -m "feat: HTTP entry point — FastMCP(auth=provider) + custom /oauth/login"
git push -u origin feat/remote-server
gh pr create --title "feat: remote HTTP+OAuth MCP server entry point" \
  --body "$(cat <<'EOF'
## Summary
- New `remote_server.py` (~200 LOC). `FastMCP(auth=provider)` triggers auto-mounting of OAuth and discovery routes.
- Custom routes only for `/oauth/login` (GET+POST), `/health`, and root-path mirrors of `.well-known/oauth-*` (defensive against non-spec-compliant MCP clients).
- Lifespan-managed: TokenStore connection pool + cleanup loop start with the server, tear down on shutdown. Pool is closed even on partial-startup failure.
- Validated locally with MCP Inspector against `taiga.shikenso.org`: OAuth flow end-to-end, two test users see distinct project-scoped data.

## Test plan
- [ ] §3.2 local validation per plan
- [ ] Existing unit + integration tests pass

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 4: Helm Chart for `taiga-mcp`

**Goal:** `deployment/helm/taiga-mcp/` in the **`taiga` repo**. Deploys MCP next to Taiga in the `taiga` namespace, embedded Postgres subchart, path-suffix ingress on `taiga.shikenso.org/mcp` (reuses existing `taiga-tls` secret).

**Branch (in `taiga` repo):** `feat/taiga-mcp-helm`

> All paths in this PR are relative to `/home/wahed/workspace/taiga/`.

## Task 4.1: Chart skeleton

- [ ] **Step 1: Create branch**

```bash
cd /home/wahed/workspace/taiga
git checkout main && git pull
git checkout -b feat/taiga-mcp-helm
mkdir -p deployment/helm/taiga-mcp/{templates,charts}
```

- [ ] **Step 2: `Chart.yaml`**

```yaml
apiVersion: v2
name: taiga-mcp
description: MCP server for Taiga — per-user Taiga tools via Model Context Protocol with OAuth 2.1.
type: application
version: 0.1.0
appVersion: "1.9.1"
dependencies:
  - name: postgresql
    version: 13.2.24
    repository: file://charts/postgresql-13.2.24.tgz
    condition: postgresql.enabled
home: https://github.com/Shikenso-Analytics/langchain-taiga
sources:
  - https://github.com/Shikenso-Analytics/langchain-taiga
  - https://github.com/Shikenso-Analytics/taiga
maintainers:
  - name: Shikenso Analytics
```

- [ ] **Step 3: Copy Postgres subchart from existing Taiga chart**

```bash
cp deployment/helm/taiga/charts/postgresql-13.2.24.tgz deployment/helm/taiga-mcp/charts/
```

- [ ] **Step 4: `values.yaml`**

```yaml
# IMPORTANT: replicaCount is hard-pinned to 1 because the OAuth bridge stores
# in-flight Authorize state (PKCE challenges, ~10-min lifetime) in a per-pod
# in-memory dict. Scaling to >1 pod would silently break OAuth flows for users
# whose login form lands on a different pod than their /authorize redirect.
# This is a v1 design constraint; v2 will move state to Postgres.
replicaCount: 1

image:
  repository: registry.shikenso.org/taiga-mcp
  tag: ""  # set by Jenkins; falls back to .Chart.AppVersion
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  host: taiga.shikenso.org
  pathPrefix: /mcp
  tlsSecretName: taiga-tls   # SHARED with Taiga chart — pre-existing secret
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"

resources:
  requests: {cpu: 100m, memory: 256Mi}
  limits:   {cpu: 1000m, memory: 1Gi}

# Public base URL the MCP server reports as its OAuth issuer
baseUrl: "https://taiga.shikenso.org/mcp"
taigaApiUrl: "http://taiga-back.taiga.svc.cluster.local:8000"
taigaUrl: "https://taiga.shikenso.org"

# Pre-create this secret with the following keys:
#   OPENAI_API_KEY          — for LLM helpers
#   TAIGA_MCP_TOKEN_SECRET  — 64-hex HMAC key (python -c 'import secrets; print(secrets.token_hex(32))')
#   TAIGA_MCP_FERNET_KEY    — Fernet key (python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')
existingSecret: taiga-mcp-secrets

postgresql:
  enabled: true
  auth:
    username: mcp
    database: mcp
    existingSecret: taiga-mcp-postgresql-auth
    secretKeys:
      adminPasswordKey: postgres-password
      userPasswordKey: password
  primary:
    persistence:
      enabled: true
      size: 1Gi

networkPolicy:
  enabled: true
  allowedClaudeAi:
    - 160.79.104.0/21

podDisruptionBudget:
  enabled: false  # single replica → no PDB

serviceAccount:
  create: true
  annotations: {}
  name: ""
```

- [ ] **Step 5: `_helpers.tpl`** (standard pattern, copied from Taiga chart)

```yaml
{{- define "taiga-mcp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "taiga-mcp.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "taiga-mcp.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "taiga-mcp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "taiga-mcp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "taiga-mcp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "taiga-mcp.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "taiga-mcp.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
```

- [ ] **Step 6: Lint**

```bash
helm dependency update deployment/helm/taiga-mcp
helm lint deployment/helm/taiga-mcp --set image.tag=1.9.1
```

- [ ] **Step 7: Commit**

```bash
git add deployment/helm/taiga-mcp/
git commit -m "feat(helm): scaffold taiga-mcp chart with single-replica constraint"
```

## Task 4.2: Deployment, Service, Ingress, ConfigMap, Secret, ServiceAccount

`templates/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "taiga-mcp.fullname" . }}
  labels: {{- include "taiga-mcp.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}  # MUST stay 1 — see values.yaml comment
  selector:
    matchLabels: {{- include "taiga-mcp.selectorLabels" . | nindent 6 }}
  strategy:
    type: Recreate  # single replica + state coupling → no rolling
  template:
    metadata:
      labels: {{- include "taiga-mcp.selectorLabels" . | nindent 8 }}
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      serviceAccountName: {{ include "taiga-mcp.serviceAccountName" . }}
      containers:
        - name: taiga-mcp
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command: ["langchain-taiga-mcp-remote"]
          ports:
            - name: http
              containerPort: 8000
          envFrom:
            - configMapRef:
                name: {{ include "taiga-mcp.fullname" . }}
            - secretRef:
                name: {{ .Values.existingSecret }}
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: {{ .Release.Name }}-postgresql
                  key: password
            - name: TAIGA_MCP_DB_URL
              value: "postgresql://{{ .Values.postgresql.auth.username }}:$(POSTGRES_PASSWORD)@{{ .Release.Name }}-postgresql:5432/{{ .Values.postgresql.auth.database }}"
          readinessProbe:
            httpGet: {path: /mcp/health, port: http}
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet: {path: /mcp/health, port: http}
            initialDelaySeconds: 30
            periodSeconds: 30
          resources: {{- toYaml .Values.resources | nindent 12 }}
```

`templates/service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "taiga-mcp.fullname" . }}
  labels: {{- include "taiga-mcp.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - {name: http, port: {{ .Values.service.port }}, targetPort: http, protocol: TCP}
  selector: {{- include "taiga-mcp.selectorLabels" . | nindent 4 }}
```

`templates/ingress.yaml`:
```yaml
{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "taiga-mcp.fullname" . }}
  labels: {{- include "taiga-mcp.labels" . | nindent 4 }}
  annotations: {{- toYaml .Values.ingress.annotations | nindent 4 }}
spec:
  ingressClassName: {{ .Values.ingress.className }}
  tls:
    - hosts: [{{ .Values.ingress.host }}]
      secretName: {{ .Values.ingress.tlsSecretName }}
  rules:
    - host: {{ .Values.ingress.host }}
      http:
        paths:
          # MCP tool path + auto-mounted OAuth endpoints (FastMCP serves them under /mcp/...)
          - {path: {{ .Values.ingress.pathPrefix }}, pathType: Prefix, backend: {service: {name: {{ include "taiga-mcp.fullname" . }}, port: {number: {{ .Values.service.port }}}}}}
          # Path-aware discovery (RFC 8414 §3.1)
          - {path: /.well-known/oauth-authorization-server/mcp, pathType: Exact, backend: {service: {name: {{ include "taiga-mcp.fullname" . }}, port: {number: {{ .Values.service.port }}}}}}
          - {path: /.well-known/oauth-protected-resource/mcp, pathType: Exact, backend: {service: {name: {{ include "taiga-mcp.fullname" . }}, port: {number: {{ .Values.service.port }}}}}}
          # Defensive root-path mirror (v3 review item)
          - {path: /.well-known/oauth-authorization-server, pathType: Exact, backend: {service: {name: {{ include "taiga-mcp.fullname" . }}, port: {number: {{ .Values.service.port }}}}}}
          - {path: /.well-known/oauth-protected-resource, pathType: Exact, backend: {service: {name: {{ include "taiga-mcp.fullname" . }}, port: {number: {{ .Values.service.port }}}}}}
{{- end }}
```

`templates/configmap.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "taiga-mcp.fullname" . }}
  labels: {{- include "taiga-mcp.labels" . | nindent 4 }}
data:
  TAIGA_API_URL: {{ .Values.taigaApiUrl | quote }}
  TAIGA_URL: {{ .Values.taigaUrl | quote }}
  TAIGA_MCP_BASE_URL: {{ .Values.baseUrl | quote }}
  TAIGA_MCP_HOST: "0.0.0.0"
  TAIGA_MCP_PORT: "8000"
```

`templates/secret.yaml` (placeholder if operator forgot to pre-create):
```yaml
{{- if not (lookup "v1" "Secret" .Release.Namespace .Values.existingSecret) -}}
apiVersion: v1
kind: Secret
metadata:
  name: {{ .Values.existingSecret }}
  labels: {{- include "taiga-mcp.labels" . | nindent 4 }}
  annotations:
    helm.sh/hook: pre-install
    helm.sh/hook-weight: "-5"
type: Opaque
stringData:
  OPENAI_API_KEY: "REPLACE_ME"
  # 64-hex (32 bytes): python -c 'import secrets; print(secrets.token_hex(32))'
  TAIGA_MCP_TOKEN_SECRET: "REPLACE_ME_64_HEX_CHARS"
  # Fernet key: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
  TAIGA_MCP_FERNET_KEY: "REPLACE_ME_FERNET_KEY"
{{- end }}
```

`templates/serviceaccount.yaml`:
```yaml
{{- if .Values.serviceAccount.create -}}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "taiga-mcp.serviceAccountName" . }}
  labels: {{- include "taiga-mcp.labels" . | nindent 4 }}
{{- end }}
```

- [ ] **Lint + commit:**
```bash
helm lint deployment/helm/taiga-mcp --set image.tag=1.9.1
helm template taiga-mcp deployment/helm/taiga-mcp --set image.tag=1.9.1 | head -100
git add deployment/helm/taiga-mcp/templates/
git commit -m "feat(helm): deployment, service, ingress (with root well-known mirrors), configmap, secret, sa"
```

## Task 4.3: NetworkPolicy

```yaml
{{- if .Values.networkPolicy.enabled -}}
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {{ include "taiga-mcp.fullname" . }}-ingress
  labels: {{- include "taiga-mcp.labels" . | nindent 4 }}
spec:
  podSelector:
    matchLabels: {{- include "taiga-mcp.selectorLabels" . | nindent 6 }}
  policyTypes: [Ingress, Egress]
  ingress:
    - from:
        - namespaceSelector:
            matchLabels: {kubernetes.io/metadata.name: ingress-nginx}
      ports: [{port: 8000, protocol: TCP}]
    - from:
        {{- range .Values.networkPolicy.allowedClaudeAi }}
        - {ipBlock: {cidr: {{ . | quote }}}}
        {{- end }}
      ports: [{port: 8000, protocol: TCP}]
  egress:
    - to:
        - podSelector:
            matchLabels:
              app.kubernetes.io/name: postgresql
              app.kubernetes.io/instance: {{ .Release.Name }}
      ports: [{port: 5432, protocol: TCP}]
    - to:
        - namespaceSelector:
            matchLabels: {kubernetes.io/metadata.name: taiga}
          podSelector:
            matchLabels: {app.kubernetes.io/name: taiga-back}
      ports: [{port: 8000, protocol: TCP}]
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels: {k8s-app: kube-dns}
      ports: [{port: 53, protocol: UDP}]
    # Egress to OpenAI + external Taiga (defense-in-depth: tighten in v2 with egress gateway)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except: [10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16]
      ports: [{port: 443, protocol: TCP}]
{{- end }}
```

```bash
git add deployment/helm/taiga-mcp/templates/networkpolicy.yaml
git commit -m "feat(helm): NetworkPolicy with claude.ai allowlist"
```

## Task 4.4: Cluster dry-run

- [ ] **Step 1: Verify ingress-nginx namespace label**

```bash
KUBECONFIG=~/Desktop/kenso-cluster.yaml kubectl get ns --show-labels | grep -i nginx
```
If your cluster uses a non-`ingress-nginx` namespace name, override `from.namespaceSelector.matchLabels` in `values.yaml`.

- [ ] **Step 2: Pre-create secret**

```bash
TOKEN_SECRET=$(python -c 'import secrets; print(secrets.token_hex(32))')
FERNET_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')

KUBECONFIG=~/Desktop/kenso-cluster.yaml kubectl -n taiga create secret generic taiga-mcp-secrets \
  --from-literal=OPENAI_API_KEY=$OPENAI_API_KEY \
  --from-literal=TAIGA_MCP_TOKEN_SECRET=$TOKEN_SECRET \
  --from-literal=TAIGA_MCP_FERNET_KEY=$FERNET_KEY \
  --dry-run=client -o yaml
```

- [ ] **Step 3: Helm dry-run**

```bash
KUBECONFIG=~/Desktop/kenso-cluster.yaml helm install taiga-mcp \
  deployment/helm/taiga-mcp --namespace taiga \
  --set image.tag=1.9.1 --dry-run --debug 2>&1 | head -100
```

- [ ] **Step 4: Open PR**

```bash
git push -u origin feat/taiga-mcp-helm
gh pr create --title "feat(helm): taiga-mcp chart — single-replica, path-suffix ingress, Fernet key" \
  --body "$(cat <<'EOF'
## Summary
- New `deployment/helm/taiga-mcp/`. Same namespace as Taiga.
- Path-suffix ingress on `taiga.shikenso.org/mcp` reuses existing `taiga-tls` secret. Routes: `/mcp/*`, path-aware `.well-known/*` AND root-path `.well-known/*` (defensive against non-spec-compliant MCP clients).
- Embedded Bitnami Postgres subchart (1Gi PVC), decoupled from Taiga's Postgres.
- `replicaCount: 1` hard-pinned with explicit comment about in-memory authorize-state coupling.
- NetworkPolicy: ingress from nginx + Anthropic CIDR; egress to Postgres, Taiga backend, OpenAI/HTTPS.
- Required secrets pre-created: `OPENAI_API_KEY`, `TAIGA_MCP_TOKEN_SECRET`, `TAIGA_MCP_FERNET_KEY`.

## Test plan
- [ ] `helm lint` passes
- [ ] `helm install --dry-run` against OVH cluster
- [ ] No conflict with existing Taiga ingress at the same host
EOF
)"
```

---

# PR 5: Dockerfile + Jenkins Pipeline

(Unchanged from v2 — reproduced for completeness.)

## Task 5.1: Dockerfile (in `taiga` repo)

`deployment/helm/taiga-mcp/Dockerfile`:
```dockerfile
FROM python:3.12-slim
ARG TAIGA_MCP_VERSION
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir "langchain-taiga==${TAIGA_MCP_VERSION}"
RUN useradd --system --uid 1000 app
USER app
EXPOSE 8000
CMD ["langchain-taiga-mcp-remote"]
```

```bash
cd /home/wahed/workspace/taiga
git checkout -b feat/taiga-mcp-jenkins
docker build --build-arg TAIGA_MCP_VERSION=1.9.1 \
  -t taiga-mcp:dev deployment/helm/taiga-mcp/
docker run --rm taiga-mcp:dev python -c "import langchain_taiga; print(langchain_taiga.__version__)"
git add deployment/helm/taiga-mcp/Dockerfile
git commit -m "feat(taiga-mcp): pip-install Dockerfile"
```

## Task 5.2: Jenkinsfile.taiga-mcp

```groovy
@Library('shikenso-ci@1.0.1') _

pipeline {
    agent { label 'docker-helm' }
    parameters {
        string(name: 'TAIGA_MCP_VERSION', defaultValue: '1.9.1',
               description: 'langchain-taiga PyPI version + image tag.')
        choice(name: 'ACTION', choices: ['deploy', 'validate'],
               description: 'deploy = upgrade --install, validate = template + dry-run')
    }
    environment {
        REGISTRY     = "registry.shikenso.org"
        IMAGE_REPO   = "taiga-mcp"
        IMAGE_TAG    = "${params.TAIGA_MCP_VERSION}"
        CHART_DIR    = "deployment/helm/taiga-mcp"
        DOCKERFILE   = "deployment/helm/taiga-mcp/Dockerfile"
        NAMESPACE    = "taiga"
        RELEASE_NAME = "taiga-mcp"
    }
    options { timeout(time: 30, unit: 'MINUTES'); ansiColor('xterm') }
    stages {
        stage('Verify PyPI release') {
            steps {
                sh '''
                    curl -s --fail "https://pypi.org/pypi/langchain-taiga/${TAIGA_MCP_VERSION}/json" >/dev/null \
                      || (echo "ERROR: langchain-taiga==${TAIGA_MCP_VERSION} not on PyPI"; exit 1)
                '''
            }
        }
        stage('Build') {
            steps {
                sh '''
                    docker build \
                        --build-arg TAIGA_MCP_VERSION="${TAIGA_MCP_VERSION}" \
                        --tag "${REGISTRY}/${IMAGE_REPO}:${IMAGE_TAG}" \
                        --tag "${REGISTRY}/${IMAGE_REPO}:latest" \
                        --file "${DOCKERFILE}" deployment/helm/taiga-mcp/
                '''
            }
        }
        stage('Push') {
            when { expression { params.ACTION == 'deploy' } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'shikenso-registry',
                                                  usernameVariable: 'REG_USER',
                                                  passwordVariable: 'REG_PASS')]) {
                    sh '''
                        echo "${REG_PASS}" | docker login "${REGISTRY}" -u "${REG_USER}" --password-stdin
                        docker push "${REGISTRY}/${IMAGE_REPO}:${IMAGE_TAG}"
                        docker push "${REGISTRY}/${IMAGE_REPO}:latest"
                    '''
                }
            }
        }
        stage('Update chart deps') {
            steps { sh 'cd "${CHART_DIR}" && helm dependency update' }
        }
        stage('Validate') {
            steps {
                withCredentials([file(credentialsId: 'kubeconfig-ovh-mks', variable: 'KUBECONFIG')]) {
                    sh '''
                        helm template "${RELEASE_NAME}" "${CHART_DIR}" \
                            --namespace "${NAMESPACE}" \
                            --set image.tag="${IMAGE_TAG}" > rendered.yaml
                        kubectl --kubeconfig "${KUBECONFIG}" apply --dry-run=server -f rendered.yaml
                    '''
                }
            }
        }
        stage('Deploy') {
            when { expression { params.ACTION == 'deploy' } }
            steps {
                withCredentials([file(credentialsId: 'kubeconfig-ovh-mks', variable: 'KUBECONFIG')]) {
                    sh '''
                        # Bitnami PG password passthrough (mirrors taiga repo Jenkinsfile commit 90c13cc)
                        PG_PASS=$(kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" \
                            get secret "${RELEASE_NAME}-postgresql" \
                            -o jsonpath="{.data.password}" 2>/dev/null | base64 -d || echo "")
                        PG_ADMIN_PASS=$(kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" \
                            get secret "${RELEASE_NAME}-postgresql" \
                            -o jsonpath="{.data.postgres-password}" 2>/dev/null | base64 -d || echo "")

                        helm upgrade --install "${RELEASE_NAME}" "${CHART_DIR}" \
                            --namespace "${NAMESPACE}" --kubeconfig "${KUBECONFIG}" \
                            --set image.tag="${IMAGE_TAG}" \
                            ${PG_PASS:+--set postgresql.auth.password="$PG_PASS" --set global.postgresql.auth.password="$PG_PASS"} \
                            ${PG_ADMIN_PASS:+--set postgresql.auth.postgresPassword="$PG_ADMIN_PASS"} \
                            --wait --timeout 10m
                    '''
                }
            }
        }
    }
    post { always { sh 'docker logout "${REGISTRY}" || true'; cleanWs() } }
}
```

```bash
git add Jenkinsfile.taiga-mcp
git commit -m "ci(taiga-mcp): Jenkins pipeline (Dockerfile + helm upgrade)"
git push -u origin feat/taiga-mcp-jenkins
gh pr create --title "ci(taiga-mcp): Dockerfile + Jenkins pipeline" \
  --body "Pipeline + thin pip-install Dockerfile. PyPI-version pre-flight, Bitnami-PG password passthrough."
```

---

# Validation: claude.ai Connector Wiring

After PR 5 deploy, register the connector in claude.ai (manual):

- [ ] **Step 1:** Settings → Connectors → Add Custom Connector
  - Name: `Taiga (Shikenso)`
  - URL: `https://taiga.shikenso.org/mcp`
- [ ] **Step 2:** Click "Connect" — claude.ai performs DCR + Authorize → redirects to login form
- [ ] **Step 3:** Sign in with Taiga credentials → claude.ai exchanges code for access token
- [ ] **Step 4:** Smoke test in chat: "List my user stories in shikenso-development"
- [ ] **Step 5:** Multi-user verification with a second team member

---

# Risks & Watchpoints

| Risk | Mitigation |
|---|---|
| FastMCP `OAuthProvider` API differs across patch versions (e.g., `ClientRegistrationOptions` import path, exact ABC method names) | Verification probe at start of Task 2.5 reads installed source; logic is correct, only imports/types adapt. |
| MCP-SDK type names (`OAuthClientMetadata`, `AuthorizationParams`, `AccessToken.claims` shape) | Same — verify against installed `mcp` SDK. The `claims` dict field is documented stable in v2.11+. |
| MCP TS SDK RFC-8414 path-issuer bug (#822) | Defensive root-path mirror of `.well-known/oauth-*` in both Ingress and `remote_server.py`. |
| `nginx.ingress.kubernetes.io/rewrite-target` on existing Taiga ingress would strip `/mcp/` | Check during Task 4.4 dry-run; if present, MCP ingress declares no rewrite (longest-prefix wins). |
| Cache leak between users | `_user_scoped_key` uses `user_id` from claims; E2E test asserts outbound headers per user. |
| Bitnami Postgres "PASSWORDS ERROR" on upgrade | Jenkinsfile passthrough pattern from `taiga` repo `90c13cc`. |
| Anthropic IP CIDR changes | `values.yaml` `networkPolicy.allowedClaudeAi` is a list — operators update via `--set` or values override. |
| LLM cost in multi-tenant mode | Server-paid in v1; per-user OpenAI key deferred to v2. |
| Refresh-token race | `refresh_lock` SELECT FOR UPDATE; race-condition test. |
| `_authorize_states` in-memory + 1 replica | Hard-pinned `replicaCount: 1`; documented constraint. v2 moves state to Postgres if scaling needed. |
| Taiga JWT/refresh DB compromise | **Now Fernet-encrypted at rest.** DB compromise + K8s secret read both required. |
| Plaintext `print(...)` calls in `taiga_tools.py` (`find_users` LLM debug, `get_project` error log) leak user-context output to stdout | Pre-deployment audit before PR 3 merges: replace with `logging.getLogger(__name__).debug(...)`. |
| No rate-limit on `/oauth/register` | NetworkPolicy restricts to claude.ai/nginx; flood-from-Anthropic-IPs is implausible. v2 hardening: per-IP token bucket. |
| `TAIGA_MCP_TOKEN_SECRET` / `TAIGA_MCP_FERNET_KEY` rotation invalidates all live state | Documented in `values.yaml`; treat as long-lived secrets. Rotate on suspected compromise only. |
| FastMCP `mcp.run(lifespan=...)` arg may not exist in installed version | Phase 0 probe identifies this. `remote_server.py` uses `mcp.run_async()` exclusively; if missing, this is a plan-update event, not a runtime fallback (see "Why no Path B fallback" in Task 3.1). |
| **FastMCP attaches OAuth + discovery routes when ASGI app is built (construction time), not at request time** | `mcp.py` is a factory (`make_mcp(auth=provider, lifespan=...)`); `remote_server.py` bootstraps the `TaigaOAuthProvider` synchronously **before** calling `make_mcp`. Setting `.auth` after construction is fundamentally too late and will silently 404 the OAuth endpoints — Task 3.0 prevents this. |
| **MCP access-token TTL must stay ≤ Taiga JWT TTL (1h ≤ 1d default)** | Documented in `load_access_token` docstring + plan. Server-side transparent refresh is **NOT** implemented; if Taiga TTL ever shortens below MCP TTL, lower `ACCESS_TOKEN_TTL` to match — don't reintroduce refresh-on-the-fly because it desyncs server expiry from claude.ai's `expires_in` claim. Refresh-token grant is a v2 feature that requires schema split (`mcp_expires_at` + `taiga_expires_at`). |
| MCP-SDK type imports differ across patch versions (`OAuthClientMetadata`, `AuthorizationParams`, `AccessToken`) | Phase 0 probe confirms canonical paths; Task 2.5 Step 1 re-verifies before coding. Imports are localized to one file (`provider.py`). |

---

# Self-Review Notes

**Spec coverage check:**

- ✅ Phase 1 (refactor `get_taiga_api`) → Task 1.2
- ✅ Phase 2 (per-request JWT propagation) → Tasks 1.3–1.4 — **massively simplified**: no kwarg threading, no ContextVar, no custom middleware
- ✅ Phase 3 (auth module) → Tasks 2.1–2.5
- ✅ Phase 4 (HTTP entry point) → Task 3.1 — **80% smaller than v2**: FastMCP auto-mounts; only `/oauth/login` is custom
- ✅ Phase 5 (MCP Inspector validation) → Task 3.2
- ✅ Phase 6 (Helm chart) → Tasks 4.1–4.4 with `replicaCount: 1` constraint and Fernet key
- ✅ Phase 7 (Jenkins) → Tasks 5.1–5.2
- ✅ Phase 8 (claude.ai connector) → "Validation" section, with `claude.ai` and `claude.com` callbacks documented

**v3 architecture-review action items:**

1. ✅ `OAuthProvider` subclass + `FastMCP(auth=...)` (Task 2.5)
2. ✅ Custom Bearer-to-ContextVar middleware deleted; tools use `get_access_token()` (Tasks 1.3–1.4)
3. ✅ Taiga JWT carried in `AccessToken.claims["taiga_jwt"]` (Task 2.5 `load_access_token`)
4. ✅ `https://claude.com/api/mcp/auth_callback` in redirect_uri allowlist (Task 2.5 `DEFAULT_ALLOWED_REDIRECT_PREFIXES`)
5. ✅ `none` advertised in `token_endpoint_auth_methods_supported` (Task 3.1 metadata fallback)
6. ✅ Discovery metadata mirrored at root path (Task 3.1 + Task 4.2 ingress)
7. ✅ Upstream Taiga JWT and refresh tokens encrypted with Fernet (Task 2.2)
8. ✅ `_authorize_states` in-memory; Helm `replicaCount: 1` with comment (Task 4.1)
9. ⏭ Authlib `AuthorizationServer` — deferred to v2 follow-up (could replace ~300 LOC of grant logic)
10. ✅ HTTP 400 `invalid_client` for unknown DCR clients (Task 2.5 `get_client` returns None; FastMCP renders the error)

**Items deferred to v2 follow-ups:**

- Sentry / `kenso_utils` integration
- Per-user OpenAI keys
- Wildcard cert via DNS-01
- Authlib substitution for grant logic
- alembic / numbered SQL migrations
- Move `_authorize_states` to Postgres (only when scaling >1 replica)
- Rate-limit on `/oauth/register`
- Refresh-token rotation (RFC 6819 §5.2.2.3)
- `TAIGA_MCP_TOKEN_SECRET` / `TAIGA_MCP_FERNET_KEY` rotation runbook
- Replace `print(...)` calls with structured logging

No placeholder TODOs remain. The OAuth wiring is fully explicit (auto-mounted by FastMCP + one custom login route). The Bearer → AccessToken propagation is fully explicit (FastMCP auth-context middleware → `get_access_token()` → tool body). Net code reduction vs. v2: ~40%.

**v3.1 follow-up corrections:**

| Bug | Fix location |
|---|---|
| `mcp.run(lifespan=...)` may not exist in installed FastMCP | Task 3.1 main() — uses `mcp.run_async()` (single path; no broken fallback) controlled by Phase 0 probe |
| `mcp.auth = provider` set inside lifespan = too late, OAuth routes never registered | Task 3.0 introduces `make_mcp(auth=, lifespan=)` factory; remote_server bootstraps provider eagerly before factory call |
| `load_access_token` rebumped MCP-token expiry on Taiga refresh — dead code at v1 TTLs and would desync claude.ai's expires_in | Task 2.5 — refresh branch deleted; docstring documents MCP TTL ≤ Taiga TTL invariant |

**v3.2 follow-up corrections (this revision):**

| Bug | Fix location |
|---|---|
| Path B (Starlette `Mount("/mcp", asgi)`) would route OAuth + well-known to `/mcp/...` instead of root → silently breaks RFC 8414 §3.1 discovery | Task 3.1 — Path B deleted; if `run_async` missing, plan-update required, not a runtime workaround |
| E2E test mocked outbound via respx, but python-taiga uses `requests` (synchronous) which respx does not intercept — tests would hit the real network or fail with ConnectionError | Task 2.6 / 2.7 — added `responses` test dep; E2E uses `respx.mock()` for httpx (TaigaClient) AND `responses.RequestsMock()` for requests (python-taiga) |
| `make_mcp` docstring claimed unsupported kwargs are "dropped silently" but the code passes them straight through (TypeError on unsupported FastMCP version) | `mcp.py` — docstring rewritten to honestly state that the `pyproject.toml` `fastmcp = ">=2.14.0,<3.0.0"` pin is the contract; no try/except theatre in the factory |
| Phase 0 probe missed: OAuthProvider constructor params, `run_async` signature, MCP-SDK Pydantic field shapes | Phase 0 probe extended to inspect all three; required-field listings catch e.g. `AuthorizationCode.redirect_uri_provided_explicitly` if present |
| `taiga_tools.py` top-level `from langchain_taiga.mcp import mcp` would deterministically fail post-factory-refactor (circular import during module load) | Task 3.0 Step 4a promoted from caveat to required step: drop the top-level import |
| `_register_mcp_tools` rewrite was prescriptive about call shape (`mcp.tool()(structured_tool.func)`) and would lose `args_schema`/`description` if the existing function does anything fancier | Task 3.0 Step 4b reframed as "preserve existing shape, just parametrize the FastMCP target" with explicit instruction to keep StructuredTool wrapping intact |
| FastMCP version pin not bumped explicitly in PR 1, so PR 1's `from fastmcp.server.dependencies import get_access_token` could fail on an older floor | Task 1.0 (new) — verify pin, bump to `>=2.14.0,<3.0.0`, run Phase 0 probe, commit `phase0-probe.txt` for PR description |
