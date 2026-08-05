# AGENTS.md

## Project

Python package providing two surfaces:
1. **LangChain tools** for Taiga (create / search / update entities, wiki, custom attributes, members, whoami).
2. **Remote MCP server** (`langchain-taiga-mcp-remote`) — multi-tenant OAuth bridge that lets claude.ai / VSCode / Claude Desktop talk to Taiga via the MCP protocol.

Default branch: `main`. Tests run in conda env `langchain_taiga`.

## Test

CI-canonical (matches `make test` in `ci_publish.yml`):

```bash
make test
# expands to:
poetry run pytest --disable-socket --allow-unix-socket tests/unit_tests/
```

`--disable-socket` (with `--allow-unix-socket` for asyncio-postgres style local sockets) blocks accidental network calls; tests that monkey-patch HTTP clients break loudly without it. Always use this flag locally too.

For Shikenso's local conda env (alternative when Poetry isn't set up):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate langchain_taiga && python -m pytest --disable-socket --allow-unix-socket tests/unit_tests/
```

`taiga_tools.TAIGA_URL` is captured at import via `os.getenv("TAIGA_URL")`. Tests must `monkeypatch.setattr(taiga_tools, "TAIGA_URL", "https://...")` — `monkeypatch.setenv` runs after import and is a no-op for the captured attribute (likely `None` in CI without the env var, which then `AttributeError`s at `.rstrip("/")`).

## CI / Release

- **`.github/workflows/ci_publish.yml` auto-publishes to PyPI on push to `main`.** Never push directly — always PR. A direct push immediately ships a release.
- After merge, PyPI `/simple/` index lags `/json` by ~1 min (different CDN). `pip install <pkg>==<new-ver>` may fail right after publish; check `pip index versions langchain-taiga` before chained Jenkins builds in `taiga` repo.
- Trusted-publishing (OIDC) occasionally returns `504 upstream request timeout` — re-run the failed workflow run, no code change needed.
- `pypa/gh-action-pypi-publish` is configured `skip-existing: true` (`ci_publish.yml:66`) — README/docs-only commits to `main` are safe without a version bump; the publish step no-ops on duplicate version.

## Branch / PR conventions

- Branch names use `/` (`feat/whoami-and-list-members`) — slashes are fine here because this repo doesn't deploy to K8s. The "no slashes" rule in workspace `CLAUDE.md` applies to repos with K8s deploys (e.g. `taiga`).
- Copilot is auto-assigned reviewer on every PR. After replying to comments, resolve threads via GraphQL `resolveReviewThread` (NOT `minimizeComment`).

## MCP-SDK / FastMCP gotchas

When working on `langchain_taiga/auth/` or `remote_server.py`:

- **`mcp.server.auth.routes.MetadataHandler` hardcodes `Cache-Control: public, max-age=3600`** on discovery responses → connected MCP clients (claude.ai, VSCode) cache server-side discovery changes for up to 1h. Server patches don't take effect until cache expiry or client restart.
- **`mcp.server.auth.routes.build_metadata` hardcodes `token_endpoint_auth_methods_supported = ["client_secret_post"]`.** Public clients (VSCode) need `"none"` advertised or they reject DCR. We monkey-patch at startup — see `_patch_metadata_to_advertise_none_auth` in `remote_server.py`.
- **FastMCP `OAuthProvider` splits issuer/base_url:**
  - AS metadata `issuer` is built from `base_url` (server origin / root)
  - PRM `authorization_servers` is built from `issuer_url`
  - If those don't match, RFC-8414-strict clients (VSCode 1.107+) reject the discovery as invalid → "DCR not supported".
  - `TaigaOAuthProvider.__init__` pins `self.issuer_url = self.base_url` post-super-init to keep them aligned.
- **VSCode MCP submits THREE redirect URIs in DCR**: `http://127.0.0.1:<port>`, `https://vscode.dev/redirect`, `https://insiders.vscode.dev/redirect` (the last whenever the user has Insiders installed or Settings Sync turned on). The redirect-URI allowlist in `provider.py:DEFAULT_ALLOWED_REDIRECT_TARGETS` MUST include all three — any rejected URI fails the entire registration with `ValueError` → 500 → VSCode shows "DCR not supported".

## Taiga API & python-taiga gotchas

When working on `langchain_taiga/tools/taiga_tools.py`:

- **`userstory.points` requires `role.computable=True`.** Taiga's PATCH validator rejects non-computable role IDs in the `points` payload with the misleading server-side message `Invalid role id '<id>'` (the role exists, just isn't points-eligible). Toggle in Taiga admin → Members → Roles → "Compute story points for this role". Since 2.3.1, `set_userstory_points_tool` rejects upfront with a 400 + `non_computable_roles` so this no longer manifests as the tool's wrapping-exception 500. Burned a full diagnosis cycle on `wahed` project before this was caught.
- **`Resource.patch(fields)` does NOT auto-include `version` like `update()` does.** `update()` sends `to_dict()` (every `allowed_param` including `version`); `patch(["foo"])` sends ONLY `{"foo": ...}`. For any optimistic-locked field (userstory, task, issue, epic), pass `["<field>", "version"]` explicitly. Skipping `version` strips the optimistic-lock check and Taiga rejects the request mid-validation. Caught by Codex review on the 2.3.0 PR.
- **`UserStory.points` wire shape is `{stringified_role_id: point_id}`** — NOT role-names, NOT point-values. Translation tables come from `project.list_roles()` (id ↔ name) and `project.list_points()` (id ↔ value, where `value=None` is the "?" unestimated point). `_format_userstory_points(entity, project)` produces the human-readable `{role_name: value}` shape used by `get_entity_by_ref_tool`'s response and as input to `set_userstory_points_tool`.
- **`langchain-core`'s `_parse_google_docstring` rejects `:`-bearing lines in `Args:`** as new arg names with `ValueError: Arg ... in docstring not found in function signature`. Keep literal dict examples (e.g. `{"Developer": 5}`) in the `Examples:` section, NEVER in any `Args:` line. There's an inline guard comment above `set_userstory_points_tool` to prevent regression.
- **`tags` is read as `[name, color]` pairs but written as flat names.** Taiga hands back `[["jobs_manager", null], ["voice", "#845EF7"]]` and accepts `["jobs_manager", "voice"]`. The colour is NOT per-entity — it lives in the project-level `tags_colors` registry (`list_all_tags`) and is joined in on read, which is why writing names back never loses it. Always flatten through `_normalize_tag_names` before comparing or rewriting: `"voice" in entity.tags` against the pair shape is silently always false, and that is exactly how `search_entities_tool`'s tag filter sat dead until 2.14.0. `get_entity_by_ref_tool` returns the flat names so a read feeds straight into `manage_tags_by_ref_tool`.
- **`owner` is the creator and is read-only through this package.** Taiga tracks authorship (`owner`) separately from responsibility (`assigned_to`), and both ship in *every* list and detail payload as `owner` (a user id) plus `owner_extra_info` (username + `full_name_display`). python-taiga `setattr`s every key the API returns, so both are already on the object — `allowed_params` only governs what a write sends back, which is why `owner` is absent there and `update_entity_by_ref_tool` cannot reassign authorship (pinned by a canary test in `test_entity_owner.py`). Resolve via `_owner_summary`, which prefers the embedded blob so it stays free inside the per-match search loop; only fall back to `get_user` when the blob is missing. Before 2.15.0 nothing surfaced it and the creator had to be recovered by walking history — one call per entity, and wrong by default because history comes back newest-first. The REST list endpoints also accept `owner=<id>`, so a single resolved owner is pushed server-side for user stories — but **never for tasks**, which are reached by walking user stories, where the param would filter the *stories* by creator and silently drop tasks living under someone else's story.
- **`find_users` only searches `project.members`, and lies about its return type.** It is annotated `-> List[Dict]` but `return`s a plain **string** on both LLM failure paths (`"Error decoding LLM response: …"`, `"LLM returned JSON that is not a list."`), so the idiomatic `[u["id"] for u in find_users(...)]` iterates that string's characters and dies on `u["id"]` with a `TypeError` outside the caller's error handling — guard with `isinstance(..., list)`. Separately, resolving only against *current* members means anyone who has left the project resolves to nobody, while their entities keep carrying the original `owner`/`owner_extra_info` — so an id-only filter answers "they filed nothing", which is indistinguishable from the truth. `_owner_matches` falls back to matching the requested name against the embedded owner blob (and accepts a bare numeric id, which needs no lookup at all and keeps its server-side pushdown). The pre-existing `assigned_to` filter has neither guard: it tests truthiness, so an unresolvable assignee silently disables the filter and returns the whole project.
- **An unknown tag is created implicitly by Taiga on write** — there is no "tag not found" error, so a typo becomes a permanent project tag. `manage_tags_by_ref_tool` reports these back as `created_tags` rather than blocking, mirroring the Taiga UI. **Read the registry BEFORE the write** — Taiga registers the new tag as part of that same save, so a lookup afterwards always finds it already there and reports nothing (and `list_all_tags` is TTL-cached 10 min, so this fails exactly on a cold cache, i.e. the first tag edit after a pod restart). The same pre-read supplies the canonical spelling for a tag the project knows but the entity doesn't carry yet. The lookup stays informational: a failure yields `created_tags: null` and must never fail the edit. After a write that creates a project tag, **evict the cache entry** (`_invalidate_tag_cache`) — Taiga registers the tag behind `list_all_tags`' back, so the stale entry would both re-report it as new on the next edit and lose its canonical spelling, writing a differently-cased duplicate. `create_entity_tool` is the other write path that registers tags and evicts too.

## Architecture pointers

| Concern | Location |
|---|---|
| Tool definitions (`@tool`) | `langchain_taiga/tools/taiga_tools.py` |
| MCP factory + tool registration | `langchain_taiga/mcp.py` (`make_mcp`, `_register_mcp_tools`) |
| Remote OAuth bridge entry point | `langchain_taiga/remote_server.py` |
| OAuth provider (Taiga creds → JWT) | `langchain_taiga/auth/provider.py` (`TaigaOAuthProvider`) |
| In-memory token store (dev/tests) | `langchain_taiga/auth/store.py` (`InMemoryStore`) |
| Durable token store (prod) | `langchain_taiga/auth/postgres_store.py` (`PostgresStore`) |
| Backend selection | `remote_server.py` (`_build_store`) |
| Login HTML | `langchain_taiga/auth/login_page.py` + `templates/` |

OAuth state backend is chosen by **`TAIGA_MCP_STATE_BACKEND`** (`remote_server._build_store`):

- `postgres` → `PostgresStore`. **This is what production runs.** DCR client registrations, access tokens and refresh tokens survive a pod restart, so a deploy or a liveness-probe kill no longer forces every connected client through a fresh browser login. Before 2.13.0 they did, which broke unattended agents.
- `memory` (the default) → `InMemoryStore`, for local dev and the test suite. Startup logs a warning.

The switch is **explicit, never inferred from the connection variables**. Inferring it would mean the likeliest operator error — a dropped env var, a chart refactor moving the `env:` block — silently downgrades prod to the in-memory store: green pod, passing probes, and everyone logged out on every restart. `TAIGA_MCP_STATE_BACKEND=postgres` with no connection config is a startup error instead. Don't "helpfully" add a fallback.

Connection config comes in two forms: the discrete `TAIGA_MCP_PG_*` vars (what the Helm chart passes, so what production uses) or a full `TAIGA_MCP_DATABASE_URL` DSN (CI and local tests). Discrete exists because a DSN must percent-encode `@ : / ? #` in the password, and the password comes from a Secret whose charset the chart doesn't control.

`tests/unit_tests/test_store_contract.py` runs one suite against **both** implementations so they cannot drift. The Postgres half needs a real database (`TAIGA_MCP_TEST_DATABASE_URL`, provided in CI by a `postgres:16` service container) and skips without one — the properties under test are SQL-level atomicity guarantees that a mock cannot demonstrate. Locally:

```bash
docker run -d --name pg -e POSTGRES_PASSWORD=pw -p 55432:5432 postgres:16
export TAIGA_MCP_TEST_DATABASE_URL=postgres://postgres:pw@127.0.0.1:55432/postgres
```

`InMemoryStore` gets its atomicity from asyncio being single-threaded (its critical sections contain no `await` between read and mutation). `PostgresStore` re-expresses each as a transaction — `SELECT ... FOR UPDATE` for refresh-token rotation, `DELETE ... RETURNING` for the single-use auth-code pop, and a per-family `pg_advisory_xact_lock` shared by `revoke_token_family` / `issue_new_generation`. **Preserve those semantics in any future edit**: losing them silently breaks refresh-token reuse-detection rather than failing a test.

`_SCHEMA` in `postgres_store.py` is **append-only** — `CREATE TABLE IF NOT EXISTS` no-ops against an existing table, so editing a `CREATE` does nothing to an already-booted environment. Add columns via an appended `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

`PostgresStore` persists Taiga JWTs and DCR client secrets **at rest** in plaintext columns — an accepted risk, decided deliberately: the same database already holds Django session keys and password hashes for the same users, so encrypting only these four tables would move no real trust boundary while adding a key to manage and rotate. The mitigation that does the work is scoping — the bridge connects as a dedicated role owning nothing but `mcp_oauth`. Don't point it at the database owner.

Deployment: see `taiga` repo's helm chart `taiga-mcp`.

## Related

- `taiga` repo: deploys `langchain-taiga-mcp-remote` to OVH MKS via Helm (chart `taiga-mcp`). Bump default `TAIGA_MCP_VERSION` in its `Jenkinsfile` after each PyPI release.
- python-taiga: third-party SDK we wrap. Has fragile `created_date` regex that leaves microsecond timestamps as raw strings — see `_coerce_to_aware_datetime` in `taiga_tools.py`.
