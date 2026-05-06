# AGENTS.md

## Project

Python package providing two surfaces:
1. **LangChain tools** for Taiga (create / search / update entities, wiki, custom attributes, members, whoami).
2. **Remote MCP server** (`langchain-taiga-mcp-remote`) — multi-tenant OAuth bridge that lets claude.ai / VSCode / Claude Desktop talk to Taiga via the MCP protocol.

Default branch: `main`. Tests run in conda env `langchain_taiga`.

## Test

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate langchain_taiga && python -m pytest tests/ -q
```

## CI / Release

- **`.github/workflows/ci_publish.yml` auto-publishes to PyPI on push to `main`.** Never push directly — always PR. A direct push immediately ships a release.
- After merge, PyPI `/simple/` index lags `/json` by ~1 min (different CDN). `pip install <pkg>==<new-ver>` may fail right after publish; check `pip index versions langchain-taiga` before chained Jenkins builds in `taiga` repo.
- Trusted-publishing (OIDC) occasionally returns `504 upstream request timeout` — re-run the failed workflow run, no code change needed.

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

## Architecture pointers

| Concern | Location |
|---|---|
| Tool definitions (`@tool`) | `langchain_taiga/tools/taiga_tools.py` |
| MCP factory + tool registration | `langchain_taiga/mcp.py` (`make_mcp`, `_register_mcp_tools`) |
| Remote OAuth bridge entry point | `langchain_taiga/remote_server.py` |
| OAuth provider (Taiga creds → JWT) | `langchain_taiga/auth/provider.py` (`TaigaOAuthProvider`) |
| In-memory token store | `langchain_taiga/auth/store.py` (`InMemoryStore`) |
| Login HTML | `langchain_taiga/auth/login_page.py` + `templates/` |

The remote server stores OAuth state in-memory (single-replica deploy) — see `taiga` repo's helm chart `taiga-mcp` for the deployment.

## Related

- `taiga` repo: deploys `langchain-taiga-mcp-remote` to OVH MKS via Helm (chart `taiga-mcp`). Bump default `TAIGA_MCP_VERSION` in its `Jenkinsfile` after each PyPI release.
- python-taiga: third-party SDK we wrap. Has fragile `created_date` regex that leaves microsecond timestamps as raw strings — see `_coerce_to_aware_datetime` in `taiga_tools.py`.
