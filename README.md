# langchain-taiga

[![PyPI version](https://badge.fury.io/py/langchain-taiga.svg)](https://pypi.org/project/langchain-taiga/)

[Taiga](https://docs.taiga.io/) integration for [LangChain](https://github.com/langchain-ai/langchain) and the [Model Context Protocol](https://modelcontextprotocol.io/).

The package ships three things in one install:

1. **20 LangChain tools** for Taiga (entities, wiki, custom attributes, members, sprint planning).
2. **A `TaigaToolkit`** that bundles them for one-line LangChain agent setup.
3. **An MCP server in two flavours:**
   - **Stdio mode** — single-user, local credentials in env vars. For Claude Desktop, Claude Code, VSCode local.
   - **Remote mode** — multi-tenant HTTP server with OAuth 2.1 + PKCE + Dynamic Client Registration. For [claude.ai Custom Connectors](https://support.anthropic.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp), [VSCode Web](https://vscode.dev/), Claude Desktop with HTTP transport, etc. Each user signs in with their own Taiga credentials; the server stores no static API key.

The 20 tools:

- **`create_entity_tool`**: Creates user stories, tasks and issues in Taiga.
- **`search_entities_tool`**: Searches for user stories, tasks and issues in Taiga. Returns `{matches, count, max_results, truncated}`. Supports `max_results` and `include_custom_attributes` (default `False` — opt-in to avoid an N+1 fetch storm). Date filters are tz-aware.
- **`get_entity_by_ref_tool`**: Gets a user story, task or issue by reference. For user stories, the response also includes a `points` field (`{role_name: value}` shape, symmetric to `set_userstory_points_tool`'s input) so a read + write round-trip stays in the same vocabulary.
- **`update_entity_by_ref_tool`**: Updates a user story, task or issue by reference.
- **`add_comment_by_ref_tool`**: Adds a comment to a user story, task or issue.
- **`add_attachment_by_ref_tool`**: Adds an attachment to a user story, task or issue by downloading a **public URL** (the server fetches it via `requests.get`).
- **`list_attachments_by_ref_tool`**: List all attachments on an entity with fresh signed download URLs. URL tokens from the Taiga UI/webhook diff expire after ~6 min; this tool re-mints them on every call.
- **`get_attachment_by_ref_tool`**: Fetch a specific attachment by ID and return its content base64-encoded inline. Refuses files larger than `TAIGA_MAX_INLINE_ATTACHMENT_BYTES` (default 10 MB) — for larger files use `list_attachments_by_ref_tool` and `curl` the `download_url` out-of-band.
- **`promote_issue_to_userstory_tool`**: Promotes an issue to a user story, preserving comments and history.
- **`list_custom_attributes_tool`**: Lists custom-attribute definitions for a project + entity type, including dropdown `choices` (parsed) and the raw `extra` field — so an agent can discover the valid options for a dropdown attribute before writing via `set_custom_attributes_tool`.
- **`set_custom_attributes_tool`**: Sets custom-attribute values on a user story, task or issue.
- **`get_custom_attributes_tool`**: Reads custom-attribute values from a user story, task or issue.
- **`sort_kanban_by_rice_tool`**: Re-orders Kanban swimlanes using a RICE-style score. Closed status columns (Done, Cancelled, …) are skipped — re-ranking already-completed work has no value. Each entry in `columns_updated` carries `status_name` alongside `status_id` so consumers don't need to round-trip through Taiga to translate the id.
- **`set_userstory_points_tool`**: Sets Taiga story points on a user story for one or more roles (Developer, UX, Design, …). Resolves role names + point values to internal IDs at runtime so it adapts to per-project scales. The only programmatic path to set the field that `sort_kanban_by_rice_tool` reads as effort. The target role must be configured as **computable** in the project (Taiga admin → Members → Roles → "Compute story points for this role"); non-computable roles are rejected upfront with a 400 + `non_computable_roles` diagnostic, instead of letting Taiga's PATCH endpoint return its cryptic `Invalid role id` server-side rejection (which would otherwise surface as a generic 500 via this tool's exception wrapper).
- **`list_wiki_pages_tool`**: Lists wiki pages in a project.
- **`get_wiki_page_tool`**: Reads a wiki page by slug.
- **`create_wiki_page_tool`**: Creates a wiki page.
- **`update_wiki_page_tool`**: Updates a wiki page.
- **`whoami_tool`**: Returns the currently authenticated Taiga user.
- **`list_project_members_tool`**: Lists all members of a project with their roles (email opt-in).

---

## Installation

```bash
pip install -U langchain-taiga
```

---

## Environment Variables

For direct LangChain tool use, the toolkit, and **stdio MCP mode** (single-user, local credentials):

```bash
export TAIGA_URL="https://taiga.xyz.org/"
export TAIGA_API_URL="https://taiga.xyz.org/"
export TAIGA_USERNAME="username"
export TAIGA_PASSWORD="pw"
export OPENAI_API_KEY="..."   # used by some tools' LLM-powered helpers
```

If `TAIGA_USERNAME` / `TAIGA_PASSWORD` are not set, the tools raise `ValueError` on call.

**Remote MCP mode is different** — see the [Remote Mode](#remote-mode-multi-tenant-oauth) section. There, end-users supply their own Taiga credentials interactively at the `<mcp-path>/oauth/login` form (e.g. `https://your-server/mcp/oauth/login`); only `TAIGA_API_URL`, `TAIGA_URL`, `TAIGA_MCP_BASE_URL`, and `OPENAI_API_KEY` are server-side env.

---

## Usage

### Direct Tool Usage

Each tool is a `@tool`-decorated function callable via `.invoke({...})`. A few representative examples:

```python
from langchain_taiga.tools.taiga_tools import (
    create_entity_tool,
    search_entities_tool,
    whoami_tool,
)

# Create — write
create_entity_tool.invoke({
    "project_slug": "shikenso-development",
    "entity_type": "us",
    "subject": "Add /metrics endpoint",
    "status": "New",
    "description": "Prometheus-scrape-friendly text format.",
    "tags": ["backend", "observability"],
})

# Search — natural-language query, server-side caps + truncation flag in response
search_entities_tool.invoke({
    "project_slug": "shikenso-development",
    "query": "open issues created after 2026-03-01",
    "entity_type": "issue",
    "max_results": 50,
})

# Whoami — verify auth wiring end-to-end
whoami_tool.invoke({})
```

For the full set of 20 tools see the list at the top of this README, the docstrings in [`taiga_tools.py`](./langchain_taiga/tools/taiga_tools.py), or just grab them all via the toolkit below.

### Using the Toolkit

You can also use `TaigaToolkit` to automatically gather both tools:

```python
from langchain_taiga.toolkits import TaigaToolkit

toolkit = TaigaToolkit()
tools = toolkit.get_tools()
```

## MCP Server

The package ships an MCP server powered by [`fastmcp`](https://pypi.org/project/fastmcp/). All 20 tools above are exposed as MCP tools without changing their behaviour. There are **two transport modes** with different auth models:

| Mode | Transport | Auth | Use case |
|---|---|---|---|
| **Stdio** | stdin/stdout | env-var credentials | One developer running locally — Claude Desktop, Claude Code, VSCode local |
| **Remote** | HTTP (streamable) | OAuth 2.1 + PKCE + DCR | Multi-tenant — claude.ai Custom Connectors, VSCode Web, hosted teams |

### Stdio Mode (single-user, local)

Run the server:

```bash
python -m langchain_taiga.mcp_server
```

Or without installing into your project (using [uv](https://docs.astral.sh/uv/)):

```bash
uv run --with langchain-taiga python -m langchain_taiga.mcp_server
```

The Taiga credentials come from the `TAIGA_USERNAME` / `TAIGA_PASSWORD` env vars set in the wrapping process. All requests use the same single user.

#### VSCode

Add the following to your `.vscode/mcp.json` (or via the VSCode MCP settings UI):

```json
{
  "servers": {
    "taiga": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "langchain-taiga",
        "python",
        "-m",
        "langchain_taiga.mcp_server"
      ],
      "env": {
        "TAIGA_API_URL": "${input:taiga_api_url}",
        "TAIGA_URL": "${input:taiga_url}",
        "TAIGA_USERNAME": "${input:taiga_username}",
        "TAIGA_PASSWORD": "${input:taiga_password}"
      }
    }
  },
  "inputs": [
    {
      "id": "taiga_api_url",
      "type": "promptString",
      "description": "Taiga API URL (e.g. https://api.taiga.io)",
      "password": false
    },
    {
      "id": "taiga_url",
      "type": "promptString",
      "description": "Taiga Web URL (e.g. https://tree.taiga.io)",
      "password": false
    },
    {
      "id": "taiga_username",
      "type": "promptString",
      "description": "Taiga Username",
      "password": false
    },
    {
      "id": "taiga_password",
      "type": "promptString",
      "description": "Taiga Password",
      "password": true
    }
  ]
}
```

#### Claude Code

Add the Taiga MCP server via the CLI:

```bash
claude mcp add taiga -- uv run --with langchain-taiga python -m langchain_taiga.mcp_server
```

This adds the server to your project's `.claude/mcp.json`. Make sure the Taiga environment variables are set in your shell, or pass them explicitly:

```bash
claude mcp add taiga -e TAIGA_API_URL=https://api.taiga.io -e TAIGA_URL=https://tree.taiga.io -e TAIGA_USERNAME=your_user -e TAIGA_PASSWORD=your_pass -- uv run --with langchain-taiga python -m langchain_taiga.mcp_server
```

Alternatively, add the entry manually to `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "taiga": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "langchain-taiga",
        "python",
        "-m",
        "langchain_taiga.mcp_server"
      ],
      "env": {
        "TAIGA_API_URL": "https://api.taiga.io",
        "TAIGA_URL": "https://tree.taiga.io",
        "TAIGA_USERNAME": "your_user",
        "TAIGA_PASSWORD": "your_pass"
      }
    }
  }
}
```

#### Claude Desktop / GitHub Copilot Chat

Add a similar entry to your MCP configuration, pointing to
`uv run --with langchain-taiga python -m langchain_taiga.mcp_server`.

---

### Remote Mode (multi-tenant, OAuth)

Run the server as a long-lived HTTP service:

```bash
TAIGA_API_URL=https://taiga.example.org \
TAIGA_URL=https://taiga.example.org \
TAIGA_MCP_BASE_URL=https://mcp.example.org/mcp \
OPENAI_API_KEY=sk-... \
langchain-taiga-mcp-remote
```

The server speaks MCP over the streamable-HTTP transport at the URL given by `TAIGA_MCP_BASE_URL` and implements:

- **OAuth 2.1 + PKCE (S256)** — `/authorize`, `/token`
- **OAuth 2.1 refresh-token rotation with reuse-detection** (2.5.0+) — connected MCP clients (claude.ai, VSCode, Claude Desktop) stay authenticated across hours without forcing the user back through the OAuth flow. Each refresh rotates the token; a replayed refresh token revokes the entire token family per OAuth 2.1 best practice for public clients.
- **RFC 7591 Dynamic Client Registration** — `/register`
- **RFC 8414 / RFC 9728 discovery docs** — `/.well-known/oauth-authorization-server[/<mcp-path>]` and `/.well-known/oauth-protected-resource[/<mcp-path>]`
- **A login form** at `<mcp-path>/oauth/login` where each user signs in with their own Taiga credentials. The Taiga JWT is then carried per-request via the MCP `AccessToken` claims and used by every tool call — so two users connected to the same server see only their own projects.

OAuth state lives in-process (per-pod in-memory dict). For production, run a single replica; users re-authorize after a Helm deploy or pod restart (refresh tokens are wiped along with everything else).

#### Connecting clients to a remote server

| Client | How to connect |
|---|---|
| **claude.ai** | Settings → Connectors → Add custom connector → URL `https://your-server/mcp` → sign in with Taiga creds. [Anthropic docs](https://support.anthropic.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp). |
| **VSCode (stable + Insiders + Web)** | MCP UI → "Add server" → HTTP → `https://your-server/mcp`. DCR runs automatically; the redirect URIs (`vscode.dev/redirect`, `insiders.vscode.dev/redirect`, `127.0.0.1:<port>`) are pre-allowed by the server's allowlist. |
| **Claude Desktop** | MCP config: `"transport": {"type": "http", "url": "https://your-server/mcp"}` |
| **MCP Inspector** | `npx @modelcontextprotocol/inspector` → URL `https://your-server/mcp` → "Quick OAuth Flow" |

#### Production deployment

For Shikenso's deployment to OVH MKS, see the `taiga` repo's `deployment/helm/taiga-mcp` chart and `Jenkinsfile`. Bump the chart's `TAIGA_MCP_VERSION` parameter after each langchain-taiga release.

---

## Tests

The CI-aligned command (matches `make test` in CI):

```bash
make test
# expands to:
poetry run pytest --disable-socket --allow-unix-socket tests/unit_tests/
```

The `--disable-socket` flag blocks real network calls — the unit tests rely on it being on. Running `pytest` without it can hide tests that accidentally talk to live services.

For Shikenso's conda env (alternative local setup): `source ~/miniconda3/etc/profile.d/conda.sh && conda activate langchain_taiga && python -m pytest --disable-socket --allow-unix-socket tests/unit_tests/`.

---

## License

[MIT License](./LICENSE)

---

## Further Documentation

- **`AGENTS.md`** in this repo — onboarding for AI coding assistants and humans (test command, CI flow, MCP-SDK gotchas, architecture pointers).
- Tool docstrings in [`langchain_taiga/tools/taiga_tools.py`](./langchain_taiga/tools/taiga_tools.py).
- [`TaigaToolkit`](./langchain_taiga/toolkits.py).
- Remote OAuth bridge entry point: [`langchain_taiga/remote_server.py`](./langchain_taiga/remote_server.py).
- Official Taiga Developer Docs: <https://docs.taiga.io/api.html>.
- [LangChain GitHub](https://github.com/langchain-ai/langchain) for general LangChain usage.
- [Model Context Protocol](https://modelcontextprotocol.io/) spec.