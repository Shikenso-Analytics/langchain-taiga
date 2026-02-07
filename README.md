# langchain-taiga

[![PyPI version](https://badge.fury.io/py/langchain-taiga.svg)](https://pypi.org/project/langchain-taiga/)

This package provides [Taiga](https://docs.taiga.io/) tools and a toolkit for use with LangChain. It includes:

- **`create_entity_tool`**: Creates user stories, tasks and issues in Taiga.
- **`search_entities_tool`**: Searches for user stories, tasks and issues in Taiga.
- **`get_entity_by_ref_tool`**: Gets a user story, task or issue by reference.
- **`update_entity_by_ref_tool`**: Updates a user story, task or issue by reference.
- **`add_comment_by_ref_tool`**: Adds a comment to a user story, task or issue.
- **`add_attachment_by_ref_tool`**: Adds an attachment to a user story, task or issue.

---

## Installation

```bash
pip install -U langchain-taiga
```

---

## Environment Variable

Export your taiga logins:

```bash
export TAIGA_URL="https://taiga.xyz.org/"
export TAIGA_API_URL="https://taiga.xyz.org/"
export TAIGA_USERNAME="username"
export TAIGA_PASSWORD="pw"
export OPENAI_API_KEY="OPENAI_API_KEY"
```

If this environment variable is not set, the tools will raise a `ValueError` when instantiated.

---

## Usage

### Direct Tool Usage

```python
from langchain_taiga.tools.taiga_tools import create_entity_tool, search_entities_tool, get_entity_by_ref_tool, update_entity_by_ref_tool, add_comment_by_ref_tool, add_attachment_by_ref_tool

response = create_entity_tool({"project_slug": "slug",
                       "entity_type": "us",
                       "subject": "subject",
                       "status": "new",
                       "description": "desc",
                       "parent_ref": 5,
                       "assign_to": "user",
                       "due_date": "2022-01-01",
                       "tags": ["tag1", "tag2"]})

response = search_entities_tool({"project_slug": "slug", "query": "query", "entity_type": "task"})

response = get_entity_by_ref_tool({"entity_type": "user_story", "project_id": 1, "ref": "1"})

response = update_entity_by_ref_tool({"project_slug": "slug", "entity_ref": 555, "entity_type": "us"})

response = add_comment_by_ref_tool({"project_slug": "slug", "entity_ref": 3, "entity_type": "us",
                "comment": "new"})

response = add_attachment_by_ref_tool({"project_slug": "slug", "entity_ref": 3, "entity_type": "us",
                "attachment_url": "url", "content_type": "png", "description": "desc"})
```

### Using the Toolkit

You can also use `TaigaToolkit` to automatically gather both tools:

```python
from langchain_taiga.toolkits import TaigaToolkit

toolkit = TaigaToolkit()
tools = toolkit.get_tools()
```

### MCP Server

The package ships with a [Model Context Protocol](https://modelcontextprotocol.io/) server powered by
[`fastmcp`](https://pypi.org/project/fastmcp/). It exposes the same Taiga tools without changing their
behaviour.

#### Running the Server

```bash
python -m langchain_taiga.mcp_server
```

Or without installing into your project (using [uv](https://docs.astral.sh/uv/)):

```bash
uv run --with langchain-taiga python -m langchain_taiga.mcp_server
```

The server exports the following tools for MCP clients: `create_entity_tool`, `search_entities_tool`, `get_entity_by_ref_tool`,
`update_entity_by_ref_tool`, `add_comment_by_ref_tool`, `add_attachment_by_ref_tool`, `list_wiki_pages_tool`, `get_wiki_page_tool`,
`create_wiki_page_tool`, and `update_wiki_page_tool`.

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

## Tests

If you have a tests folder (e.g. `tests/unit_tests/`), you can run them (assuming Pytest) with:

```bash
pytest --maxfail=1 --disable-warnings -q
```

---

## License

[MIT License](./LICENSE)

---

## Further Documentation

- For more details, see the docstrings in:
  - [`taiga_tools.py`](./langchain_taiga/tools/taiga_tools.py)
  - [`toolkits.py`](./langchain_taiga/toolkits.py) for `TaigaToolkit`

- Official Taiga Developer Docs: <https://docs.taiga.io/api.html>
- [LangChain GitHub](https://github.com/hwchase17/langchain) for general LangChain usage and tooling.