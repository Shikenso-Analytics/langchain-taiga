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

### MCP server (Claude CLI & GitHub Copilot Chat)

The package now ships with a [Model Context Protocol](https://modelcontextprotocol.io/) server powered by
[`fastmcp`](https://pypi.org/project/fastmcp/). It exposes the same Taiga tools without changing their
behaviour.

1. Ensure the Taiga environment variables above are set for the process running the server.
2. Start the server:

   ```bash
   python -m langchain_taiga.mcp_server
   ```

3. Point your MCP client at the command:
   - **Claude CLI/Desktop**: add to `~/.config/claude/claude.json` under `"mcpServers"`:

     ```json
     {
       "mcpServers": {
         "taiga": {
           "command": "python",
           "args": ["-m", "langchain_taiga.mcp_server"]
         }
       }
     }
     ```

   - **GitHub Copilot Chat (CLI/IDE)**: add a similar entry to your Copilot MCP configuration, pointing to
     `python -m langchain_taiga.mcp_server` so the client can discover the Taiga tools.

The server exports the following tools for MCP clients: `create_entity`, `search_entities`, `get_entity_by_ref`,
`update_entity_by_ref`, `add_comment_by_ref`, and `add_attachment_by_ref`.

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