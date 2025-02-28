# langchain-taiga

[![PyPI version](https://badge.fury.io/py/langchain-discord-shikenso.svg)](https://pypi.org/project/langchain-discord-shikenso/)

This package provides Discord tools and a toolkit for use with LangChain. It includes:

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
  - [`toolkits.py`](./langchain_discord/toolkits.py) for `TaigaToolkit`

- Official Taiga Developer Docs: <https://docs.taiga.io/api.html>
- [LangChain GitHub](https://github.com/hwchase17/langchain) for general LangChain usage and tooling.