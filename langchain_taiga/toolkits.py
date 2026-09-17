"""Taiga toolkit."""

from typing import List

from langchain_core.tools import BaseTool, BaseToolkit

from langchain_taiga.tools.taiga_tools import (
    add_attachment_by_ref_tool,
    add_comment_by_ref_tool,
    create_attachment_upload_by_ref_tool,
    create_entity_tool,
    create_wiki_page_tool,
    get_attachment_by_ref_tool,
    get_custom_attributes_tool,
    get_entity_by_ref_tool,
    get_kanban_board_tool,
    get_wiki_page_tool,
    list_attachments_by_ref_tool,
    list_custom_attributes_tool,
    list_project_members_tool,
    list_wiki_pages_tool,
    manage_tags_by_ref_tool,
    manage_watchers_by_ref_tool,
    promote_issue_to_userstory_tool,
    search_entities_tool,
    set_custom_attributes_tool,
    set_userstory_points_tool,
    sort_kanban_by_rice_tool,
    update_entity_by_ref_tool,
    update_entities_by_ref_tool,
    update_wiki_page_tool,
    whoami_tool,
)


class TaigaToolkit(BaseToolkit):
    # https://github.com/langchain-ai/langchain/blob/c123cb2b304f52ab65db4714eeec46af69a861ec/libs/community/langchain_community/agent_toolkits/sql/toolkit.py#L19
    """Taiga toolkit: the package's agent-facing Taiga tools in one list.

    It holds the same tools the MCP server exposes. A few library helpers, such as
    ``add_attachment_inline_by_ref_tool``, stay importable from ``langchain_taiga`` but are
    not part of it.

    Setup:
        Install ``langchain-taiga`` and set the Taiga environment variables.

        .. code-block:: bash

            pip install -U langchain-taiga
            export TAIGA_URL="https://taiga.example.com/"
            export TAIGA_API_URL="https://taiga.example.com/"
            export TAIGA_USERNAME="username"
            export TAIGA_PASSWORD="pw"
            export OPENAI_API_KEY="..."  # used by some tools' LLM-powered helpers

    Instantiate:
        .. code-block:: python

            from langchain_taiga.toolkits import TaigaToolkit

            toolkit = TaigaToolkit()

    Tools:
        .. code-block:: python

            tools = toolkit.get_tools()

    Use within an agent:
        .. code-block:: python

            from langgraph.prebuilt import create_react_agent

            agent_executor = create_react_agent(llm, tools)

            example_query = "List the open issues in my-project"

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

    """  # noqa: E501

    def get_tools(self) -> List[BaseTool]:
        return [
            create_entity_tool,
            search_entities_tool,
            get_kanban_board_tool,
            get_entity_by_ref_tool,
            update_entity_by_ref_tool,
            update_entities_by_ref_tool,
            manage_watchers_by_ref_tool,
            manage_tags_by_ref_tool,
            add_comment_by_ref_tool,
            add_attachment_by_ref_tool,
            create_attachment_upload_by_ref_tool,
            list_attachments_by_ref_tool,
            get_attachment_by_ref_tool,
            promote_issue_to_userstory_tool,
            list_custom_attributes_tool,
            set_custom_attributes_tool,
            get_custom_attributes_tool,
            sort_kanban_by_rice_tool,
            set_userstory_points_tool,
            list_wiki_pages_tool,
            get_wiki_page_tool,
            create_wiki_page_tool,
            update_wiki_page_tool,
            whoami_tool,
            list_project_members_tool,
        ]
