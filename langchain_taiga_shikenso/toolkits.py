"""TaigaShikenso toolkits."""

from typing import List

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_taiga_shikenso.tools.taiga_tools import (create_entity_tool,
                                                        search_entities_tool,
                                                        get_entity_by_ref_tool,
                                                        update_entity_by_ref_tool,
                                                        add_comment_by_ref_tool,
                                                        add_attachment_by_ref_tool)


class TaigaShikensoToolkit(BaseToolkit):
    # https://github.com/langchain-ai/langchain/blob/c123cb2b304f52ab65db4714eeec46af69a861ec/libs/community/langchain_community/agent_toolkits/sql/toolkit.py#L19
    """TaigaShikenso toolkit.

    Setup:
        Install ``langchain-taiga-shikenso`` and set environment variable ``TAIGASHIKENSO_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-taiga-shikenso
            export TAIGA_URL="taiga url"
            export TAIGA_API_URL="taiga api url"
            export TAIGA_USERNAME="username"
            export TAIGA_PASSWORD="pw"

    # TODO: Populate with relevant params.
    Key init args:
        arg 1: type
            description
        arg 2: type
            description

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain-taiga-shikenso import TaigaShikensoToolkit

            toolkit = TaigaShikensoToolkit(
                # ...
            )

    Tools:
        .. code-block:: python

            toolkit.get_tools()

        .. code-block:: none

            # TODO: Example output.

    Use within an agent:
        .. code-block:: python

            from langgraph.prebuilt import create_react_agent

            agent_executor = create_react_agent(llm, tools)

            example_query = "..."

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

        .. code-block:: none

             # TODO: Example output.

    """  # noqa: E501

    def get_tools(self) -> List[BaseTool]:
        return [
            create_entity_tool,
            search_entities_tool,
            get_entity_by_ref_tool,
            update_entity_by_ref_tool,
            add_comment_by_ref_tool,
            add_attachment_by_ref_tool,
        ]
