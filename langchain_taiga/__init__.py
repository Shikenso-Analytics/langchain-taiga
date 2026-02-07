from importlib import metadata

from langchain_taiga.tools.taiga_tools import (add_attachment_by_ref_tool,
                                               add_comment_by_ref_tool,
                                               create_entity_tool,
                                               create_wiki_page_tool,
                                               get_entity_by_ref_tool,
                                               get_wiki_page_tool,
                                               list_wiki_pages_tool,
                                               search_entities_tool,
                                               update_entity_by_ref_tool,
                                               update_wiki_page_tool)


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "add_attachment_by_ref_tool",
    "add_comment_by_ref_tool",
    "create_entity_tool",
    "create_wiki_page_tool",
    "get_entity_by_ref_tool",
    "get_wiki_page_tool",
    "list_wiki_pages_tool",
    "search_entities_tool",
    "update_entity_by_ref_tool",
    "update_wiki_page_tool",
    "__version__",
]
