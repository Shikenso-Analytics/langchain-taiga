from importlib import metadata

from langchain_taiga_shikenso.chat_models import ChatTaigaShikenso
from langchain_taiga_shikenso.document_loaders import TaigaShikensoLoader
from langchain_taiga_shikenso.embeddings import TaigaShikensoEmbeddings
from langchain_taiga_shikenso.retrievers import TaigaShikensoRetriever
from langchain_taiga_shikenso.toolkits import TaigaShikensoToolkit
from langchain_taiga_shikenso.tools import TaigaShikensoTool
from langchain_taiga_shikenso.vectorstores import TaigaShikensoVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatTaigaShikenso",
    "TaigaShikensoVectorStore",
    "TaigaShikensoEmbeddings",
    "TaigaShikensoLoader",
    "TaigaShikensoRetriever",
    "TaigaShikensoToolkit",
    "TaigaShikensoTool",
    "__version__",
]
