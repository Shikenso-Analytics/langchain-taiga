"""In-chat file uploads: fastmcp's ``FileUpload`` app (MCP Apps), scoped per Taiga user.

The drop zone (``file_manager``) posts the bytes to the app-only ``store_files``
tool, so a file never passes through the model's context; the model hands the
upload's name to ``create_attachment_upload_by_ref_tool(uploaded_file=...)``.
Needs the ``[apps]`` extra (prefab-ui); without it this module fails to import
and ``make_mcp`` serves the Taiga tools alone.

Changes to the stock provider: scope by Taiga user, because ``/mcp`` is
stateless and a session id never repeats; ``read_file`` previews text too,
because "text" is the client's claim; and ``store_files`` is dropped from
``tools/list`` only, because since fastmcp 4.1.6 ``disable()`` also blocks the
hashed call the drop zone makes.
"""

from __future__ import annotations

import base64
import threading
from time import monotonic
from typing import Any, Optional

from fastmcp.apps.file_upload import FileUpload
from fastmcp.server.middleware import Middleware

from langchain_taiga import upload_tickets
from langchain_taiga.tools.taiga_tools import _current_user_scope

# Memory budget for a 1 Gi pod, next to the server's own baseline and the curl
# path's ~200 MB (2 x 25 MB x 4, see upload_tickets). Stored uploads are kept
# base64, ~4/3 of these sizes.
# ponytail: fixed caps; raise them together with the pod's memory limit.
MAX_TOTAL_BYTES = 50 * 1024 * 1024  # all users together: two full-size files
TTL_SECONDS = 900.0

# One attach from the store at a time: requests' multipart encode holds the
# file twice in RSS, on top of the curl path's own UPLOAD_CONCURRENCY slots.
ATTACH_SLOTS = threading.BoundedSemaphore(1)

_PREVIEW_CHARS = 200
_PREVIEW_B64 = 1104  # 828 bytes: 200 characters even at 4 bytes each
_TEXT_EXTENSIONS = (".csv", ".json", ".txt", ".md", ".py", ".yaml", ".yml", ".toml")


class TaigaFileUpload(FileUpload):
    """``FileUpload`` scoped by Taiga user, with preview-only reads."""

    def __init__(self) -> None:
        super().__init__(
            max_file_size=upload_tickets.MAX_UPLOAD_BYTES,
            max_total_size=MAX_TOTAL_BYTES,
            file_ttl_seconds=TTL_SECONDS,
            description=(
                "Drop files here to attach them to a Taiga ticket. Uploads are "
                "kept for 15 minutes; ask the assistant to attach one by name."
            ),
        )

    def _get_scope_key(self, ctx: Any) -> str:
        return _current_user_scope()

    def get_upload(self, name: str) -> Optional[dict]:
        """The caller's stored upload ``name`` (base64 ``data``), or ``None``."""
        scope = self._get_scope_key(None)
        with self._store_lock:
            self._prune_expired(monotonic())
            return self._store.get(scope, {}).get(name)

    def upload_names(self) -> list:
        return [f["name"] for f in self.on_list(None)]

    def discard(self, name: str) -> None:
        scope = self._get_scope_key(None)
        with self._store_lock:
            self._store.get(scope, {}).pop(name, None)

    def on_read(self, name: str, ctx: Any) -> dict:
        entry = self.get_upload(name)
        if entry is None:
            raise ValueError(
                f"File {name!r} not found. Available: {self.upload_names()}"
            )
        data = entry["data"]
        result = {k: entry[k] for k in ("name", "size", "type", "uploaded_at")}
        if entry["type"].startswith("text/") or entry["name"].endswith(
            _TEXT_EXTENSIONS
        ):
            text = base64.b64decode(data[:_PREVIEW_B64]).decode("utf-8", "replace")
            cut = len(data) > _PREVIEW_B64 or len(text) > _PREVIEW_CHARS
            result["content"] = text[:_PREVIEW_CHARS] + ("..." if cut else "")
        else:
            cut = len(data) > _PREVIEW_CHARS
            result["content_base64"] = data[:_PREVIEW_CHARS] + ("..." if cut else "")
        return result


# One store for the process: the stdio singleton and the remote server are
# separate ``make_mcp`` instances, and the attach tool must see what either
# drop zone stored.
PROVIDER = TaigaFileUpload()


# ponytail: drop this once fastmcp leaves app-only tools out of tools/list itself.
class HideAppOnlyTools(Middleware):
    """Drop app-only tools (``visibility == ["app"]``) from ``tools/list``."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        return [
            t
            for t in tools
            if ((t.meta or {}).get("ui") or {}).get("visibility") != ["app"]
        ]


def mount(mcp_instance: Any) -> None:
    """Add the drop zone to ``mcp_instance``, with its receiver kept off the listing."""
    mcp_instance.add_provider(PROVIDER)
    mcp_instance.add_middleware(HideAppOnlyTools())
