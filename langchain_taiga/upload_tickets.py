"""One-shot upload tickets for out-of-band attachment uploads.

Why this exists
---------------
MCP tool arguments are emitted by the model, so any file inlined as
base64 into a tool call is paid for twice — once in tokens, once in
latency. A 500 KB screenshot costs roughly 200k tokens that way. The
published MCP spec has no client-to-server upload primitive (SEP-2631
proposes ``files/authorizeUpload`` but is still a draft), and the
maintainers' standing recommendation is the handle/upload pattern: keep
the bytes out of JSON-RPC, let the tool argument carry an opaque handle.

``create_attachment_upload_by_ref_tool`` issues a ticket here and returns
its URL; the client POSTs the bytes straight at
``POST /mcp/upload/{token}`` (see ``remote_server._attach_custom_routes``)
and the server attaches them. Token cost is constant at roughly 300
tokens no matter how large the file is.

Why an in-process dict and not Postgres
---------------------------------------
The Helm chart pins ``replicas: 1`` at template level, and the in-flight
OAuth authorize state next door is already a per-pod dict for exactly
that reason. A ticket lives for seconds: a pod restart in that window
invalidates a pending ticket and the caller repeats one cheap tool call.
Persisting it would buy schema DDL, two ``StateStore`` implementations, a
contract-test extension and a second copy of the Taiga JWT in the
database — to protect a trivially retryable flow.

Threading
---------
Tickets are issued from a tool body, which ``_register_mcp_tools`` runs
in a worker thread via ``asyncio.to_thread``, and consumed from the
Starlette route handler on the event loop. Two different threads touch
the dict, so it is guarded by a ``threading.Lock`` rather than relying on
asyncio's single-threaded execution the way ``InMemoryStore`` does.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

# How long an issued ticket stays valid. The flow is "tool call, then
# curl", so this only has to cover an agent thinking in between.
UPLOAD_TICKET_TTL_SECONDS = float(os.getenv("TAIGA_MCP_UPLOAD_TICKET_TTL", "600"))

# Hard ceiling on an uploaded body.
#
# Deliberately well under the ingress' own ``proxy-body-size: 100m``. The
# INBOUND leg streams to disk and costs nothing, but the OUTBOUND leg does
# not: ``entity.attach`` hands the path to python-taiga, which calls
# ``requests.post(files=...)``, and ``requests.models._encode_files`` does
# ``fdata = fp.read()`` before ``encode_multipart_formdata`` copies that into
# a fresh ``BytesIO``. So one upload peaks at roughly twice the file size in
# resident memory inside the worker thread, against a 1Gi pod limit that also
# carries the uvicorn/FastMCP/langchain baseline. 25 MB keeps a single
# transfer near 50 MB and leaves room for several at once; nothing here bounds
# concurrency, so the cap is the only thing standing between a few
# simultaneous large uploads and the limit.
#
# This bounds ONE upload. UPLOAD_CONCURRENCY below bounds how many run at
# once, and the pod's real exposure is the PRODUCT of the two — roughly
# ``2 * MAX_UPLOAD_BYTES * UPLOAD_CONCURRENCY``. Raising either alone raises
# that product: to allow bigger files at the same memory ceiling, lower
# UPLOAD_CONCURRENCY to compensate, and only raise both together when the
# pod's memory limit goes up with them.
MAX_UPLOAD_BYTES = int(os.getenv("TAIGA_MCP_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# How many uploads may be in their memory-heavy phase at once.
#
# The per-request cap above bounds ONE upload; without this it would bound
# nothing in aggregate. ``asyncio.to_thread`` uses the default executor,
# which runs up to ``min(32, cpu_count + 4)`` jobs concurrently, and a ticket
# costs one cheap tool call — so an authenticated user doing the obvious
# thing (mint several tickets, upload them in parallel) could put ~20
# multipart encodes in flight and OOM the pod for every tenant. That is a
# denial of service reachable through entirely legitimate use, not an abuse
# case.
#
# 4 x ~50 MB peak leaves the 1Gi pod comfortable headroom over its
# uvicorn/FastMCP/langchain baseline. Keep that product in mind before
# changing either number — see MAX_UPLOAD_BYTES above. Excess uploads queue
# rather than fail:
# they are already streamed to disk by then and hold no significant memory
# while they wait, and the ingress' 300s proxy timeout bounds the wait.
UPLOAD_CONCURRENCY = int(os.getenv("TAIGA_MCP_UPLOAD_CONCURRENCY", "4"))


@dataclass(frozen=True)
class UploadTicket:
    """A single-use authorization to attach one file to one entity.

    Everything except the bytes is fixed at issue time. The upload request
    carries no parameters at all, so a guessed token cannot retarget the
    upload at a different entity, rename the file, or borrow a different
    user's Taiga session.
    """

    token: str
    taiga_jwt: Optional[str]
    project_slug: str
    entity_type: str
    entity_ref: int
    filename: str
    description: str
    expires_at: float

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires_at


_tickets: Dict[str, UploadTicket] = {}
_lock = threading.Lock()


def issue(
    *,
    taiga_jwt: Optional[str],
    project_slug: str,
    entity_type: str,
    entity_ref: int,
    filename: str,
    description: str = "",
    ttl_seconds: Optional[float] = None,
) -> UploadTicket:
    """Mint a ticket and return it.

    ``ttl_seconds`` defaults to the module-level
    ``UPLOAD_TICKET_TTL_SECONDS``, read at call time so tests (and a
    future env reload) can monkeypatch it. No production caller passes it
    explicitly; it exists so a test can mint an already-expired ticket
    without either sleeping or juggling the module constant around a call
    that also needs a live one.
    """
    ttl = UPLOAD_TICKET_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    ticket = UploadTicket(
        # Same entropy as the OAuth authorization codes minted in
        # provider.py — this token IS the authorization for the upload.
        token=secrets.token_urlsafe(32),
        taiga_jwt=taiga_jwt,
        project_slug=project_slug,
        entity_type=entity_type,
        entity_ref=entity_ref,
        filename=filename,
        description=description,
        expires_at=time.time() + ttl,
    )
    with _lock:
        _tickets[ticket.token] = ticket
    return ticket


def consume(token: str) -> Optional[UploadTicket]:
    """Atomically pop a ticket. Returns ``None`` if unknown or expired.

    Single-use is enforced here rather than after a successful upload: the
    ticket is gone the moment a request presents it, so a replay — including
    a retry of an upload that failed downstream — finds nothing. Callers
    request a fresh ticket instead.
    """
    with _lock:
        ticket = _tickets.pop(token, None)
    if ticket is None or ticket.is_expired():
        return None
    return ticket


def prune(now: Optional[float] = None) -> int:
    """Drop expired tickets. Returns how many were removed.

    Called from ``run_cleanup_loop`` on the same schedule that prunes the
    provider's abandoned authorize states. Without it, tickets that are
    issued but never used would accumulate until pod restart.
    """
    now = time.time() if now is None else now
    with _lock:
        expired = [t for t, tk in _tickets.items() if tk.is_expired(now)]
        for token in expired:
            del _tickets[token]
    return len(expired)


def clear() -> None:
    """Drop every ticket. For tests."""
    with _lock:
        _tickets.clear()


__all__ = [
    "UploadTicket",
    "UPLOAD_TICKET_TTL_SECONDS",
    "MAX_UPLOAD_BYTES",
    "UPLOAD_CONCURRENCY",
    "issue",
    "consume",
    "prune",
    "clear",
]
