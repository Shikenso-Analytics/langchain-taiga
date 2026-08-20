import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shlex
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import PureWindowsPath
from typing import Any, Dict, List, Optional

import httpx
import requests
from cachetools import TTLCache, cached
from dotenv import load_dotenv
from fastmcp.server.dependencies import get_access_token
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from taiga import TaigaAPI
from taiga.exceptions import TaigaRestException
from taiga.models import Project, EpicStatuses, Epics, Issues

from langchain_taiga import upload_tickets

logger = logging.getLogger(__name__)


# --- python-taiga has no HTTP timeouts. Give it some. ----------------------
#
# ``taiga/requestmaker.py`` calls ``requests.get/post/put/patch/delete``
# without ``timeout=``, and requests then waits forever. A Taiga backend that
# accepts a connection and goes silent hangs the calling thread for good.
#
# That was survivable while every caller was a short tool invocation. It is
# not survivable for ``POST /mcp/upload/{token}``: attaches run under a
# semaphore sized by ``UPLOAD_CONCURRENCY``, so a handful of wedged requests
# would hold every slot permanently and take the upload endpoint down until
# the pod restarts.
#
# Scoped by swapping the ``requests`` reference inside python-taiga's own
# module namespace rather than setting ``requests.post.timeout`` globally --
# ``taiga.requestmaker.requests`` IS the shared requests module, so patching
# an attribute on it would silently re-time-out every other library in the
# process. ``requestmaker`` is the only file in python-taiga that imports
# requests, so this one swap covers all of it.
#
# Installed once at import and never restored, which is what makes it safe
# under threads -- a context manager would race its own restore between the
# concurrent attach workers.
#
# The read timeout is generous on purpose: requests applies it BETWEEN bytes,
# not to the whole transfer, so an upload that is merely slow keeps going and
# only a genuinely stalled connection trips it.
TAIGA_HTTP_CONNECT_TIMEOUT = float(os.getenv("TAIGA_HTTP_CONNECT_TIMEOUT", "10"))
TAIGA_HTTP_READ_TIMEOUT = float(os.getenv("TAIGA_HTTP_READ_TIMEOUT", "300"))


class _TimeoutDefaultingRequests:
    """Forwards to ``requests``, defaulting a timeout on the HTTP verbs.

    ``setdefault`` rather than an override, so an explicit timeout from a
    future python-taiga release still wins.
    """

    _VERBS = ("get", "post", "put", "patch", "delete")

    def __init__(self, wrapped: Any, timeout: tuple) -> None:
        self._wrapped = wrapped
        self._timeout = timeout

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._wrapped, name)
        if name not in self._VERBS:
            return attr

        def _with_timeout(*args, **kwargs):
            kwargs.setdefault("timeout", self._timeout)
            return attr(*args, **kwargs)

        return _with_timeout


def _install_taiga_http_timeouts() -> None:
    """Idempotent; safe to call more than once (module reload, tests)."""
    import taiga.requestmaker as _requestmaker

    if isinstance(_requestmaker.requests, _TimeoutDefaultingRequests):
        return
    _requestmaker.requests = _TimeoutDefaultingRequests(
        _requestmaker.requests,
        (TAIGA_HTTP_CONNECT_TIMEOUT, TAIGA_HTTP_READ_TIMEOUT),
    )


_install_taiga_http_timeouts()

load_dotenv()

TAIGA_URL = os.getenv("TAIGA_URL")
TAIGA_API_URL = os.getenv("TAIGA_API_URL")
TAIGA_TOKEN = os.getenv("TAIGA_TOKEN")
TAIGA_USERNAME = os.getenv("TAIGA_USERNAME")
TAIGA_PASSWORD = os.getenv("TAIGA_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Maximum size of an attachment that ``get_attachment_by_ref_tool`` will
# inline as base64. Larger files force the caller to use the signed
# download URL from ``list_attachments_by_ref_tool`` and fetch
# out-of-band. ENV-overridable so we can tune without a re-release if
# claude.ai / FastMCP enforces a lower payload cap.
MAX_INLINE_ATTACHMENT_BYTES = int(os.getenv("TAIGA_MAX_INLINE_ATTACHMENT_BYTES", 10 * 1024 * 1024))

# Translation table used by ``add_attachment_inline_by_ref_tool`` to
# strip ASCII whitespace from inline base64 input in a single pass.
# ``str.translate`` allocates exactly one new string (length ≤ input),
# unlike ``str.split() + "".join()`` which materializes a list of
# substring objects — pathological inputs interleaving single chars
# with whitespace (``"A\nA\nA..."``) generate millions of substring
# objects (~50 bytes each of header overhead in CPython), defeating
# the size guard and OOM-ing the MCP worker.
_INLINE_B64_WHITESPACE_DELETE = str.maketrans("", "", " \t\n\r\v\f")

# The two jobs this model does — turning a short query into a filter dict
# (``search_entities_tool``) and picking members out of a roster
# (``find_users``) — are classification, not reasoning, so they do not need
# a flagship model. ``gpt-5.6-luna`` is the current generation's smallest
# tier and measured at least as accurate as the ``gpt-5.1`` it replaces on
# both, at 1/6 the input and 1/8 the output price:
#
#   8-case parse suite, 3 runs   luna 8/8 8/8 8/8   gpt-5.1 8/8
#   negated-status queries       luna 5/5           gpt-5.1 3/3
#   "meine Tickets" (no name)    luna 5/5           gpt-5.1 0/3  <- emits
#                                                     assigned_to="me"
#   find_users roster lookups    luna 5/5           gpt-5.1 5/5
#
# ``gpt-5.4-nano`` is the same price tier but wobbles on German phrasings
# ("an Tobi zugewiesene Issues" -> owner "von Tobi"), and gpt-5.6-terra is
# the mini tier — *more* expensive than gpt-5.1, and no better here.
#
# NB the reported gpt-5.6-luna structured-output corruption applies to the
# Responses API with a strict JSON schema. This package uses Chat
# Completions and no structured-output mode at all (it prompts for JSON and
# extracts it with a regex), so neither precondition holds.
if OPENAI_API_KEY:
    small_llm = ChatOpenAI(model=os.getenv("TAIGA_SMALL_LLM_MODEL", "gpt-5.6-luna"))
else:
    small_llm = ChatOllama(model="llama3.2:3b")

# Configure caches.
#
# Single shared re-entrant lock guards every @cached helper because
# ``cachetools.TTLCache`` is NOT thread-safe and 2.3.3 introduces
# cross-thread access patterns: ``sort_kanban_by_rice_tool`` runs on
# a worker thread (via ``asyncio.to_thread`` at the FastMCP
# registration layer) while other MCP tools concurrently execute on
# the asyncio event-loop thread. Without locking, concurrent
# mutation/expiry of the same TTL entry can raise or corrupt cached
# values (cachetools 5.x docs §"Thread Safety"). Re-entrant so
# cached helpers that call other cached helpers don't self-deadlock.
_cache_lock = threading.RLock()

taiga_api_cache = TTLCache(maxsize=100, ttl=timedelta(hours=2).total_seconds())
project_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())
status_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())
list_all_statuses_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())
list_all_tags_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=10).total_seconds())

find_issue_type_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_severity_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_priority_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_status_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
milestone_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())

user_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_user_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
custom_attr_definitions_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=10).total_seconds())

# Cache owned by ``sort_kanban_by_rice_tool`` (introduced in 2.3.4).
# Project-level RICE/blocked-by/multiplicator attribute IDs change at
# project-config edit time, which is rare — 5 min TTL skips two GETs
# (``list_user_story_attributes`` + ``list_epic_attributes``) on every
# repeat invocation within the window.
#
# Per-epic *Multiplicator values* are deliberately NOT cached: the user
# flow "sort → edit an epic's Multiplicator → re-sort" needs to reflect
# the edit immediately, which it can't if a TTL is sitting on the dict.
# A draft 2.3.4 added a 60 s cache here for the marginal speedup; Codex
# review caught that it ignored same-minute multiplicator edits, and
# ~10 parallel httpx GETs against Taiga is already <1 s wall time, so
# the cache wasn't worth the correctness regression.
sort_attr_def_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())


def _current_taiga_jwt() -> Optional[str]:
    """Return the per-request Taiga JWT, or None outside an authenticated request.

    Behavior matrix:
      - No HTTP request context (stdio path) → None (caller falls back to ENV)
      - HTTP context, no verified AccessToken → None (caller falls back to ENV;
        FastMCP's auth middleware should reject unauthenticated /mcp calls before
        we reach this point, so this is unreachable in practice)
      - HTTP context with a verified AccessToken whose claims contain "taiga_jwt"
        → that JWT
      - HTTP context with a verified AccessToken whose claims DO NOT contain
        "taiga_jwt" → raise PermissionError (fail-closed, prevents fallback to
        server's ENV credentials in case of future provider regression)
    """
    try:
        tok = get_access_token()
    except (LookupError, RuntimeError):
        return None
    except Exception:  # pragma: no cover — unexpected runtime error
        logger.exception("get_access_token() raised; treating as no context")
        return None
    if tok is None:
        return None
    claims = getattr(tok, "claims", None)
    taiga_jwt = (claims or {}).get("taiga_jwt")
    if not taiga_jwt:
        raise PermissionError(
            "Authenticated request is missing taiga_jwt claim — refusing to fall "
            "back to server ENV credentials. This indicates a malformed access "
            "token from the OAuth provider."
        )
    return taiga_jwt


def _current_user_scope() -> str:
    """Return a 16-hex user-scope for cache keying, or 'default' outside auth context.

    Behavior matrix mirrors _current_taiga_jwt: stdio → 'default'; verified token
    with user_id → sha256(user_id)[:16]; verified token without user_id → raise
    PermissionError so we never cross-scope an authenticated user with stdio.
    """
    try:
        tok = get_access_token()
    except (LookupError, RuntimeError):
        return "default"
    except Exception:  # pragma: no cover
        logger.exception("get_access_token() raised in scope; treating as no context")
        return "default"
    if tok is None:
        return "default"
    claims = getattr(tok, "claims", None)
    uid = (claims or {}).get("user_id")
    if uid is None:
        raise PermissionError(
            "Authenticated request is missing user_id claim — refusing to use "
            "default cache scope. Cache scoping requires user identity."
        )
    return hashlib.sha256(str(uid).encode()).hexdigest()[:16]


def _user_scoped_key(*args: Any, **kwargs: Any) -> tuple:
    """cachetools key function that prepends the current user scope.

    Delegates the scope computation to :func:`_current_user_scope`; this
    function only assembles the cachetools-shaped key tuple.
    """
    return (_current_user_scope(), *args, *sorted(kwargs.items()))


# Mapping of acceptable entity types (singular or plural) to normalized form.
ENTITY_TYPE_MAPPING = {
    "task": "task",
    "tasks": "task",
    "userstory": "us",
    "userstories": "us",
    "us": "us",
    "issue": "issue",
    "issues": "issue",
    "epic": "epic",
    "epics": "epic",
}


def normalize_entity_type(entity_type: str) -> Optional[str]:
    """Return the normalized entity type, or None if unsupported."""
    return ENTITY_TYPE_MAPPING.get(entity_type.lower())


def get_custom_attribute_definitions(project: Project, norm_type: str) -> Dict[str, Dict]:
    """
    Get custom attribute definitions for an entity type (cached by project.id + norm_type).

    Returns a dict mapping attribute ID (as string) to {name, description, type}.
    """
    cache_key = (_current_user_scope(), project.id, norm_type)
    if cache_key in custom_attr_definitions_cache:
        return custom_attr_definitions_cache[cache_key]

    try:
        if norm_type == "us":
            attrs = project.list_user_story_attributes()
        elif norm_type == "task":
            attrs = project.list_task_attributes()
        elif norm_type == "issue":
            attrs = project.list_issue_attributes()
        elif norm_type == "epic":
            attrs = project.list_epic_attributes()
        else:
            return {}

        result = {
            str(attr.id): {
                "name": attr.name,
                "description": getattr(attr, "description", ""),
                "type": getattr(attr, "type", "text"),
            }
            for attr in attrs
        }

        custom_attr_definitions_cache[cache_key] = result
        return result
    except Exception:
        return {}


def get_formatted_custom_attributes(entity, project: Project, norm_type: str) -> List[Dict]:
    """
    Get custom attribute values for an entity, formatted with name and description.

    Returns a list of dicts with id, name, description, type, and value.
    """
    try:
        # Get attribute definitions (cached by project.id + norm_type)
        definitions = get_custom_attribute_definitions(project, norm_type)
        if not definitions:
            return []

        # Get current values
        attrs_data = entity.get_attributes()
        values = attrs_data.get("attributes_values", {})

        result = []
        for attr_id, definition in definitions.items():
            value = values.get(attr_id)
            if value is not None:
                result.append(
                    {
                        "id": int(attr_id),
                        "name": definition["name"],
                        "description": definition["description"],
                        "type": definition["type"],
                        "value": value,
                    }
                )

        return result
    except Exception:
        return []


def fetch_entity(project: Project, norm_type: str, entity_ref: int):
    """Retrieve an entity from a project given its normalized type and visible reference.

    Returns ``None`` when the ref does not exist. Taiga answers a missing ref
    with HTTP 404 and python-taiga raises that as ``TaigaRestException``
    rather than returning ``None`` — but every caller documents 404 for
    "entity not found" and reaches that branch by testing the RETURN VALUE,
    with the raise landing in their generic ``except`` and surfacing as a 500
    instead. Translated here, once, so all callers honour their own contract.

    Only 404 is translated; every other status still raises, so a genuine
    Taiga fault is never disguised as a missing entity.
    """
    try:
        return _fetch_entity_uncaught(project, norm_type, entity_ref)
    except TaigaRestException as e:
        if getattr(e, "status_code", None) == 404:
            return None
        raise


def _fetch_entity_uncaught(project: Project, norm_type: str, entity_ref: int):
    if norm_type == "task":
        return project.get_task_by_ref(entity_ref)
    elif norm_type == "us":
        return project.get_userstory_by_ref(entity_ref)
    elif norm_type == "issue":
        return project.get_issue_by_ref(entity_ref)
    elif norm_type == "epic":
        return project.get_epic_by_ref(entity_ref)
    return None


@cached(cache=taiga_api_cache, lock=_cache_lock)
def _get_taiga_api_from_env() -> TaigaAPI:
    """ENV-credentialed client, cached. Used by stdio mode.

    Reads credentials at call time (not module-level constants) so test
    fixtures can inject env via ``monkeypatch.setenv`` without import-order
    surprises.
    """
    username = os.getenv("TAIGA_USERNAME")
    password = os.getenv("TAIGA_PASSWORD")
    token_env = os.getenv("TAIGA_TOKEN")
    api_url = os.getenv("TAIGA_API_URL")
    if username and password:
        taiga_api = TaigaAPI(host=api_url)
        taiga_api.auth(username, password)
    elif token_env:
        taiga_api = TaigaAPI(host=api_url, token=token_env)
    else:
        raise ValueError("Taiga credentials not provided.")
    return taiga_api


def get_taiga_api(token: Optional[str] = None) -> TaigaAPI:
    """Get a Taiga API client.

    - No ``token`` → ENV-cached singleton (stdio path).
    - With ``token`` → fresh per-request ``TaigaAPI(host=..., token=token)``,
      uncached. Multi-tenant HTTP path.
    """
    if token is not None:
        return TaigaAPI(host=os.getenv("TAIGA_API_URL"), token=token)
    return _get_taiga_api_from_env()


@cached(cache=project_cache, key=_user_scoped_key, lock=_cache_lock)
def get_project(slug: str) -> Optional[Project]:
    """Get project by slug with auto-refreshing 5-minute, user-scoped cache."""
    # Extract slug from URL if present
    if "/project/" in slug:
        match = re.search(r"/project/([^/]+)", slug)
        if match:
            slug = match.group(1)

    try:
        project = get_taiga_api(token=_current_taiga_jwt()).projects.get_by_slug(slug)
        return project

    except Exception as e:
        print(f"Error fetching project {slug}: {e}")
        return None


@cached(cache=user_cache, key=_user_scoped_key, lock=_cache_lock)
def get_user(user_id: int) -> Optional[Dict]:
    """
    Get user by ID.

    Args:
        user_id: User ID.

    Returns:
        Dictionary with user details or an error dict.
    """
    try:
        user = get_taiga_api(token=_current_taiga_jwt()).users.get(user_id)
        user_dict = user.to_dict()
        user_dict["id"] = user.id
        user_dict["full_name"] = user.full_name
        user_dict["username"] = user.username
        return user_dict
    except Exception as e:
        return {"error": str(e), "code": 500}


@cached(cache=find_user_cache, key=_user_scoped_key, lock=_cache_lock)
def find_users(project_slug: str, query: Optional[str] = None) -> List[Dict]:
    """
    List all users in a Taiga project, optionally filtered by a query string.

    Args:
        project_slug: Project identifier.
        query: A string to filter users by name, username, or ID.

    Returns:
        str: A JSON-formatted string containing the list of users matching the query.
    """
    users = get_project(project_slug).members
    user_list = []
    for user in users:
        user_list.append({"id": user.id, "full_name": user.full_name, "username": user.username})

    if query:
        # Use a small LLM to filter the user list based on the query. Query is usually a name or username or id.
        prompt = f"""
You are given a list of users from a Taiga project as valid JSON.
The user's filter query is: {query!r}.
# Examples:
# 1) If the user query is "John Doe", it should match users with names containing "John Doe".
# 2) If the user query is "johndoe", it should match users with usernames containing "johndoe".
# 3) If the user query is "1234", it should match users with IDs containing "1234".

Return a JSON list of only those users that match the user's filter. Sort the list by relevance.
(semantically or by name or username or ID). Output must be valid JSON, with the same keys.

List of users (JSON):
{json.dumps(user_list, indent=2)}

Now filter them based on the user query "{query}".
Return only the filtered items in valid JSON (e.g., [{{"id":..., "full_name":..., "username":..., ...}}, ...]).
Do NOT include any extra commentary, just the JSON list without formatting.
        """
        response = small_llm.invoke([HumanMessage(content=prompt)])
        print(f"LLM response: {response}")
        response_str = response.content

        try:
            filtered_users = json.loads(response_str)
            print(f"Filtered users: {filtered_users}")
            if not isinstance(filtered_users, list):
                return "LLM returned JSON that is not a list."
        except json.JSONDecodeError as e:
            return f"Error decoding LLM response: {e}"
        return filtered_users
    return user_list


@cached(cache=status_cache, key=_user_scoped_key, lock=_cache_lock)
def get_status(project_slug: str, entity_type: str, status_id: int) -> Optional[Dict]:
    """
    Get status by ID for a specific entity type in a project.

    Args:
        project_slug: Project identifier.
        entity_type: 'task', 'userstory', or 'issue'.
        status_id: ID of the status.

    Returns:
        Dictionary with status details or an error dict.
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return {"error": f"Entity type '{entity_type}' is not supported.", "code": 400}

    project = get_project(project_slug)
    if not project:
        return None

    try:
        api = get_taiga_api(token=_current_taiga_jwt())
        if norm_type == "task":
            return api.task_statuses.get(status_id).to_dict()
        elif norm_type == "us":
            return api.user_story_statuses.get(status_id).to_dict()
        elif norm_type == "issue":
            return api.issue_statuses.get(status_id).to_dict()
        elif norm_type == "epic":
            return EpicStatuses(api.raw_request).get(status_id).to_dict()
    except Exception as e:
        return {"error": str(e), "code": 500}
    return None


def _find_attribute_ids(project: Project, items: list, query: str, attribute_type: str) -> List[int]:
    """Generic helper for finding attribute IDs using LLM semantic matching."""
    # Try exact match first
    exact_match = next((item for item in items if item.name.lower() == query.lower()), None)
    if exact_match:
        return [exact_match.id]

    # Prepare items for LLM processing
    item_dicts = [
        {
            "id": item.id,
            "name": item.name,
            "description": getattr(item, "description", ""),
        }
        for item in items
    ]

    prompt = f"""
Match Taiga {attribute_type} entries to query. Rules:
1. Exact name matches first
2. Partial matches (e.g. 'progress' → 'In Progress')
3. Semantic similarity (e.g. 'urgent' → 'Critical', or 'closed' → 'Done')

Available {attribute_type} entries (JSON):
{json.dumps(item_dicts, indent=2)}

Query: {query}

Return ONLY a JSON list of numeric IDs (e.g. [13, 14]) with no extra formatting.
"""

    try:
        response = small_llm.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content.strip())
    except Exception as e:
        print(f"Error finding {attribute_type} IDs: {e}")
        return []


@cached(cache=find_issue_type_cache, key=_user_scoped_key, lock=_cache_lock)
def find_issue_type_ids(project_slug: str, query: str) -> List[int]:
    """Find issue type IDs by semantic matching."""
    project = get_project(project_slug)
    if not project:
        return []
    return _find_attribute_ids(project, project.list_issue_types(), query, "issue_type")


@cached(cache=find_severity_cache, key=_user_scoped_key, lock=_cache_lock)
def find_severity_ids(project_slug: str, query: str) -> List[int]:
    """Find severity IDs by semantic matching."""
    project = get_project(project_slug)
    if not project:
        return []
    return _find_attribute_ids(project, project.list_severities(), query, "severity")


@cached(cache=find_priority_cache, key=_user_scoped_key, lock=_cache_lock)
def find_priority_ids(project_slug: str, query: str) -> List[int]:
    """Find priority IDs by semantic matching."""
    project = get_project(project_slug)
    if not project:
        return []
    return _find_attribute_ids(project, project.list_priorities(), query, "priority")


def _get_epic_statuses(project_id: int) -> list:
    """Get epic statuses for a project using the EpicStatuses factory."""
    api = get_taiga_api(token=_current_taiga_jwt())
    return EpicStatuses(api.raw_request).list(project=project_id)


@cached(cache=find_status_cache, key=_user_scoped_key, lock=_cache_lock)
def find_status_ids(project_slug: str, entity_type: str, query: str) -> List[int]:
    """Find status IDs by semantic matching for any entity type."""
    norm_type = normalize_entity_type(entity_type)
    project = get_project(project_slug)

    if not norm_type or not project:
        return []

    if norm_type == "epic":
        statuses = _get_epic_statuses(project.id)
    else:
        status_map = {
            "task": project.list_task_statuses,
            "us": project.list_user_story_statuses,
            "issue": project.list_issue_statuses,
        }
        statuses = status_map[norm_type]()

    return _find_attribute_ids(project, statuses, query, "status")


@cached(cache=milestone_cache, key=_user_scoped_key, lock=_cache_lock)
def list_milestones(project_slug: str) -> List[Dict]:
    """List all milestones (sprints) for a project, returning id, name, closed status, and dates."""
    project = get_project(project_slug)
    if not project:
        return []
    milestones = project.list_milestones()
    return [
        {
            "id": m.id,
            "name": m.name,
            "closed": m.closed,
            "estimated_start": getattr(m, "estimated_start", None),
            "estimated_finish": getattr(m, "estimated_finish", None),
        }
        for m in milestones
    ]


#: Entity types Taiga lets you put in a sprint. Epics carry no ``milestone``
#: key at all, and a task's sprint is denormalised from its user story — set
#: it there, or the two drift apart with nothing reporting it.
MILESTONE_CAPABLE_TYPES = ("us", "issue")

#: Values that mean "take this out of its sprint" rather than naming one.
_MILESTONE_CLEAR_WORDS = ("", "none", "null", "backlog")


def resolve_milestone(project_slug: str, milestone: str):
    """Resolve a sprint name to its id.

    Accepts anything ``find_milestone_id`` understands — a numeric id, the
    sprint name, a substring of it, or a "current sprint" phrasing — plus
    ``""``/``"none"``/``"null"``/``"backlog"`` to remove the entity from its
    sprint.

    Returns ``(milestone_id, None)`` on success — where ``milestone_id`` is
    ``None`` for the clear case — or ``(None, error_dict)``. The error lists
    the open sprints by name, because the caller is usually a model that has
    no other way to discover them and would otherwise guess again.
    """
    wanted = (milestone or "").strip()
    if wanted.lower() in _MILESTONE_CLEAR_WORDS:
        return None, None

    # Matching itself is ``find_milestone_id``'s job — it already handles a
    # numeric id, an exact name, a substring ("83" -> "Sprint 83") and the
    # "current sprint" / "aktueller Sprint" phrasings.
    #
    # What it cannot do is be used directly on a write: it answers ``None``
    # for BOTH "no match" and "empty query". On a read that is harmless, on a
    # write the two are opposites — one must fail, the other must clear the
    # sprint. Collapsing them would mean a typo'd sprint name silently pulls
    # the entity out of its sprint instead of erroring. Hence the tri-state
    # here rather than a call to ``find_milestone_id`` at the call sites.
    milestone_id = find_milestone_id(project_slug, wanted)
    if milestone_id is not None:
        return milestone_id, None

    milestones = list_milestones(project_slug)
    if not milestones:
        return None, {
            "error": f"Project '{project_slug}' has no sprints",
            "code": 404,
        }
    return None, {
        "error": f"Sprint '{milestone}' not found",
        "code": 404,
        "open_sprints": [m["name"] for m in milestones if not m["closed"]],
        "hint": "Pass the sprint name, or 'current' for the sprint covering today.",
    }


def _milestone_update_for(norm_type: str, project_slug: str, milestone: str):
    """Shared validation + resolution for the two write tools.

    Returns ``({"milestone": <id or None>}, None)`` or ``(None, error_dict)``.
    """
    if norm_type not in MILESTONE_CAPABLE_TYPES:
        detail = (
            "a task's sprint follows its user story — set it on the story instead"
            if norm_type == "task"
            else "epics have no sprint in Taiga"
        )
        return None, {
            "error": f"Cannot set a sprint on a {norm_type}: {detail}",
            "code": 400,
        }
    milestone_id, err = resolve_milestone(project_slug, milestone)
    if err:
        return None, err
    return {"milestone": milestone_id}, None


def get_current_milestone(project_slug: str) -> Optional[Dict]:
    """Return the milestone (sprint) whose date range includes today, or the nearest upcoming one."""
    milestones = list_milestones(project_slug)
    if not milestones:
        return None

    today = datetime.now().strftime("%Y-%m-%d")
    open_milestones = [m for m in milestones if not m["closed"]]

    # First: find a sprint where today falls within estimated_start..estimated_finish
    for m in open_milestones:
        start = m.get("estimated_start")
        finish = m.get("estimated_finish")
        if start and finish and start <= today <= finish:
            return m

    # Fallback: nearest future open milestone by start date
    future = [m for m in open_milestones if m.get("estimated_start") and m["estimated_start"] >= today]
    if future:
        return min(future, key=lambda m: m["estimated_start"])

    # Last resort: most recently started open milestone
    with_start = [m for m in open_milestones if m.get("estimated_start")]
    if with_start:
        return max(with_start, key=lambda m: m["estimated_start"])

    return None


# Patterns that indicate the user means "current sprint"
_CURRENT_SPRINT_PATTERNS = re.compile(
    r"^(?:current|aktuell|aktuelle|laufend|laufende|this|jetzt|now|heute|today)"
    r"(?:\s+(?:sprint|milestone|iteration))?"
    r"|(?:sprint|milestone|iteration)\s+"
    r"(?:current|aktuell|aktuelle|laufend|laufende|this|jetzt|now|heute|today)$",
    re.IGNORECASE,
)


def find_milestone_id(project_slug: str, milestone_query: str) -> Optional[int]:
    """Resolve a milestone name or ID to a milestone ID.

    Supports: 'current sprint', exact ID (int/string), exact name match, and fuzzy substring match.
    """
    if not milestone_query:
        return None

    # Handle "current sprint" / "aktueller Sprint" etc.
    if _CURRENT_SPRINT_PATTERNS.search(milestone_query.strip()):
        current = get_current_milestone(project_slug)
        if current:
            return current["id"]

    milestones = list_milestones(project_slug)
    if not milestones:
        return None

    # Try direct ID match
    try:
        milestone_id = int(milestone_query)
        if any(m["id"] == milestone_id for m in milestones):
            return milestone_id
    except (ValueError, TypeError):
        pass

    # Try exact name match (case-insensitive)
    query_lower = milestone_query.lower().strip()
    for m in milestones:
        if m["name"].lower().strip() == query_lower:
            return m["id"]

    # Try substring match (e.g. "83" matches "Sprint 83")
    for m in milestones:
        if query_lower in m["name"].lower():
            return m["id"]

    return None


@cached(cache=list_all_statuses_cache, key=_user_scoped_key, lock=_cache_lock)
def list_all_statuses(project_slug: str, entity_type: Optional[str]) -> Dict[str, List[Dict]]:
    """
    List all statuses for tasks, userstories, and issues in a project.
    Output is a dictionary with keys 'task_statuses', 'userstory_statuses', and 'issue_statuses'.
    Example:
    {
        "task_statuses": [
            {
            "name": "New",
            "order": 0,
            "is_closed": false,
            "color": "#70728F",
            "project": 3,
            "id": 11
            },
            {
            "name": "In progress",
            "order": 1,
            "is_closed": false,
            "color": "#E47C40",
            "project": 3,
            "id": 12
            },
            ...
        ],
        "userstory_statuses": [
            {
            "name": "New",
            "order": 1,
            "is_closed": false,
            "color": "#70728F",
            "wip_limit": null,
            "project": 3,
            "id": 13
            },
            {
            "name": "Ready",
            "order": 2,
            "is_closed": false,
            "color": "#E44057",
            "wip_limit": null,
            "project": 3,
            "id": 14
            },
           ...
        ],
        "issue_statuses": [
            {
            "name": "New",
            "order": 0,
            "is_closed": false,
            "color": "#70728F",
            "project": 3,
            "id": 15
            },
            {
            "name": "In progress",
            "order": 2,
            "is_closed": false,
            "color": "#40A8E4",
            "project": 3,
            "id": 16
            },
            ...
        ]
        }

    Args:
        project_slug: Project identifier.

    Returns:
        Dictionary with lists of statuses for each entity type.
    """
    project = get_project(project_slug)
    if not project:
        return {}

    output = {}
    if not entity_type or normalize_entity_type(entity_type) == "task":
        task_statuses = [{**status.to_dict(), "id": status.id} for status in project.list_task_statuses()]
        output["task_statuses"] = task_statuses
    if not entity_type or normalize_entity_type(entity_type) == "us":
        us_statuses = [{**status.to_dict(), "id": status.id} for status in project.list_user_story_statuses()]
        output["us_statuses"] = us_statuses
    if not entity_type or normalize_entity_type(entity_type) == "issue":
        issue_statuses = [{**status.to_dict(), "id": status.id} for status in project.list_issue_statuses()]
        output["issue_statuses"] = issue_statuses
    if not entity_type or normalize_entity_type(entity_type) == "epic":
        epic_statuses = [{**status.to_dict(), "id": status.id} for status in _get_epic_statuses(project.id)]
        output["epic_statuses"] = epic_statuses

    return output


@cached(cache=list_all_tags_cache, key=_user_scoped_key, lock=_cache_lock)
def list_all_tags(project_slug: str) -> List[str]:
    """
    List all tags used in a Taiga project.

    Args:
        project_slug: Project identifier.
    Returns:
        List of tag strings.
    """
    project = get_project(project_slug)
    if not project:
        return []

    return list(project.list_tags().keys())


def _invalidate_tag_cache(project_slug: str) -> None:
    """Drop this user's cached tag registry for ``project_slug``.

    Taiga registers a new tag as a side effect of saving an entity that
    carries it, which happens behind :func:`list_all_tags`' back — the
    10-minute TTL would otherwise keep serving a registry that predates
    the write. Two things break while it is stale: the same tag is
    reported as newly created again on the next edit, and the canonical
    spelling is missing, so a differently-cased spelling of a tag that
    now exists gets written as a second project tag.
    """
    with _cache_lock:
        list_all_tags_cache.pop(_user_scoped_key(project_slug), None)


def _normalize_tag_names(raw: Any) -> List[str]:
    """Flatten a Taiga ``tags`` payload down to plain tag names.

    Taiga is asymmetric here: it **reads** tags back as ``[name, color]``
    pairs (e.g. ``[["jobs_manager", null], ["voice", "#845EF7"]]``) but
    **accepts** a flat list of names on write. The colour is not a
    property of the entity at all — it lives in the project-level
    ``tags_colors`` registry (see :func:`list_all_tags`) and is joined in
    on read, which is why writing names back never loses it.

    Every comparison or read-modify-write of tags has to flatten first.
    Testing ``"voice" in entity.tags`` against the pair shape is silently
    always false, which is exactly how the search tool's tag filter went
    unnoticed as dead code.

    Args:
        raw: A ``tags`` payload — pairs, plain strings, or a mix. ``None``
            is tolerated (entities that never had tags).

    Returns:
        De-duplicated, whitespace-stripped tag names in first-seen order.
    """
    names: List[str] = []
    for item in raw or []:
        if isinstance(item, (list, tuple)):
            candidate = item[0] if item else None
        else:
            candidate = item
        if candidate is None:
            continue
        name = str(candidate).strip()
        if name and name not in names:
            names.append(name)
    return names


def _owner_summary(entity: Any) -> Optional[Dict]:
    """Resolve an entity's **owner** — the person who filed it.

    Taiga tracks authorship (``owner``) separately from responsibility
    (``assigned_to``), and the two routinely disagree: filing a ticket for
    someone else leaves you the owner forever. Neither the Taiga UI nor,
    before 2.15.0, any tool here surfaced it, so "what did I file that
    nobody picked up?" had to be answered by walking each entity's
    history — one API call per ticket, and easy to get wrong because the
    history comes back newest-first.

    Every list and detail response already carries both ``owner`` (a user
    id) and ``owner_extra_info`` (a nested blob with the username and
    display name). python-taiga ``setattr``s every key the API returns, so
    both are on the object — prefer the blob and fall back to
    :func:`get_user` only when it is absent, which keeps this genuinely
    free inside the per-match loop of a search rather than merely cheap
    (``get_user`` is TTL-cached for a day, so the fallback would cost one
    call per *distinct* user, not one per match).

    Args:
        entity: A python-taiga entity (user story, task, issue or epic).

    Returns:
        ``{"id", "username", "full_name"}``, or ``None`` for an entity
        with no owner (possible on very old tickets and on entities
        created by a since-deleted account).
    """
    owner_id = getattr(entity, "owner", None)
    if not owner_id:
        return None

    extra = getattr(entity, "owner_extra_info", None)
    if isinstance(extra, dict) and extra.get("username"):
        return {
            "id": extra.get("id", owner_id),
            "username": extra.get("username"),
            "full_name": extra.get("full_name_display") or extra.get("full_name"),
        }

    # No embedded blob: pay for the lookup rather than report nothing.
    # ``get_user`` returns a bare {"error", "code"} dict on failure, which
    # falls through this mapping to the same id-only answer an explicit
    # error branch would have produced.
    user = get_user(owner_id) or {}
    return {
        "id": user.get("id", owner_id),
        "username": user.get("username"),
        "full_name": user.get("full_name"),
    }


def _member_ids(matches: Any) -> List[int]:
    """Pull integer user ids out of whatever :func:`find_users` handed back.

    ``find_users`` ``json.loads``es the LLM's reply and returns it if it is
    a *list* — the elements are never checked. So a well-formed list can
    still hold ``{}`` (``KeyError``), ``"user"`` (``TypeError``), or an id
    the model stringified to ``"51"``, which is the quiet one: it survives
    the comprehension and then never equals an integer ``entity.owner``,
    silently matching nothing. Skip junk, coerce digit strings.

    Args:
        matches: The value returned by :func:`find_users` (already known
            to be a list).

    Returns:
        Integer ids, in the order given, junk dropped.
    """
    ids: List[int] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        uid = match.get("id")
        # bool is an int subclass; a stray ``true`` is not a user id.
        if isinstance(uid, bool):
            continue
        if isinstance(uid, int):
            ids.append(uid)
        elif isinstance(uid, str) and uid.strip().isdigit():
            ids.append(int(uid.strip()))
    return ids


def _owner_matches(entity: Any, owner_ids: List[int], name_key: Optional[str]) -> bool:
    """Does ``entity`` belong to the requested creator?

    ``name_key`` is the fallback for a person :func:`find_users` could not
    resolve. That lookup only searches ``project.members``, so anyone who
    has since left the project resolves to nobody — while their tickets
    keep carrying the original ``owner`` id and ``owner_extra_info``. Left
    at ids alone, "what did <departed colleague> file?" answers "nothing",
    which is indistinguishable from the truth and therefore worse than an
    error. Matching the name off the embedded blob keeps those searchable.

    Args:
        entity: A python-taiga entity.
        owner_ids: Resolved member ids; may be empty.
        name_key: Case-folded name to match against the embedded owner
            blob, or ``None`` when the ids are authoritative.

    Returns:
        True when the entity was filed by the requested person.
    """
    if getattr(entity, "owner", None) in owner_ids:
        return True
    if not name_key:
        return False
    summary = _owner_summary(entity) or {}
    # Containment, not equality, to stay symmetric with the member path:
    # ``find_users``' prompt matches names by containment, so requiring an
    # exact match here would make a first name ("Walid") find a current
    # colleague but not a departed one — the very case this fallback is
    # for.
    return any(
        name_key in str(value or "").casefold()
        for value in (summary.get("username"), summary.get("full_name"))
    )


def _assignee_matches(
    entity: Any, assigned_to_ids: List[int], name_key: Optional[str]
) -> bool:
    """Is ``entity`` assigned to the requested person?

    The mirror image of :func:`_owner_matches`, and it exists for the same
    reason: ``find_users`` only resolves *current* ``project.members``, so
    without a fallback "what is assigned to <departed colleague>?" answers
    "nothing" — a different wrong answer than the whole-project dump this
    filter used to produce, but still a wrong one. Entities keep carrying
    ``assigned_to_extra_info`` after the person leaves, so match the name
    off that blob.

    Args:
        entity: A python-taiga entity.
        assigned_to_ids: Resolved member ids; may be empty.
        name_key: Case-folded name to match against the embedded assignee
            blob, or ``None`` when the ids are authoritative.

    Returns:
        True when the entity is assigned to the requested person.
    """
    if getattr(entity, "assigned_to", None) in assigned_to_ids:
        return True
    if not name_key:
        return False
    extra = getattr(entity, "assigned_to_extra_info", None)
    if not isinstance(extra, dict):
        return False
    # Containment, for symmetry with ``find_users``' own prompt and with
    # ``_owner_matches``.
    return any(
        name_key in str(extra.get(key) or "").casefold()
        for key in ("username", "full_name_display", "full_name")
    )


def _assignee_summary(entity: Any) -> Optional[Dict]:
    """Who is responsible for ``entity``, resolved without a request.

    The mirror of :func:`_owner_summary`. Taiga embeds
    ``assigned_to_extra_info`` in every list and detail row, so reading it
    keeps this free inside the per-match loop; ``get_user`` is the fallback
    for the row that somehow lacks the blob. Doing it the other way round
    costs one sequential GET per distinct assignee per user-scoped cache
    entry, which on a 200-match search is the largest remaining per-match
    round-trip.

    Args:
        entity: A python-taiga entity.

    Returns:
        ``{"id", "username", "full_name"}``, or None when unassigned.
    """
    assigned_to_id = getattr(entity, "assigned_to", None)
    if not assigned_to_id:
        return None
    extra = getattr(entity, "assigned_to_extra_info", None)
    if isinstance(extra, dict) and extra.get("username"):
        return {
            "id": extra.get("id", assigned_to_id),
            "username": extra.get("username"),
            "full_name": extra.get("full_name_display") or extra.get("full_name"),
        }
    user = get_user(assigned_to_id) or {}
    return {
        "id": user.get("id", assigned_to_id),
        "username": user.get("username"),
        "full_name": user.get("full_name"),
    }


def _status_summary(project_slug: str, norm_type: str, entity: Any) -> Dict:
    """Name + closedness for ``entity``'s status, preferring the embedded blob.

    Taiga ships ``status_extra_info`` — ``{"name", "color", "is_closed"}`` —
    in every list and detail row, so both answers are already on the object.
    Falling back to the TTL-cached status registry costs a request per
    distinct status, and that cache is only 5 minutes, so on the per-match
    path the blob is what keeps the loop free.

    ``is_closed`` stays ``None`` when neither source knows, so a caller can
    tell "not closed" from "no idea" rather than silently treating an
    unknown status as open.

    Args:
        project_slug: Project identifier.
        norm_type: Normalised entity type.
        entity: A python-taiga entity.

    Returns:
        ``{"name": str, "is_closed": Optional[bool]}``.
    """
    extra = getattr(entity, "status_extra_info", None)
    if isinstance(extra, dict) and extra.get("name"):
        return {"name": extra["name"], "is_closed": extra.get("is_closed")}
    registry = get_status(project_slug, norm_type, getattr(entity, "status", None)) or {}
    return {
        "name": registry.get("name", "Unknown"),
        "is_closed": registry.get("is_closed"),
    }


def _milestone_label(entity: Any, milestone_names: Dict[int, str]) -> Optional[str]:
    """Human-readable sprint name for ``entity``, or None for the backlog.

    User stories ship ``milestone_name`` inline. Issues carry only the
    ``milestone`` id, so it is resolved against a map the caller builds
    once — passing the data in rather than reaching back into
    ``list_milestones`` per entity, whose cache is both short-lived and
    backed by a 4-page/~1 MB fetch. Epics have no milestone field at all
    (the key is absent from the payload, not null) and always land on None.

    Args:
        entity: A python-taiga entity.
        milestone_names: ``{milestone_id: name}`` for the project.

    Returns:
        The sprint name, or None when the entity is in the backlog or its
        type has no milestone concept.
    """
    inline = getattr(entity, "milestone_name", None)
    if inline:
        return str(inline)
    milestone_id = getattr(entity, "milestone", None)
    if not milestone_id:
        return None
    return milestone_names.get(milestone_id)


def _list_project_entities(project: Any, norm_type: str, **queryparams: Any) -> Any:
    """List a project's entities of ``norm_type`` with filters pushed server-side.

    python-taiga's convenience wrappers are asymmetric: ``list_user_stories``
    forwards ``**queryparams`` to the REST call, but ``Project.list_issues``
    and ``Project.list_epics`` are declared ``(self)`` and accept nothing.
    That is a *wrapper* limitation, not an API one — ``/issues`` and ``/epics``
    honour the same ``owner`` / ``assigned_to`` / ``status__is_closed`` params
    as ``/userstories``. Going through the resource managers directly keeps
    the filter on the server instead of paging the entire project down and
    discarding almost all of it in the client-side loop below.

    The difference is not marginal. On shikenso-development (4168 issues, 139
    sequential pages of 30, refetched in full on every search because no
    entity list is cached) one owner-filtered issue query measured 0.2s
    against 45.9s for the unfiltered walk, returning the identical rows.

    Args:
        project: A python-taiga ``Project``.
        norm_type: Normalised entity type: ``'us'``, ``'issue'`` or ``'epic'``.
            ``'task'`` is not routed here — tasks are reached by walking user
            stories, so they have no project-level list endpoint to filter.
        **queryparams: REST query params to push server-side.

    Returns:
        The entity list.

    Raises:
        ValueError: for a type with no project-level list endpoint. Raising
            beats returning ``[]``: the caller already converts exceptions
            into a structured 500, whereas an empty list would turn a
            programming error into a plausible-looking "no matches" — the
            exact silent-wrong-answer shape the rest of this tool fights.
    """
    if norm_type == "us":
        return project.list_user_stories(**queryparams)
    if norm_type == "issue":
        return Issues(project.requester).list(project=project.id, **queryparams)
    if norm_type == "epic":
        return Epics(project.requester).list(project=project.id, **queryparams)
    raise ValueError(f"no project-level list endpoint for norm_type {norm_type!r}")


def get_severity(project_slug: str, severity_id: int) -> Optional[Dict]:
    """
    Get severity by ID for a specific project.

    Args:
        project_slug: Project identifier.
        severity_id: ID of the severity.

    Returns:
        Dictionary with severity details or an error dict.
    """
    project = get_project(project_slug)
    if not project:
        return None

    try:
        return project.severities.get(severity_id).to_dict()
    except Exception as e:
        return {"error": str(e), "code": 500}
    # return None


@tool(parse_docstring=True)
def create_entity_tool(
    project_slug: str,
    entity_type: str,
    subject: str,
    status: str,
    description: Optional[str] = "",
    parent_ref: Optional[int] = None,
    assign_to: Optional[str] = None,
    due_date: Optional[str] = None,
    tags: List[str] = [],
    color: Optional[str] = None,
    severity: Optional[str] = None,
    issue_type: Optional[str] = None,
    priority: Optional[str] = None,
    milestone: Optional[str] = None,
) -> str:
    """
    Create new userstory, tasks, issues or epics.
    Use when:
      - User requests creation of new work items
      - Need to break down userstories into tasks
      - Reporting new issues/bugs
      - Creating epics to group user stories

    Args:
        project_slug: Project identifier
        entity_type: 'userstory', 'task', 'issue' or 'epic'
        subject: Short title/name
        status: State of the entity
        description: Detailed description (optional)
        parent_ref: For tasks - userstory reference
        assign_to: Username to assign (optional)
        due_date: Deadline for the task (Format: YYYY-MM-DD) (optional)
        tags: List of tags (optional)
        color: Color for the epic (hex format, e.g. '#FF0000') (optional, epics only)
        severity: Severity name for issues (optional, uses first available if omitted)
        issue_type: Issue type name for issues (optional, uses first available if omitted)
        priority: Priority name for issues (optional, uses first available if omitted)
        milestone: Sprint to put it in (optional). The sprint name exactly, or
            'current' for the sprint covering today. User stories and issues
            only. Omit to leave it in the backlog.

    Returns:
        JSON with created entity details. Errors: 400 (entity type, or a sprint
        on a task/epic), 404 (project, parent, status, user or sprint not found).
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps({"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2)

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    # Resolve parent userstory if needed
    parent_us = None
    if parent_ref and norm_type == "task":
        parent_us = project.get_userstory_by_ref(parent_ref)
        if not parent_us:
            return json.dumps(
                {"error": f"Parent userstory {parent_ref} not found", "code": 404},
                indent=2,
            )

    # Resolve assignee
    assignee_id = None
    if assign_to:
        users = find_users(project_slug, assign_to)
        if not users:
            return json.dumps({"error": f"User '{assign_to}' not found", "code": 404}, indent=2)
        assignee_id = users[0]["id"]

    # Base creation data
    create_data = {
        "subject": subject[:500],
        "description": description[:2000],
        # Same invariant as manage_tags_by_ref_tool: strip, drop blanks,
        # de-duplicate. Taiga creates unknown tags implicitly, so passing
        # the caller's list through raw mints '  voice ' and 'voice' as
        # two permanent project tags.
        "tags": _normalize_tag_names(tags),
        "assigned_to": assignee_id,
        "due_date": due_date,
    }

    # Resolve the status ONCE, for every entity type.
    #
    # This used to be done per-branch, three different ways, and only one of
    # them was right: user stories dropped ``status`` on the floor entirely
    # (created in the project's default and silently, so US #8130 sat in
    # ``New`` for a whole sprint and then jumped straight to ``Done``, which
    # distorts sprint statistics), tasks indexed ``[0]`` into a possibly
    # empty list and surfaced an unknown status as "Creation failed: list
    # index out of range", and epics ignored an unresolvable one. Resolving
    # here means one behaviour and one error for all four.
    status_ids = find_status_ids(
        project_slug=project_slug, entity_type=entity_type, query=status
    )
    if not status_ids:
        return json.dumps(
            {"error": f"Status '{status}' not found", "code": 404}, indent=2
        )
    create_data["status"] = status_ids[0]

    if milestone is not None:
        milestone_update, err = _milestone_update_for(
            norm_type, project_slug, milestone
        )
        if err:
            return json.dumps(err, indent=2)
        create_data.update(milestone_update)

    try:
        if norm_type == "task":
            if not parent_us:
                return json.dumps({"error": "Tasks require a parent userstory", "code": 400}, indent=2)
            entity = parent_us.add_task(**create_data)
        elif norm_type == "us":
            entity = project.add_user_story(**create_data)
        elif norm_type == "issue":
            # Resolve issue type
            if issue_type:
                issue_type_ids = find_issue_type_ids(project_slug, issue_type)
                if not issue_type_ids:
                    return json.dumps({"error": f"Issue type '{issue_type}' not found"}, indent=2)
                create_data["issue_type"] = issue_type_ids[0]
            else:
                # Use first available issue type from project
                available_issue_types = project.list_issue_types()
                if not available_issue_types:
                    return json.dumps({"error": "No issue types available in project"}, indent=2)
                create_data["issue_type"] = available_issue_types[0].id

            # Resolve severity
            if severity:
                severity_ids = find_severity_ids(project_slug, severity)
                if not severity_ids:
                    return json.dumps({"error": f"Severity '{severity}' not found"}, indent=2)
                create_data["severity"] = severity_ids[0]
            else:
                # Use first available severity from project
                available_severities = project.list_severities()
                if not available_severities:
                    return json.dumps({"error": "No severities available in project"}, indent=2)
                create_data["severity"] = available_severities[0].id

            # Resolve priority
            if priority:
                priority_ids = find_priority_ids(project_slug, priority)
                if priority_ids:
                    create_data["priority"] = priority_ids[0]
            else:
                # Use first available priority from project
                available_priorities = project.list_priorities()
                if available_priorities:
                    create_data["priority"] = available_priorities[0].id

            entity = project.add_issue(**create_data)
        elif norm_type == "epic":
            # Add color if provided
            if color:
                create_data["color"] = color

            # Remove due_date as epics don't have it
            create_data.pop("due_date", None)

            entity = project.add_epic(**create_data)
        else:
            return json.dumps({"error": "Unsupported entity type", "code": 400}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Creation failed: {str(e)}", "code": 500}, indent=2)

    # Creation is the other write path that can register a project tag, and
    # Taiga does it as a side effect of the save — see _invalidate_tag_cache.
    # Unconditional here (unlike manage_tags_by_ref_tool, which knows what it
    # added) because checking would cost the very fetch this avoids.
    if create_data["tags"]:
        _invalidate_tag_cache(project_slug)

    return json.dumps(
        {
            "created": True,
            "type": norm_type,
            "ref": entity.ref,
            "subject": entity.subject,
            "due_date": due_date,
            "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity.ref}",
            "assigned_to": assign_to,
            "parent": parent_ref,
        },
        indent=2,
    )


def _format_userstory_points(entity: Any, project: Any) -> Dict[str, float]:
    """Resolve a user story's per-role ``points`` dict to the human-readable
    ``{role_name: point_value}`` shape that ``set_userstory_points_tool``
    accepts as input. Returns an empty dict if ``entity.points`` is empty
    or unset.

    Roles whose ID is no longer in ``project.list_roles()`` (deleted/stale)
    and assignments pointing to the "?" (unestimated, ``value=None``)
    point are silently dropped — consumers see only role-name keys with
    concrete numeric values, never partial or null entries.
    """
    raw_points = getattr(entity, "points", None) or {}
    if not raw_points:
        return {}
    role_id_to_name = {str(r.id): r.name for r in project.list_roles()}
    point_id_to_value = {p.id: p.value for p in project.list_points() if p.value is not None}
    out: Dict[str, float] = {}
    for role_id, point_id in raw_points.items():
        role_name = role_id_to_name.get(str(role_id))
        point_value = point_id_to_value.get(point_id)
        if role_name and point_value is not None:
            out[role_name] = point_value
    return out


def _coerce_to_aware_datetime(value: Any) -> Optional[datetime]:
    """Best-effort coerce a Taiga timestamp value to tz-aware UTC datetime.

    python-taiga's ``Resource.__init__`` parses ``created_date`` /
    ``modified_date`` via a strict regex
    (``r"\\d+-\\d+-\\d+T\\d+:\\d+:\\d+\\+0000"``) — anything outside that
    exact shape (microsecond precision, ``Z`` suffix, ``+00:00`` with
    colon, …) leaves the attribute as a raw string. Comparing that
    string against a datetime then raises
    ``TypeError: '<' not supported between instances of 'str' and
    'datetime.datetime'`` mid-loop and silently truncates results.

    This helper is the rescue: accept either a datetime or a string,
    normalize to tz-aware UTC, return None on anything unparseable.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            # dateutil is already a transitive dependency via python-taiga;
            # it handles every flavour of ISO-8601 Taiga has been observed
            # to emit (microseconds, Z, +0000 with or without colon).
            from dateutil.parser import parse as _du_parse

            dt = _du_parse(value)
        except Exception:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


@tool(parse_docstring=True)
def search_entities_tool(
    project_slug: str,
    query: str,
    entity_type: str = "task",
    max_results: int = 200,
    include_custom_attributes: bool = False,
    open_only: bool = False,
) -> str:
    """
    Search tasks/userstories/issues/epics using natural language filters with client-side matching.
    Use when:
      - Looking for items matching complex criteria
      - Needing flexible search beyond API filter capabilities
      - Searching across multiple entity relationships

    Performance note: each match enriched with custom attributes triggers
    one extra Taiga API call. For overview/aggregation queries
    (counting, grouping, ranking) keep ``include_custom_attributes=False``
    to avoid an N+1 round-trip storm — typically a 5-10x speedup on
    projects with hundreds of entities. Set it True only when the
    custom-attribute values are actually needed in the answer.

    Args:
        project_slug: Project identifier (e.g. 'mobile-app').
        query: Natural language query (e.g. 'UX tasks in progress assigned to @john').
        entity_type: 'task', 'userstory', 'issue', or 'epic'.
        max_results: Cap on number of matched entities returned. Defaults
            to 200. The response payload includes a ``truncated`` flag so
            callers can detect when more matches exist beyond the cap.
        include_custom_attributes: If True, fetch each match's custom
            attribute values via an extra API call per entity. Default
            False — leave off unless the values are actually needed.
        open_only: If True, return only entities whose status is not a
            closed one, decided by Taiga's own ``is_closed`` flag rather
            than by status names. Always prefer this over phrasing the
            exclusion in the query, because negation is not expressible
            in the query language — a query such as "not closed and not
            archived" is resolved by dropping only the status names that
            literally appear in it, so sibling terminal statuses like
            "Done" or "Rejected" survive the filter and the result
            quietly includes finished work.

    Returns:
        JSON object with ``matches`` (list of entities), ``truncated``
        (bool — was the max_results cap hit?), ``count`` (length of
        matches), and ``max_results`` (the cap that was applied).

        Each match carries both ``owner`` (the username of whoever filed
        it) and ``assigned_to`` (whoever is responsible for it now).
        Those routinely differ, and the query language filters on either:
        "issues created by jdoe" narrows on owner, "assigned to jdoe" on
        the assignee.

        A match also carries ``is_closed`` (Taiga's own flag for the
        entity's status, so "is this finished?" needs no status-name
        table), plus ``milestone`` (sprint id, null for the backlog) and
        ``milestone_name`` (the sprint's name). Those three are read off
        the payload Taiga already sent — they exist so that grouping by
        sprint or dropping finished work does not require a
        ``get_entity_by_ref_tool`` call per match. Epics have no
        milestone concept at all and always report null for both.

        Two limits worth knowing before composing a query:

        - The query is parsed without any notion of who is asking, so a
          first-person query ("tickets I created") cannot resolve. Call
          ``whoami_tool`` first and put the returned username into the
          query.
        - Negation is not expressible. For "created by me but NOT
          assigned to me", filter on the owner and drop the matches whose
          ``assigned_to`` is that same person — both fields are in the
          response precisely so this needs no second call.
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps({"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2)

    # Reject malformed caller-controlled cap up-front. Without this
    # guard a caller passing ``max_results=0`` or negative would
    # terminate the match loop on the first iteration AND report
    # ``truncated=True`` with no matches.
    if max_results < 1:
        return json.dumps(
            {
                "error": f"max_results must be >= 1, got {max_results}",
                "code": 400,
            },
            indent=2,
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    statuses = list_all_statuses(project_slug, norm_type)
    # Hand the parser Taiga's ``is_closed`` flag, not just the names. Without
    # it the model can only reason lexically about a negated query, and
    # "not closed and not archived" gets resolved by striking the names that
    # literally appear — leaving every *other* terminal status ("Done",
    # "Rejected") in the filter and quietly returning finished work.
    status_catalog = ", ".join(
        f"{s['name']}{' [CLOSED]' if s.get('is_closed') else ''}"
        for s in statuses.get(f"{norm_type}_statuses", [])
    )
    tags = list_all_tags(project_slug)
    milestones = list_milestones(project_slug)
    open_milestones = [m for m in milestones if not m["closed"]]
    current_milestone = get_current_milestone(project_slug)
    milestone_names = ", ".join([f'"{m["name"]}" (id={m["id"]})' for m in open_milestones])
    if current_milestone:
        current_sprint_info = (
            f'The CURRENT sprint is "{current_milestone["name"]}" '
            f'({current_milestone["estimated_start"]} to {current_milestone["estimated_finish"]}).'
        )
    else:
        current_sprint_info = "No current sprint could be determined from dates."

    # Convert natural language to search criteria
    # Short-circuit: if the query is a catch-all like "all", "all epics", etc.
    # skip LLM parsing entirely and return all entities unfiltered.
    _all_pattern = re.compile(r"^(?:all|show all|list all|every|alles|alle)(?:\s+\w+)?$", re.IGNORECASE)
    if _all_pattern.match(query.strip()):
        search_params: dict = {}
    else:
        prompt = f"""
Convert this project management query to search parameters:
Query: {query}

The entity type being searched is "{norm_type}" — do NOT use the entity type as a text_search or tag filter.

Possible parameters:
- status_names: List[str] (status names)
- assigned_to: str (username/ID of the person RESPONSIBLE for the item)
- owner: str (username/ID of the person who CREATED/filed the item — the author/reporter. Use for "created by", "filed by", "reported by", "angelegt von", "erstellt von". NOT the same as assigned_to. Only set this when the query NAMES a person; a first-person query like "my tickets" carries no name and must leave this null.)
- milestone: str (sprint/milestone name, e.g. "Sprint 83")
- tags: List[str]
- text_search: str (searches subject/description). Only set text_search if the user explicitly wants to search for specific words in subjects or descriptions.
- created_after: date (YYYY-MM-DD)
- closed_before: date (YYYY-MM-DD)

IMPORTANT: Only set parameters that are explicitly mentioned or clearly implied by the query. Use null for everything else. Do NOT guess or hallucinate filter values.

Output ONLY valid JSON with parameter keys. Use null for unknown values.

IMPORTANT: The entity type ({norm_type}) is already selected — do NOT use it as a tag filter.
If the user wants "all" items, return all null values: {{"status_names": null, "assigned_to": null, "owner": null, "tags": null, "text_search": null, "created_after": null, "closed_before": null}}

Possible status names: {status_catalog}

Statuses marked [CLOSED] are Taiga's finished/terminal states. If the query asks for
items that are open / active / "not closed" / "not done" / "nicht geschlossen" /
"nicht archiviert" — or excludes a terminal state in any other wording — list ONLY
the names that are NOT marked [CLOSED]. Never include a [CLOSED] name just because
that particular word is absent from the query.

Available milestones/sprints: {milestone_names}
{current_sprint_info}

Possible tags: {', '.join(tags)}

Example response for "John's open UX tasks in Sprint 83":
"{{"status_names": ["Open"], "assigned_to": "john_doe", "milestone": "Sprint 83", "tags": ["UX"]}}"

Example response for "all items in Sprint 83":
"{{"milestone": "Sprint 83", "status_names": null, "assigned_to": null, "owner": null, "tags": null, "text_search": null}}"

Example response for "issues created by john_doe":
"{{"owner": "john_doe", "assigned_to": null, "status_names": null, "tags": null, "text_search": null}}"

IMPORTANT: When the user says "current sprint", "aktueller Sprint", "this sprint", "laufender Sprint", or similar, use the current sprint name shown above as the milestone value.
"""
        try:
            response = small_llm.invoke([HumanMessage(content=prompt)])
            content = str(response.content)
            # Try to find JSON block
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                content = match.group(0)
            search_params = json.loads(content)
        except Exception as e:
            return json.dumps({"error": f"Query parsing failed: {str(e)}", "code": 500}, indent=2)

    # Resolve milestone filter (before fetching entities for server-side
    # filtering). Tri-stated like the owner/assignee/status filters:
    # ``milestone_requested`` records that a sprint WAS asked for, so an
    # unresolvable name ("Sprint 99", a typo, a renamed sprint) matches
    # nothing instead of failing open. ``find_milestone_id`` returns None
    # for both "not asked" and "asked but unknown", and testing that for
    # truthiness is the same bug this change fixes elsewhere — it would
    # return the whole project for the single most common sprint query.
    milestone_requested = bool(search_params.get("milestone"))
    milestone_id = None
    if milestone_requested:
        milestone_id = find_milestone_id(project_slug, search_params["milestone"])

    # Resolve the owner (creator) filter here too, for the same reason:
    # the REST list endpoints accept ``owner=<id>``, so a resolvable name
    # keeps the client-side scan below off the whole project. ``None``
    # means "no filter"; an empty list means "asked for someone who isn't
    # a member" and must match nothing rather than degrade to unfiltered.
    owner_ids: Optional[List[int]] = None
    owner_name_key: Optional[str] = None
    if search_params.get("owner"):
        owner_query = str(search_params["owner"]).strip()
        if owner_query.isdigit():
            # A numeric query is already the exact answer, so skip the
            # lookup entirely. Routing it through ``find_users`` would be
            # actively wrong: that prompt matches ids by *containment*, so
            # "51" also resolves user 151 — widening the filter to a second
            # person's tickets and, with two ids, dropping the single-id
            # server-side pushdown. It also works for a departed user,
            # who is in no ``project.members`` list to be found in.
            owner_ids = [int(owner_query)]
        else:
            # ``find_users`` calls the LLM *outside* its own try, so a
            # provider timeout or rate-limit propagates from here — and this
            # block sits outside the entity-listing try below, so without a
            # guard the whole tool call raises instead of returning the
            # structured error every other lookup failure gets. The
            # query-parsing call above is wrapped for exactly this reason.
            try:
                owner_matches = find_users(project_slug, owner_query)
            except Exception as e:
                return json.dumps(
                    {"error": f"Owner lookup failed: {e}", "code": 500}, indent=2
                )
            # It is annotated ``-> List[Dict]`` but returns a plain STRING on
            # both of its parse-failure paths. Iterating that yields single
            # characters and blows up on ``u["id"]``.
            if not isinstance(owner_matches, list):
                return json.dumps(
                    {"error": f"Owner lookup failed: {owner_matches}", "code": 500},
                    indent=2,
                )
            owner_ids = _member_ids(owner_matches)
            if not owner_ids:
                # Nobody by that name is a CURRENT member — the normal case
                # for a departed colleague whose tickets are exactly what
                # someone wants to chase. Keep the name as a fallback,
                # matched against the owner blob the entities still carry.
                owner_name_key = owner_query.casefold()

    # Resolve the assignee filter the same way, and for the same two
    # reasons: one resolved id can be pushed server-side, and the tri-state
    # keeps "nobody matched that name" from degrading into "no filter".
    # It used to be resolved *below* the fetch as a plain list, and
    # ``if resolved_filters.get("assigned_to_ids"):`` then read an empty
    # list as falsy — so an unresolvable assignee silently switched the
    # filter off and handed back the entire project, labelled as that
    # person's work. Measured on shikenso-development: an unknown name
    # returned all 14 epics, and 200 (capped) user stories and issues.
    assigned_to_ids: Optional[List[int]] = None
    assigned_to_name_key: Optional[str] = None
    if search_params.get("assigned_to"):
        assignee_query = str(search_params["assigned_to"]).strip()
        if assignee_query.isdigit():
            # Same reasoning as the owner path: a numeric query IS the
            # answer, and routing it through ``find_users`` would match ids
            # by containment ("51" also resolving 151).
            assigned_to_ids = [int(assignee_query)]
        else:
            try:
                assignee_matches = find_users(project_slug, assignee_query)
            except Exception as e:
                return json.dumps(
                    {"error": f"Assignee lookup failed: {e}", "code": 500}, indent=2
                )
            # ``find_users`` is annotated ``-> List[Dict]`` but returns a
            # plain STRING on both parse-failure paths. The old code fed it
            # straight into ``[u["id"] for u in users]``, which iterates the
            # string's characters and dies on ``u["id"]`` with a TypeError
            # that escaped the tool entirely.
            if not isinstance(assignee_matches, list):
                return json.dumps(
                    {
                        "error": f"Assignee lookup failed: {assignee_matches}",
                        "code": 500,
                    },
                    indent=2,
                )
            assigned_to_ids = _member_ids(assignee_matches)
            if not assigned_to_ids:
                assigned_to_name_key = assignee_query.casefold()

    # Fetch entities (with server-side filtering when possible)
    try:
        if norm_type == "task":
            us_kwargs = {}
            if milestone_id is not None:
                us_kwargs["milestone"] = milestone_id
            # NB: none of ``owner`` / ``assigned_to`` / ``status__is_closed``
            # is pushed down here. Tasks are reached by walking user stories,
            # so every one of those params would select on the *story* and
            # silently drop tasks that are filed by, assigned to, or open
            # under somebody else's story. This branch therefore builds its
            # own kwargs rather than sharing the project-level set below.
            #
            # KNOWN GAP (pre-dates ``open_only``, unchanged here): the
            # ``us.is_closed`` skip below means tasks are only ever collected
            # from OPEN stories, so an open task parked under a finished
            # story is invisible to every task search — with or without
            # ``open_only``. Widening it would make each task search walk
            # every story in the project (1013 on shikenso-development), so
            # it is left as-is rather than changed as a side effect of this
            # commit. ``open_only`` therefore narrows tasks client-side by
            # each task's own status, but only within that already-narrowed
            # set.
            entities = []
            for us in project.list_user_stories(**us_kwargs):
                if us.is_closed:
                    continue
                entities.extend(us.list_tasks())
        else:
            # Everything resolvable to a single value is pushed server-side.
            # The client-side pass below still enforces every filter, so a
            # param the API ignores costs correctness nothing — only the
            # saved round-trips.
            list_kwargs: Dict[str, Any] = {}
            if milestone_id is not None and norm_type != "epic":
                # Epics have no milestone field at all (the key is absent
                # from the payload rather than null), so the param would be
                # meaningless.
                list_kwargs["milestone"] = milestone_id
            if owner_ids and len(owner_ids) == 1:
                list_kwargs["owner"] = owner_ids[0]
            if assigned_to_ids and len(assigned_to_ids) == 1:
                list_kwargs["assigned_to"] = assigned_to_ids[0]
            if open_only:
                list_kwargs["status__is_closed"] = "false"
            entities = _list_project_entities(project, norm_type, **list_kwargs)
    except Exception as e:
        return json.dumps({"error": f"Entity listing failed: {str(e)}", "code": 500}, indent=2)

    # Resolve filters upfront
    resolved_filters = {}

    # Status resolution. Tri-stated like the owner/assignee filters: ``None``
    # means no status filter was asked for, an empty list means "the
    # requested status names resolve to nothing here" and must match nothing.
    # Testing the old plain list for truthiness meant a renamed or misspelled
    # status silently dropped the filter and returned the whole project —
    # including the closed items the caller was trying to exclude.
    #
    # Keyed off ``isinstance(..., list)`` rather than truthiness, because an
    # explicit ``"status_names": []`` is a third case: the parser was asked
    # for a status filter and produced no names. That is exactly the shape
    # this tool's own prompt can emit for a negated query on a project whose
    # statuses are all terminal — and reading it as "no filter" hands back
    # the whole project, the failure being fixed here. Absent or ``null``
    # still means "not asked for".
    requested_status_names = search_params.get("status_names")
    status_ids: Optional[List[int]] = None
    if isinstance(requested_status_names, list):
        resolved_status_ids: List[int] = []
        for status_name in requested_status_names:
            resolved_status_ids.extend(
                find_status_ids(project_slug, norm_type, status_name)
            )
        status_ids = list(set(resolved_status_ids))

    # Tag resolution. Derived only from ``search_params``, so it belongs
    # here with the other upfront resolutions rather than being rebuilt
    # for every entity in the loop below. An all-blank list resolves to
    # nothing and leaves the filter disabled.
    if search_params.get("tags"):
        wanted_tags = {name.lower() for name in _normalize_tag_names(search_params["tags"])}
        if wanted_tags:
            resolved_filters["tag_keys"] = wanted_tags

    # Date parsing.
    # Both filter datetimes are made tz-aware (UTC). python-taiga returns
    # ``entity.created_date`` / ``entity.finished_date`` as tz-aware
    # datetimes (Taiga API ships ISO timestamps with ``+0000``), and
    # comparing tz-aware vs tz-naive raises ``TypeError: can't compare
    # offset-naive and offset-aware datetimes`` mid-loop, which silently
    # truncates results.
    date_format = "%Y-%m-%d"
    if search_params.get("created_after"):
        resolved_filters["created_after"] = datetime.strptime(search_params["created_after"], date_format).replace(
            tzinfo=timezone.utc
        )
    if search_params.get("closed_before"):
        resolved_filters["closed_before"] = datetime.strptime(search_params["closed_before"], date_format).replace(
            tzinfo=timezone.utc
        )

    # Sprint id -> name, built once from the list already fetched above.
    # ``_milestone_label`` takes this map rather than calling
    # ``list_milestones`` per entity: that cache is 5-minute TTL over a
    # 4-page/~1 MB fetch, so a long enough loop can outrun it and silently
    # re-pay mid-iteration.
    milestone_names_by_id = {m["id"]: m.get("name") for m in milestones}

    # Client-side filtering
    matches = []
    cap_hit = False
    for entity in entities:
        match = True
        status_info = _status_summary(project_slug, norm_type, entity)

        # Milestone filter (client-side fallback for entities not server-filtered).
        # Keyed off ``milestone_requested``, not off ``milestone_id``: a sprint
        # name that resolves to nothing must match nothing rather than fail open.
        # The ``is None`` arm is not redundant — backlog entities also carry
        # ``milestone = None``, so comparing against an unresolved filter would
        # match every one of them instead of none.
        if milestone_requested:
            if milestone_id is None or getattr(entity, "milestone", None) != milestone_id:
                match = False

        # Status filter
        if status_ids is not None and entity.status not in status_ids:
            match = False

        # Open-only filter. Decided by Taiga's own ``is_closed`` flag, which
        # rides along in ``status_extra_info`` on every row, so this needs no
        # lookup and — unlike a status-name list — cannot be defeated by
        # somebody adding or renaming a terminal status. An entity whose
        # closedness is genuinely undeterminable is kept rather than dropped:
        # a missing flag is not evidence the work is finished.
        if open_only and status_info["is_closed"] is True:
            match = False

        # Assignment filter. Tri-stated for the same reason as the owner
        # filter below — an empty id list is a real filter (name-only), and
        # truthiness would wave the whole project through instead.
        if assigned_to_ids is not None and not _assignee_matches(
            entity, assigned_to_ids, assigned_to_name_key
        ):
            match = False

        # Owner (creator) filter. ``owner_ids`` stays a plain local: it is
        # resolved above the fetch (to be pushed server-side) and read here,
        # both in this function's scope. ``None`` means no filter was asked
        # for — tested explicitly rather than by truthiness, because an
        # empty id list is a real filter (name-only, see _owner_matches)
        # and truthiness would wave the whole project through instead.
        if owner_ids is not None and not _owner_matches(entity, owner_ids, owner_name_key):
            match = False

        # Tag filter. ``entity.tags`` arrives as [name, color] pairs, so
        # the names have to be flattened out before comparing — the
        # pre-2.14.0 ``tag in entity.tags`` compared a name against a
        # pair and therefore matched nothing, ever.
        if resolved_filters.get("tag_keys"):
            entity_tags = {name.lower() for name in _normalize_tag_names(getattr(entity, "tags", None))}
            if not resolved_filters["tag_keys"] <= entity_tags:
                match = False

        # Text search
        if search_params.get("text_search"):
            search_text = search_params["text_search"].lower()
            subject_match = search_text in entity.subject.lower()
            desc_match = search_text in (getattr(entity, "description", "") or "").lower()
            if not (subject_match or desc_match):
                match = False

        # Date filters. Both sides must be tz-aware datetimes — Taiga
        # may return ``created_date`` / ``finished_date`` as raw strings
        # when python-taiga's strict regex doesn't match (e.g. microsecond
        # precision), which used to crash with
        # ``TypeError: '<' not supported between instances of 'str' and
        # 'datetime.datetime'``. Coerce both sides up-front.
        if resolved_filters.get("created_after"):
            ec = _coerce_to_aware_datetime(entity.created_date)
            if ec is None or ec < resolved_filters["created_after"]:
                match = False
        if resolved_filters.get("closed_before"):
            ef = _coerce_to_aware_datetime(entity.finished_date)
            if ef is None or ef > resolved_filters["closed_before"]:
                match = False

        if match:
            description = getattr(entity, "description", "") or ""
            custom_attributes: List[Dict] = []

            # Per-match enrichment is opt-in: ``fetch_entity`` triggers
            # one extra API request per match, which on a 200-match
            # search is an N+1 storm that dominates response time. Only
            # do it when the caller asked for custom attributes.
            if include_custom_attributes:
                try:
                    full_entity = fetch_entity(project, norm_type, entity.ref)
                    if full_entity:
                        if not description:
                            description = getattr(full_entity, "description", "") or ""
                        custom_attributes = get_formatted_custom_attributes(full_entity, project, norm_type)
                except Exception:
                    pass

            matches.append(
                {
                    "ref": entity.ref,
                    "subject": entity.subject,
                    "description": description,
                    "status": status_info["name"],
                    # Both people come off the blob Taiga already sent, so
                    # neither costs a lookup inside this loop.
                    "assigned_to": (_assignee_summary(entity) or {}).get("username"),
                    # Creator, not assignee.
                    "owner": (_owner_summary(entity) or {}).get("username"),
                    # Taiga's own terminal-status flag, so a caller can drop
                    # finished work without a status-name table of its own.
                    "is_closed": status_info["is_closed"],
                    # Sprint id + name, both already in the payload. Null on
                    # both means the backlog — or an epic, which has no
                    # milestone concept at all.
                    "milestone": getattr(entity, "milestone", None),
                    "milestone_name": _milestone_label(entity, milestone_names_by_id),
                    "created_date": (entity.created_date if entity.created_date else None),
                    "due_date": getattr(entity, "due_date", None),
                    "custom_attributes": custom_attributes,
                    "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity.ref}",
                }
            )

            # Cap is caller-controlled. ``cap_hit`` is set ONLY when we
            # actually break out early — relying on ``len == cap`` after
            # the loop reports false-positive ``truncated=True`` for a
            # search whose total result set happens to land exactly on
            # the cap (e.g. ``max_results=200`` with exactly 200 real
            # matches and no remaining entities to test).
            if len(matches) >= max_results:
                cap_hit = True
                break

    return json.dumps(
        {
            "matches": matches,
            "count": len(matches),
            "max_results": max_results,
            "truncated": cap_hit,
        },
        indent=2,
        default=str,
    )


def fetch_history(entity, norm_type):
    """
    Return the full history list for a Taiga entity.

    Parameters
    ----------
    entity : taiga.models.models.BaseEntity
        The already‑fetched Taiga object (UserStory, Task, Issue …).
    norm_type : str
        Normalised entity type: `'us'`, `'task'`, or `'issue'`.

    Returns
    -------
    list[taiga.models.models.HistoryEntity]
        A list of history entries, newest first. If the entity type is not
        supported, an empty list is returned.

    Notes
    -----
    * Taiga stores comments (and other changes) as history entries.
    * The helper does **not** filter for comments – callers can filter with
      ``[h for h in history if getattr(h, "comment", None)]`` when needed.
    """
    api = get_taiga_api(token=_current_taiga_jwt())

    # Map normalised type to the corresponding history accessor
    history_fetcher = {
        "us": api.history.user_story.get,
        "task": api.history.task.get,
        "issue": api.history.issue.get,
        "epic": api.history.epic.get,
    }.get(norm_type)

    return history_fetcher(entity.id) if history_fetcher else []


@tool(parse_docstring=True)
def get_kanban_board_tool(project_slug: str, include_closed: bool = True) -> str:
    """Return the user-story Kanban board grouped into ordered status columns.

    Mirrors the Taiga UI board: one column per user-story status in
    ``order``, each holding that column's user stories sorted by
    ``kanban_order``. User stories only (Kanban is always user stories);
    swimlanes, custom attributes and story points are intentionally left
    out to keep the payload a fast board snapshot.

    Use when:
      - The user wants the Kanban board layout, not a flat item list.
      - You need columns (including empty ones), WIP limits, and per-card
        order the way the UI shows them.

    Args:
        project_slug: Project identifier (the URL slug).
        include_closed: Include closed-status columns such as Done or
            Archived. Default True. Pass False to see only active columns.

    Returns:
        JSON object with ``project`` (name) and ``columns`` (ordered by
        status order). Each column carries ``status``, ``status_id``,
        ``order``, ``is_closed``, ``wip_limit`` and ``cards`` — each card
        having ``ref``, ``subject``, ``assigned_to`` (username or null)
        and ``kanban_order``. ``orphan_cards`` appears only when a story
        references a status id absent from the board.
    """
    try:
        project = get_project(project_slug)
        if not project:
            return json.dumps(
                {"error": f"Project '{project_slug}' not found", "code": 404},
                indent=2,
            )

        statuses = sorted(project.list_user_story_statuses(), key=lambda s: (s.order, s.id))
        known_ids = {s.id for s in statuses}
        shown = [s for s in statuses if include_closed or not s.is_closed]
        columns = {
            s.id: {
                "status": s.name,
                "status_id": s.id,
                "order": s.order,
                "is_closed": s.is_closed,
                "wip_limit": s.wip_limit,
                "cards": [],
            }
            for s in shown
        }

        # id -> username from the already-hydrated member list (no extra API
        # call — the same source find_users / list_project_members_tool use);
        # get_user is the cached fallback for ex-members still stamped on old
        # stories.
        member_names = {u.id: u.username for u in project.members}

        orphans = []
        for us in project.list_user_stories():
            column = columns.get(us.status)
            is_orphan = us.status not in known_ids
            if column is None and not is_orphan:
                # Card sits in a closed column hidden by include_closed=False —
                # skip it before touching the assignee (avoids a wasted
                # get_user fallback for a card that won't appear).
                continue

            assignee = None
            if us.assigned_to:
                assignee = member_names.get(us.assigned_to)
                if assignee is None:
                    # get_user is TTL-cached and returns an {"error"...} dict
                    # on failure — treat any non-username result as unassigned
                    # rather than crashing the whole board render.
                    user = get_user(us.assigned_to)
                    assignee = user.get("username") if isinstance(user, dict) else None
            card = {
                "ref": us.ref,
                "subject": us.subject,
                "assigned_to": assignee,
                "kanban_order": us.kanban_order,
            }
            if column is not None:
                column["cards"].append(card)
            else:
                # Orphan: status id matches no column at all (only reachable in
                # a rename/delete cache race) — surface it, never drop it.
                orphans.append({**card, "status_id": us.status})

        for column in columns.values():
            column["cards"].sort(key=lambda c: (c["kanban_order"] is None, c["kanban_order"] or 0))

        result = {"project": project.name, "columns": list(columns.values())}
        if orphans:
            result["orphan_cards"] = orphans
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "code": 500}, indent=2)


@tool(parse_docstring=True)
def get_entity_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    include_history: bool = True,
) -> str:
    """
    Retrieve any Taiga entity (task/userstory/issue/epic) by its visible reference number.
    Use when:
      - A direct URL to an entity is provided.
      - Verifying existence of specific items.
      - Looking up details before modifications.

    Args:
        project_slug (str): Project identifier.
        entity_ref (int): Visible reference number (not the database ID).
        entity_type (str): 'task', 'userstory', 'issue', or 'epic'.
        include_history (bool): Whether to fetch and embed the change log.
            Default True. History dominates this payload — measured at
            84-97% of the returned JSON on real tickets, up to ~190 KB for
            a single busy story — so pass False whenever the answer only
            needs the entity's current state. Doing so also skips one API
            round-trip per call. When False the ``history`` key is absent
            rather than empty, so a caller cannot mistake "not fetched"
            for "nothing ever happened".

    Returns:
        JSON structure with entity details, for example:
        {
            "project": "Project Name",
            "project_slug": "project-slug",
            "type": "task",
            "ref": 123,
            "status": "Status Name",
            "subject": "Entity subject",
            "description": "Entity description",
            "due_date": "2022-12-31",
            "url": "http://TAIGA_URL/project/project-slug/task/123",
            "owner": {"id": 5, "username": "jdoe", "full_name": "Jane Doe"},
            "related": {
                "comments": 3,
                "tasks": [
                    {
                        "ref": 1234,
                        "subject": "Task subject",
                        "status": "Status Name"
                    },
                    ...
                ]
            },
            "history": [
                {
                    "id": "ad932dcc-…",
                    "created_at": "2025-04-19T09:35:49.276Z",
                    "type": 1,
                    "comment": "Updated description",
                    "diff": { "description": ["", "Updated description"] },
                },
                ...
            ],
            // For ``entity_type='userstory'`` only — per-role story
            // points, role-name keys mapping to numeric values,
            // symmetric to ``set_userstory_points_tool``'s input.
            // Empty dict when nothing is set; roles assigned to "?"
            // (unestimated) or stale role-ids are silently dropped.
            "points": {"Developer": 5, "UX": 2}
        }
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Entity type '{entity_type}' is not supported.", "code": 400},
            indent=2,
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Error fetching {norm_type} {entity_ref}: {str(e)}",
                "code": 500,
            },
            indent=2,
        )

    if not entity:
        return json.dumps(
            {
                "error": f"{entity_type} {entity_ref} not found in {project_slug}",
                "code": 404,
            },
            indent=2,
        )

    # Retrieve status name (or fallback to "Unknown")
    status_info = get_status(project_slug, norm_type, entity.status)
    status_name = status_info.get("name", "Unknown") if status_info else "Unknown"

    # Get custom attributes with formatted output
    custom_attributes = get_formatted_custom_attributes(entity, project, norm_type)

    result = {
        "project": project.name,
        "project_slug": project.slug,
        "type": norm_type,
        "ref": entity.ref,
        "status": status_name,
        "subject": entity.subject,
        "description": entity.description,
        "due_date": getattr(entity, "due_date", None),
        "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity.ref}",
        "custom_attributes": custom_attributes,
        "related": {},
        # Flat names, not Taiga's [name, color] read shape: this is the
        # exact list ``manage_tags_by_ref_tool`` takes as input, so a
        # caller can round-trip what it reads here without translating.
        # The colour is a project-level attribute (``tags_colors``) that
        # no tool in this package edits.
        "tags": _normalize_tag_names(getattr(entity, "tags", None)),
    }

    # Omitted entirely rather than set to [] when not requested: an empty
    # list is a real answer here (Taiga writes no history entry for
    # creation, so a never-edited ticket genuinely has none), and conflating
    # the two would let a caller read "not fetched" as "nothing happened".
    if include_history:
        result["history"] = fetch_history(entity, norm_type)

    # Add milestone/sprint info for userstories
    entity_milestone = getattr(entity, "milestone", None)
    if entity_milestone:
        milestones = list_milestones(project_slug)
        milestone_info = next((m for m in milestones if m["id"] == entity_milestone), None)
        result["milestone"] = milestone_info if milestone_info else {"id": entity_milestone}
    else:
        result["milestone"] = None

    assigned_to = entity.assigned_to
    if assigned_to:
        assigned_to = get_user(assigned_to)
    result["assigned_to"] = assigned_to

    # Who filed it, as opposed to who is working on it. Read straight off
    # the payload Taiga already sent, so this costs no extra request.
    result["owner"] = _owner_summary(entity)

    watchers = entity.watchers
    if watchers:
        watchers = [get_user(w) for w in watchers]
    result["watchers"] = watchers

    # For userstories, include the count of related tasks AND the
    # per-role story points (symmetric to set_userstory_points_tool's
    # input shape: {"Developer": 5, "UX": 2}).
    if norm_type == "us":
        result["related"]["tasks"] = [
            {
                **task.to_dict(),
                "ref": task.ref,
                "status": get_status(project_slug, "task", task.status).get("name", "Unknown"),
                # to_dict() hands back python-taiga's raw field, so without
                # this the same response carries two different shapes under
                # the same key: flat names at the top level, [name, color]
                # pairs for each related task.
                "tags": _normalize_tag_names(getattr(task, "tags", None)),
            }
            for task in entity.list_tasks()
        ]
        result["points"] = _format_userstory_points(entity, project)
    if norm_type == "task":
        result["user_story_extra_info"] = entity.user_story_extra_info
    if norm_type == "epic":
        # Add epic-specific fields
        result["color"] = getattr(entity, "color", None)
        result["is_closed"] = getattr(entity, "is_closed", False)
        # Get related user stories for this epic
        try:
            related_us = entity.list_user_stories()
            result["related"]["user_stories"] = [
                {
                    "ref": us.ref,
                    "subject": us.subject,
                    "status": get_status(project_slug, "us", us.status).get("name", "Unknown"),
                }
                for us in related_us
            ]
        except Exception:
            result["related"]["user_stories"] = []

    return json.dumps(result, indent=2)


@tool(parse_docstring=True)
def update_entity_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    assign_to: Optional[str] = None,
    status: Optional[str] = None,
    due_date: Optional[str] = None,
    epic_ref: Optional[int] = None,
    milestone: Optional[str] = None,
) -> str:
    """
    Update a Taiga entity (task/userstory/issue/epic) by its visible reference number.
    Use when:
      - Specific fields of an entity need to be modified (e.g., status, assignee, description).
      - Linking a user story to an epic.

    Args:
        project_slug (str): Project identifier.
        entity_ref (int): Visible reference number (not the database ID).
        entity_type (str): 'task', 'userstory', 'issue', or 'epic'.
        subject (str): New title/subject for the entity.
        description (str): New description for the entity.
        assign_to (str): Username of the user to assign the entity to.
        status (str): New status for the entity.
        due_date (str): New due date for the entity (Format YYYY-MM-DD).
        epic_ref (int): Epic reference number to link a user story to (userstory only).
        milestone (str): Sprint to move it into — the sprint name, or 'current'
            for the sprint covering today. Pass an empty string to take it out
            of its sprint. User stories and issues only; a task's sprint follows
            its user story and epics have none.

    Returns:
        A JSON message indicating success or an error message.
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Entity type '{entity_type}' is not supported.", "code": 400},
            indent=2,
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Error fetching {norm_type} {entity_ref}: {str(e)}",
                "code": 500,
            },
            indent=2,
        )

    if not entity:
        return json.dumps(
            {
                "error": f"{entity_type} {entity_ref} not found in {project_slug}",
                "code": 404,
            },
            indent=2,
        )

    updates = {}
    if subject:
        updates["subject"] = subject

    if status:
        status_ids = find_status_ids(project_slug, entity_type, status)
        if not status_ids:
            return json.dumps({"error": f"Status '{status}' not found", "code": 404}, indent=2)
        updates["status"] = status_ids[0]

    if description:
        updates["description"] = description

    if assign_to:
        user = find_users(project_slug, assign_to)
        if not user:
            return json.dumps({"error": f"User '{assign_to}' not found", "code": 404}, indent=2)
        updates["assigned_to"] = user[0]["id"]

    if due_date:
        updates["due_date"] = due_date

    # Sprint membership. Checked for ``is not None`` rather than truthiness
    # because an empty string is the documented way to pull something OUT of
    # its sprint — a truthiness test would silently ignore exactly that.
    if milestone is not None:
        milestone_update, err = _milestone_update_for(
            norm_type, project_slug, milestone
        )
        if err:
            return json.dumps(err, indent=2)
        updates.update(milestone_update)

    # Link user story to epic using Taiga's related_userstories endpoint
    epic_link_result = None
    if epic_ref is not None and norm_type == "us":
        epic = project.get_epic_by_ref(epic_ref)
        if not epic:
            return json.dumps({"error": f"Epic {epic_ref} not found", "code": 404}, indent=2)
        # Use the Taiga API's related_userstories endpoint
        try:
            api = get_taiga_api(token=_current_taiga_jwt())
            api.raw_request.post(
                "/{endpoint}/{epic_id}/related_userstories",
                endpoint="epics",
                epic_id=epic.id,
                payload={"epic": epic.id, "user_story": entity.id},
            )
            epic_link_result = f"User story {entity_ref} linked to epic {epic_ref}."
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Error linking user story to epic: {str(e)}",
                    "code": 500,
                },
                indent=2,
            )

    # Apply other updates if any
    if updates:
        try:
            # Scoped PATCH of only the changed fields (+ version for the
            # OCC check) instead of update()'s full PUT of to_dict(): a
            # targeted field edit must not re-send/reset every other
            # allowed field of the fetched entity. patch() does NOT
            # auto-include version, so it is listed explicitly (see
            # AGENTS.md python-taiga gotchas).
            entity.patch(["version"], **updates)
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Error updating {norm_type} {entity_ref}: {str(e)}",
                    "code": 500,
                },
                indent=2,
            )

    message = f"{norm_type.capitalize()} {entity_ref} updated successfully."
    if epic_link_result:
        message += f" {epic_link_result}"
    return json.dumps({"message": message}, indent=2)


_VALID_WATCHER_MODES = ("add", "replace", "remove")


def _resolve_watcher_ids(members: List[Any], identifiers: List[str]):
    """Resolve watcher identifiers against a project's member list.

    Deterministic and LLM-free (unlike ``find_users``): each identifier
    is matched case-insensitively, in priority order, as a numeric member
    id, then an exact ``username``, then an exact ``full_name``.

    Args:
        members: The project's member objects (``.id``/``.username``/``.full_name``).
        identifiers: Usernames, full names, or numeric-id strings to resolve.

    Returns:
        Tuple ``(resolved_ids, unresolved, ambiguous)`` where ``resolved_ids``
        is the de-duplicated list of member ids in first-seen order,
        ``unresolved`` lists identifiers that matched no member, and
        ``ambiguous`` lists identifiers that matched more than one member.
    """
    resolved_ids: List[int] = []
    unresolved: List[str] = []
    ambiguous: List[str] = []
    for ident in identifiers:
        key = str(ident).strip()
        if not key:
            continue
        key_low = key.lower()
        matches = []
        if key.isdigit():
            wanted = int(key)
            matches = [m for m in members if m.id == wanted]
        if not matches:
            matches = [m for m in members if (getattr(m, "username", None) or "").lower() == key_low]
        if not matches:
            matches = [m for m in members if (getattr(m, "full_name", None) or "").lower() == key_low]
        unique_ids = list(dict.fromkeys(m.id for m in matches))
        if not unique_ids:
            unresolved.append(key)
        elif len(unique_ids) > 1:
            ambiguous.append(key)
        elif unique_ids[0] not in resolved_ids:
            resolved_ids.append(unique_ids[0])
    return resolved_ids, unresolved, ambiguous


@tool(parse_docstring=True)
def manage_watchers_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    watchers: List[str],
    mode: str = "add",
) -> str:
    """Add, replace, or remove watchers on a Taiga entity by its visible reference number.

    Use when a user wants to change who watches (follows) a task, user
    story, issue, or epic. Watcher identifiers are resolved against the
    project's members, so pass usernames, full names, or numeric user
    ids. Resolution is exact and case-insensitive (no fuzzy matching).

    Args:
        project_slug (str): Project identifier.
        entity_ref (int): Visible reference number (not the database ID).
        entity_type (str): One of 'task', 'userstory', 'issue', or 'epic'.
        watchers (List[str]): Usernames, full names, or numeric user ids to apply. May be empty only when mode is 'replace' (clears all watchers).
        mode (str): 'add' merges the given watchers into the existing set, 'replace' sets the watchers to exactly the given users, 'remove' drops the given users from the existing set. Defaults to 'add'.

    Returns:
        A JSON message with the resulting watcher list, or an error message.
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Entity type '{entity_type}' is not supported.", "code": 400},
            indent=2,
        )

    mode_norm = (mode or "").strip().lower()
    if mode_norm not in _VALID_WATCHER_MODES:
        return json.dumps(
            {
                "error": f"Mode '{mode}' is not supported. Use one of {list(_VALID_WATCHER_MODES)}.",
                "code": 400,
            },
            indent=2,
        )

    identifiers = [str(w).strip() for w in (watchers or []) if str(w).strip()]
    if not identifiers and mode_norm != "replace":
        return json.dumps(
            {"error": f"Mode '{mode_norm}' requires at least one watcher.", "code": 400},
            indent=2,
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {"error": f"Error fetching {norm_type} {entity_ref}: {str(e)}", "code": 500},
            indent=2,
        )

    if not entity:
        return json.dumps(
            {"error": f"{entity_type} {entity_ref} not found in {project_slug}", "code": 404},
            indent=2,
        )

    target_ids, unresolved, ambiguous = _resolve_watcher_ids(project.members, identifiers)
    if unresolved or ambiguous:
        return json.dumps(
            {
                "error": "Could not resolve some watchers against the project members.",
                "unresolved": unresolved,
                "ambiguous": ambiguous,
                "code": 404,
            },
            indent=2,
        )

    current_ids = list(entity.watchers or [])
    current_set = set(current_ids)
    if mode_norm == "add":
        result_ids = current_ids + [i for i in target_ids if i not in current_set]
    elif mode_norm == "remove":
        remove_set = set(target_ids)
        result_ids = [i for i in current_ids if i not in remove_set]
    else:  # replace
        result_ids = list(dict.fromkeys(target_ids))

    if set(result_ids) == current_set:
        return json.dumps(
            {
                "message": f"No watcher change on {norm_type} {entity_ref} (mode={mode_norm}).",
                "watchers": [get_user(w) for w in current_ids],
            },
            indent=2,
        )

    try:
        # Scoped PATCH of only the watchers field (+ version for the OCC
        # check) rather than update()'s full PUT of to_dict() — a
        # watcher change must not re-send/reset every other allowed field
        # of the fetched entity. patch() does NOT auto-include version, so
        # it is listed explicitly (see AGENTS.md python-taiga gotchas).
        entity.patch(["version"], watchers=result_ids)
    except Exception as e:
        return json.dumps(
            {"error": f"Error updating watchers on {norm_type} {entity_ref}: {str(e)}", "code": 500},
            indent=2,
        )

    return json.dumps(
        {
            "message": f"Watchers updated on {norm_type} {entity_ref} (mode={mode_norm}).",
            "watchers": [get_user(w) for w in result_ids],
        },
        indent=2,
    )


_VALID_TAG_MODES = ("add", "replace", "remove")


@tool(parse_docstring=True)
def manage_tags_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    tags: List[str],
    mode: str = "add",
) -> str:
    """Add, replace, or remove tags on a Taiga entity by its visible reference number.

    Use when a user wants to change the tags (labels) on a task, user
    story, issue, or epic. The default mode is 'add', which merges into
    whatever tags the entity already carries — prefer it over 'replace'
    unless the user explicitly wants the tag list overwritten, because
    'replace' drops every tag not listed in this call.

    Tag matching is case-insensitive and the spelling already known to
    Taiga wins, so adding 'Voice' to a project that already uses 'voice'
    reuses the existing tag rather than minting a second one or renaming
    it. A tag the project has never seen is created implicitly by Taiga;
    those are listed back in 'created_tags' so a typo is visible instead
    of silently becoming a permanent project tag. A 'created_tags' of
    null means the project's tag list could not be read, so nothing was
    verified — that is different from an empty list, which means nothing
    new was created.

    Args:
        project_slug (str): Project identifier.
        entity_ref (int): Visible reference number (not the database ID).
        entity_type (str): One of 'task', 'userstory', 'issue', or 'epic'.
        tags (List[str]): Tag names to apply. May be empty only when mode is 'replace' (clears all tags).
        mode (str): 'add' merges the given tags into the existing ones, 'replace' sets the tags to exactly the given list, 'remove' drops the given tags from the existing ones. Defaults to 'add'.

    Returns:
        A JSON message with the resulting tag list and any newly created
        tags, or an error message.
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Entity type '{entity_type}' is not supported.", "code": 400},
            indent=2,
        )

    mode_norm = (mode or "").strip().lower()
    if mode_norm not in _VALID_TAG_MODES:
        return json.dumps(
            {
                "error": f"Mode '{mode}' is not supported. Use one of {list(_VALID_TAG_MODES)}.",
                "code": 400,
            },
            indent=2,
        )

    requested = _normalize_tag_names(tags)
    if not requested and mode_norm != "replace":
        return json.dumps(
            {"error": f"Mode '{mode_norm}' requires at least one tag.", "code": 400},
            indent=2,
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {"error": f"Error fetching {norm_type} {entity_ref}: {str(e)}", "code": 500},
            indent=2,
        )

    if not entity:
        return json.dumps(
            {"error": f"{entity_type} {entity_ref} not found in {project_slug}", "code": 404},
            indent=2,
        )

    current = _normalize_tag_names(getattr(entity, "tags", None))
    current_keys = {name.lower() for name in current}

    # The project-level tag registry, read BEFORE the write. Taiga creates
    # an unknown tag implicitly as part of the very save below, so reading
    # afterwards would always find it already there and report nothing —
    # the whole point of ``created_tags`` is to catch a typo before it
    # becomes permanent. It also supplies the canonical spelling for tags
    # the project knows but this entity doesn't carry yet, so adding
    # 'Voice' to a project that already has 'voice' doesn't mint a second
    # tag. ``None`` means "couldn't check": informational only, never let
    # it cost the caller an edit. 'remove' can only shrink the list, so it
    # needs neither and skips the (cached) call.
    registry: Optional[List[str]] = None
    if mode_norm != "remove":
        try:
            registry = list_all_tags(project_slug)
        except Exception:
            registry = None

    # Case-insensitive index onto the spelling already in Taiga, with what
    # is stored on the entity winning over the project registry. Add and
    # replace resolve every requested tag through this, so an edit never
    # renames a tag as a side effect of the caller's capitalisation.
    entity_spelling: Dict[str, str] = {}
    for name in current:
        # setdefault, not assignment: an entity from before this tool can
        # carry case-variant duplicates, and the first one listed should
        # decide the survivor rather than whichever happens to come last.
        entity_spelling.setdefault(name.lower(), name)
    # What is actually stored on the entity beats what the project registry
    # calls it; the registry only fills in tags this entity doesn't carry.
    spelling = {str(name).lower(): str(name) for name in (registry or [])}
    spelling.update(entity_spelling)

    if mode_norm == "remove":
        drop = {name.lower() for name in requested}
        result = [name for name in current if name.lower() not in drop]
    else:
        # 'add' keeps what is already on the entity, 'replace' starts from
        # nothing; both then fold the request in, skipping any tag already
        # present under a different capitalisation.
        result = list(current) if mode_norm == "add" else []
        seen = set(current_keys) if mode_norm == "add" else set()
        for name in requested:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                result.append(spelling.get(key, name))

    # Compared as ordered lists, not sets: an entity that predates this tool
    # can carry case-variant duplicates ('voice' AND 'Voice'), which collapse
    # to one key and would make every replace look like a no-op — leaving the
    # duplicate uncleanable. Lists also honour the explicit ordering a caller
    # asks for in replace mode.
    if [name.lower() for name in result] == [name.lower() for name in current]:
        return json.dumps(
            {
                "message": f"No tag change on {norm_type} {entity_ref} (mode={mode_norm}).",
                "tags": current,
                # Always present, same as the success path below. A no-op
                # cannot create anything, so [] is the honest answer rather
                # than an omission the caller has to interpret.
                "created_tags": [],
            },
            indent=2,
        )

    try:
        # Scoped PATCH of only the tags field (+ version for the OCC
        # check) rather than update()'s full PUT of to_dict() — a tag
        # change must not re-send/reset every other allowed field of the
        # fetched entity. patch() does NOT auto-include version, so it is
        # listed explicitly (see AGENTS.md python-taiga gotchas).
        entity.patch(["version"], tags=result)
    except Exception as e:
        return json.dumps(
            {"error": f"Error updating tags on {norm_type} {entity_ref}: {str(e)}", "code": 500},
            indent=2,
        )

    # Which of the newly-attached tags did the project not know yet?
    # The key is always present: ``null`` specifically means "the registry
    # could not be read", which a caller must be able to tell apart from
    # "nothing new was created".
    added = [name for name in result if name.lower() not in current_keys]
    if not added:
        created_tags: Optional[List[str]] = []
    elif registry is None:
        created_tags = None
    else:
        known = {str(name).lower() for name in registry}
        created_tags = [name for name in added if name.lower() not in known]

    # Only a tag that is new to the PROJECT changes the registry; attaching
    # one the project already knows leaves it untouched and the cache valid.
    # ``None`` means the check couldn't run, so drop the entry rather than
    # risk serving a stale one — the cost of being wrong is one re-fetch.
    if created_tags is None or created_tags:
        _invalidate_tag_cache(project_slug)

    return json.dumps(
        {
            "message": f"Tags updated on {norm_type} {entity_ref} (mode={mode_norm}).",
            "tags": result,
            "created_tags": created_tags,
        },
        indent=2,
    )


@tool(parse_docstring=True)
def add_comment_by_ref_tool(project_slug: str, entity_ref: int, entity_type: str, comment: str) -> str:
    """
    Add comment to any Taiga entity using its visible reference. Use when:
    - User provides direct URL to an item
    - Need to document decisions on specific tasks/issues/userstories/epics
    - Providing status updates via comments

    Args:
        project_slug: From URL path (e.g. 'development')
        entity_ref: Visible number in entity URL
        entity_type: 'task', 'userstory', 'issue', or 'epic'
        comment: Text to add (supports Markdown)

    Returns:
        JSON structure: {
            "added": bool,
            "project": str,
            "type": str,
            "ref": int,
            "url": str,
            "comment_preview": str
        }

    Examples:
        add_comment_by_ref("mobile-app", 1421, "task", "QA verified fix")
        add_comment_by_ref("docs", 887, "userstory", "UX review completed")
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps({"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2)

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps({"error": f"Error fetching entity: {str(e)}", "code": 500}, indent=2)

    if not entity:
        return json.dumps({"error": f"{entity_type} {entity_ref} not found", "code": 404}, indent=2)

    try:
        entity.add_comment(comment)
    except Exception as e:
        return json.dumps({"error": f"Comment failed: {str(e)}", "code": 500}, indent=2)

    return json.dumps(
        {
            "added": True,
            "project": project.name,
            "type": norm_type,
            "ref": entity_ref,
            "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
            "comment_preview": (f"{comment[:50]}..." if len(comment) > 50 else comment),
        },
        indent=2,
    )


def _attachment_envelope(
    *,
    project_name: str,
    project_slug: str,
    norm_type: str,
    entity_ref: int,
    attachment: Any,
) -> dict:
    """Build the response every attachment-add path returns.

    All three paths — URL download, inline base64, and out-of-band upload —
    answer with the same shape, so a caller handling one handles all of them
    without a branch. Shared rather than copied because a future field would
    otherwise have to be added in three places.

    ``attachment.url`` is stripped: Taiga's signed URL tokens expire after
    roughly 6 minutes, so a URL captured at upload time is stale by the time
    anyone follows it. ``list_attachments_by_ref_tool`` re-mints a fresh one
    on demand, which is the supported way to get a working link.
    """
    att_dict = attachment.to_dict()
    att_dict.pop("url", None)
    return {
        "added": True,
        "project": project_name,
        "type": norm_type,
        "ref": entity_ref,
        "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
        "attachments": att_dict,
    }

@tool(parse_docstring=True)
def add_attachment_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    attachment_url: str,
    content_type: str,
    description: str = "",
) -> str:
    """
    Add attachment (images and other files) to any Taiga entity using its visible reference. Use when:
    - User provides direct URL to an item
    - Need to share screenshots, logs, or other files
    - Providing additional context to tasks/issues/userstories/epics

    Args:
        project_slug: From URL path (e.g. 'development')
        entity_ref: Visible number in entity URL
        entity_type: 'task', 'userstory', 'issue', or 'epic'
        attachment_url: Attachment URL to add
        content_type: Content type of the attachment (e.g. 'image/png', 'application/pdf')
        description: Description of the attachment (optional)

    Returns:
        JSON structure: {
            "added": bool,
            "project": str,
            "type": str,
            "ref": int,
            "url": str,
            "attachments": dict
        }

    Examples:
        add_attachment_by_ref_tool("mobile-app", 1421, "task", "http://www.xyz.com/screenshot.png", "image/png")
        add_attachment_by_ref_tool("docs", 887, "userstory", "http://www.xyz.com/specs.pdf", "application/pdf")
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps({"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2)

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps({"error": f"Error fetching entity: {str(e)}", "code": 500}, indent=2)

    if not entity:
        return json.dumps({"error": f"{entity_type} {entity_ref} not found", "code": 404}, indent=2)

    temp_file_path = None
    try:
        # converts response headers mime type to an extension (may not work with everything)
        ext = content_type.split("/")[-1]
        r = requests.get(attachment_url, stream=True, timeout=60)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_file:
            for chunk in r.iter_content(1024):  # iterate on stream using 1KB packets
                tmp_file.write(chunk)
            temp_file_path = tmp_file.name
        attachment = entity.attach(temp_file_path, description=description)
        # entity.add_comment(truncated_comment)
    except Exception as e:
        return json.dumps({"error": f"Comment failed: {str(e)}", "code": 500}, indent=2)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return json.dumps(
        _attachment_envelope(
            project_name=project.name,
            project_slug=project_slug,
            norm_type=norm_type,
            entity_ref=entity_ref,
            attachment=attachment,
        ),
        indent=2,
    )


@tool(parse_docstring=True)
def add_attachment_inline_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    attachment_filename: str,
    attachment_content_base64: str,
    description: str = "",
) -> str:
    """
    Upload a LOCAL file as an attachment to a Taiga entity by inlining
    its bytes as base64. Symmetric counterpart to
    ``get_attachment_by_ref_tool``: where that one returns inline-base64
    DOWN, this one accepts inline-base64 UP.

    Use when:
      - You have a file on the calling client (Claude Code, claude.ai)
        and need to attach it to a Taiga ticket without exposing it via
        a public URL or a paste service.
      - You produced a file in-session (analysis output, screenshot,
        diff dump, log capture) and need to ship it to a ticket in
        one tool call.

    The Taiga-side ``content_type`` is determined by the filename
    extension (Taiga sniffs from the uploaded file's name). Pass the
    desired extension via ``attachment_filename`` (e.g. ``foo.md``,
    ``screenshot.png``) — there is no separate MIME-type parameter,
    because ``python-taiga``'s multipart upload does not expose one.

    Refuses payloads whose decoded size exceeds
    ``TAIGA_MAX_INLINE_ATTACHMENT_BYTES`` (default 10 MB). Above that
    threshold, host the file externally and use
    ``add_attachment_by_ref_tool`` with the resulting URL.

    Args:
        project_slug: From URL path (e.g. 'shikenso-development').
        entity_ref: Visible number in entity URL (e.g. 7398).
        entity_type: 'task', 'userstory', 'issue', or 'epic'.
        attachment_filename: File name to display in Taiga (e.g.
            'handover.md'). Path components are stripped — only the
            basename reaches Taiga. Both POSIX and Windows separators
            are stripped (a Windows client passing a backslash path
            still uploads as the bare filename).
        attachment_content_base64: File bytes, base64-encoded (standard
            alphabet, padded). Plain text files should be encoded the
            same way — no URL-safe variant.
        description: Optional attachment description shown in Taiga.

    Returns:
        JSON envelope mirroring ``add_attachment_by_ref_tool``:
        ``{added, project, type, ref, url, attachments}`` where
        ``attachments`` is the python-taiga attachment dict
        (id, name, size, description, …) with the signed-url field
        stripped. Errors: 400 (entity_type / empty filename / invalid
        base64 / empty payload), 404 (project/entity not found), 413
        (decoded size > cap), 500 (Taiga upload failed).

    Examples:
        add_attachment_inline_by_ref_tool("shikenso-development", 7398,
            "issue", "log.txt", "aGVsbG8=")
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Invalid entity type '{entity_type}'", "code": 400},
            indent=2,
        )

    # Strip any path components so a caller can't smuggle "../etc/passwd"
    # or weird path-shaped names into Taiga's attachment display name, and
    # reject names that resolve to nothing usable. Shared with
    # ``create_attachment_upload_by_ref_tool`` — see
    # ``_safe_attachment_basename`` for why ``PureWindowsPath`` and why the
    # dot-segments are a 400 rather than a later 500.
    safe_filename = _safe_attachment_basename(attachment_filename)
    if not safe_filename:
        return json.dumps(
            {
                "error": (
                    "attachment_filename must resolve to a non-empty, "
                    "non-dot basename (path components are not allowed)"
                ),
                "code": 400,
            },
            indent=2,
        )

    # Two-stage size guard.
    #
    # Stage 1: cheap O(1) raw-length ceiling on the input. The
    # whitespace-stripping ``str.translate`` call in stage 2 still
    # allocates O(input) extra memory (one new string of length ≤
    # original). For wrapped GNU-base64 input of a 10 MB file
    # (~13.5 MB raw chars including newlines), the cleaned form is
    # well under the 10 MB cap — but we MUST refuse arbitrarily-large
    # raw inputs BEFORE that allocation, or a 1 GB whitespace-padded
    # payload would still OOM the worker.
    #
    # The threshold is ``cap * 1.5`` worth of raw chars (i.e. raw
    # upper-bound > cap * 1.5). That window is wide enough to absorb
    # all common base64 wrapping styles (GNU's 76-char wrap is ~1.3%
    # overhead; MIME 7-bit transfer ~3%; we allow up to 50% as buffer)
    # AND narrow enough that the subsequent translate allocation is
    # bounded by ~2 × cap of memory.
    raw_upper_bound = len(attachment_content_base64) * 3 // 4
    if raw_upper_bound > MAX_INLINE_ATTACHMENT_BYTES * 3 // 2:
        return json.dumps(
            {
                "error": (
                    f"Payload exceeds TAIGA_MAX_INLINE_ATTACHMENT_BYTES="
                    f"{MAX_INLINE_ATTACHMENT_BYTES} (raw upper-bound "
                    f"estimate {raw_upper_bound} bytes). Host the file "
                    f"externally and use add_attachment_by_ref_tool with "
                    f"the URL."
                ),
                "code": 413,
                "size": raw_upper_bound,
                "max_bytes": MAX_INLINE_ATTACHMENT_BYTES,
            },
            indent=2,
        )

    # Stage 2: strip ASCII whitespace and refine the bound. GNU
    # ``base64`` and many MIME tools wrap encoded output at 76 chars
    # with newlines, and a single trailing ``\n`` is common in pasted
    # payloads. With ``validate=True`` ``b64decode`` would reject these
    # — that's a usability footgun for a "upload local file" tool.
    # Stripping the whitespace:
    #   - tightens the upper-bound estimate (avoids counting newlines
    #     against the cap)
    #   - keeps ``validate=True`` strict against actual invalid chars
    #   - matches the de-facto base64 contract that "newlines may
    #     appear in the encoded data" (RFC 4648 §3.3).
    #
    # Uses ``str.translate`` (single allocation of length ≤ input) NOT
    # ``str.split() + "".join()`` — the latter materializes a list of
    # substring objects (~50 bytes header each in CPython) for inputs
    # like ``"A\nA\nA..."``, which can amplify a 20 MB input into
    # hundreds of MB of resident heap and OOM the worker even though
    # the cleaned form is small.
    #
    # The cleaned upper-bound check uses ``> MAX + 3`` (3-byte slack
    # absorbs padding + alignment overshoot of the loose formula) so
    # exactly-at-cap valid payloads still go through — the post-decode
    # ``>`` check below catches anything genuinely over.
    b64_clean = attachment_content_base64.translate(_INLINE_B64_WHITESPACE_DELETE)
    b64_upper_bound = len(b64_clean) * 3 // 4
    if b64_upper_bound > MAX_INLINE_ATTACHMENT_BYTES + 3:
        return json.dumps(
            {
                "error": (
                    f"Payload exceeds TAIGA_MAX_INLINE_ATTACHMENT_BYTES="
                    f"{MAX_INLINE_ATTACHMENT_BYTES} (upper-bound estimate "
                    f"{b64_upper_bound} bytes). Host the file externally "
                    f"and use add_attachment_by_ref_tool with the URL."
                ),
                "code": 413,
                "size": b64_upper_bound,
                "max_bytes": MAX_INLINE_ATTACHMENT_BYTES,
            },
            indent=2,
        )

    try:
        payload = base64.b64decode(b64_clean, validate=True)
    except Exception as e:
        return json.dumps(
            {
                "error": f"attachment_content_base64 is not valid base64: {str(e)}",
                "code": 400,
            },
            indent=2,
        )

    if not payload:
        return json.dumps(
            {"error": "attachment_content_base64 decodes to zero bytes", "code": 400},
            indent=2,
        )

    # Post-decode cap — precise ``>`` check on the actual decoded
    # length. This is reachable: the stage-2 pre-decode bound allows a
    # 3-byte slack (``MAX + 3``) to absorb padding + alignment
    # overshoot of the loose ``len(b64) * 3 // 4`` formula, so valid
    # b64 whose decoded length is 1–3 bytes over the cap will slip past
    # the pre-check and land here. Kept identical in shape to
    # ``get_attachment_by_ref_tool``'s mid-stream guard so reviewers
    # don't have to reason about which tools enforce the cap which way.
    if len(payload) > MAX_INLINE_ATTACHMENT_BYTES:
        return json.dumps(
            {
                "error": (
                    f"Decoded payload is {len(payload)} bytes, exceeds "
                    f"TAIGA_MAX_INLINE_ATTACHMENT_BYTES="
                    f"{MAX_INLINE_ATTACHMENT_BYTES}."
                ),
                "code": 413,
                "size": len(payload),
                "max_bytes": MAX_INLINE_ATTACHMENT_BYTES,
            },
            indent=2,
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404},
            indent=2,
        )

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {"error": f"Error fetching entity: {str(e)}", "code": 500},
            indent=2,
        )
    if not entity:
        return json.dumps(
            {"error": f"{entity_type} {entity_ref} not found", "code": 404},
            indent=2,
        )

    # ``TemporaryDirectory`` + joined path means python-taiga's
    # ``Attachments._new_resource`` will use ``os.path.basename(file_path)``
    # — i.e. ``safe_filename`` — as the multipart ``name``, so the
    # attachment shows up in Taiga with the caller's filename instead of
    # a ``tmpXXXX`` random suffix (the failure mode of the existing
    # ``add_attachment_by_ref_tool``).
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, safe_filename)
            with open(file_path, "wb") as f:
                f.write(payload)
            attachment = entity.attach(file_path, description=description)
    except Exception as e:
        return json.dumps(
            {"error": f"Attachment upload failed: {str(e)}", "code": 500},
            indent=2,
        )

    return json.dumps(
        _attachment_envelope(
            project_name=project.name,
            project_slug=project_slug,
            norm_type=norm_type,
            entity_ref=entity_ref,
            attachment=attachment,
        ),
        indent=2,
    )


def _safe_attachment_basename(filename: str) -> Optional[str]:
    """Strip path components from a caller-supplied filename.

    ``PureWindowsPath`` treats BOTH ``/`` and ``\\`` as separators, unlike
    ``os.path.basename`` which only knows the host OS's — on the Linux MCP
    pod that would let a Windows client smuggle ``C:\\tmp\\foo.txt`` through
    as a full path. Returns ``None`` for names that resolve to nothing
    usable (empty, ``.``, ``..``), which would otherwise blow up later as
    an ``IsADirectoryError`` surfacing as a misleading 500.

    A NUL byte is rejected too: ``"\u0000"`` is legal JSON, survives
    ``PureWindowsPath``, and only blows up later at ``open()`` with
    ``ValueError: embedded null byte`` — outside the caller's error mapping,
    so it would surface as a framework 500 instead of the documented 400.

    Shared by the inline-base64 tool and the upload-ticket tool so the two
    can't drift apart on what a legal attachment name is.
    """
    if "\x00" in (filename or ""):
        return None
    safe = PureWindowsPath(filename or "").name
    if not safe or safe in {".", ".."}:
        return None
    return safe


def attach_file_for_ticket(ticket: Any, file_path: str) -> dict:
    """Attach an already-downloaded file on behalf of an upload ticket.

    Runs the blocking python-taiga calls; the HTTP route wraps this in
    ``asyncio.to_thread``. Lives here rather than in ``remote_server`` so
    all Taiga API knowledge stays in one module.

    Note this deliberately does NOT go through ``get_project``: that helper
    resolves credentials from the ambient FastMCP request context via
    ``_current_taiga_jwt()``, and an upload request carries no MCP access
    token — the ticket is the only credential. Bypassing it also skips the
    user-scoped project cache, which is correct here: a stale cache entry
    would be attributed to the wrong session.

    Raises whatever python-taiga raises; the caller maps it to a 502.
    """
    api = get_taiga_api(token=ticket.taiga_jwt)
    project = api.projects.get_by_slug(ticket.project_slug)
    entity = fetch_entity(project, ticket.entity_type, ticket.entity_ref)
    if not entity:
        raise ValueError(
            f"{ticket.entity_type} {ticket.entity_ref} not found in "
            f"'{ticket.project_slug}'"
        )
    attachment = entity.attach(file_path, description=ticket.description)
    return _attachment_envelope(
        project_name=project.name,
        project_slug=ticket.project_slug,
        norm_type=ticket.entity_type,
        entity_ref=ticket.entity_ref,
        attachment=attachment,
    )


@tool(parse_docstring=True)
def create_attachment_upload_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    filename: str,
    description: str = "",
) -> str:
    """
    Get a one-time URL for uploading a LOCAL file to a Taiga entity, so the
    file's bytes never pass through the conversation. Use when:
    - You have a file on disk (screenshot, CSV, log dump, report) and want it
      attached to a ticket.
    - The file is anything but tiny — this costs the same few hundred tokens
      whether the file is 2 KB or 20 MB.

    Returns an ``upload_url`` plus a ready-to-run ``curl`` command. Run the
    command (or POST the raw file bytes to ``upload_url`` yourself), and the
    server attaches the file. The URL is single-use, expires in
    ``expires_in`` seconds, and is already bound to this project, entity and
    filename — the upload request itself takes no parameters.

    Requires a shell on the calling client. Clients without one (claude.ai on
    the web) cannot perform the upload step; use
    ``add_attachment_by_ref_tool`` with a public URL there instead.

    Args:
        project_slug: From URL path (e.g. 'shikenso-development').
        entity_ref: Visible number in entity URL (e.g. 7398).
        entity_type: 'task', 'userstory', 'issue', or 'epic'.
        filename: Path or name of the local file. The basename becomes the
            attachment name in Taiga (Taiga sniffs the content type from the
            extension, so keep it), and the value you pass is echoed into the
            returned curl command as the source path.
        description: Optional attachment description shown in Taiga.

    Returns:
        JSON structure: {
            "upload_url": str,
            "curl": str,
            "expires_in": int,
            "max_bytes": int,
            "filename": str,
            "project": str,
            "type": str,
            "ref": int
        }
        Errors: 400 (entity_type / unusable filename), 404 (project or entity
        not found), 500 (server not running in remote HTTP mode).

    Examples:
        create_attachment_upload_by_ref_tool("shikenso-development", 7398,
            "issue", "./rca.md")
        create_attachment_upload_by_ref_tool("mobile-app", 1421, "task",
            "/tmp/screenshot.png")
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2
        )

    safe_filename = _safe_attachment_basename(filename)
    if not safe_filename:
        return json.dumps(
            {
                "error": (
                    "filename must resolve to a non-empty, non-dot basename "
                    "(path components are not allowed)"
                ),
                "code": 400,
            },
            indent=2,
        )

    # The public base URL the pod advertises. Absent in stdio mode, where
    # there is no HTTP server to upload to — fail loudly rather than hand
    # back a URL that resolves to nothing.
    base_url = (os.getenv("TAIGA_MCP_BASE_URL") or "").rstrip("/")
    if not base_url:
        return json.dumps(
            {
                "error": (
                    "TAIGA_MCP_BASE_URL is not set — upload tickets require the "
                    "remote HTTP server. Use add_attachment_by_ref_tool instead."
                ),
                "code": 500,
            },
            indent=2,
        )

    # Validate the target BEFORE minting a ticket, so a typo surfaces as a
    # precise 404 here instead of a confusing 502 after the bytes are
    # already on the wire.
    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )
    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {"error": f"Error fetching entity: {str(e)}", "code": 500}, indent=2
        )
    if not entity:
        return json.dumps(
            {"error": f"{entity_type} {entity_ref} not found", "code": 404}, indent=2
        )

    ticket = upload_tickets.issue(
        taiga_jwt=_current_taiga_jwt(),
        project_slug=project_slug,
        entity_type=norm_type,
        entity_ref=entity_ref,
        filename=safe_filename,
        description=description,
    )
    upload_url = f"{base_url}/upload/{ticket.token}"
    return json.dumps(
        {
            "upload_url": upload_url,
            "curl": (
                f"curl -sf -X POST --data-binary @{shlex.quote(filename)} "
                f"{shlex.quote(upload_url)}"
            ),
            "expires_in": int(ticket.expires_at - time.time()),
            "max_bytes": upload_tickets.MAX_UPLOAD_BYTES,
            "filename": safe_filename,
            "project": project.name,
            "type": norm_type,
            "ref": entity_ref,
        },
        indent=2,
    )


@tool(parse_docstring=True)
def list_attachments_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
) -> str:
    """
    List all attachments on a Taiga entity using its visible reference.

    Returns each attachment's id, name, size, description, content_type,
    owner, dates, and a freshly-signed ``download_url``.

    Use when:
      - You need to read or download a file attached to a ticket.
      - The token in an attachment URL from the Taiga UI / webhook diff
        has expired (HTTP 403 on click).
      - Inspecting what files exist on an entity before deciding to
        download one.

    The ``download_url`` is signed by ``taiga-protected`` with TTL ~6 min
    (360 s). Do NOT cache it, do NOT forward it across turns. If you
    actually need the file CONTENT, prefer ``get_attachment_by_ref_tool``
    — it re-mints the URL just before downloading, so the freshness
    window is closed inside the call.

    Args:
        project_slug: From URL path (e.g. 'volleyball-world-11-25').
        entity_ref: Visible number in entity URL (e.g. 7398).
        entity_type: 'task', 'userstory', 'issue', or 'epic'.

    Returns:
        JSON string with project / type / ref / url / count / attachments,
        where each attachment has id, name, size, content_type,
        description, owner, created_date, modified_date, download_url.

    Examples:
        list_attachments_by_ref_tool("volleyball-world-11-25", 7398, "issue")
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Invalid entity type '{entity_type}'", "code": 400},
            indent=2,
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404},
            indent=2,
        )

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {"error": f"Error fetching entity: {str(e)}", "code": 500},
            indent=2,
        )

    if not entity:
        return json.dumps(
            {"error": f"{entity_type} {entity_ref} not found", "code": 404},
            indent=2,
        )

    try:
        attachments = entity.list_attachments() or []
    except Exception as e:
        return json.dumps(
            {"error": f"Could not list attachments: {str(e)}", "code": 500},
            indent=2,
        )

    result = []
    for a in attachments:
        result.append(
            {
                "id": a.id,
                "name": getattr(a, "name", None),
                "size": getattr(a, "size", None),
                "description": getattr(a, "description", "") or "",
                "content_type": getattr(a, "content_type", None),
                "owner": getattr(a, "owner", None),
                "created_date": str(getattr(a, "created_date", "") or ""),
                "modified_date": str(getattr(a, "modified_date", "") or ""),
                "download_url": getattr(a, "url", None),
            }
        )

    return json.dumps(
        {
            "project": project.name,
            "type": norm_type,
            "ref": entity_ref,
            "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
            "count": len(result),
            "attachments": result,
        },
        indent=2,
    )


@tool(parse_docstring=True)
def get_attachment_by_ref_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    attachment_id: int,
) -> str:
    """
    Fetch a specific attachment from a Taiga entity and return its
    content base64-encoded inline. The download URL is re-minted
    inside this call (NOT reused from a previous list call), so the
    6-minute URL TTL is never user-visible.

    Use when:
      - You need the actual file content to parse (spreadsheet, PDF,
        image, JSON, ...).
      - The attachment was found via ``list_attachments_by_ref_tool``
        and you have its numeric ``attachment_id``.

    Refuses files larger than ``TAIGA_MAX_INLINE_ATTACHMENT_BYTES``
    (default 10 MB). For larger files, use ``list_attachments_by_ref_tool``
    to obtain a fresh signed ``download_url`` and fetch out-of-band.

    Args:
        project_slug: From URL path (e.g. 'volleyball-world-11-25').
        entity_ref: Visible number in entity URL (e.g. 7398).
        entity_type: 'task', 'userstory', 'issue', or 'epic'.
        attachment_id: Numeric attachment ID (from ``list_attachments_by_ref_tool``).

    Returns:
        JSON string with id, name, size, content_type, content_base64,
        encoding. Errors: 400 (entity_type), 404 (project/entity/attachment),
        413 (size > cap), 502 (HTTP error from taiga-protected), 500 (other).

    Examples:
        get_attachment_by_ref_tool("volleyball-world-11-25", 7398, "issue", 10334)
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Invalid entity type '{entity_type}'", "code": 400},
            indent=2,
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404},
            indent=2,
        )

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {"error": f"Error fetching entity: {str(e)}", "code": 500},
            indent=2,
        )
    if not entity:
        return json.dumps(
            {"error": f"{entity_type} {entity_ref} not found", "code": 404},
            indent=2,
        )

    # Re-fetch the attachment list so the URL token is freshly-minted
    # by taiga-protected; the alternative — caching the URL from a prior
    # list call — would race the 360 s TTL.
    try:
        attachments = entity.list_attachments() or []
    except Exception as e:
        return json.dumps(
            {"error": f"Could not list attachments: {str(e)}", "code": 500},
            indent=2,
        )

    attachment = next((a for a in attachments if a.id == attachment_id), None)
    if attachment is None:
        return json.dumps(
            {
                "error": (f"Attachment {attachment_id} not found on " f"{entity_type} {entity_ref}"),
                "code": 404,
            },
            indent=2,
        )

    name = getattr(attachment, "name", None)
    size = getattr(attachment, "size", None)
    content_type = getattr(attachment, "content_type", None)
    download_url = getattr(attachment, "url", None)

    # Pre-check size against the cap. Skip the check if ``size`` is None /
    # missing — the mid-stream cap below catches that case anyway, and
    # refusing every size-less attachment would be over-strict.
    if size is not None and size > MAX_INLINE_ATTACHMENT_BYTES:
        return json.dumps(
            {
                "error": (
                    f"Attachment is {size} bytes, exceeds "
                    f"TAIGA_MAX_INLINE_ATTACHMENT_BYTES="
                    f"{MAX_INLINE_ATTACHMENT_BYTES}. Use "
                    f"list_attachments_by_ref_tool and download the URL "
                    f"out-of-band."
                ),
                "code": 413,
                "size": size,
                "max_bytes": MAX_INLINE_ATTACHMENT_BYTES,
            },
            indent=2,
        )

    if not download_url:
        return json.dumps(
            {"error": "Attachment has no download URL", "code": 500},
            indent=2,
        )

    # Both auth paths are sent: the URL ``?token=…`` (taiga-protected's
    # signed query string) is the canonical browser path; Bearer JWT is
    # belt-and-braces in case the URL token races with expiry between
    # mint and download.
    jwt = _current_taiga_jwt()
    headers = {"Authorization": f"Bearer {jwt}"} if jwt else {}

    # The ``with`` block guarantees ``resp.close()`` on every exit path —
    # critical for the mid-stream 413 below, which would otherwise leak
    # the streaming connection / open file descriptor under repeated
    # oversized downloads. ``content_type`` must be captured INSIDE the
    # block because the response is closed by the time we return.
    resp_content_type = None
    try:
        with requests.get(download_url, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            # Mid-stream cap so a misreported ``size`` (or no size at all)
            # cannot OOM the worker. 1 KiB grace above the configured cap
            # so attachments exactly at the cap still succeed if their
            # last chunk pushes slightly past the announced size.
            cap = MAX_INLINE_ATTACHMENT_BYTES + 1024
            buf = bytearray()
            for chunk in resp.iter_content(64 * 1024):
                buf.extend(chunk)
                if len(buf) > cap:
                    return json.dumps(
                        {
                            "error": (
                                "Attachment exceeded "
                                "TAIGA_MAX_INLINE_ATTACHMENT_BYTES during "
                                "stream. Use list_attachments_by_ref_tool "
                                "and download out-of-band."
                            ),
                            "code": 413,
                            "max_bytes": MAX_INLINE_ATTACHMENT_BYTES,
                        },
                        indent=2,
                    )
            resp_content_type = resp.headers.get("Content-Type")
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        return json.dumps(
            {"error": f"Download failed: HTTP {status}", "code": 502},
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": f"Download failed: {str(e)}", "code": 500},
            indent=2,
        )

    return json.dumps(
        {
            "id": attachment.id,
            "name": name,
            "size": len(buf),
            "content_type": content_type or resp_content_type,
            "content_base64": base64.b64encode(bytes(buf)).decode("ascii"),
            "encoding": "base64",
        },
        indent=2,
    )


@tool(parse_docstring=True)
def promote_issue_to_userstory_tool(
    project_slug: str,
    issue_ref: int,
    project_id: Optional[int] = None,
) -> str:
    """
    Promote a Taiga issue to a user story using Taiga's native promote feature.
    This creates a new user story from an existing issue, preserving the link.
    Use when:
      - Converting an issue/bug report into a user story for development
      - Moving inbox items (issues) to the backlog (user stories)

    Args:
        project_slug: Project identifier (e.g. 'mobile-app')
        issue_ref: Visible issue reference number (not the database ID)
        project_id: Optional project ID for the new user story (defaults to same project)

    Returns:
        JSON structure with the newly created user story details:
        {
            "promoted": bool,
            "project": str,
            "issue_ref": int,
            "userstory": {
                "ref": int,
                "subject": str,
                "status": str,
                "url": str
            }
        }

    Examples:
        promote_issue_to_userstory_tool("mobile-app", 29)
        promote_issue_to_userstory_tool("wahed", 15, project_id=123)
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        issue = project.get_issue_by_ref(issue_ref)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Error fetching issue {issue_ref}: {str(e)}",
                "code": 500,
            },
            indent=2,
        )

    if not issue:
        return json.dumps(
            {
                "error": f"Issue {issue_ref} not found in {project_slug}",
                "code": 404,
            },
            indent=2,
        )

    try:
        api = get_taiga_api(token=_current_taiga_jwt())

        # Prepare payload - use project.id (database ID) if not specified
        payload = {"project_id": project_id if project_id else project.id}

        # Call the promote_to_user_story endpoint using issue.id (database ID)
        response = api.raw_request.post(
            "/{endpoint}/{id}/promote_to_user_story",
            endpoint="issues",
            id=issue.id,  # Database ID required for API
            payload=payload,
        )

        # The response contains a list of user story REFs (not database IDs!)
        # See: taiga-back/tests/integration/test_issues.py#L935-L953
        us_refs = response.json()

        if not us_refs:
            return json.dumps(
                {"error": "Empty response from promote endpoint", "code": 500},
                indent=2,
            )

        # Get the last ref (newest promotion) - this is the visible ref, not the DB id
        if isinstance(us_refs, list):
            new_us_ref = us_refs[-1]
        else:
            new_us_ref = us_refs

        # Fetch the user story by its ref (visible reference number)
        us = project.get_userstory_by_ref(new_us_ref)

        if us:
            us_ref = us.ref
            us_subject = us.subject
            us_status_info = getattr(us, "status_extra_info", None)
            us_status = us_status_info.get("name", "Unknown") if isinstance(us_status_info, dict) else "New"
        else:
            # Fallback: return basic info from ref
            us_ref = new_us_ref
            us_subject = issue.subject
            us_status = "New"

        return json.dumps(
            {
                "promoted": True,
                "project": project.name,
                "issue_ref": issue_ref,
                "userstory": {
                    "ref": us_ref,
                    "subject": us_subject,
                    "status": us_status,
                    "url": f"{TAIGA_URL}/project/{project_slug}/us/{us_ref}",
                },
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {
                "error": f"Error promoting issue to user story: {str(e)}",
                "code": 500,
            },
            indent=2,
        )


@tool(parse_docstring=True)
def list_custom_attributes_tool(
    project_slug: str,
    entity_type: str = "userstory",
) -> str:
    """
    List all custom attribute definitions for a project.
    Use when:
      - Need to find custom attribute IDs for setting values
      - Want to see what custom fields are available (e.g., RICE fields)
      - Documenting custom attribute configuration

    Args:
        project_slug: Project identifier (e.g. 'wahed')
        entity_type: 'userstory', 'task', 'issue', or 'epic'

    Returns:
        JSON envelope ``{project, entity_type, custom_attributes}``
        where ``custom_attributes`` is the list of definitions. Each
        entry has id, name, description, type, order, extra, choices.
        For type='dropdown', ``choices`` is the parsed list of valid
        option strings — use one of them as the value to
        ``set_custom_attributes_tool``. ``extra`` is the raw
        newline-delimited string returned by the Taiga API. For
        non-dropdown types, ``choices`` is an empty list and
        ``extra`` is null or empty.

    Examples:
        list_custom_attributes_tool("wahed", "userstory")
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity_type_lower = entity_type.lower()
        if entity_type_lower in ("userstory", "us", "user_story"):
            attrs = project.list_user_story_attributes()
        elif entity_type_lower in ("task", "tasks"):
            attrs = project.list_task_attributes()
        elif entity_type_lower in ("issue", "issues"):
            attrs = project.list_issue_attributes()
        elif entity_type_lower in ("epic", "epics"):
            attrs = project.list_epic_attributes()
        else:
            return json.dumps(
                {"error": f"Unsupported entity type: {entity_type}", "code": 400},
                indent=2,
            )

        result = []
        for attr in attrs:
            attr_type = getattr(attr, "type", "text")
            extra_raw = getattr(attr, "extra", None)
            if attr_type == "dropdown" and isinstance(extra_raw, str):
                # Taiga stores dropdown options as a newline-delimited
                # string. Normalize CRLF, trim per-line whitespace, drop
                # empties so a trailing newline (common in the admin UI)
                # doesn't produce a phantom "" choice.
                choices = [
                    line.strip()
                    for line in extra_raw.replace("\r\n", "\n").split("\n")
                    if line.strip()
                ]
            else:
                choices = []
            result.append(
                {
                    "id": attr.id,
                    "name": attr.name,
                    "description": getattr(attr, "description", ""),
                    "type": attr_type,
                    "order": getattr(attr, "order", 0),
                    "extra": extra_raw,
                    "choices": choices,
                }
            )

        return json.dumps(
            {
                "project": project.name,
                "entity_type": entity_type,
                "custom_attributes": result,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": f"Error listing custom attributes: {str(e)}", "code": 500},
            indent=2,
        )


@tool(parse_docstring=True)
def set_custom_attributes_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
    attributes: Dict[str, Any],
) -> str:
    """
    Set custom attribute values for an entity (userstory, task, issue, epic).
    Use when:
      - Setting RICE scores (Reach, Impact, Confidence, Effort)
      - Filling in any custom fields on entities
      - Updating custom metadata

    Args:
        project_slug: Project identifier (e.g. 'wahed')
        entity_ref: Visible reference number of the entity
        entity_type: 'userstory', 'task', 'issue', or 'epic'
        attributes: Dictionary mapping attribute IDs (as strings) to values

    Returns:
        JSON with updated custom attribute values

    Examples:
        set_custom_attributes_tool("wahed", 34, "userstory", {"1": 4, "2": 5})
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Unsupported entity type: {entity_type}", "code": 400},
            indent=2,
        )

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Error fetching {entity_type} {entity_ref}: {str(e)}",
                "code": 500,
            },
            indent=2,
        )

    if not entity:
        return json.dumps(
            {
                "error": f"{entity_type} {entity_ref} not found in {project_slug}",
                "code": 404,
            },
            indent=2,
        )

    try:
        # Get current version for optimistic locking
        current_attrs = entity.get_attributes()
        version = current_attrs.get("version", 1)

        # Set each attribute
        updated_values = {}
        for attr_id, value in attributes.items():
            result = entity.set_attribute(str(attr_id), value, version=version)
            # Update version for next attribute
            version = result.get("version", version)
            updated_values[attr_id] = value

        return json.dumps(
            {
                "updated": True,
                "project": project.name,
                "entity_type": entity_type,
                "ref": entity_ref,
                "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
                "attributes_set": updated_values,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": f"Error setting custom attributes: {str(e)}", "code": 500},
            indent=2,
        )


@tool(parse_docstring=True)
def get_custom_attributes_tool(
    project_slug: str,
    entity_ref: int,
    entity_type: str,
) -> str:
    """
    Get current custom attribute values for an entity.
    Use when:
      - Reading RICE scores or other custom field values
      - Checking what custom data is set on an entity
      - Debugging custom attribute issues

    Args:
        project_slug: Project identifier (e.g. 'wahed')
        entity_ref: Visible reference number of the entity
        entity_type: 'userstory', 'task', 'issue', or 'epic'

    Returns:
        JSON with custom attribute values

    Examples:
        get_custom_attributes_tool("wahed", 34, "userstory")
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Unsupported entity type: {entity_type}", "code": 400},
            indent=2,
        )

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Error fetching {entity_type} {entity_ref}: {str(e)}",
                "code": 500,
            },
            indent=2,
        )

    if not entity:
        return json.dumps(
            {
                "error": f"{entity_type} {entity_ref} not found in {project_slug}",
                "code": 404,
            },
            indent=2,
        )

    try:
        attrs = entity.get_attributes()
        return json.dumps(
            {
                "project": project.name,
                "entity_type": entity_type,
                "ref": entity_ref,
                "subject": getattr(entity, "subject", ""),
                "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
                "attributes_values": attrs.get("attributes_values", {}),
                "version": attrs.get("version", 1),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": f"Error getting custom attributes: {str(e)}", "code": 500},
            indent=2,
        )


# =============================================================================
# Urgency Calculation
# =============================================================================

STORY_POINTS_TO_EVENINGS = {
    2: 1,  # 2 SP = 1 evening
    4: 2,  # 4 SP = 2 evenings
    5: 3,  # 5 SP = 3 evenings
    6: 4,  # 6 SP = 4 evenings
    7: 6,  # 7 SP = 6 evenings
    8: 10,  # 8 SP = 10 evenings
    9: 15,  # 9 SP = 15 evenings
    10: 20,  # 10 SP = 20 evenings
}


def calculate_urgency(due_date_str: Optional[str], story_points: int = 2) -> float:
    """
    Calculate urgency multiplier based on buffer (days remaining - work evenings needed).

    Simple formula: Buffer = Days until deadline - Work evenings needed

    Aggressive multipliers ensure deadline stories rise to the top:
    - Buffer < 0:   50.0  (Impossible - must be top priority!)
    - Buffer 0-1:   25.0  (Extremely tight)
    - Buffer 2-3:   10.0  (Very tight)
    - Buffer 4-7:    5.0  (Tight)
    - Buffer 8-14:   2.0  (Soon)
    - Buffer > 14:   1.5  (Has deadline but comfortable)
    - No deadline:   1.0  (Normal)

    Args:
        due_date_str: Due date string (YYYY-MM-DD) or None
        story_points: Total story-point effort (sum across roles) for the user story

    Returns:
        Urgency multiplier
    """
    if not due_date_str:
        return 1.0

    work_evenings_needed = STORY_POINTS_TO_EVENINGS.get(story_points, 1)

    try:
        if isinstance(due_date_str, str):
            due_date = datetime.strptime(due_date_str[:10], "%Y-%m-%d").date()
        else:
            due_date = due_date_str

        today = datetime.now().date()
        days_remaining = (due_date - today).days
        buffer_days = days_remaining - work_evenings_needed

        if buffer_days < 0:
            return 50.0  # Impossible without overtime!
        elif buffer_days <= 1:
            return 25.0  # Extremely tight
        elif buffer_days <= 3:
            return 10.0  # Very tight
        elif buffer_days <= 7:
            return 5.0  # Tight
        elif buffer_days <= 14:
            return 2.0  # Soon
        else:
            return 1.5  # Has deadline but comfortable
    except (ValueError, TypeError):
        return 1.0


def calculate_completion_bonus(completion_pct: float) -> float:
    """
    Calculate a completion bonus for an epic based on how far it is done.

    Uses a quadratic curve so that nearly-finished epics get a much stronger
    boost than epics that are just getting started ("finish what you started").

    Formula: 1.0 + 0.5 × completion²

    | Completion | Bonus |
    |------------|-------|
    |   0%       | 1.00  |
    |  20%       | 1.02  |
    |  50%       | 1.13  |
    |  75%       | 1.28  |
    |  80%       | 1.32  |
    |  90%       | 1.41  |
    |  95%       | 1.45  |
    | 100%       | 1.50  |

    Args:
        completion_pct: Fraction of closed user stories (0.0–1.0)

    Returns:
        Completion bonus multiplier (1.0–1.5)
    """
    pct = max(0.0, min(1.0, completion_pct))
    return 1.0 + 0.5 * pct**2


@cached(cache=sort_attr_def_cache, key=_user_scoped_key, lock=_cache_lock)
def _discover_sort_attr_ids(project_slug: str) -> Optional[Dict[str, Any]]:
    """Discover RICE/blocked-by/multiplicator custom-attribute IDs for a project.

    These are project-config-level metadata that almost never change at
    runtime — caching for 5 min skips two GET requests
    (``list_user_story_attributes`` + ``list_epic_attributes``) on every
    repeat invocation of :func:`sort_kanban_by_rice_tool`.

    Returns ``None`` if the project itself can't be loaded; the caller
    is responsible for the 404 response in that case.
    """
    project = get_project(project_slug)
    if project is None:
        return None

    rice_attrs: Dict[str, str] = {}
    blocked_by_attr_id: Optional[str] = None
    rice_keys = {"reach", "impact", "confidence"}
    for attr in project.list_user_story_attributes():
        name = attr.name.lower()
        if name in rice_keys:
            rice_attrs[name] = str(attr.id)
        elif name == "blocked by":
            blocked_by_attr_id = str(attr.id)

    multiplicator_attr_id: Optional[str] = None
    try:
        for attr in project.list_epic_attributes():
            if attr.name.lower() == "multiplicator":
                multiplicator_attr_id = str(attr.id)
                break
    except Exception:
        # Multiplicator is optional — projects without an "epic" panel
        # raise here. Treat as "no multiplicator" rather than failing.
        pass

    return {
        "rice_attrs": rice_attrs,
        "blocked_by_attr_id": blocked_by_attr_id,
        "multiplicator_attr_id": multiplicator_attr_id,
    }


async def _fetch_us_attrs_async(
    base_url: str,
    token: str,
    stories: List[Any],
    *,
    timeout_s: float = 30.0,
    max_concurrency: int = 30,
) -> tuple:
    """Concurrent per-userstory custom-attribute fetch via httpx + asyncio.gather.

    Replaces the pre-2.3.4 ``ThreadPoolExecutor(max_workers=10)`` + per-thread
    ``us.get_attributes()`` (sync requests) approach. On a 40-story project
    this drops the per-US fetch wall time from ~25 s (4 batches × ~6 s) to
    ~3 s (one round-trip with all requests in flight) — well under the
    upstream Anthropic streaming-response timeout that was killing the
    tool with ``stream timeout`` in 2.3.3.

    Concurrency is capped at 30 because the OVH-hosted Taiga has returned
    502s under bursty 50+ parallel reads in our prod testing.

    Returns ``(values_by_us_ref, errors)`` where:
      - ``values_by_us_ref`` maps ``us.ref → {attr_id: value}``; entries
        for failed fetches default to ``{}`` (so RICE falls back to
        1×1×1 for those, same shape as the threaded version)
      - ``errors`` is a list of ``{"ref": <us-ref>, "error": "<msg>"}``
        for fetches that didn't return 2xx; surfaced unchanged in the
        tool's response so the caller sees partial-failure detail.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    limits = httpx.Limits(max_connections=max_concurrency)
    timeout = httpx.Timeout(timeout_s, connect=10.0)

    values_by_ref: Dict[int, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=base_url, headers=headers, limits=limits, timeout=timeout) as client:

        async def _one(us: Any) -> None:
            try:
                resp = await client.get(f"/api/v1/userstories/custom-attributes-values/{us.id}")
                resp.raise_for_status()
                body = resp.json()
                values_by_ref[us.ref] = body.get("attributes_values", {}) or {}
            except Exception as exc:
                values_by_ref[us.ref] = {}
                errors.append({"ref": us.ref, "error": f"{type(exc).__name__}: {exc}"})

        await asyncio.gather(*(_one(us) for us in stories))

    return values_by_ref, errors


async def _fetch_epic_multiplicators_async(
    base_url: str,
    token: str,
    epics: List[Any],
    multiplicator_attr_id: str,
    *,
    timeout_s: float = 30.0,
    max_concurrency: int = 15,
) -> Dict[int, float]:
    """Concurrent per-epic multiplicator fetch. Failures default to 1.0.

    Epic count is bounded (typically <10) so even the threaded version
    was fast — going async here is mostly for code symmetry with the
    per-US fetcher and to keep the whole pipeline on one event loop.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    limits = httpx.Limits(max_connections=max_concurrency)
    timeout = httpx.Timeout(timeout_s, connect=10.0)

    result: Dict[int, float] = {}

    async with httpx.AsyncClient(base_url=base_url, headers=headers, limits=limits, timeout=timeout) as client:

        async def _one(epic: Any) -> None:
            try:
                resp = await client.get(f"/api/v1/epics/custom-attributes-values/{epic.id}")
                resp.raise_for_status()
                body = resp.json()
                attrs = body.get("attributes_values", {}) or {}
                mult = attrs.get(multiplicator_attr_id, 1.0) or 1.0
                result[epic.id] = float(mult)
            except Exception:
                # Per-epic multiplicator is optional metadata; a fetch
                # failure shouldn't poison the sort. Default to neutral
                # (1.0) so the affected epic's stories sort by raw RICE.
                result[epic.id] = 1.0

        await asyncio.gather(*(_one(e) for e in epics))

    return result


def _resolve_taiga_api_base_url() -> str:
    """Pick the host the new async fetchers should hit.

    python-taiga's ``TaigaAPI(host=...)`` is constructed from
    ``TAIGA_API_URL`` (see :func:`_get_taiga_api_from_env` /
    :func:`get_taiga_api`), so the existing ``us.get_attributes()``
    path went through the API origin. The async refactor must do the
    same — using ``TAIGA_URL`` would break the documented split
    deployment (``tree.taiga.io`` UI / ``api.taiga.io`` API, or the
    cluster-internal-API setup we run in remote-MCP mode), where the
    UI host doesn't speak the v1 API at all.

    Falls back to ``TAIGA_URL`` when ``TAIGA_API_URL`` is unset, which
    matches the single-host Shikenso deployment shape. Raises
    ``ValueError`` when neither is set so the caller sees a clear
    config error instead of an ``AttributeError`` on ``None.rstrip``.
    """
    base = TAIGA_API_URL or TAIGA_URL
    if not base:
        raise ValueError(
            "Taiga URL is not configured: set TAIGA_API_URL "
            "(preferred) or TAIGA_URL (fallback) in the environment "
            "before invoking sort_kanban_by_rice_tool."
        )
    return base.rstrip("/")


@tool(parse_docstring=True)
def sort_kanban_by_rice_tool(
    project_slug: str,
    descending: bool = True,
) -> str:
    """
    Sort user stories in the Kanban board by their RICE score.
    RICE = (Reach × Impact × Confidence) / Effort
    (Confidence is optional — defaults to 1 when the project has no
    Confidence custom attribute; Reach and Impact are required.
    Attribute discovery is cached ~5 min, so a newly-added Confidence
    attribute may take up to that long to affect scoring.)
    Final Priority = RICE × Epic Multiplicator × Completion Bonus × Urgency Multiplier

    Closed status columns (Done, Cancelled, …) are skipped — re-ranking
    already-completed work has no value. Each entry in ``columns_updated``
    carries the resolved ``status_name`` alongside ``status_id``.

    Completion Bonus rewards nearly-finished epics ("finish what you started"):
    Formula: 1.0 + 0.5 × (closed_stories / total_stories)²
    - 0% complete: 1.00 (no bonus)
    - 50% complete: 1.13
    - 80% complete: 1.32
    - 100% complete: 1.50

    Urgency Multiplier based on Buffer (Days remaining - Work evenings needed):
    - 50.0: Buffer < 0 (Impossible without overtime!)
    - 25.0: Buffer 0-1 (Extremely tight)
    - 10.0: Buffer 2-3 (Very tight)
    - 5.0: Buffer 4-7 (Tight)
    - 2.0: Buffer 8-14 (Soon)
    - 1.5: Buffer > 14 (Has deadline but comfortable)
    - 1.0: No due date (Normal)

    Blocked stories (via "Blocked By" custom attribute) are automatically placed
    immediately below their blocker, regardless of their own RICE score.

    Use when:
      - Reordering the Kanban board after setting RICE scores
      - Weekly review to prioritize the backlog visually
      - Ensuring highest-priority items are at the top

    Args:
        project_slug: Project identifier (e.g. 'wahed')
        descending: If True, highest RICE first. If False, lowest first.

    Returns:
        JSON with sorting results per status column

    Examples:
        sort_kanban_by_rice_tool("wahed")
        sort_kanban_by_rice_tool("wahed", descending=False)
    """
    # ``asyncio.run`` is safe here because the @tool wrapper is sync and
    # FastMCP's registration layer offloads us to a worker thread via
    # ``asyncio.to_thread`` (see ``_register_mcp_tools``) — no outer
    # event loop is running on this thread. See ``_fetch_us_attrs_async``
    # for the rationale behind the 2.3.4 async migration.
    return asyncio.run(_sort_kanban_async_impl(project_slug, descending))


async def _sort_kanban_async_impl(project_slug: str, descending: bool) -> str:
    """Async body of :func:`sort_kanban_by_rice_tool`. See that tool's
    docstring for the user-facing contract; the inline ``--- N.``
    section comments below document each pipeline step.

    The bulk-update POST stays on ``requests.post`` (sync) — column
    count is bounded (<10) so async there isn't worth the test churn,
    and it reuses ``base_url``/``token`` from the async fetcher so we
    hit the same API host as the GETs.
    """
    import requests
    import traceback
    from collections import defaultdict

    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404},
            indent=2,
        )

    # Catch-all so any unexpected failure produces a JSON payload instead
    # of bubbling up as the harness-generic "Error occurred during tool
    # execution" — without this the caller saw nothing useful when the
    # FastMCP worker died or got killed by k8s liveness probe (the prod
    # symptom that prompted this refactor in 2.3.2/2.3.3).
    try:
        # --- 1. Discover relevant custom-attribute IDs (cached 5 min).
        attr_defs = _discover_sort_attr_ids(project_slug)
        if attr_defs is None:
            # Should be unreachable since the project lookup above
            # already returned; defensive in case of a cache-key drift.
            return json.dumps(
                {"error": f"Project '{project_slug}' not found", "code": 404},
                indent=2,
            )
        rice_attrs = attr_defs["rice_attrs"]
        blocked_by_attr_id = attr_defs["blocked_by_attr_id"]
        multiplicator_attr_id = attr_defs["multiplicator_attr_id"]

        # Reach + Impact are mandatory. Confidence is OPTIONAL and defaults
        # to 1 in the RICE product when the board has no Confidence custom
        # attribute (the shikenso-development board dropped to Reach+Impact
        # only). Pre-2.10.0 this gate required all three and 400'd boards
        # without Confidence.
        #
        # Accepted limitation: rice_attrs comes from the 5-min-cached
        # _discover_sort_attr_ids. A board sorted while it has no Confidence
        # attribute, then given one, keeps scoring confidence=1 until the
        # cache TTL expires (~5 min). This is a bounded transient — the same
        # staleness already applies to renaming Reach/Impact — and boards
        # that intentionally omit Confidence score 1 correctly and forever.
        required_attrs = {"reach", "impact"}
        missing_required = sorted(required_attrs - set(rice_attrs))
        if missing_required:
            return json.dumps(
                {
                    "error": "RICE custom attributes not fully configured",
                    "found": sorted(rice_attrs.keys()),
                    "required": sorted(required_attrs),
                    "missing": missing_required,
                    "code": 400,
                },
                indent=2,
            )

        # --- 2. Fetch all stories ONCE. The list endpoint already inlines
        # ``points``, ``total_points``, ``epics``, ``status``, ``swimlane``,
        # ``due_date``, ``is_closed`` — so we don't need any per-US calls
        # for those, only for the custom-attribute values (RICE).
        all_stories = list(project.list_user_stories())

        # ``stories`` is the working list for the per-US async fetch and
        # the eventual sort/POST loop — closed stories are dropped here
        # to (a) avoid ~N redundant ``custom-attributes-values`` GETs
        # on closed-heavy boards and (b) prevent the section-3
        # total-failure guard from firing falsely on a closed-only
        # board during a transient attr-fetch outage. The section-7
        # ``status_by_id`` filter below still runs as defense-in-depth
        # (handles the rare case where ``us.is_closed`` and
        # ``status.is_closed`` disagree, e.g. right after an admin
        # toggles a status's closed flag and the per-story field hasn't
        # caught up yet).
        #
        # ``all_stories`` is preserved for section-4 epic-completion
        # math: closed stories MUST count toward their epic's
        # completion_pct, otherwise an 80%-complete epic looks like 0%
        # to its remaining open stories and the documented "finish what
        # you started" boost (1.0–1.5×) is silently disabled.
        stories = [us for us in all_stories if not getattr(us, "is_closed", False)]

        # --- 3. Async-parallel per-US custom-attribute fetch.
        # Per-story failures are RECORDED in ``attribute_fetch_errors``
        # rather than swallowed: a missing reach/impact/confidence
        # defaults RICE to 1×1×1, which would silently shove a real
        # high-priority story to the bottom on a transient outage. The
        # caller decides whether to retry; partial failures must not be
        # invisible. ``base_url`` (API host, prefers TAIGA_API_URL) is
        # reused by the section-9 bulk-update POST — pre-2.3.4 that
        # POST used TAIGA_URL directly, a latent split-host bug fixed
        # here as a drive-by since the v1 endpoint lives on the API
        # origin, not the UI.
        base_url = _resolve_taiga_api_base_url()
        api = get_taiga_api(token=_current_taiga_jwt())
        token = api.token

        us_attr_values, attr_fetch_errors = await _fetch_us_attrs_async(base_url, token, stories)

        # Total-failure guard: if every per-story fetch failed, the RICE
        # numbers we'd compute are uniformly garbage and reordering the
        # board would be actively harmful. Bail loudly.
        if stories and len(attr_fetch_errors) == len(stories):
            return json.dumps(
                {
                    "error": (
                        "All per-story custom-attribute fetches failed; "
                        "cannot compute RICE scores reliably. Likely a "
                        "Taiga API outage or auth problem — refusing to "
                        "reorder the Kanban from defaults."
                    ),
                    "code": 500,
                    "attribute_fetch_errors": attr_fetch_errors[:10],
                },
                indent=2,
            )

        # --- 4. Pre-group stories by epic from inline ``us.epics`` data.
        # Pre-2.3.2 the per-epic ``epic.list_user_stories()`` was an
        # additional N+1 over epics; this groups in-memory from the
        # already-fetched stories list, no extra HTTP.
        #
        # Iterates ``all_stories`` (NOT the closed-filtered ``stories``)
        # so that closed stories still count toward their epic's
        # completion ratio. Filtering them out here would zero the
        # closed-count for any epic that has both open and closed work
        # → ``completion_pct = 0`` → ``completion_bonus = 1.0`` → the
        # documented "finish what you started" boost (1.0–1.5×) would
        # silently disappear, regressing RICE order for active stories.
        epic_to_stories: defaultdict = defaultdict(list)
        for us in all_stories:
            for e in getattr(us, "epics", None) or []:
                epic_id = e.get("id") if isinstance(e, dict) else getattr(e, "id", None)
                if epic_id:
                    epic_to_stories[epic_id].append(us)

        epic_completions = {
            epic_id: (sum(1 for us in uss if getattr(us, "is_closed", False)) / len(uss) if uss else 0.0)
            for epic_id, uss in epic_to_stories.items()
        }

        # --- 5. Epic Multiplicator dict (always fresh — see the
        # ``sort_attr_def_cache`` module-level comment for why this
        # isn't TTL-cached). Multiplicator is optional metadata, so a
        # fetch failure is non-fatal — just skip the epic-level boost.
        epic_multiplicators: Dict[int, float] = {}
        if multiplicator_attr_id:
            try:
                epics_list = list(project.list_epics())
                epic_multiplicators = await _fetch_epic_multiplicators_async(
                    base_url, token, epics_list, multiplicator_attr_id
                )
            except Exception:
                pass

        # --- 6. Build the per-story RICE rows from already-fetched data.
        # ``us.total_points`` is what Taiga itself shows in the UI as the
        # story's effort (sum across computable roles). We use it directly
        # instead of re-summing ``us.points.values()`` against a separate
        # ``project.list_points()`` lookup — eliminates one HTTP call AND
        # one source of bug surface (the role-id-19 hardcoded lookup that
        # 2.3.0 removed had its own zero-effort regression).
        stories_with_rice = []
        for us in stories:
            attr_values = us_attr_values.get(us.ref, {})

            reach = attr_values.get(rice_attrs["reach"], 1) or 1
            impact = attr_values.get(rice_attrs["impact"], 1) or 1
            # Confidence is optional (see the gate above). A board without a
            # Confidence attribute scores confidence as 1 instead of raising
            # KeyError on rice_attrs["confidence"].
            conf_id = rice_attrs.get("confidence")
            confidence = (attr_values.get(conf_id, 1) if conf_id else 1) or 1

            effort = getattr(us, "total_points", None) or 0

            if effort and effort > 0:
                rice_score = (reach * impact * confidence) / effort
            else:
                rice_score = 0

            epic_mult = 1.0
            completion_pct = 0.0
            completion_bonus = 1.0
            epic_ref = None
            epics = getattr(us, "epics", None)
            if epics and len(epics) > 0:
                # User story can be linked to multiple epics, use the first one
                epic_info = epics[0]
                epic_id = epic_info.get("id") if isinstance(epic_info, dict) else getattr(epic_info, "id", None)
                epic_ref = epic_info.get("ref") if isinstance(epic_info, dict) else getattr(epic_info, "ref", None)
                if epic_id and epic_id in epic_multiplicators:
                    epic_mult = epic_multiplicators[epic_id]
                if epic_id and epic_id in epic_completions:
                    completion_pct = epic_completions[epic_id]
                    completion_bonus = calculate_completion_bonus(completion_pct)

            due_date = getattr(us, "due_date", None)
            urgency = calculate_urgency(due_date, int(effort) if effort else 2)
            final_priority = rice_score * epic_mult * completion_bonus * urgency

            blocked_by_ref = None
            if blocked_by_attr_id:
                blocked_by_url = attr_values.get(blocked_by_attr_id, None)
                if blocked_by_url:
                    # Extract ref number from URL like https://taiga.shikenso.org/project/wahed/us/26
                    match = re.search(r"/us/(\d+)", blocked_by_url)
                    if match:
                        blocked_by_ref = int(match.group(1))

            stories_with_rice.append(
                {
                    "ref": us.ref,
                    "id": us.id,
                    "subject": us.subject,
                    "rice": rice_score,
                    "effort": effort,
                    "epic_ref": epic_ref,
                    "epic_mult": epic_mult,
                    "completion_pct": completion_pct,
                    "completion_bonus": completion_bonus,
                    "due_date": due_date,
                    "urgency": urgency,
                    "final_priority": final_priority,
                    "status_id": us.status,
                    "swimlane_id": getattr(us, "swimlane", None),
                    "blocked_by_ref": blocked_by_ref,
                }
            )

        # --- 7. Group by (status_id, swimlane_id) and sort each group.
        grouped = defaultdict(list)
        for s in stories_with_rice:
            key = (s["status_id"], s["swimlane_id"])
            grouped[key].append(s)

        # Build a status-id → status-dict lookup. The same call serves
        # the skip-closed filter below AND the ``status_name``
        # augmentation in section 9. ``list_all_statuses`` is cached
        # for 5 min (``list_all_statuses_cache``), so back-to-back sort
        # calls share it. A failure here intentionally bubbles to the
        # outer try/except — silently sorting closed columns on a
        # transient failure would defeat the point of this filter.
        #
        # Short-circuit on empty ``grouped``: if the board has no
        # stories left to sort (closed-only project, or every story
        # filtered out earlier), there's nothing to filter or augment,
        # so we skip the ``list_all_statuses`` call entirely. Without
        # this, an empty/closed-only board on a transient
        # statuses-endpoint outage would surface as a 500 even though
        # the right answer is "nothing to sort". Section 9's per-column
        # loop also doesn't run on empty ``grouped``, so leaving
        # ``status_by_id`` as ``{}`` is safe.
        status_by_id: Dict[int, Dict[str, Any]] = {}
        if grouped:
            status_by_id = {s["id"]: s for s in list_all_statuses(project_slug, "us").get("us_statuses", [])}

            # Drop groups whose status is closed (Done, Cancelled, ...).
            # Re-ranking completed work has no value and would generate a
            # redundant bulk-update POST per closed column. Orphan
            # status_ids — present in ``grouped`` but absent from
            # ``status_by_id`` (e.g. after an admin renamed the status
            # mid-cache-window) — are FAIL-OPEN: sorted normally rather
            # than dropped, so we never silently lose work on a transient
            # mismatch. The matching ``status_name`` falls back to None
            # in section 9.
            grouped = defaultdict(
                list,
                {
                    (sid, swimlane): rows
                    for (sid, swimlane), rows in grouped.items()
                    if not status_by_id.get(sid, {}).get("is_closed", False)
                },
            )

        for key in grouped:
            stories = grouped[key]
            stories.sort(key=lambda x: x["final_priority"], reverse=descending)

            # Reorder: place blocked stories immediately after their blocker
            # ONLY if the blocked story would otherwise appear ABOVE the blocker.
            ref_to_story = {s["ref"]: s for s in stories}
            blocked_stories = [s for s in stories if s["blocked_by_ref"] is not None]
            for blocked in blocked_stories:
                blocker_ref = blocked["blocked_by_ref"]
                if blocker_ref in ref_to_story:
                    blocker = ref_to_story[blocker_ref]
                    blocked_idx = stories.index(blocked)
                    blocker_idx = stories.index(blocker)
                    if blocked_idx < blocker_idx:
                        stories.remove(blocked)
                        # Re-find blocker's index after removal.
                        blocker_idx = stories.index(blocker)
                        stories.insert(blocker_idx + 1, blocked)
            grouped[key] = stories

        # --- 8. Warn on dependency-vs-deadline conflicts.
        warnings = []
        for stories in grouped.values():
            ref_to_story = {s["ref"]: s for s in stories}
            for story in stories:
                if story["blocked_by_ref"] and story["due_date"]:
                    blocker_ref = story["blocked_by_ref"]
                    if blocker_ref in ref_to_story:
                        blocker = ref_to_story[blocker_ref]
                        if not blocker["due_date"]:
                            warnings.append(
                                f"⚠️ US #{story['ref']} has due_date ({story['due_date']}) but is blocked by "
                                f"US #{blocker_ref} which has NO due_date. Consider adding a due_date to #{blocker_ref}."
                            )

        # --- 9. Push the new order to Taiga, one bulk-call per group.
        # Reuses ``base_url``/``token`` from section 3 — single
        # ``get_taiga_api`` call per invocation, and the POST hits the
        # same API host as the GETs (fixes the pre-2.3.4 split-host bug).
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        results = []
        for (status_id, swimlane_id), stories in grouped.items():
            if not stories:
                continue
            bulk_ids = [s["id"] for s in stories]
            data = {
                "project_id": project.id,
                "status_id": status_id,
                "bulk_userstories": bulk_ids,
            }
            if swimlane_id:
                data["swimlane_id"] = swimlane_id
            try:
                resp = requests.post(
                    f"{base_url}/api/v1/userstories/bulk_update_kanban_order",
                    json=data,
                    headers=headers,
                )
                results.append(
                    {
                        "status_id": status_id,
                        "status_name": status_by_id.get(status_id, {}).get("name"),
                        "swimlane_id": swimlane_id,
                        "success": resp.status_code == 200,
                        "order": [
                            {
                                "ref": s["ref"],
                                "rice": round(s["rice"], 2),
                                "effort": s["effort"],
                                "epic_ref": s["epic_ref"],
                                "epic_mult": s["epic_mult"],
                                "completion_pct": round(s["completion_pct"] * 100),
                                "completion_bonus": round(s["completion_bonus"], 2),
                                "due_date": s["due_date"],
                                "urgency": s["urgency"],
                                "final": round(s["final_priority"], 2),
                                "blocked_by": s["blocked_by_ref"],
                            }
                            for s in stories
                        ],
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "status_id": status_id,
                        "status_name": status_by_id.get(status_id, {}).get("name"),
                        "swimlane_id": swimlane_id,
                        "success": False,
                        "error": str(e),
                    }
                )

        return json.dumps(
            {
                "sorted": True,
                "project": project.name,
                "direction": ("descending (highest first)" if descending else "ascending (lowest first)"),
                "formula": "Final Priority = RICE × Epic Multiplicator × Completion Bonus × Urgency Multiplier",
                "completion_bonus_formula": "1.0 + 0.5 × (closed_stories / total_stories)²",
                "urgency_formula": "Buffer = Days remaining - Work evenings needed",
                "story_points_to_evenings": {"2": 1, "4": 2, "5": 3, "6": 4, "7": 6, "8": 10, "9": 15, "10": 20},
                "completion_bonus_scale": {
                    "0%": 1.0,
                    "20%": 1.02,
                    "50%": 1.13,
                    "75%": 1.28,
                    "80%": 1.32,
                    "90%": 1.41,
                    "100%": 1.5,
                },
                "urgency_multipliers": {
                    "buffer_negative": 50.0,
                    "buffer_0_1": 25.0,
                    "buffer_2_3": 10.0,
                    "buffer_4_7": 5.0,
                    "buffer_8_14": 2.0,
                    "buffer_over_14": 1.5,
                    "no_deadline": 1.0,
                },
                "total_stories": len(stories_with_rice),
                "deadline_stories": len([s for s in stories_with_rice if s["due_date"]]),
                "epic_multiplicators_used": len(epic_multiplicators) > 0,
                "epic_completions": {str(k): round(v * 100) for k, v in epic_completions.items()},
                "warnings": warnings if warnings else None,
                # ``None`` when every per-story fetch succeeded; otherwise
                # a list of ``{"ref": <us-ref>, "error": "<msg>"}`` for
                # the stories whose RICE custom attributes couldn't be
                # read. Those stories sorted with reach/impact/confidence
                # defaulted to 1, so their placement is unreliable; the
                # caller may want to retry the tool or reorder by hand.
                "attribute_fetch_errors": (attr_fetch_errors if attr_fetch_errors else None),
                "columns_updated": results,
            },
            indent=2,
        )

    except Exception as e:
        # Outer safety net so any uncaught failure produces a JSON
        # payload — without this the FastMCP harness reports the generic
        # "Error occurred during tool execution" with no diagnostic.
        return json.dumps(
            {
                "error": (f"Unexpected error in sort_kanban_by_rice_tool: " f"{type(e).__name__}: {str(e)}"),
                "code": 500,
                "trace_tail": traceback.format_exc().splitlines()[-5:],
            },
            indent=2,
        )


# NOTE: keep literal dict examples (e.g. ``{"Developer": 5}``) in the
# ``Examples:`` section, NOT in any ``Args:`` line. langchain-core's
# ``_parse_google_docstring`` treats every ":"-bearing line in ``Args:``
# as a new arg name and rejects the function with
# ``ValueError: Arg ... in docstring not found in function signature``.
@tool(parse_docstring=True)
def set_userstory_points_tool(
    project_slug: str,
    user_story_ref: int,
    points: Dict[str, float],
) -> str:
    """
    Set Taiga story points on a user story for one or more roles.

    Use when:
      - Setting Developer story points (the field
        ``sort_kanban_by_rice_tool`` reads as effort). Without this,
        points have to be set manually in the Taiga UI.
      - Estimating effort across multiple roles (Design, UX, Developer, ...).

    Role names are matched case-insensitively against the project's role
    names. Point values must match a value configured in the project
    (Taiga's default scale is the Fibonacci-like 0, 1/2, 1, 2, 3, 5, 8,
    10, 15, 20, 40 — your project's exact scale may differ). The ``?``
    (unestimated) point has value=None and cannot be set via this tool;
    use the Taiga UI for that.

    The target role MUST be configured as **computable** in the project
    (Taiga admin → Members/Roles → "Compute story points for this role").
    Non-computable roles are rejected by Taiga's userstory PATCH endpoint
    with a generic ``Invalid role id`` error; this tool surfaces that
    upfront as a 400 with ``non_computable_roles`` so the user knows to
    flip the project setting.

    Existing points for roles NOT included in the ``points`` dict are
    preserved.

    Args:
        project_slug: Project identifier (e.g. 'wahed').
        user_story_ref: Visible reference number of the user story
            (the number after ``/us/`` in the URL, NOT the database ID).
        points: Dictionary mapping role names to point values. See the
            ``Examples`` section below for the literal dict shape.

    Returns:
        JSON with the resolved points and a URL to the user story. On
        failure: a 400-coded error with the following diagnostic fields
        (each populated only for the failure modes that occurred, so
        the LLM can branch its retry strategy by which list is
        non-empty):

        - ``unresolved_roles``: role-name strings the caller passed
          that don't exist in the project at all.
        - ``unresolved_values``: ``[{role, value}]`` pairs whose value
          isn't part of the project's configured points scale.
        - ``non_computable_roles``: role names that DO exist but have
          ``computable=False``; Taiga rejects points assignment for
          those. Remediate via Taiga project admin → Members/Roles →
          "Compute story points for this role".
        - ``available_roles``: every role name the project knows about
          (sorted alphabetically; deterministic for the LLM to grep).
        - ``computable_roles``: subset of ``available_roles`` whose
          ``computable`` flag is True — these are the only role names
          the call could possibly have succeeded with.
        - ``available_point_values``: every numeric value in the
          project's points scale, sorted ascending.

    Examples:
        set_userstory_points_tool("wahed", 34, {"Developer": 5})
        set_userstory_points_tool("wahed", 34, {"Developer": 5, "UX": 2})
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404},
            indent=2,
        )

    try:
        us = project.get_userstory_by_ref(user_story_ref)
    except Exception as e:
        return json.dumps(
            {
                "error": f"Error fetching user story {user_story_ref}: {str(e)}",
                "code": 500,
            },
            indent=2,
        )
    if not us:
        return json.dumps(
            {
                "error": (f"User story {user_story_ref} not found in {project_slug}"),
                "code": 404,
            },
            indent=2,
        )

    try:
        # Resolve role names -> role IDs (project-scoped, case-insensitive).
        roles = project.list_roles()
        role_by_name = {r.name.lower(): r for r in roles}

        # Resolve point values -> point IDs. Skip the special "?" point
        # whose value is None (clearing not supported).
        point_id_by_value: Dict[float, int] = {}
        for p in project.list_points():
            if p.value is not None:
                point_id_by_value[p.value] = p.id

        # Build the resolved {role_id: point_id} map. Start from the
        # current us.points so roles NOT being changed are preserved.
        # Keys stringified to match Taiga's wire format.
        existing_points = us.points or {}
        new_points: Dict[str, int] = {str(k): v for k, v in existing_points.items()}
        unresolved_roles: List[str] = []
        unresolved_values: List[Dict[str, Any]] = []
        non_computable_roles: List[str] = []
        resolved: Dict[str, float] = {}

        for role_name, value in points.items():
            role = role_by_name.get(role_name.lower())
            if role is None:
                unresolved_roles.append(role_name)
                continue
            # Taiga's userstory PATCH only accepts role IDs whose
            # ``computable`` flag is True. Non-computable roles fail
            # server-side with a generic "Invalid role id" message —
            # we catch them here so the caller knows to flip the
            # project setting instead of debugging payload shape.
            if not getattr(role, "computable", True):
                non_computable_roles.append(role_name)
                continue
            point_id = point_id_by_value.get(value)
            if point_id is None:
                unresolved_values.append({"role": role_name, "value": value})
                continue
            new_points[str(role.id)] = point_id
            resolved[role_name] = value

        if unresolved_roles or unresolved_values or non_computable_roles:
            error_parts = []
            if non_computable_roles:
                error_parts.append(
                    f"Roles not computable for points "
                    f"(enable 'Compute story points for this role' in "
                    f"Taiga project admin -> Members/Roles): "
                    f"{non_computable_roles}"
                )
            if unresolved_roles:
                error_parts.append(f"Unknown roles: {unresolved_roles}")
            if unresolved_values:
                error_parts.append(f"Unknown point values: {unresolved_values}")
            return json.dumps(
                {
                    "error": "; ".join(error_parts),
                    "unresolved_roles": unresolved_roles,
                    "unresolved_values": unresolved_values,
                    "non_computable_roles": non_computable_roles,
                    "available_roles": sorted(r.name for r in roles),
                    "computable_roles": sorted(r.name for r in roles if getattr(r, "computable", True)),
                    "available_point_values": sorted(point_id_by_value.keys()),
                    "code": 400,
                },
                indent=2,
            )

        # PATCH only the points field plus the optimistic-lock version —
        # narrower than update()'s PUT-everything to avoid stomping
        # concurrent edits to other fields. Taiga's userstory PATCH
        # endpoint requires ``version`` on every modifying request, so
        # we explicitly include it in the field list (python-taiga's
        # ``Resource.patch`` does NOT auto-add version the way
        # ``update()`` does).
        us.points = new_points
        us.patch(["points", "version"])

        return json.dumps(
            {
                "updated": True,
                "project": project.name,
                "user_story_ref": user_story_ref,
                "points_set": resolved,
                "new_version": getattr(us, "version", None),
                "url": f"{TAIGA_URL}/project/{project_slug}/us/{user_story_ref}",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"error": f"Error setting story points: {str(e)}", "code": 500},
            indent=2,
        )


# ---------------------------------------------------------------------------
# User / membership helpers
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
def whoami_tool() -> str:
    """
    Get the currently authenticated Taiga user.
    Use when:
      - User asks "who am I" / "wer bin ich" / "what's my username"
      - Need to know the current user before assigning items to them
      - Verifying that auth is working end-to-end after an OAuth flow

    Returns:
        JSON with id, username, full_name, email of the current user.
    """
    try:
        api = get_taiga_api(token=_current_taiga_jwt())
        me = api.me()
        return json.dumps(
            {
                "id": me.id,
                "username": me.username,
                "full_name": me.full_name,
                "email": getattr(me, "email", None),
            },
            indent=2,
        )
    except Exception as e:
        # TODO(v2.2): distinguish 401 (token expired → claude.ai re-auth)
        # from 500 (real server error). Mirrors existing tool pattern.
        return json.dumps(
            {"error": f"Could not fetch current user: {str(e)}", "code": 500},
            indent=2,
        )


@tool(parse_docstring=True)
def list_project_members_tool(
    project_slug: str,
    include_email: bool = False,
) -> str:
    """
    List all members of a Taiga project with their roles.
    Use when:
      - Need to know who can be assigned to a task/issue/userstory/epic
      - User asks "wer ist im Projekt" / "who is on this project"
      - Looking up the right username/full_name to pass as assign_to

    Joins two python-taiga sources because neither alone has all the
    fields needed: ``project.members`` provides username/full_name/email
    (User objects), ``Project.list_memberships()`` provides role_name
    and is_admin. Joined by user_id.

    Args:
        project_slug: Project identifier (the URL slug).
        include_email: If True, include each member's email. Default
            False to avoid leaking other users' emails to the LLM/MCP
            client. The current user's own email is available via
            whoami_tool.

    Returns:
        JSON list of members with user_id, username, full_name, role,
        is_admin per entry; email only when include_email=True.
    """
    project = get_project(project_slug)
    if not project:
        # NOTE: ``get_project`` swallows every exception and returns None
        # (see helper at top of this file), so ``project is None`` here
        # can also mean an expired token / lack of permission / Taiga
        # outage — not strictly "not found". The error message is worded
        # to reflect that ambiguity. v2.2 should split get_project so
        # auth/permission errors propagate as their own status codes.
        return json.dumps(
            {
                "error": (
                    f"Project '{project_slug}' is not accessible (not found, "
                    "no permission, or auth/connection failure)."
                ),
                "code": 404,
            },
            indent=2,
        )
    try:
        # project.members → User objects with username/full_name/email.
        # Membership.user_email is unreliable for accepted members
        # (Taiga only populates it for pending invitations), so we
        # always pull email from the User object instead.
        users_by_id = {
            u.id: {
                "username": u.username,
                "full_name": u.full_name,
                "email": getattr(u, "email", None),
            }
            for u in project.members
        }
        # list_memberships() → role/is_admin per user
        result = []
        for m in project.list_memberships():
            uid = getattr(m, "user", None)
            user = users_by_id.get(uid, {})
            entry = {
                "user_id": uid,
                "username": user.get("username"),
                "full_name": user.get("full_name") or getattr(m, "full_name", None),
                "role": getattr(m, "role_name", None),
                "is_admin": bool(getattr(m, "is_admin", False)),
            }
            if include_email:
                entry["email"] = user.get("email")
            result.append(entry)
        return json.dumps(result, indent=2)
    except Exception as e:
        # TODO(v2.2): distinguish 401 (token expired → claude.ai re-auth)
        # from 500 (real server error). Mirrors existing tool pattern.
        return json.dumps(
            {"error": f"Error listing members: {str(e)}", "code": 500},
            indent=2,
        )


# ---------------------------------------------------------------------------
# Wiki helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Sanitize *text* into a valid Taiga wiki slug (lowercase, hyphenated)."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")


def _get_wiki_page(project: Project, wiki_slug: str):
    """Return the WikiPage with the given *slug*, or ``None``."""
    for wp in project.list_wikipages():
        if wp.slug == wiki_slug:
            return wp
    return None


# ---------------------------------------------------------------------------
# Wiki tools
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
def list_wiki_pages_tool(project_slug: str) -> str:
    """
    List all wiki pages in a Taiga project.
    Use when:
      - You need to know which wiki pages exist before reading or editing
      - User asks to see the project documentation overview

    Args:
        project_slug: Project identifier (the URL slug)

    Returns:
        JSON array of wiki page summaries (slug, modified_date, url).
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found."})

    try:
        pages = project.list_wikipages()
        result = []
        for wp in pages:
            result.append(
                {
                    "slug": wp.slug,
                    "modified_date": str(getattr(wp, "modified_date", "")),
                    "url": f"{TAIGA_URL}/project/{project_slug}/wiki/{wp.slug}",
                }
            )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(parse_docstring=True)
def get_wiki_page_tool(project_slug: str, wiki_slug: str) -> str:
    """
    Read a wiki page by its slug.
    Use when:
      - You need to read the current content of a wiki page
      - You need the version number before updating a wiki page (required for optimistic locking)

    Args:
        project_slug: Project identifier (the URL slug)
        wiki_slug: Slug of the wiki page (the part after /wiki/ in the URL)

    Returns:
        JSON with slug, content (Markdown), version, modified_date, url and attachments.
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found."})

    wp = _get_wiki_page(project, wiki_slug)
    if not wp:
        return json.dumps({"error": f"Wiki page '{wiki_slug}' not found in project '{project_slug}'."})

    try:
        attachments = [{"filename": a.name, "url": a.url} for a in (wp.list_attachments() or [])]
    except Exception:
        attachments = []

    return json.dumps(
        {
            "slug": wp.slug,
            "content": wp.content,
            "version": wp.version,
            "modified_date": str(getattr(wp, "modified_date", "")),
            "url": f"{TAIGA_URL}/project/{project_slug}/wiki/{wp.slug}",
            "attachments": attachments,
        },
        indent=2,
    )


@tool(parse_docstring=True)
def create_wiki_page_tool(
    project_slug: str,
    wiki_slug: str,
    content: str,
) -> str:
    """
    Create a new wiki page in a Taiga project.
    Use when:
      - User asks to create project documentation
      - You need to write a new wiki article or knowledge-base entry
    The slug will be sanitized automatically (lowercased, spaces become hyphens).

    Args:
        project_slug: Project identifier (the URL slug)
        wiki_slug: URL slug for the new page (e.g. "sprint-retrospective")
        content: Markdown content of the wiki page

    Returns:
        JSON with the created page slug and URL, or an error message.
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found."})

    slug = _slugify(wiki_slug)
    if not slug:
        return json.dumps({"error": "wiki_slug is empty after sanitization."})

    existing = _get_wiki_page(project, slug)
    if existing:
        return json.dumps(
            {
                "error": f"Wiki page '{slug}' already exists. Use update_wiki_page_tool to edit it.",
                "url": f"{TAIGA_URL}/project/{project_slug}/wiki/{slug}",
            }
        )

    try:
        wp = project.add_wikipage(slug, content)
        return json.dumps(
            {
                "slug": wp.slug,
                "url": f"{TAIGA_URL}/project/{project_slug}/wiki/{wp.slug}",
                "message": f"Wiki page '{wp.slug}' created successfully.",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool(parse_docstring=True)
def update_wiki_page_tool(
    project_slug: str,
    wiki_slug: str,
    content: str,
    version: int,
) -> str:
    """
    Update the content of an existing wiki page.
    Use when:
      - User asks to edit or update project documentation
      - You need to modify the content of a wiki page
    IMPORTANT: You must first call get_wiki_page_tool to obtain the current
    version number. Passing the correct version prevents overwriting
    concurrent edits (optimistic locking).

    Args:
        project_slug: Project identifier (the URL slug)
        wiki_slug: Slug of the wiki page to update
        content: New Markdown content (replaces the entire page)
        version: Current version number obtained from get_wiki_page_tool

    Returns:
        JSON with updated slug, new version, and URL, or an error message.
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found."})

    wp = _get_wiki_page(project, wiki_slug)
    if not wp:
        return json.dumps({"error": f"Wiki page '{wiki_slug}' not found in project '{project_slug}'."})

    try:
        wp.content = content
        wp.version = version
        wp.update()
        return json.dumps(
            {
                "slug": wp.slug,
                "new_version": wp.version,
                "url": f"{TAIGA_URL}/project/{project_slug}/wiki/{wp.slug}",
                "message": f"Wiki page '{wp.slug}' updated successfully.",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


_MCP_REGISTERED_INSTANCES: set[int] = set()


def _register_mcp_tools(mcp_instance) -> None:
    """Register LangChain Taiga tools with the given FastMCP instance.

    Idempotent per-instance: re-calling with the same instance is a no-op.
    The eager module-level call was removed when ``mcp.py`` became a
    factory — ``make_mcp()`` now invokes this against the freshly-built
    instance, which avoids the import cycle that would otherwise occur if
    we kept the legacy ``from langchain_taiga.mcp import mcp`` shape.

    ``sort_kanban_by_rice_tool`` is registered through an async wrapper
    that offloads its sync body via ``asyncio.to_thread``. Without this,
    the tool's parallel HTTP fetches against Taiga (~5–30s wall time on
    realistically-sized projects) block the FastMCP event loop, the
    ``/mcp/health`` endpoint stops responding, and the k8s liveness
    probe kills the pod mid-call. Other tools complete in <1s of HTTP
    work and don't need the wrapper. See taiga#5 for the temporary
    probe-budget relaxation that bought time before this fix.
    """
    if id(mcp_instance) in _MCP_REGISTERED_INSTANCES:
        return

    import functools
    import inspect

    def _async_offload(sync_func):
        """Wrap ``sync_func`` so FastMCP sees an ``async def`` that
        offloads execution to a worker thread. Preserves the
        function's name, docstring, signature, and annotations so
        FastMCP/Pydantic still build the same JSON-schema as if the
        sync function were registered directly."""

        @functools.wraps(sync_func)
        async def _wrapper(*args, **kwargs):
            return await asyncio.to_thread(sync_func, *args, **kwargs)

        # functools.wraps copies __name__/__doc__/__qualname__/etc but
        # NOT __signature__; FastMCP introspects via inspect.signature
        # so we attach it explicitly to keep the schema identical.
        _wrapper.__signature__ = inspect.signature(sync_func)
        return _wrapper

    # Tools that need their sync body offloaded to a worker thread when
    # registered with FastMCP. Identity comparison (``is``) over the
    # StructuredTool objects — not name comparison — so a future rename
    # of the underlying function CAN'T silently skip the offload and
    # re-introduce the event-loop block.
    # All three attachment tools do synchronous ``requests`` I/O against
    # external storage (download for ``add``/``get``, Taiga API for
    # ``list``) and can easily run multiple seconds on large files or
    # slow OVH egress. Without the offload they block the FastMCP event
    # loop, ``/mcp/health`` stops responding, and the k8s liveness probe
    # kills the pod — the exact failure mode that ``sort_kanban_by_rice_tool``
    # had pre-2.3.4.
    # ``get_kanban_board_tool`` does the same uncapped ``list_user_stories()``
    # fetch as ``sort_kanban_by_rice_tool`` (plus per-ex-member ``get_user``
    # fallbacks), so it gets the same offload to keep the event loop free.
    _TOOLS_NEEDING_ASYNC_OFFLOAD = frozenset(
        {
            id(sort_kanban_by_rice_tool),
            id(get_kanban_board_tool),
            id(add_attachment_by_ref_tool),
            id(list_attachments_by_ref_tool),
            id(get_attachment_by_ref_tool),
            # Validates project + entity against the Taiga API before it
            # mints a ticket, so it does the same blocking ``requests`` I/O
            # as its siblings even though it never touches the file itself.
            id(create_attachment_upload_by_ref_tool),
        }
    )

    for structured_tool in (
        create_entity_tool,
        search_entities_tool,
        get_kanban_board_tool,
        get_entity_by_ref_tool,
        update_entity_by_ref_tool,
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
    ):
        sync_func = structured_tool.func
        if sync_func is None:
            # Defensive: ``StructuredTool.func`` is None when the tool
            # was created from a coroutine. Currently no tool in this
            # package is async, but if one ever is, fail loud here
            # instead of silently registering ``None`` with FastMCP.
            raise RuntimeError(
                f"StructuredTool {structured_tool.name!r} has no .func "
                "attribute (likely an async-only tool). _register_mcp_tools "
                "doesn't support async tools yet — extend the offload "
                "machinery first."
            )
        if id(structured_tool) in _TOOLS_NEEDING_ASYNC_OFFLOAD:
            registered = mcp_instance.tool()(_async_offload(sync_func))
        else:
            registered = mcp_instance.tool()(sync_func)

        _copy_arg_descriptions(structured_tool, registered)

    _MCP_REGISTERED_INSTANCES.add(id(mcp_instance))


def _copy_arg_descriptions(structured_tool: Any, registered_tool: Any) -> int:
    """Publish each parameter's ``Args:`` text in the MCP input schema.

    ``@tool(parse_docstring=True)`` already parses the Google-style ``Args:``
    block into per-field descriptions on the LangChain ``args_schema``. But
    ``_register_mcp_tools`` hands FastMCP the *raw function*, so FastMCP
    re-derives its schema from the signature and type hints alone and every
    one of those descriptions is dropped — measured at 90 of 90 parameters
    across all 23 tools, all of them recoverable from the LangChain schema.

    The text still reaches the model inside the tool's monolithic
    ``description`` (which is why tool-choice evals pass without this), but
    a client that renders parameter help separately shows nothing, and a
    model scanning per-argument metadata has to find e.g. ``open_only``'s
    meaning inside a 3.4k-character blob. Copying is preferable to
    annotating every parameter with ``Annotated[..., Field(description=…)]``:
    that would restate ~90 descriptions in the signatures and leave two
    copies to drift apart. FastMCP 2.13 exposes no input-schema override on
    ``tool()``, so the schema dict is updated in place after registration.

    Existing descriptions are never overwritten, so a future explicit
    ``Field(description=…)`` still wins.

    Args:
        structured_tool: The LangChain ``StructuredTool`` holding the parsed
            docstring.
        registered_tool: The FastMCP tool object returned by registration.

    Returns:
        How many descriptions were copied.
    """
    schema = getattr(structured_tool, "args_schema", None)
    params = getattr(registered_tool, "parameters", None)
    if schema is None or not isinstance(params, dict):
        return 0
    try:
        source = schema.model_json_schema().get("properties", {})
    except Exception:
        # A tool whose args_schema cannot be rendered must not take the whole
        # server down at import time; it just keeps the status quo.
        return 0
    copied = 0
    for name, spec in params.get("properties", {}).items():
        if not isinstance(spec, dict) or spec.get("description"):
            continue
        description = (source.get(name) or {}).get("description")
        if description:
            spec["description"] = description
            copied += 1
    return copied


if __name__ == "__main__":
    # Simple test
    # statuses = list_all_statuses("shikenso-development")
    # print(json.dumps(statuses, indent=2))
    pass
