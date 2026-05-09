import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timedelta, timezone
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
from taiga.models import Project, EpicStatuses

logger = logging.getLogger(__name__)

load_dotenv()

TAIGA_URL = os.getenv("TAIGA_URL")
TAIGA_API_URL = os.getenv("TAIGA_API_URL")
TAIGA_TOKEN = os.getenv("TAIGA_TOKEN")
TAIGA_USERNAME = os.getenv("TAIGA_USERNAME")
TAIGA_PASSWORD = os.getenv("TAIGA_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    small_llm = ChatOpenAI(model="gpt-5.1")
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
list_all_statuses_cache = TTLCache(
    maxsize=100, ttl=timedelta(minutes=5).total_seconds()
)
list_all_tags_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=10).total_seconds())

find_issue_type_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_severity_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_priority_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_status_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
milestone_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())

user_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_user_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
custom_attr_definitions_cache = TTLCache(
    maxsize=100, ttl=timedelta(minutes=10).total_seconds()
)

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
sort_attr_def_cache = TTLCache(
    maxsize=100, ttl=timedelta(minutes=5).total_seconds()
)


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


def get_custom_attribute_definitions(
    project: Project, norm_type: str
) -> Dict[str, Dict]:
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


def get_formatted_custom_attributes(
    entity, project: Project, norm_type: str
) -> List[Dict]:
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
    """Retrieve an entity from a project given its normalized type and visible reference."""
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
        user_list.append(
            {"id": user.id, "full_name": user.full_name, "username": user.username}
        )

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


def _find_attribute_ids(
    project: Project, items: list, query: str, attribute_type: str
) -> List[int]:
    """Generic helper for finding attribute IDs using LLM semantic matching."""
    # Try exact match first
    exact_match = next(
        (item for item in items if item.name.lower() == query.lower()), None
    )
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
    future = [
        m
        for m in open_milestones
        if m.get("estimated_start") and m["estimated_start"] >= today
    ]
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
def list_all_statuses(
    project_slug: str, entity_type: Optional[str]
) -> Dict[str, List[Dict]]:
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
        task_statuses = [
            {**status.to_dict(), "id": status.id}
            for status in project.list_task_statuses()
        ]
        output["task_statuses"] = task_statuses
    if not entity_type or normalize_entity_type(entity_type) == "us":
        us_statuses = [
            {**status.to_dict(), "id": status.id}
            for status in project.list_user_story_statuses()
        ]
        output["us_statuses"] = us_statuses
    if not entity_type or normalize_entity_type(entity_type) == "issue":
        issue_statuses = [
            {**status.to_dict(), "id": status.id}
            for status in project.list_issue_statuses()
        ]
        output["issue_statuses"] = issue_statuses
    if not entity_type or normalize_entity_type(entity_type) == "epic":
        epic_statuses = [
            {**status.to_dict(), "id": status.id}
            for status in _get_epic_statuses(project.id)
        ]
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

    Returns:
        JSON with created entity details
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2
        )

    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )

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
            return json.dumps(
                {"error": f"User '{assign_to}' not found", "code": 404}, indent=2
            )
        assignee_id = users[0]["id"]

    # Base creation data
    create_data = {
        "subject": subject[:500],
        "description": description[:2000],
        "tags": tags,
        "assigned_to": assignee_id,
        "due_date": due_date,
    }

    try:
        if norm_type == "task":
            if not parent_us:
                return json.dumps(
                    {"error": "Tasks require a parent userstory", "code": 400}, indent=2
                )
            create_data["status"] = find_status_ids(
                project_slug=project_slug, entity_type=entity_type, query=status
            )[0]
            entity = parent_us.add_task(**create_data)
        elif norm_type == "us":
            entity = project.add_user_story(**create_data)
        elif norm_type == "issue":
            # Resolve issue type
            if issue_type:
                issue_type_ids = find_issue_type_ids(project_slug, issue_type)
                if not issue_type_ids:
                    return json.dumps(
                        {"error": f"Issue type '{issue_type}' not found"}, indent=2
                    )
                create_data["issue_type"] = issue_type_ids[0]
            else:
                # Use first available issue type from project
                available_issue_types = project.list_issue_types()
                if not available_issue_types:
                    return json.dumps(
                        {"error": "No issue types available in project"}, indent=2
                    )
                create_data["issue_type"] = available_issue_types[0].id

            # Resolve severity
            if severity:
                severity_ids = find_severity_ids(project_slug, severity)
                if not severity_ids:
                    return json.dumps(
                        {"error": f"Severity '{severity}' not found"}, indent=2
                    )
                create_data["severity"] = severity_ids[0]
            else:
                # Use first available severity from project
                available_severities = project.list_severities()
                if not available_severities:
                    return json.dumps(
                        {"error": "No severities available in project"}, indent=2
                    )
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

            # Status resolution (existing)
            status_ids = find_status_ids(
                project_slug=project_slug, entity_type=entity_type, query=status
            )
            if not status_ids:
                return json.dumps({"error": f"Status '{status}' not found"}, indent=2)
            create_data["status"] = status_ids[0]

            entity = project.add_issue(**create_data)
        elif norm_type == "epic":
            # Resolve status for epic
            status_ids = find_status_ids(
                project_slug=project_slug, entity_type=entity_type, query=status
            )
            if status_ids:
                create_data["status"] = status_ids[0]

            # Add color if provided
            if color:
                create_data["color"] = color

            # Remove due_date as epics don't have it
            create_data.pop("due_date", None)

            entity = project.add_epic(**create_data)
        else:
            return json.dumps(
                {"error": "Unsupported entity type", "code": 400}, indent=2
            )
    except Exception as e:
        return json.dumps(
            {"error": f"Creation failed: {str(e)}", "code": 500}, indent=2
        )

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
    point_id_to_value = {
        p.id: p.value for p in project.list_points() if p.value is not None
    }
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

    Returns:
        JSON object with ``matches`` (list of entities), ``truncated``
        (bool — was the max_results cap hit?), ``count`` (length of
        matches), and ``max_results`` (the cap that was applied).
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps(
            {"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2
        )

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
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )

    statuses = list_all_statuses(project_slug, norm_type)
    tags = list_all_tags(project_slug)
    milestones = list_milestones(project_slug)
    open_milestones = [m for m in milestones if not m["closed"]]
    current_milestone = get_current_milestone(project_slug)
    milestone_names = ", ".join(
        [f'"{m["name"]}" (id={m["id"]})' for m in open_milestones]
    )
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
    _all_pattern = re.compile(
        r"^(?:all|show all|list all|every|alles|alle)(?:\s+\w+)?$", re.IGNORECASE
    )
    if _all_pattern.match(query.strip()):
        search_params: dict = {}
    else:
        prompt = f"""
Convert this project management query to search parameters:
Query: {query}

The entity type being searched is "{norm_type}" — do NOT use the entity type as a text_search or tag filter.

Possible parameters:
- status_names: List[str] (status names)
- assigned_to: str (username/ID)
- milestone: str (sprint/milestone name, e.g. "Sprint 83")
- tags: List[str]
- text_search: str (searches subject/description). Only set text_search if the user explicitly wants to search for specific words in subjects or descriptions.
- created_after: date (YYYY-MM-DD)
- closed_before: date (YYYY-MM-DD)

IMPORTANT: Only set parameters that are explicitly mentioned or clearly implied by the query. Use null for everything else. Do NOT guess or hallucinate filter values.

Output ONLY valid JSON with parameter keys. Use null for unknown values.

IMPORTANT: The entity type ({norm_type}) is already selected — do NOT use it as a tag filter.
If the user wants "all" items, return all null values: {{"status_names": null, "assigned_to": null, "tags": null, "text_search": null, "created_after": null, "closed_before": null}}

Possible status names: {', '.join([s['name'] for s in statuses.get(f'{norm_type}_statuses', [])])}

Available milestones/sprints: {milestone_names}
{current_sprint_info}

Possible tags: {', '.join(tags)}

Example response for "John's open UX tasks in Sprint 83":
"{{"status_names": ["Open"], "assigned_to": "john_doe", "milestone": "Sprint 83", "tags": ["UX"]}}"

Example response for "all items in Sprint 83":
"{{"milestone": "Sprint 83", "status_names": null, "assigned_to": null, "tags": null, "text_search": null}}"

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
            return json.dumps(
                {"error": f"Query parsing failed: {str(e)}", "code": 500}, indent=2
            )

    # Resolve milestone filter (before fetching entities for server-side filtering)
    milestone_id = None
    if search_params.get("milestone"):
        milestone_id = find_milestone_id(project_slug, search_params["milestone"])

    # Fetch entities (with server-side milestone filtering when possible)
    try:
        if norm_type == "task":
            entities = []
            us_kwargs = {}
            if milestone_id is not None:
                us_kwargs["milestone"] = milestone_id
            for us in project.list_user_stories(**us_kwargs):
                if us.is_closed:
                    continue
                entities.extend(us.list_tasks())
        elif norm_type == "us":
            us_kwargs = {}
            if milestone_id is not None:
                us_kwargs["milestone"] = milestone_id
            entities = project.list_user_stories(**us_kwargs)
        elif norm_type == "issue":
            entities = project.list_issues()
        elif norm_type == "epic":
            entities = project.list_epics()
        else:
            entities = []
    except Exception as e:
        return json.dumps(
            {"error": f"Entity listing failed: {str(e)}", "code": 500}, indent=2
        )

    # Resolve filters upfront
    resolved_filters = {}

    # Milestone resolution (store for client-side fallback on issues)
    if milestone_id is not None:
        resolved_filters["milestone_id"] = milestone_id

    # Status resolution
    if search_params.get("status_names"):
        status_ids = []
        for status_name in search_params["status_names"]:
            ids = find_status_ids(project_slug, norm_type, status_name)
            status_ids.extend(ids)
        resolved_filters["status_ids"] = list(set(status_ids))

    # User resolution
    if search_params.get("assigned_to"):
        users = find_users(project_slug, search_params["assigned_to"])
        resolved_filters["assigned_to_ids"] = [u["id"] for u in users] if users else []

    # Date parsing.
    # Both filter datetimes are made tz-aware (UTC). python-taiga returns
    # ``entity.created_date`` / ``entity.finished_date`` as tz-aware
    # datetimes (Taiga API ships ISO timestamps with ``+0000``), and
    # comparing tz-aware vs tz-naive raises ``TypeError: can't compare
    # offset-naive and offset-aware datetimes`` mid-loop, which silently
    # truncates results.
    date_format = "%Y-%m-%d"
    if search_params.get("created_after"):
        resolved_filters["created_after"] = datetime.strptime(
            search_params["created_after"], date_format
        ).replace(tzinfo=timezone.utc)
    if search_params.get("closed_before"):
        resolved_filters["closed_before"] = datetime.strptime(
            search_params["closed_before"], date_format
        ).replace(tzinfo=timezone.utc)

    # Client-side filtering
    matches = []
    cap_hit = False
    for entity in entities:
        match = True

        # Milestone filter (client-side fallback for entities not server-filtered)
        if resolved_filters.get("milestone_id"):
            entity_milestone = getattr(entity, "milestone", None)
            if entity_milestone != resolved_filters["milestone_id"]:
                match = False

        # Status filter
        if resolved_filters.get("status_ids"):
            if entity.status not in resolved_filters["status_ids"]:
                match = False

        # Assignment filter
        if resolved_filters.get("assigned_to_ids"):
            if entity.assigned_to not in resolved_filters["assigned_to_ids"]:
                match = False

        # Tag filter
        if search_params.get("tags"):
            if not all(tag in entity.tags for tag in search_params["tags"]):
                match = False

        # Text search
        if search_params.get("text_search"):
            search_text = search_params["text_search"].lower()
            subject_match = search_text in entity.subject.lower()
            desc_match = (
                search_text in (getattr(entity, "description", "") or "").lower()
            )
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
            # Get status name for display
            status_info = get_status(project_slug, norm_type, entity.status)
            status_name = (
                status_info.get("name", "Unknown") if status_info else "Unknown"
            )

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
                            description = (
                                getattr(full_entity, "description", "") or ""
                            )
                        custom_attributes = get_formatted_custom_attributes(
                            full_entity, project, norm_type
                        )
                except Exception:
                    pass

            matches.append(
                {
                    "ref": entity.ref,
                    "subject": entity.subject,
                    "description": description,
                    "status": status_name,
                    "assigned_to": (
                        get_user(entity.assigned_to)["username"]
                        if entity.assigned_to
                        else None
                    ),
                    "created_date": (
                        entity.created_date if entity.created_date else None
                    ),
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
def get_entity_by_ref_tool(project_slug: str, entity_ref: int, entity_type: str) -> str:
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
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )

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
        "history": fetch_history(entity, norm_type),
        "tags": entity.tags,
    }

    # Add milestone/sprint info for userstories
    entity_milestone = getattr(entity, "milestone", None)
    if entity_milestone:
        milestones = list_milestones(project_slug)
        milestone_info = next(
            (m for m in milestones if m["id"] == entity_milestone), None
        )
        result["milestone"] = (
            milestone_info if milestone_info else {"id": entity_milestone}
        )
    else:
        result["milestone"] = None

    assigned_to = entity.assigned_to
    if assigned_to:
        assigned_to = get_user(assigned_to)
    result["assigned_to"] = assigned_to

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
                "status": get_status(project_slug, "task", task.status).get(
                    "name", "Unknown"
                ),
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
                    "status": get_status(project_slug, "us", us.status).get(
                        "name", "Unknown"
                    ),
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
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )

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
            return json.dumps(
                {"error": f"Status '{status}' not found", "code": 404}, indent=2
            )
        updates["status"] = status_ids[0]

    if description:
        updates["description"] = description

    if assign_to:
        user = find_users(project_slug, assign_to)
        if not user:
            return json.dumps(
                {"error": f"User '{assign_to}' not found", "code": 404}, indent=2
            )
        updates["assigned_to"] = user[0]["id"]

    if due_date:
        updates["due_date"] = due_date

    # Link user story to epic using Taiga's related_userstories endpoint
    epic_link_result = None
    if epic_ref is not None and norm_type == "us":
        epic = project.get_epic_by_ref(epic_ref)
        if not epic:
            return json.dumps(
                {"error": f"Epic {epic_ref} not found", "code": 404}, indent=2
            )
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
            entity.update(**updates)
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


@tool(parse_docstring=True)
def add_comment_by_ref_tool(
    project_slug: str, entity_ref: int, entity_type: str, comment: str
) -> str:
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
        return json.dumps(
            {"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2
        )

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
        return json.dumps(
            {"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2
        )

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

    try:
        # converts response headers mime type to an extension (may not work with everything)
        ext = content_type.split("/")[-1]
        r = requests.get(attachment_url, stream=True)
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_file:
            for chunk in r.iter_content(1024):  # iterate on stream using 1KB packets
                tmp_file.write(chunk)
            temp_file_path = tmp_file.name
        attachment = entity.attach(temp_file_path, description=description)
        # entity.add_comment(truncated_comment)
    except Exception as e:
        return json.dumps({"error": f"Comment failed: {str(e)}", "code": 500}, indent=2)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    att_dict = attachment.to_dict()
    att_dict.pop("url", None)
    return json.dumps(
        {
            "added": True,
            "project": project.name,
            "type": norm_type,
            "ref": entity_ref,
            "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
            "attachments": att_dict,
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
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )

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
            us_status = (
                us_status_info.get("name", "Unknown")
                if isinstance(us_status_info, dict)
                else "New"
            )
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
        JSON list of custom attributes with id, name, description, and type

    Examples:
        list_custom_attributes_tool("wahed", "userstory")
    """
    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )

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
            result.append({
                "id": attr.id,
                "name": attr.name,
                "description": getattr(attr, "description", ""),
                "type": getattr(attr, "type", "text"),
                "order": getattr(attr, "order", 0),
            })

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
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )

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
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404}, indent=2
        )

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
    2: 1,   # 2 SP = 1 evening
    4: 2,   # 4 SP = 2 evenings
    5: 3,   # 5 SP = 3 evenings
    6: 4,   # 6 SP = 4 evenings
    7: 6,   # 7 SP = 6 evenings
    8: 10,  # 8 SP = 10 evenings
    9: 15,  # 9 SP = 15 evenings
    10: 20, # 10 SP = 20 evenings
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
    return 1.0 + 0.5 * pct ** 2


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

    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, limits=limits, timeout=timeout
    ) as client:
        async def _one(us: Any) -> None:
            try:
                resp = await client.get(
                    f"/api/v1/userstories/custom-attributes-values/{us.id}"
                )
                resp.raise_for_status()
                body = resp.json()
                values_by_ref[us.ref] = body.get("attributes_values", {}) or {}
            except Exception as exc:
                values_by_ref[us.ref] = {}
                errors.append(
                    {"ref": us.ref, "error": f"{type(exc).__name__}: {exc}"}
                )

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

    async with httpx.AsyncClient(
        base_url=base_url, headers=headers, limits=limits, timeout=timeout
    ) as client:
        async def _one(epic: Any) -> None:
            try:
                resp = await client.get(
                    f"/api/v1/epics/custom-attributes-values/{epic.id}"
                )
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
    Final Priority = RICE × Epic Multiplicator × Completion Bonus × Urgency Multiplier

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


async def _sort_kanban_async_impl(
    project_slug: str, descending: bool
) -> str:
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

        if len(rice_attrs) < 3:
            return json.dumps(
                {
                    "error": "RICE custom attributes not fully configured",
                    "found": list(rice_attrs.keys()),
                    "required": ["reach", "impact", "confidence"],
                    "code": 400,
                },
                indent=2,
            )

        # --- 2. Fetch all stories ONCE. The list endpoint already inlines
        # ``points``, ``total_points``, ``epics``, ``status``, ``swimlane``,
        # ``due_date``, ``is_closed`` — so we don't need any per-US calls
        # for those, only for the custom-attribute values (RICE).
        stories = list(project.list_user_stories())

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

        us_attr_values, attr_fetch_errors = await _fetch_us_attrs_async(
            base_url, token, stories
        )

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
        epic_to_stories: defaultdict = defaultdict(list)
        for us in stories:
            for e in (getattr(us, "epics", None) or []):
                epic_id = (
                    e.get("id") if isinstance(e, dict) else getattr(e, "id", None)
                )
                if epic_id:
                    epic_to_stories[epic_id].append(us)

        epic_completions = {
            epic_id: (
                sum(1 for us in uss if getattr(us, "is_closed", False)) / len(uss)
                if uss
                else 0.0
            )
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
            confidence = attr_values.get(rice_attrs["confidence"], 1) or 1

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
                epic_id = (
                    epic_info.get("id")
                    if isinstance(epic_info, dict)
                    else getattr(epic_info, "id", None)
                )
                epic_ref = (
                    epic_info.get("ref")
                    if isinstance(epic_info, dict)
                    else getattr(epic_info, "ref", None)
                )
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
                        "swimlane_id": swimlane_id,
                        "success": False,
                        "error": str(e),
                    }
                )

        return json.dumps(
            {
                "sorted": True,
                "project": project.name,
                "direction": (
                    "descending (highest first)"
                    if descending
                    else "ascending (lowest first)"
                ),
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
                "attribute_fetch_errors": (
                    attr_fetch_errors if attr_fetch_errors else None
                ),
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
                "error": (
                    f"Unexpected error in sort_kanban_by_rice_tool: "
                    f"{type(e).__name__}: {str(e)}"
                ),
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
                "error": (
                    f"User story {user_story_ref} not found in {project_slug}"
                ),
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
        new_points: Dict[str, int] = {
            str(k): v for k, v in existing_points.items()
        }
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
                    "computable_roles": sorted(
                        r.name for r in roles if getattr(r, "computable", True)
                    ),
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
                "full_name": user.get("full_name")
                or getattr(m, "full_name", None),
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
        return json.dumps(
            {"error": f"Wiki page '{wiki_slug}' not found in project '{project_slug}'."}
        )

    try:
        attachments = [
            {"filename": a.name, "url": a.url} for a in (wp.list_attachments() or [])
        ]
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
        return json.dumps(
            {"error": f"Wiki page '{wiki_slug}' not found in project '{project_slug}'."}
        )

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
    _TOOLS_NEEDING_ASYNC_OFFLOAD = frozenset({id(sort_kanban_by_rice_tool)})

    for structured_tool in (
        create_entity_tool,
        search_entities_tool,
        get_entity_by_ref_tool,
        update_entity_by_ref_tool,
        add_comment_by_ref_tool,
        add_attachment_by_ref_tool,
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
            mcp_instance.tool()(_async_offload(sync_func))
        else:
            mcp_instance.tool()(sync_func)

    _MCP_REGISTERED_INSTANCES.add(id(mcp_instance))


if __name__ == "__main__":
    # Simple test
    # statuses = list_all_statuses("shikenso-development")
    # print(json.dumps(statuses, indent=2))
    pass
