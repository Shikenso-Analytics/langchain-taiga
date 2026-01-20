import json
import os
import re
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from cachetools import TTLCache, cached
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from taiga import TaigaAPI
from taiga.models import Project, EpicStatuses

from langchain_taiga.mcp import mcp

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

# Configure caches
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

user_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_user_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
custom_attr_definitions_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=10).total_seconds())

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
    cache_key = (project.id, norm_type)
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
                result.append({
                    "id": int(attr_id),
                    "name": definition["name"],
                    "description": definition["description"],
                    "type": definition["type"],
                    "value": value,
                })
        
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


@cached(cache=taiga_api_cache)
def get_taiga_api() -> TaigaAPI:
    """Get the Taiga API client."""
    # Initialize the main Taiga API client
    if TAIGA_USERNAME and TAIGA_PASSWORD:
        taiga_api = TaigaAPI(host=TAIGA_API_URL)
        taiga_api.auth(TAIGA_USERNAME, TAIGA_PASSWORD)
    elif TAIGA_TOKEN:
        taiga_api = TaigaAPI(host=TAIGA_API_URL, token=TAIGA_TOKEN)
    else:
        raise ValueError("Taiga credentials not provided.")
    return taiga_api


@cached(cache=project_cache)
def get_project(slug: str) -> Optional[Project]:
    """Get project by slug with auto-refreshing 5-minute cache."""
    # Extract slug from URL if present
    if "/project/" in slug:
        match = re.search(r"/project/([^/]+)", slug)
        if match:
            slug = match.group(1)

    try:
        project = get_taiga_api().projects.get_by_slug(slug)
        return project

    except Exception as e:
        print(f"Error fetching project {slug}: {e}")
        return None


@cached(cache=user_cache)
def get_user(user_id: int) -> Optional[Dict]:
    """
    Get user by ID.

    Args:
        user_id: User ID.

    Returns:
        Dictionary with user details or an error dict.
    """
    try:
        user = get_taiga_api().users.get(user_id)
        user_dict = user.to_dict()
        user_dict["id"] = user.id
        user_dict["full_name"] = user.full_name
        user_dict["username"] = user.username
        return user_dict
    except Exception as e:
        return {"error": str(e), "code": 500}


@cached(cache=find_user_cache)
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


@cached(cache=status_cache)
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
        if norm_type == "task":
            return get_taiga_api().task_statuses.get(status_id).to_dict()
        elif norm_type == "us":
            return get_taiga_api().user_story_statuses.get(status_id).to_dict()
        elif norm_type == "issue":
            return get_taiga_api().issue_statuses.get(status_id).to_dict()
        elif norm_type == "epic":
            api = get_taiga_api()
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


@cached(cache=find_issue_type_cache)
def find_issue_type_ids(project_slug: str, query: str) -> List[int]:
    """Find issue type IDs by semantic matching."""
    project = get_project(project_slug)
    if not project:
        return []
    return _find_attribute_ids(project, project.list_issue_types(), query, "issue_type")


@cached(cache=find_severity_cache)
def find_severity_ids(project_slug: str, query: str) -> List[int]:
    """Find severity IDs by semantic matching."""
    project = get_project(project_slug)
    if not project:
        return []
    return _find_attribute_ids(project, project.list_severities(), query, "severity")


@cached(cache=find_priority_cache)
def find_priority_ids(project_slug: str, query: str) -> List[int]:
    """Find priority IDs by semantic matching."""
    project = get_project(project_slug)
    if not project:
        return []
    return _find_attribute_ids(project, project.list_priorities(), query, "priority")


def _get_epic_statuses(project_id: int) -> list:
    """Get epic statuses for a project using the EpicStatuses factory."""
    api = get_taiga_api()
    return EpicStatuses(api.raw_request).list(project=project_id)


@cached(cache=find_status_cache)
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


@cached(cache=list_all_statuses_cache)
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


@cached(cache=list_all_tags_cache)
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
            issue_type_ids = find_issue_type_ids(project_slug, "Bug")  # Example value
            if not issue_type_ids:
                return json.dumps({"error": "Issue type 'Bug' not found"}, indent=2)
            create_data["issue_type"] = issue_type_ids[0]

            # Resolve severity
            severity_ids = find_severity_ids(project_slug, "Normal")  # Example value
            if not severity_ids:
                return json.dumps({"error": "Severity 'High' not found"}, indent=2)
            create_data["severity"] = severity_ids[0]

            # Resolve priority
            priority_ids = find_priority_ids(project_slug, "Normal")  # Example value
            if priority_ids:
                create_data["priority"] = priority_ids[0]

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


@tool(parse_docstring=True)
def search_entities_tool(
    project_slug: str, query: str, entity_type: str = "task"
) -> str:
    """
    Search tasks/userstories/issues/epics using natural language filters with client-side matching.
    Use when:
      - Looking for items matching complex criteria
      - Needing flexible search beyond API filter capabilities
      - Searching across multiple entity relationships

    Args:
        project_slug: Project identifier (e.g. 'mobile-app')
        query: Natural language query (e.g. 'UX tasks in progress assigned to @john')
        entity_type: 'task', 'userstory', 'issue', or 'epic'

    Returns:
        JSON list of matching entities with essential details
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

    statuses = list_all_statuses(project_slug, norm_type)
    tags = list_all_tags(project_slug)

    # Convert natural language to search criteria
    prompt = f"""
Convert this project management query to search parameters:
Query: {query}

Possible parameters:
- status_names: List[str] (status names)
- assigned_to: str (username/ID)
- tags: List[str]
- text_search: str (searches subject/description). Only set text_search if explicitly requested to search text.
- created_after: date (YYYY-MM-DD)
- closed_before: date (YYYY-MM-DD)

Output ONLY valid JSON with parameter keys. Use null for unknown values.

Possible status names: {', '.join([s['name'] for s in statuses.get(f'{norm_type}_statuses', [])])}

Possible tags: {', '.join(tags)}

Example response for "John's open UX tasks":
"{{"status_names": ["Open"], "assigned_to": "john_doe", "tags": ["UX"]}}"
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

    # Fetch all entities first
    try:
        if norm_type == "task":
            entities = []
            for us in project.list_user_stories():
                if us.is_closed:
                    continue
                entities.extend(us.list_tasks())
        elif norm_type == "us":
            entities = project.list_user_stories()  # Correct method name
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

    # Date parsing
    date_format = "%Y-%m-%d"
    if search_params.get("created_after"):
        resolved_filters["created_after"] = datetime.strptime(
            search_params["created_after"], date_format
        )
    if search_params.get("closed_before"):
        resolved_filters["closed_before"] = datetime.strptime(
            search_params["closed_before"], date_format
        )

    # Client-side filtering
    matches = []
    for entity in entities:
        match = True

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

        # Date filters
        if resolved_filters.get("created_after"):
            if entity.created_date < resolved_filters["created_after"]:
                match = False
        if resolved_filters.get("closed_before"):
            if (
                not entity.finished_date
                or entity.finished_date > resolved_filters["closed_before"]
            ):
                match = False

        if match:
            # Get status name for display
            status_info = get_status(project_slug, norm_type, entity.status)
            status_name = (
                status_info.get("name", "Unknown") if status_info else "Unknown"
            )

            # Fetch full entity details to get description and custom attributes
            description = getattr(entity, "description", "") or ""
            custom_attributes = []
            full_entity = None
            
            try:
                full_entity = fetch_entity(project, norm_type, entity.ref)
                if full_entity:
                    if not description:
                        description = getattr(full_entity, "description", "") or ""
                    # Get custom attributes for this entity
                    custom_attributes = get_formatted_custom_attributes(full_entity, project, norm_type)
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
                    "due_date": entity.due_date,
                    "custom_attributes": custom_attributes,
                    "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity.ref}",
                }
            )

            # Limit results for performance
            if len(matches) >= 100:
                break

    return json.dumps(matches, indent=2, default=str)


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
    api = get_taiga_api()

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
            ]
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

    assigned_to = entity.assigned_to
    if assigned_to:
        assigned_to = get_user(assigned_to)
    result["assigned_to"] = assigned_to

    watchers = entity.watchers
    if watchers:
        watchers = [get_user(w) for w in watchers]
    result["watchers"] = watchers

    # For userstories, include the count of related tasks.
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
            api = get_taiga_api()
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
        comment: Text to add (max 500 chars)

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
        # Truncate comments over 500 chars to match Taiga API limits
        truncated_comment = comment[:500]
        entity.add_comment(truncated_comment)
    except Exception as e:
        return json.dumps({"error": f"Comment failed: {str(e)}", "code": 500}, indent=2)

    return json.dumps(
        {
            "added": True,
            "project": project.name,
            "type": norm_type,
            "ref": entity_ref,
            "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
            "comment_preview": (
                f"{truncated_comment[:50]}..."
                if len(truncated_comment) > 50
                else truncated_comment
            ),
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
        api = get_taiga_api()
        
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
            us_status_info = getattr(us, 'status_extra_info', None)
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
            {"error": f"Error fetching {entity_type} {entity_ref}: {str(e)}", "code": 500},
            indent=2,
        )

    if not entity:
        return json.dumps(
            {"error": f"{entity_type} {entity_ref} not found in {project_slug}", "code": 404},
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
            {"error": f"Error fetching {entity_type} {entity_ref}: {str(e)}", "code": 500},
            indent=2,
        )

    if not entity:
        return json.dumps(
            {"error": f"{entity_type} {entity_ref} not found in {project_slug}", "code": 404},
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


@tool(parse_docstring=True)
def sort_kanban_by_rice_tool(
    project_slug: str,
    descending: bool = True,
) -> str:
    """
    Sort user stories in the Kanban board by their RICE score.
    RICE = (Reach × Impact × Confidence) / Effort
    Final Priority = RICE × Epic Multiplicator (if user story is linked to an epic)
    
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
    import requests
    from collections import defaultdict

    project = get_project(project_slug)
    if not project:
        return json.dumps(
            {"error": f"Project '{project_slug}' not found", "code": 404},
            indent=2,
        )

    # Get RICE custom attribute IDs for user stories
    rice_attrs = {}
    blocked_by_attr_id = None
    try:
        for attr in project.list_user_story_attributes():
            name_lower = attr.name.lower()
            if name_lower == "reach":
                rice_attrs["reach"] = str(attr.id)
            elif name_lower == "impact":
                rice_attrs["impact"] = str(attr.id)
            elif name_lower == "confidence":
                rice_attrs["confidence"] = str(attr.id)
            elif name_lower == "effort":
                rice_attrs["effort"] = str(attr.id)
            elif name_lower == "blocked by":
                blocked_by_attr_id = str(attr.id)
    except Exception as e:
        return json.dumps(
            {"error": f"Error listing custom attributes: {str(e)}", "code": 500},
            indent=2,
        )

    if len(rice_attrs) < 4:
        return json.dumps(
            {
                "error": "RICE custom attributes not fully configured",
                "found": list(rice_attrs.keys()),
                "required": ["reach", "impact", "confidence", "effort"],
                "code": 400,
            },
            indent=2,
        )

    # Get Epic Multiplicator custom attribute ID
    multiplicator_attr_id = None
    try:
        for attr in project.list_epic_attributes():
            if attr.name.lower() == "multiplicator":
                multiplicator_attr_id = str(attr.id)
                break
    except Exception:
        pass  # Multiplicator is optional

    # Build epic multiplicator cache (epic_id -> multiplicator value)
    epic_multiplicators = {}
    if multiplicator_attr_id:
        try:
            for epic in project.list_epics():
                attrs = epic.get_attributes()
                attr_values = attrs.get("attributes_values", {})
                mult = attr_values.get(multiplicator_attr_id, 1.0) or 1.0
                epic_multiplicators[epic.id] = float(mult)
        except Exception:
            pass  # If we can't get epics, continue without multiplicators

    # Get all user stories with their RICE scores
    stories_with_rice = []
    try:
        for us in project.list_user_stories():
            attrs = us.get_attributes()
            attr_values = attrs.get("attributes_values", {})

            reach = attr_values.get(rice_attrs["reach"], 1) or 1
            impact = attr_values.get(rice_attrs["impact"], 1) or 1
            confidence = attr_values.get(rice_attrs["confidence"], 1) or 1
            effort = attr_values.get(rice_attrs["effort"], 1) or 1

            if effort > 0:
                rice_score = (reach * impact * confidence) / effort
            else:
                rice_score = 0

            # Get Epic Multiplicator if user story is linked to an epic
            epic_mult = 1.0
            epic_ref = None
            epics = getattr(us, "epics", None)
            if epics and len(epics) > 0:
                # User story can be linked to multiple epics, use the first one
                epic_info = epics[0]
                epic_id = epic_info.get("id") if isinstance(epic_info, dict) else getattr(epic_info, "id", None)
                epic_ref = epic_info.get("ref") if isinstance(epic_info, dict) else getattr(epic_info, "ref", None)
                if epic_id and epic_id in epic_multiplicators:
                    epic_mult = epic_multiplicators[epic_id]

            # Final priority = RICE × Epic Multiplicator
            final_priority = rice_score * epic_mult

            # Get Blocked By reference (extract ref from URL if present)
            blocked_by_ref = None
            if blocked_by_attr_id:
                blocked_by_url = attr_values.get(blocked_by_attr_id, None)
                if blocked_by_url:
                    # Extract ref number from URL like https://taiga.shikenso.org/project/wahed/us/26
                    match = re.search(r'/us/(\d+)', blocked_by_url)
                    if match:
                        blocked_by_ref = int(match.group(1))

            stories_with_rice.append(
                {
                    "ref": us.ref,
                    "id": us.id,
                    "subject": us.subject,
                    "rice": rice_score,
                    "epic_ref": epic_ref,
                    "epic_mult": epic_mult,
                    "final_priority": final_priority,
                    "status_id": us.status,
                    "swimlane_id": getattr(us, "swimlane", None),
                    "blocked_by_ref": blocked_by_ref,
                }
            )
    except Exception as e:
        return json.dumps(
            {"error": f"Error fetching user stories: {str(e)}", "code": 500},
            indent=2,
        )

    # Group by status_id and swimlane
    grouped = defaultdict(list)
    for s in stories_with_rice:
        key = (s["status_id"], s["swimlane_id"])
        grouped[key].append(s)

    # Sort each group by Final Priority (RICE × Epic Multiplicator)
    # Then reorder to place blocked stories immediately after their blocker
    for key in grouped:
        stories = grouped[key]
        # First, sort by final_priority
        stories.sort(key=lambda x: x["final_priority"], reverse=descending)
        
        # Build a ref->story mapping for quick lookup
        ref_to_story = {s["ref"]: s for s in stories}
        
        # Find blocked stories and their blockers
        blocked_stories = [s for s in stories if s["blocked_by_ref"] is not None]
        
        # Reorder: place blocked stories immediately after their blocker
        # ONLY if the blocked story would appear ABOVE the blocker
        for blocked in blocked_stories:
            blocker_ref = blocked["blocked_by_ref"]
            if blocker_ref in ref_to_story:
                blocker = ref_to_story[blocker_ref]
                blocked_idx = stories.index(blocked)
                blocker_idx = stories.index(blocker)
                
                # Only move if blocked story is currently ABOVE the blocker
                if blocked_idx < blocker_idx:
                    # Remove blocked story from current position
                    stories.remove(blocked)
                    # Find blocker's NEW position (shifted after removal) and insert after it
                    blocker_idx = stories.index(blocker)
                    stories.insert(blocker_idx + 1, blocked)
        
        grouped[key] = stories

    # Call the bulk_update_kanban_order API for each group
    base_url = TAIGA_URL.rstrip("/")
    api = get_taiga_api()
    headers = {"Authorization": f"Bearer {api.token}", "Content-Type": "application/json"}

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
                            "epic_ref": s["epic_ref"],
                            "epic_mult": s["epic_mult"],
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
            "direction": "descending (highest first)" if descending else "ascending (lowest first)",
            "formula": "Final Priority = RICE × Epic Multiplicator",
            "total_stories": len(stories_with_rice),
            "epic_multiplicators_used": len(epic_multiplicators) > 0,
            "columns_updated": results,
        },
        indent=2,
    )


_MCP_REGISTERED = False


def _register_mcp_tools() -> None:
    """Register LangChain Taiga tools with the FastMCP server once."""

    global _MCP_REGISTERED

    if _MCP_REGISTERED:
        return

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
    ):
        mcp.tool()(structured_tool.func)

    _MCP_REGISTERED = True


_register_mcp_tools()


if __name__ == "__main__":
    # Simple test
    # statuses = list_all_statuses("shikenso-development")
    # print(json.dumps(statuses, indent=2))
    pass
