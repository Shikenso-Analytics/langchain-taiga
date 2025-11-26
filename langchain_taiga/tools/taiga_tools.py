import json
import os
import tempfile
from datetime import timedelta, datetime
from typing import Optional, Dict, List

import requests
from cachetools import TTLCache, cached
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_taiga.mcp import mcp
from taiga import TaigaAPI
from taiga.models import Project

load_dotenv()

TAIGA_URL = os.getenv("TAIGA_URL")
TAIGA_API_URL = os.getenv("TAIGA_API_URL")
TAIGA_TOKEN = os.getenv("TAIGA_TOKEN")
TAIGA_USERNAME = os.getenv("TAIGA_USERNAME")
TAIGA_PASSWORD = os.getenv("TAIGA_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    small_llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.2)
else:
    small_llm = ChatOllama(model="llama3.2:3b")

# Configure caches
taiga_api_cache = TTLCache(maxsize=100, ttl=timedelta(hours=2).total_seconds())
project_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())
status_cache = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())

find_issue_type_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_severity_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_priority_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_status_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())

user_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())
find_user_cache = TTLCache(maxsize=100, ttl=timedelta(days=1).total_seconds())

# Mapping of acceptable entity types (singular or plural) to normalized form.
ENTITY_TYPE_MAPPING = {
    "task": "task",
    "tasks": "task",
    "userstory": "us",
    "userstories": "us",
    "issue": "issue",
    "issues": "issue",
}


def normalize_entity_type(entity_type: str) -> Optional[str]:
    """Return the normalized entity type, or None if unsupported."""
    return ENTITY_TYPE_MAPPING.get(entity_type.lower())


def fetch_entity(project: Project, norm_type: str, entity_ref: int):
    """Retrieve an entity from a project given its normalized type and visible reference."""
    if norm_type == "task":
        return project.get_task_by_ref(entity_ref)
    elif norm_type == "us":
        return project.get_userstory_by_ref(entity_ref)
    elif norm_type == "issue":
        return project.get_issue_by_ref(entity_ref)
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
    except Exception as e:
        return {"error": str(e), "code": 500}
    return None


def _find_attribute_ids(
        project: Project,
        items: list,
        query: str,
        attribute_type: str
) -> List[int]:
    """Generic helper for finding attribute IDs using LLM semantic matching."""
    # Try exact match first
    exact_match = next((item for item in items if item.name.lower() == query.lower()), None)
    if exact_match:
        return [exact_match.id]

    # Prepare items for LLM processing
    item_dicts = [{
        "id": item.id,
        "name": item.name,
        "description": getattr(item, "description", "")
    } for item in items]

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


@cached(cache=find_status_cache)
def find_status_ids(project_slug: str, entity_type: str, query: str) -> List[int]:
    """Find status IDs by semantic matching for any entity type."""
    norm_type = normalize_entity_type(entity_type)
    project = get_project(project_slug)

    if not norm_type or not project:
        return []

    status_map = {
        "task": project.list_task_statuses,
        "us": project.list_user_story_statuses,
        "issue": project.list_issue_statuses
    }

    return _find_attribute_ids(
        project,
        status_map[norm_type](),
        query,
        "status"
    )


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
def create_entity_tool(project_slug: str,
                       entity_type: str,
                       subject: str,
                       status: str,
                       description: Optional[str] = "",
                       parent_ref: Optional[int] = None,
                       assign_to: Optional[str] = None,
                       due_date: Optional[str] = None,
                       tags: List[str] = []) -> str:
    """
    Create new userstory, tasks or issues.
    Use when:
      - User requests creation of new work items
      - Need to break down userstories into tasks
      - Reporting new issues/bugs

    Args:
        project_slug: Project identifier
        entity_type: 'userstory' , 'task' or 'issue'
        subject: Short title/name
        status: State of the entity
        description: Detailed description (optional)
        parent_ref: For tasks - userstory reference
        assign_to: Username to assign (optional)
        due_date: Deadline for the task (Format: YYYY-MM-DD) (optional)
        tags: List of tags (optional)

    Returns:
        JSON with created entity details
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
            return json.dumps({"error": f"Parent userstory {parent_ref} not found", "code": 404}, indent=2)

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
        "tags": tags,
        "assigned_to": assignee_id,
        "due_date": due_date
    }

    try:
        if norm_type == "task":
            if not parent_us:
                return json.dumps({"error": "Tasks require a parent userstory", "code": 400}, indent=2)
            create_data["status"] = find_status_ids(project_slug=project_slug, entity_type=entity_type, query=status)[0]
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
                project_slug=project_slug,
                entity_type=entity_type,
                query=status
            )
            if not status_ids:
                return json.dumps({"error": f"Status '{status}' not found"}, indent=2)
            create_data["status"] = status_ids[0]

            entity = project.add_issue(**create_data)
        else:
            return json.dumps({"error": "Unsupported entity type", "code": 400}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Creation failed: {str(e)}", "code": 500}, indent=2)

    return json.dumps({
        "created": True,
        "type": norm_type,
        "ref": entity.ref,
        "subject": entity.subject,
        "due_date": due_date,
        "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity.ref}",
        "assigned_to": assign_to,
        "parent": parent_ref
    }, indent=2)


@tool(parse_docstring=True)
def search_entities_tool(project_slug: str,
                         query: str,
                         entity_type: str = "task") -> str:
    """
    Search tasks/userstories/issues using natural language filters with client-side matching.
    Use when:
      - Looking for items matching complex criteria
      - Needing flexible search beyond API filter capabilities
      - Searching across multiple entity relationships

    Args:
        project_slug: Project identifier (e.g. 'mobile-app')
        query: Natural language query (e.g. 'UX tasks in progress assigned to @john')
        entity_type: 'task', 'userstory', or 'issue'

    Returns:
        JSON list of matching entities with essential details
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps({"error": f"Invalid entity type '{entity_type}'", "code": 400}, indent=2)

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    # Convert natural language to search criteria
    prompt = f"""
Convert this project management query to search parameters:
Query: {query}

Possible parameters:
- status_names: List[str] (status names)
- assigned_to: str (username/ID)
- tags: List[str]
- text_search: str (searches subject/description)
- created_after: date (YYYY-MM-DD)
- closed_before: date (YYYY-MM-DD)

Output ONLY valid JSON with parameter keys. Use null for unknown values.

Example response for "John's open UX tasks":
"{{"status_names": ["Open"], "assigned_to": "john_doe", "tags": ["UX"]}}"
"""
    try:
        response = small_llm.invoke([HumanMessage(content=prompt)])
        if "\n" in response.content:
            content = response.content.split("\n")[-2]
        else:
            content = response.content
        print(response)
        search_params = json.loads(content)
    except Exception as e:
        return json.dumps({"error": f"Query parsing failed: {str(e)}", "code": 500}, indent=2)

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
        else:
            entities = []
    except Exception as e:
        return json.dumps({"error": f"Entity listing failed: {str(e)}", "code": 500}, indent=2)

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
            desc_match = search_text in (entity.description or "").lower()
            if not (subject_match or desc_match):
                match = False

        # Date filters
        if resolved_filters.get("created_after"):
            if entity.created_date < resolved_filters["created_after"]:
                match = False
        if resolved_filters.get("closed_before"):
            if not entity.finished_date or entity.finished_date > resolved_filters["closed_before"]:
                match = False

        if match:
            # Get status name for display
            status_info = get_status(project_slug, norm_type, entity.status)
            status_name = status_info.get("name", "Unknown") if status_info else "Unknown"

            matches.append({
                "ref": entity.ref,
                "subject": entity.subject,
                "status": status_name,
                "assigned_to": get_user(entity.assigned_to)["username"] if entity.assigned_to else None,
                "created_date": entity.created_date if entity.created_date else None,
                "due_date": entity.due_date,
                "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity.ref}"
            })

            # Limit results for performance
            if len(matches) >= 10:
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
        "issue": api.history.issue.get
    }.get(norm_type)

    return history_fetcher(entity.id) if history_fetcher else []


@tool(parse_docstring=True)
def get_entity_by_ref_tool(project_slug: str, entity_ref: int, entity_type: str) -> str:
    """
    Retrieve any Taiga entity (task/userstory/issue) by its visible reference number.
    Use when:
      - A direct URL to an entity is provided.
      - Verifying existence of specific items.
      - Looking up details before modifications.

    Args:
        project_slug (str): Project identifier.
        entity_ref (int): Visible reference number (not the database ID).
        entity_type (str): 'task', 'userstory', or 'issue'.

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
        return json.dumps({"error": f"Entity type '{entity_type}' is not supported.", "code": 400}, indent=2)

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps({"error": f"Error fetching {norm_type} {entity_ref}: {str(e)}", "code": 500}, indent=2)

    if not entity:
        return json.dumps({"error": f"{entity_type} {entity_ref} not found in {project_slug}", "code": 404}, indent=2)

    # Retrieve status name (or fallback to "Unknown")
    status_info = get_status(project_slug, norm_type, entity.status)
    status_name = status_info.get("name", "Unknown") if status_info else "Unknown"

    result = {
        "project": project.name,
        "project_slug": project.slug,
        "type": norm_type,
        "ref": entity.ref,
        "status": status_name,
        "subject": entity.subject,
        "description": entity.description,
        "due_date": entity.due_date,
        "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity.ref}",
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
                "status": get_status(project_slug, "task", task.status).get("name", "Unknown")
            } for task in entity.list_tasks()]
    if norm_type == "task":
        result["user_story_extra_info"] = entity.user_story_extra_info

    return json.dumps(result, indent=2)


@tool(parse_docstring=True)
def update_entity_by_ref_tool(project_slug: str, entity_ref: int, entity_type: str, description: Optional[str] = None,
                              assign_to: Optional[str] = None, status: Optional[str] = None,
                              due_date: Optional[str] = None) -> str:
    """
    Update a Taiga entity (task/userstory/issue) by its visible reference number.
    Use when:
      - Specific fields of an entity need to be modified (e.g., status, assignee, description).

    Args:
        project_slug (str): Project identifier.
        entity_ref (int): Visible reference number (not the database ID).
        entity_type (str): 'task', 'userstory', or 'issue'.
        description (str): New description for the entity.
        assign_to (str): Username of the user to assign the entity to.
        status (str): New status for the entity.
        due_date (str): New due date for the entity (Format YYYY-MM-DD).

    Returns:
        A JSON message indicating success or an error message.
    """
    norm_type = normalize_entity_type(entity_type)
    if not norm_type:
        return json.dumps({"error": f"Entity type '{entity_type}' is not supported.", "code": 400}, indent=2)

    project = get_project(project_slug)
    if not project:
        return json.dumps({"error": f"Project '{project_slug}' not found", "code": 404}, indent=2)

    try:
        entity = fetch_entity(project, norm_type, entity_ref)
    except Exception as e:
        return json.dumps({"error": f"Error fetching {norm_type} {entity_ref}: {str(e)}", "code": 500}, indent=2)

    if not entity:
        return json.dumps({"error": f"{entity_type} {entity_ref} not found in {project_slug}", "code": 404}, indent=2)

    updates = {}
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

    try:
        entity.update(**updates)
    except Exception as e:
        return json.dumps({"error": f"Error updating {norm_type} {entity_ref}: {str(e)}", "code": 500}, indent=2)

    return json.dumps({"message": f"{norm_type.capitalize()} {entity_ref} updated successfully."}, indent=2)


@tool(parse_docstring=True)
def add_comment_by_ref_tool(project_slug: str, entity_ref: int, entity_type: str, comment: str) -> str:
    """
    Add comment to any Taiga entity using its visible reference. Use when:
    - User provides direct URL to an item
    - Need to document decisions on specific tasks/issues/userstories
    - Providing status updates via comments

    Args:
        project_slug: From URL path (e.g. 'development')
        entity_ref: Visible number in entity URL
        entity_type: 'task', 'userstory', or 'issue'
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
        # Truncate comments over 500 chars to match Taiga API limits
        truncated_comment = comment[:500]
        entity.add_comment(truncated_comment)
    except Exception as e:
        return json.dumps({"error": f"Comment failed: {str(e)}", "code": 500}, indent=2)

    return json.dumps({
        "added": True,
        "project": project.name,
        "type": norm_type,
        "ref": entity_ref,
        "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
        "comment_preview": f"{truncated_comment[:50]}..." if len(truncated_comment) > 50 else truncated_comment
    }, indent=2)


@tool(parse_docstring=True)
def add_attachment_by_ref_tool(project_slug: str, entity_ref: int, entity_type: str, attachment_url: str,
                               content_type: str, description: str = "") -> str:
    """
    Add attachment (images and other files) to any Taiga entity using its visible reference. Use when:
    - User provides direct URL to an item
    - Need to share screenshots, logs, or other files
    - Providing additional context to tasks/issues/userstories

    Args:
        project_slug: From URL path (e.g. 'development')
        entity_ref: Visible number in entity URL
        entity_type: 'task', 'userstory', or 'issue'
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

    try:
        # converts response headers mime type to an extension (may not work with everything)
        ext = content_type.split('/')[-1]
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
    return json.dumps({
        "added": True,
        "project": project.name,
        "type": norm_type,
        "ref": entity_ref,
        "url": f"{TAIGA_URL}/project/{project_slug}/{norm_type}/{entity_ref}",
        "attachments": att_dict,
    }, indent=2)


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
    ):
        mcp.tool()(structured_tool.func)

    _MCP_REGISTERED = True


_register_mcp_tools()
