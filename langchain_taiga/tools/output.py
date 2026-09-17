"""Token-lean read output: field projection and compact JSON (2.19.0).

Every byte a read tool returns lands in the calling model's context. ``fields`` lets a caller
name the dotted paths it needs (``related.tasks.ref``; a list is walked element by element) and
``compact`` drops the indentation. The tools use ``wants`` to skip fetching the parts nobody
asked for, which saves Taiga round-trips as well as tokens.

Two rules a caller depends on:

* **A requested key is never silently absent.** An element that lacks it answers ``null``, and
  ``compact`` keeps requested nulls — "unassigned" and "not asked for" stay distinguishable.
  Without ``fields``, ``compact`` drops nulls, because then nothing was asked for by name.
* **An unknown key is an error, not an empty projection** (``unknown_fields``, at every level
  the tool's schema describes). A typo that projected everything away would read as "the entity
  has no such data".
"""

from __future__ import annotations

import json
from typing import AbstractSet, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

Paths = List[List[str]]
# The keys a projection may name: a set for one level, or a mapping from each key to the schema
# below it, where ``None`` leaves everything below that key unchecked.
Schema = Union[AbstractSet[str], Mapping[str, Optional["Schema"]]]

# Marks a subtree that keeps its whole value.
_WHOLE = None


def parse_fields(fields: Optional[Sequence[str]]) -> Optional[Paths]:
    """Split ``fields`` into path segments. ``None`` means "no projection".

    Raises:
        ValueError: on an empty list or a blank segment (``""``, ``"a..b"``, ``"x."``).
    """
    if fields is None:
        return None
    if isinstance(fields, str):
        raise ValueError("fields must be a list of dotted paths, not a single string")
    if len(fields) == 0:
        raise ValueError("fields must name at least one path; omit it to get everything")
    parsed: Paths = []
    for raw in fields:
        path = str(raw).strip()
        segments = path.split(".")
        if not path or any(not segment.strip() for segment in segments):
            raise ValueError(f"invalid field path {raw!r}: every dot-separated part must be non-empty")
        parsed.append([segment.strip() for segment in segments])
    return parsed


def unknown_fields(paths: Optional[Paths], schema: Schema) -> Optional[str]:
    """An error message naming a path that leaves ``schema``, or ``None``."""
    if paths is None:
        return None
    unknown = sorted({path[0] for path in paths if path[0] not in schema})
    if unknown:
        return f"Unknown field(s) {', '.join(unknown)}. Valid top-level fields: {', '.join(sorted(schema))}."
    for path in paths:
        node: Optional[Schema] = schema
        for depth, segment in enumerate(path):
            if node is None:
                break
            if segment not in node:
                return (
                    f"Unknown field {'.'.join(path[: depth + 1])}. "
                    f"Valid fields under {'.'.join(path[:depth])}: {', '.join(sorted(node))}."
                )
            node = node[segment] if isinstance(node, Mapping) else None
    return None


def checked_fields(
    fields: Optional[Sequence[str]], schema: Schema, *, compact: bool
) -> Tuple[Optional[Paths], Optional[str]]:
    """``(paths, None)`` for valid ``fields``, else ``(None, <400 answer>)``."""
    try:
        paths = parse_fields(fields)
    except ValueError as e:
        return None, error(str(e), 400, compact=compact)
    message = unknown_fields(paths, schema)
    if message:
        return None, error(message, 400, compact=compact)
    return paths, None


def wants(paths: Optional[Paths], *prefix: str) -> bool:
    """Does the projection need anything under ``prefix``? Always true without a projection."""
    if paths is None:
        return True
    depth = len(prefix)
    for path in paths:
        shared = min(depth, len(path))
        if path[:shared] == list(prefix[:shared]):
            return True
    return False


def names(paths: Optional[Paths], *prefix: str) -> bool:
    """Does a path name ``prefix`` itself (or something below it)? An ancestor path does not count.

    For parts that cost extra requests: a caller asking for a whole parent should not trigger them.
    """
    if paths is None:
        return False
    return any(path[: len(prefix)] == list(prefix) for path in paths)


def _tree(paths: Paths) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for path in paths:
        node = tree
        for index, segment in enumerate(path):
            last = index == len(path) - 1
            if segment in node and node[segment] is _WHOLE:
                break  # an ancestor already keeps the whole value
            if last:
                node[segment] = _WHOLE
                break
            node = node.setdefault(segment, {})
    return tree


def _apply(value: Any, tree: Optional[Dict[str, Any]]) -> Any:
    if tree is _WHOLE:
        return value
    if isinstance(value, list):
        return [_apply(item, tree) for item in value]
    if isinstance(value, dict):
        return {key: _apply(value.get(key), subtree) for key, subtree in tree.items()}
    return value  # a path through a scalar or null ends here


def project(value: Any, paths: Optional[Paths]) -> Any:
    """Keep only the requested paths of ``value``; the value itself when ``paths`` is ``None``."""
    if paths is None:
        return value
    return _apply(value, _tree(paths))


def project_keeping(result: Dict[str, Any], paths: Paths, keep: Iterable[str]) -> Dict[str, Any]:
    """Project ``result``, carrying its ``keep`` keys through untouched after the projected ones.

    Those keys say whether the answer is complete (a count, a cap, the query echo), so a caller
    must see them whatever it asked for.
    """
    keep = [key for key in keep if key in result]
    answer = project({key: value for key, value in result.items() if key not in keep}, paths)
    answer.update({key: result[key] for key in keep})
    return answer


def _drop_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_nulls(item) for item in value]
    return value


def dumps(value: Any, *, compact: bool, projected: bool) -> str:
    """Serialize a tool result.

    ``compact=False`` is byte-for-byte the output every tool produced before 2.19.0. ``compact``
    removes all padding and keeps non-ASCII characters as they are (an escaped umlaut costs
    several tokens); it drops nulls only when nothing was requested by name.
    """
    if not compact:
        return json.dumps(value, indent=2, default=str)
    if not projected:
        value = _drop_nulls(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


def error(message: str, code: int, *, compact: bool = False, **extra: Any) -> str:
    """The error shape every tool already returns, honouring ``compact``."""
    return dumps({"error": message, "code": code, **extra}, compact=compact, projected=True)
