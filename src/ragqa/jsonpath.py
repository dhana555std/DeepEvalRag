"""Tiny JSON path helper used to map arbitrary RAG API payloads.

Why it exists
-------------
The retrieval and generation endpoints are described in ``config/rag_api.yaml``
by *paths* into their JSON responses, so a different payload shape needs a
config change rather than a code change.

Supported syntax
----------------
* ``a.b.c``        - nested keys
* ``a[0].b``       - list index
* ``a[*].b``       - wildcard: returns a list of ``b`` for every element of ``a``
* ``""`` / ``"."`` - the document itself
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"([^.\[\]]+)|\[(\*|\d+)\]")
_MISSING = object()


def extract(doc: Any, path: str | None, default: Any = None) -> Any:
    """Return the value at ``path`` in ``doc`` or ``default`` if absent."""
    if path in (None, "", "."):
        return doc
    result = _walk(doc, [m for m in _TOKEN.finditer(path)])
    return default if result is _MISSING else result


def _walk(node: Any, tokens: list[re.Match[str]]) -> Any:
    if not tokens:
        return node
    key, index = tokens[0].group(1), tokens[0].group(2)
    rest = tokens[1:]
    if key is not None:
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        return _walk(node[key], rest)
    if not isinstance(node, list):
        return _MISSING
    if index == "*":
        out = [_walk(item, rest) for item in node]
        return [o for o in out if o is not _MISSING]
    i = int(index)
    if i >= len(node):
        return _MISSING
    return _walk(node[i], rest)
