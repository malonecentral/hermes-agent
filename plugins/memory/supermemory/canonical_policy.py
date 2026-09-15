"""Canonical Obsidian path, audience, and identity policy."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

OWNER_CONTAINER = "owner_primary"
FAMILY_CONTAINER = "family_shared"
CANONICAL_CONTAINERS = frozenset({OWNER_CONTAINER, FAMILY_CONTAINER})
_OWNER_CANONICAL_ROOTS = frozenset({"Jarvis", "Skills"})
_FAMILY_PREFIX = ("Jarvis", "Family Shared")
_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_PERCENT_BYTE_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
SENSITIVE_CONTENT = re.compile(
    r"(?i)(?:[#?&](?:access_)?token=|(?:api[_-]?key|password|secret)\s*[:=])"
)
# These glyphs are visually usable as separators but their Unicode names do
# not contain SOLIDUS or SLASH, so keep this small exception auditable.
_UNNAMED_SEPARATOR_LOOKALIKES = frozenset(
    {
        "∖",  # U+2216 SET MINUS
        "╱",  # U+2571 BOX DRAWINGS LIGHT DIAGONAL ...
        "╲",  # U+2572 BOX DRAWINGS LIGHT DIAGONAL ...
    }
)


def _contains_unsafe_path_text(relative_path: str) -> bool:
    """Reject path syntax that could be reinterpreted across trust boundaries."""
    if _WINDOWS_DRIVE_PREFIX.match(relative_path) or _PERCENT_BYTE_ESCAPE.search(
        relative_path
    ):
        return True
    for char in relative_path:
        name = unicodedata.name(char, "")
        if (
            char in _UNNAMED_SEPARATOR_LOOKALIKES
            or (char != "/" and ("SOLIDUS" in name or "SLASH" in name))
            or unicodedata.category(char) == "Cc"
        ):
            return True
    return False


def canonical_scope_from_path(relative_path: Any) -> str | None:
    """Return the container for an approved normalized relative Markdown path."""
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path != relative_path.strip()
        or relative_path.startswith(("/", "./"))
        or "\\" in relative_path
        or _contains_unsafe_path_text(relative_path)
        or not relative_path.endswith(".md")
    ):
        return None

    parts = relative_path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    if tuple(parts[:2]) == _FAMILY_PREFIX:
        return FAMILY_CONTAINER if len(parts) > 2 else None
    if len(parts) == 1 or parts[0] in _OWNER_CANONICAL_ROOTS:
        return OWNER_CONTAINER
    return None


def canonical_visibility_from_path(relative_path: Any) -> str | None:
    """Derive canonical visibility from an approved path."""
    scope = canonical_scope_from_path(relative_path)
    if scope == FAMILY_CONTAINER:
        return "family_shared"
    if scope == OWNER_CONTAINER:
        return "owner_private"
    return None


def stable_custom_id_from_path(relative_path: Any) -> str | None:
    """Derive the stable provider custom ID for an approved canonical path."""
    if canonical_scope_from_path(relative_path) is None:
        return None
    return "obsidian-" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def canonical_frontmatter_values(source: str) -> dict[str, list[str]]:
    """Parse the importer's deliberately small deterministic frontmatter subset."""
    values: dict[str, list[str]] = {}
    if not source.startswith("---\n"):
        return values
    end = source.find("\n---\n", 4)
    if end < 0:
        raise ValueError("malformed frontmatter")
    for line in source[4:end].splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*):\s*(.*?)\s*", line)
        if not match:
            continue
        key, raw = match.groups()
        if key in values:
            raise ValueError("duplicate frontmatter field")
        if raw.startswith("["):
            if not raw.endswith("]"):
                raise ValueError("malformed frontmatter list")
            values[key] = [v.strip().strip("\"'") for v in raw[1:-1].split(",") if v.strip()]
        elif raw in {"", "null", "[]"}:
            values[key] = []
        elif raw[:1] in {"|", ">"}:
            raise ValueError("unsupported multiline frontmatter")
        else:
            values[key] = [raw.strip("\"'")]
    return values


def canonical_entity_type(relative_path: str, fields: dict[str, str]) -> str:
    """Derive the same entity type used by the canonical importer."""
    return fields.get("type") or (
        "restaurant" if "/Food/Restaurants/" in f"/{relative_path}"
        else "dish" if "/Food/Dishes/" in f"/{relative_path}"
        else "person" if "/People/" in f"/{relative_path}"
        else "document"
    )


def canonical_exclusion_reason(relative_path: str, source: str, entity_type: str) -> str:
    """Return the import exclusion reason without exposing matched content."""
    # The generated identity envelope contains source-derived values plus the
    # relative path. Its fixed field names do not themselves match this policy.
    if SENSITIVE_CONTENT.search(source) or SENSITIVE_CONTENT.search(relative_path):
        return "sensitive_content"
    if entity_type.endswith("-template") or relative_path.rsplit("/", 1)[-1][:-3].lower().endswith("template"):
        return "template"
    return ""
