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
