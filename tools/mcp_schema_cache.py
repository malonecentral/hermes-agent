"""Persistent MCP tool-schema cache for lazy server startup.

Stores per-server tool manifests on disk so Hermes can register MCP tools
into the agent snapshot without spawning the stdio child process at idle
dashboard startup. Cache entries are keyed by server name + a fingerprint
of the connection config (command/args/url/tools filters).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "mcp_schema_cache.json"
_CACHE_FORMAT_VERSION = 2
_DEFAULT_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60
_cache_lock = threading.Lock()


def _cache_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / _CACHE_FILENAME


def config_fingerprint(config: dict) -> str:
    """Stable hash of every input that can affect the advertised tool schema."""
    runtime_only = {
        "connect_timeout", "enabled", "idle_timeout_seconds", "keepalive_interval",
        "lazy", "max_lifetime_seconds", "supports_parallel_tool_calls", "timeout",
    }
    payload = {
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "config": {key: value for key, value in config.items() if key not in runtime_only},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_all() -> Dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Could not read MCP schema cache %s: %s", path, exc)
        return {}


def _save_all(data: Dict[str, Any]) -> None:
    from utils import atomic_json_write

    # Cache dir + 0o600: sibling precedent in tools/registry.py
    # _save_discovery_cache; the cache file is trusted input on the lazy
    # registration path, so keep it user-only.
    atomic_json_write(_cache_path(), data, mode=0o600)


def get_cached_entry(server_name: str, fingerprint: str) -> Optional[dict]:
    """Return cached entry when fingerprint matches (and TTL holds), else None.

    MCP 2026-07-28 (SEP-2549): ``tools/list`` results carry ``ttlMs`` as a
    freshness hint. When the live discovery path recorded one, an entry
    older than its TTL is treated as a miss so the next startup re-probes
    the server instead of serving a stale manifest forever. Entries without
    a positive server TTL use a bounded 24-hour default.
    ``cacheScope`` is irrelevant here: this cache is per-user local disk,
    which satisfies even ``private``.
    """
    with _cache_lock:
        entry = _load_all().get(server_name)
    if not isinstance(entry, dict):
        return None
    if entry.get("fingerprint") != fingerprint:
        return None
    ttl_ms = entry.get("ttl_ms")
    written_at = entry.get("written_at")
    if isinstance(written_at, (int, float)):
        max_age_ms = (
            float(ttl_ms)
            if isinstance(ttl_ms, (int, float))
            else _DEFAULT_CACHE_MAX_AGE_SECONDS * 1000.0
        )
        if (time.time() - written_at) * 1000.0 >= max_age_ms:
            return None
    else:
        # Pre-v2 entries had no creation time and could remain stale forever.
        return None
    return entry


def has_cached_entry(server_name: str, fingerprint: str) -> bool:
    return get_cached_entry(server_name, fingerprint) is not None


def write_cache_entry(
    server_name: str,
    fingerprint: str,
    *,
    tools: List[dict],
    utility_tools: Optional[List[dict]] = None,
    ttl_ms: Optional[float] = None,
    cache_scope: Optional[str] = None,
) -> None:
    """Persist tool schemas after a successful live connect.

    ``ttl_ms``/``cache_scope`` are the SEP-2549 hints from the server's
    ``tools/list`` result (2026-07-28 servers). ``written_at`` anchors TTL
    expiry in :func:`get_cached_entry`.
    """
    entry = {
        "fingerprint": fingerprint,
        "tools": tools,
        "utility_tools": utility_tools or [],
        "written_at": time.time(),
    }
    if isinstance(ttl_ms, (int, float)):
        entry["ttl_ms"] = ttl_ms

    if cache_scope:
        entry["cache_scope"] = cache_scope
    with _cache_lock:
        data = _load_all()
        # Every v2 entry has a bounded lifetime, so a successful live
        # reconfirmation must always advance written_at even when the schema
        # itself is unchanged.
        data[server_name] = entry
        _save_all(data)


def clear_cache_entry(server_name: str) -> None:
    with _cache_lock:
        data = _load_all()
        if server_name in data:
            del data[server_name]
            _save_all(data)


def tools_from_cache_entry(entry: dict) -> List[dict]:
    """Return cached MCP tool dicts (name, description, inputSchema)."""
    tools = entry.get("tools")
    return list(tools) if isinstance(tools, list) else []


def utility_tools_from_cache_entry(entry: dict) -> List[dict]:
    util = entry.get("utility_tools")
    return list(util) if isinstance(util, list) else []
