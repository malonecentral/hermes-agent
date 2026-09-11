"""Request-local, transport-neutral metadata for agent work.

The context belongs to one admitted request/turn and is intentionally opaque to
models, prompts, tool schemas, transcripts, and persistent session records.
Adapters or plugins may bind JSON-safe metadata for approved downstream
transports; consumers receive defensive copies so state cannot leak between
concurrent requests.
"""
from __future__ import annotations

import contextvars
import copy
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

_REQUEST_MCP_META: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "hermes_request_mcp_meta", default=None
)

_CLI_REQUEST_MCP_META_ENV = "HERMES_REQUEST_MCP_META"
_MAX_CLI_REQUEST_MCP_META_BYTES = 64 * 1024


def _json_object(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate/copy a JSON object without retaining a caller-owned reference."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("request MCP metadata must be an object")
    copied = copy.deepcopy(dict(value))
    try:
        encoded = json.dumps(copied, ensure_ascii=False, allow_nan=False)
        parsed = json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("request MCP metadata must be JSON-safe") from exc
    if not isinstance(parsed, dict):
        raise ValueError("request MCP metadata must be an object")
    return parsed


def get_mcp_meta(server_name: str) -> dict[str, Any] | None:
    """Return a defensive per-server metadata copy for the active request.

    The bound mapping is keyed by configured MCP server name. This prevents an
    audience token intended for one constrained service being sent to another.
    """
    if not isinstance(server_name, str) or not server_name:
        return None
    all_meta = _REQUEST_MCP_META.get()
    entry = all_meta.get(server_name) if isinstance(all_meta, dict) else None
    return _json_object(entry) if isinstance(entry, Mapping) else None


def resolve_request_mcp_meta(
    *, ingress: Mapping[str, Any] | None, platform: str = ""
) -> dict[str, Any] | None:
    """Resolve approved metadata through the opt-in request-context hook.

    ``ingress`` is adapter-owned evidence, never model input. A resolver may
    return one ``{"mcp_meta": {server_name: metadata}}`` object. Multiple
    non-empty results are rejected rather than merged: ambiguity is not an
    access-control strategy.
    """
    trusted_ingress = _json_object(ingress) or {}
    direct = (
        _json_object(trusted_ingress["mcp_meta"])
        if set(trusted_ingress) == {"mcp_meta"}
        else None
    )
    from hermes_cli.lifecycle import invoke_hook

    winner = direct
    for result in invoke_hook(
        "resolve_request_context", ingress=trusted_ingress, platform=platform
    ):
        if result is None:
            continue
        if not isinstance(result, Mapping) or set(result) != {"mcp_meta"}:
            raise ValueError("request-context resolver returned an invalid result")
        candidate = _json_object(result.get("mcp_meta"))
        if candidate is None:
            continue
        if winner is not None:
            raise ValueError("conflicting request-context resolver results")
        winner = candidate
    return winner


def consume_cli_request_context() -> dict[str, Any] | None:
    """Consume and validate the subprocess-scoped mobile handoff once."""
    raw = os.environ.pop(_CLI_REQUEST_MCP_META_ENV, None)
    if raw is None:
        return None
    if len(raw.encode("utf-8")) > _MAX_CLI_REQUEST_MCP_META_BYTES:
        raise ValueError("CLI request MCP metadata exceeds the size limit")
    try:
        parsed = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
        metadata = _json_object(parsed)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("CLI request MCP metadata is malformed") from exc
    return {"mcp_meta": metadata}


def set_mcp_meta(meta_by_server: Mapping[str, Any] | None) -> contextvars.Token:
    """Bind validated metadata and return the token required to reset it."""
    return _REQUEST_MCP_META.set(_json_object(meta_by_server))


def reset_mcp_meta(token: contextvars.Token) -> None:
    """Reset metadata previously bound by :func:`set_mcp_meta`."""
    _REQUEST_MCP_META.reset(token)


@contextmanager
def bind_mcp_meta(meta_by_server: Mapping[str, Any] | None) -> Iterator[None]:
    """Bind approved per-server MCP metadata for one request scope and reset it reliably."""
    token = set_mcp_meta(meta_by_server)
    try:
        yield
    finally:
        _REQUEST_MCP_META.reset(token)
