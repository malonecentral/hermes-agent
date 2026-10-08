"""Immutable, trusted routing contracts for Supermemory production topology.

This module selects no retrieval or capture behavior.  It projects validated
configuration and request-local authenticated identity into routes that later
integration slices can consume without accepting model/display-name input.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any, Mapping

from agent.request_context import get_mcp_meta

class RoutingProjectionError(ValueError):
    """The requested audience has no valid production topology projection."""


_REQUESTER_CONVERSATION_PREFIX = "requester_conversations_"


def validate_topology_destinations(config: Mapping[str, Any]) -> None:
    """Reject an enabled topology whose server-owned namespaces overlap."""
    if config.get("routing_projection_enabled") is not True:
        return
    names = (
        "owner_canonical_container", "owner_explicit_container",
        "family_shared_container", "owner_conversation_container",
    )
    tags = [config.get(name) for name in names]
    if any(not isinstance(tag, str) or not tag for tag in tags):
        raise RoutingProjectionError("memory topology contains an invalid destination")
    tags = [str(tag) for tag in tags]
    if len(set(tags)) != len(tags):
        raise RoutingProjectionError("memory topology destinations collide")
    if any(tag.startswith(_REQUESTER_CONVERSATION_PREFIX) for tag in tags):
        raise RoutingProjectionError("memory topology collides with a protected requester namespace")


@dataclass(frozen=True, slots=True)
class MemoryRoutingProjection:
    """One immutable, server-owned projection of allowed memory containers."""

    audience: str
    canonical_containers: tuple[str, ...]
    explicit_container: str | None
    conversation_container: str | None
    requester_conversation_capable: bool


def _trusted_principal_id(config: Mapping[str, Any]) -> str | None:
    """Read an opaque principal only through the request-context public API."""
    server_name = config.get("requester_identity_server")
    if not isinstance(server_name, str) or not server_name:
        return None
    projected = get_mcp_meta(server_name)
    requester = projected.get("jarvisRequester") if isinstance(projected, dict) else None
    principal_id = requester.get("person_id") if isinstance(requester, dict) else None
    if not isinstance(principal_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:-]{2,255}", principal_id
    ):
        return None
    return principal_id


def _requester_conversation_container(
    config: Mapping[str, Any], principal_id: str
) -> str | None:
    """Derive a stable tag without placing raw identity in the namespace."""
    key = config.get("requester_conversation_namespace_key")
    if not isinstance(key, str) or len(key.encode("utf-8")) < 32:
        return None
    digest = hmac.new(
        key.encode("utf-8"),
        b"hermes-supermemory-requester-conversation-v1\0" + principal_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"requester_conversations_{digest[:32]}"


def load_routing_projection(
    config: Mapping[str, Any], *, audience: str
) -> MemoryRoutingProjection:
    """Project one route from validated config and trusted request identity.

    Family requester-conversation capability fails closed unless it is enabled,
    a namespace key is configured, and the authenticated registry projection is
    available through :func:`agent.request_context.get_mcp_meta`.
    """
    validate_topology_destinations(config)
    if audience == "owner":
        return MemoryRoutingProjection(
            audience="owner",
            canonical_containers=(
                str(config["owner_canonical_container"]),
                str(config["family_shared_container"]),
            ),
            explicit_container=str(config["owner_explicit_container"]),
            conversation_container=str(config["owner_conversation_container"]),
            requester_conversation_capable=False,
        )
    if audience != "family":
        raise RoutingProjectionError("unknown memory audience")

    conversation = None
    if config.get("requester_conversation_projection") is True:
        principal_id = _trusted_principal_id(config)
        if principal_id is not None:
            conversation = _requester_conversation_container(config, principal_id)
    return MemoryRoutingProjection(
        audience="family",
        canonical_containers=(str(config["family_shared_container"]),),
        explicit_container=None,
        conversation_container=conversation,
        requester_conversation_capable=conversation is not None,
    )
