import json

import pytest

from agent.request_context import bind_mcp_meta
from plugins.memory.supermemory import _load_supermemory_config
from plugins.memory.supermemory.topology import (
    MemoryRoutingProjection,
    RoutingProjectionError,
    load_routing_projection,
)


def test_topology_config_defaults_preserve_production_containers_and_disable_projection(tmp_path):
    config = _load_supermemory_config(str(tmp_path))

    assert config["owner_canonical_container"] == "owner_primary"
    assert config["owner_explicit_container"] == "owner_primary"
    assert config["family_shared_container"] == "family_shared"
    assert config["owner_conversation_container"] == "owner_conversations"
    assert config["requester_conversation_projection"] is False
    assert config["requester_identity_server"] == ""
    assert config["requester_conversation_namespace_key"] == ""


def test_topology_config_accepts_valid_optional_fields(tmp_path):
    (tmp_path / "supermemory.json").write_text(json.dumps({
        "owner_canonical_container": "canonical_v2",
        "owner_explicit_container": "explicit_v2",
        "family_shared_container": "shared_v2",
        "owner_conversation_container": "owner_chat_v2",
        "requester_conversation_projection": True,
        "requester_identity_server": "family_identity",
        "requester_conversation_namespace_key": "a sufficiently long namespace secret",
    }))

    config = _load_supermemory_config(str(tmp_path))

    assert config["owner_canonical_container"] == "canonical_v2"
    assert config["owner_explicit_container"] == "explicit_v2"
    assert config["family_shared_container"] == "shared_v2"
    assert config["owner_conversation_container"] == "owner_chat_v2"
    assert config["requester_conversation_projection"] is True


def test_invalid_topology_config_falls_back_and_capability_fails_closed(tmp_path):
    (tmp_path / "supermemory.json").write_text(json.dumps({
        "owner_canonical_container": "../private",
        "owner_explicit_container": "",
        "family_shared_container": ["family_shared"],
        "owner_conversation_container": "contains spaces",
        "requester_conversation_projection": True,
        "requester_identity_server": "bad/server",
        "requester_conversation_namespace_key": "short",
    }))

    config = _load_supermemory_config(str(tmp_path))

    assert config["owner_canonical_container"] == "owner_primary"
    assert config["owner_explicit_container"] == "owner_primary"
    assert config["family_shared_container"] == "family_shared"
    assert config["owner_conversation_container"] == "owner_conversations"
    assert config["requester_conversation_projection"] is False


def test_owner_projection_is_immutable_and_matches_current_topology(tmp_path):
    projection = load_routing_projection(_load_supermemory_config(str(tmp_path)), audience="owner")

    assert projection == MemoryRoutingProjection(
        audience="owner",
        canonical_containers=("owner_primary", "family_shared"),
        explicit_container="owner_primary",
        conversation_container="owner_conversations",
        requester_conversation_capable=False,
    )
    with pytest.raises(AttributeError):
        projection.audience = "family"


def test_family_projection_uses_only_trusted_request_context_and_hmac_tag(tmp_path):
    config_path = tmp_path / "supermemory.json"
    config_path.write_text(json.dumps({
        "requester_conversation_projection": True,
        "requester_identity_server": "family_identity",
        "requester_conversation_namespace_key": "a sufficiently long namespace secret",
    }))
    config = _load_supermemory_config(str(tmp_path))
    trusted = {"family_identity": {"jarvisRequester": {"person_id": "person-aaron"}}}

    with bind_mcp_meta(trusted):
        first = load_routing_projection(config, audience="family")
        second = load_routing_projection(config, audience="family")

    assert first == second
    assert first.canonical_containers == ("family_shared",)
    assert first.explicit_container is None
    assert first.requester_conversation_capable is True
    assert first.conversation_container.startswith("requester_conversations_")
    assert "aaron" not in first.conversation_container.casefold()
    assert "person" not in first.conversation_container.casefold()


def test_family_projection_fails_closed_without_trusted_projection(tmp_path):
    (tmp_path / "supermemory.json").write_text(json.dumps({
        "requester_conversation_projection": True,
        "requester_identity_server": "family_identity",
        "requester_conversation_namespace_key": "a sufficiently long namespace secret",
    }))
    config = _load_supermemory_config(str(tmp_path))

    with bind_mcp_meta(None):
        projection = load_routing_projection(config, audience="family")
    assert projection.conversation_container is None
    assert projection.requester_conversation_capable is False

    with bind_mcp_meta({"family_identity": {"jarvisRequester": {"display_name": "Aaron"}}}):
        projection = load_routing_projection(config, audience="family")
    assert projection.conversation_container is None
    assert projection.requester_conversation_capable is False


def test_projection_rejects_unknown_audience(tmp_path):
    with pytest.raises(RoutingProjectionError, match="audience"):
        load_routing_projection(_load_supermemory_config(str(tmp_path)), audience="guest")
