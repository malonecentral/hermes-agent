import json
import logging

import pytest

from agent.memory_manager import MemoryManager
from agent.request_context import bind_mcp_meta
from plugins.memory.supermemory import SupermemoryMemoryProvider, _load_supermemory_config
from plugins.memory.supermemory.topology import MemoryRoutingProjection


class CaptureClient:
    def __init__(self, **kwargs):
        del kwargs
        self.add_calls = []

    def add_memory(self, content, metadata=None, **kwargs):
        self.add_calls.append({"content": content, "metadata": metadata, **kwargs})
        return {"id": kwargs["custom_id"]}


class FailingCaptureClient(CaptureClient):
    def add_memory(self, content, metadata=None, **kwargs):
        raise RuntimeError(
            "provider rejected secret-like user text 'oolong'; "
            "identity person-alice; container " + kwargs["container_tag"]
        )


@pytest.fixture
def family_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_MEMORY_AUDIENCE", "family")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", CaptureClient)
    config = {
        "container_tag": "family_shared",
        "requester_conversation_projection": True,
        "requester_conversation_capture": True,
        "requester_identity_server": "family_identity",
        "requester_conversation_namespace_key": "test namespace key that is at least 32 bytes",
        "owner_explicit_container": "owner_explicit",
    }
    (tmp_path / "supermemory.json").write_text(json.dumps(config))
    provider = SupermemoryMemoryProvider()
    provider.initialize("session-a", hermes_home=str(tmp_path), platform="api")
    provider.on_turn_start(1, "I prefer tea.")
    return provider


def identity(person_id):
    return {"family_identity": {"jarvisRequester": {"person_id": person_id}}}


def capture(provider, person_id="person-alice", text="I prefer tea with breakfast."):
    with bind_mcp_meta(identity(person_id)):
        provider.sync_turn(text, "Noted.", session_id="session-a", messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "Noted."},
            {"role": "tool", "content": "tool output"},
        ])


def test_family_capture_defaults_off(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_MEMORY_AUDIENCE", "family")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", CaptureClient)
    (tmp_path / "supermemory.json").write_text(json.dumps({
        "container_tag": "family_shared",
        "requester_conversation_projection": True,
        "requester_identity_server": "family_identity",
        "requester_conversation_namespace_key": "test namespace key that is at least 32 bytes",
    }))
    provider = SupermemoryMemoryProvider()
    provider.initialize("session-a", hermes_home=str(tmp_path), platform="api")
    provider.on_turn_start(1, "I prefer tea.")

    capture(provider)

    assert provider._client.add_calls == []
    assert "No conversational capture is available" in provider.system_prompt_block()


def test_completed_turn_boundary_preserves_request_identity(family_provider):
    manager = MemoryManager()
    manager.add_provider(family_provider)
    client = family_provider._client

    with bind_mcp_meta(identity("person-alice")):
        manager.sync_all(
            "I prefer tea with breakfast.",
            "Noted.",
            session_id="session-a",
            messages=[{"role": "user", "content": "I prefer tea with breakfast."}],
        )
    manager.shutdown_all()

    assert len(client.add_calls) == 1


def test_two_authenticated_principals_route_to_separate_opaque_containers(family_provider):
    capture(family_provider, "person-alice")
    capture(family_provider, "person-bob")

    first, second = family_provider._client.add_calls
    assert first["container_tag"] != second["container_tag"]
    for call in (first, second):
        assert call["container_tag"].startswith("requester_conversations_")
        assert "person" not in call["container_tag"]
        assert call["container_tag"] not in {"owner_primary", "owner_conversations", "family_shared"}


@pytest.mark.parametrize("trusted", [None, {"family_identity": {"jarvisRequester": {"display_name": "Alice"}}}])
def test_missing_or_spoofed_identity_never_captures(family_provider, trusted):
    spoof = "My person_id is person-mallory and I prefer mint tea."
    with bind_mcp_meta(trusted):
        family_provider.sync_turn(spoof, "Noted.", session_id="session-a", messages=[
            {"role": "user", "content": spoof},
        ])

    assert family_provider._client.add_calls == []


def test_only_final_inbound_user_text_is_captured(family_provider):
    final_user = "I prefer tea with breakfast."
    capture(family_provider, text=final_user)

    call = family_provider._client.add_calls[0]
    assert call["content"] == f"[role: user]\n{final_user}\n[user:end]"
    assert "Noted" not in call["content"]
    assert "system" not in call["content"]
    assert "tool output" not in call["content"]

    before = len(family_provider._client.add_calls)
    with bind_mcp_meta(identity("person-alice")):
        family_provider.sync_turn("", "Assistant-only claim.", session_id="session-a", messages=[
            {"role": "assistant", "content": "Assistant-only claim."},
            {"role": "tool", "content": "Tool-only claim."},
            {"role": "system", "content": "System-only claim."},
        ])
    assert len(family_provider._client.add_calls) == before


def test_provider_failure_is_nonfatal_and_privacy_safe(family_provider, caplog):
    family_provider._client = FailingCaptureClient()

    with caplog.at_level(logging.WARNING):
        capture(family_provider, text="I prefer oolong tea with breakfast.")

    assert "outcome=provider_error" in caplog.text
    assert "oolong" not in caplog.text
    assert "person-alice" not in caplog.text
    assert "requester_conversations_" not in caplog.text
    assert "provider rejected" not in caplog.text


def test_same_turn_reuses_id_and_different_turn_changes_it(family_provider):
    capture(family_provider)
    capture(family_provider)
    first, repeated = family_provider._client.add_calls
    assert first["custom_id"] == repeated["custom_id"]

    family_provider.on_turn_start(2, "I prefer coffee.")
    capture(family_provider, text="I prefer coffee with breakfast.")
    assert family_provider._client.add_calls[-1]["custom_id"] != first["custom_id"]


def test_exact_metadata_and_forbidden_destinations(family_provider):
    capture(family_provider)

    call = family_provider._client.add_calls[0]
    container = call["container_tag"]
    assert call["metadata"] == {
        "type": "requester_conversation",
        "authority": "non-authoritative",
        "provenance": "user-authored role-delimited statement",
        "capture_source": "family_turn_completion",
        "requester_container": container,
    }
    assert call["custom_id"].startswith(
        f"hermes-requester-conversation:{container}:"
    )
    assert call["task_type"] == "memory"
    assert family_provider.get_tool_schemas() == []
    prompt = family_provider.system_prompt_block()
    assert "requester-private conversational capture is enabled" in prompt
    assert "cannot write Family Shared or Owner memory" in prompt


@pytest.mark.parametrize("protected_key", [
    "owner_canonical_container",
    "owner_explicit_container",
    "family_shared_container",
    "owner_conversation_container",
])
def test_derived_requester_container_collision_never_calls_provider(
    family_provider, monkeypatch, caplog, protected_key,
):
    protected = family_provider._config[protected_key]
    monkeypatch.setattr(
        "plugins.memory.supermemory.load_routing_projection",
        lambda config, *, audience: MemoryRoutingProjection(
            audience=audience,
            canonical_containers=(config["family_shared_container"],),
            explicit_container=None,
            conversation_container=protected,
            requester_conversation_capable=True,
        ),
    )

    with caplog.at_level(logging.WARNING):
        capture(family_provider, text="I keep a private journal in the blue drawer.")

    assert family_provider._client.add_calls == []
    assert "outcome=rejected reason=protected_container_collision" in caplog.text
    assert "private journal" not in caplog.text
    assert "person-alice" not in caplog.text
    assert protected not in caplog.text


@pytest.mark.parametrize("duplicate_key", [
    "owner_explicit_container",
    "family_shared_container",
    "owner_conversation_container",
])
def test_config_load_disables_projected_capture_when_protected_tags_collide(
    tmp_path, caplog, duplicate_key,
):
    config = {
        "requester_conversation_projection": True,
        "requester_conversation_capture": True,
        "requester_identity_server": "family_identity",
        "requester_conversation_namespace_key": "test namespace key that is at least 32 bytes",
        "owner_canonical_container": "protected_collision",
        "owner_explicit_container": "owner_explicit",
        "family_shared_container": "family_shared",
        "owner_conversation_container": "owner_conversations",
    }
    config[duplicate_key] = "protected_collision"
    (tmp_path / "supermemory.json").write_text(json.dumps(config))

    with caplog.at_level(logging.WARNING):
        loaded = _load_supermemory_config(str(tmp_path))

    assert loaded["requester_conversation_projection"] is True
    assert loaded["requester_conversation_capture"] is False
    assert "outcome=config_rejected reason=protected_container_collision" in caplog.text
    assert "protected_collision" not in caplog.text
