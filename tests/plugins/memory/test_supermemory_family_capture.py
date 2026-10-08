import json

from plugins.memory.supermemory import SupermemoryMemoryProvider, _load_supermemory_config
from plugins.memory.supermemory.topology import load_routing_projection


class CaptureClient:
    def __init__(self, **kwargs):
        del kwargs
        self.add_calls = []

    def add_memory(self, content, metadata=None, **kwargs):
        self.add_calls.append({"content": content, "metadata": metadata, **kwargs})
        return {"id": kwargs.get("custom_id", "fixture")}


def configured_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_MEMORY_AUDIENCE", "family")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", CaptureClient)
    (tmp_path / "supermemory.json").write_text(json.dumps({
        "container_tag": "family_shared",
        "routing_projection_enabled": True,
        "requester_conversation_projection": True,
        "requester_conversation_capture": True,
        "requester_identity_server": "family_identity",
        "requester_conversation_namespace_key": "test namespace key that is at least 32 bytes",
        "owner_canonical_container": "owner_canonical",
        "owner_explicit_container": "owner_primary",
        "family_shared_container": "family_shared",
        "owner_conversation_container": "owner_conversations",
    }))
    provider = SupermemoryMemoryProvider()
    provider.initialize("session-a", hermes_home=str(tmp_path), platform="api")
    return provider


def test_plugin_forces_family_requester_capture_off(monkeypatch, tmp_path):
    provider = configured_provider(monkeypatch, tmp_path)

    assert provider._config["requester_conversation_capture"] is False
    provider.on_turn_start(1, "I prefer tea.")
    provider.sync_turn("I prefer tea.", "Noted.", session_id="session-a")

    assert provider._client.add_calls == []
    assert "capture is owned by the private executor" in provider.system_prompt_block()


def test_requester_projection_remains_available_for_retrieval(monkeypatch, tmp_path):
    provider = configured_provider(monkeypatch, tmp_path)
    config = _load_supermemory_config(str(tmp_path))
    monkeypatch.setattr(
        "plugins.memory.supermemory.topology.get_mcp_meta",
        lambda _server: {"jarvisRequester": {"person_id": "person-alice"}},
    )

    route = load_routing_projection(config, audience="family")

    assert route.canonical_containers == ("family_shared",)
    assert route.requester_conversation_capable is True
    assert route.conversation_container.startswith("requester_conversations_")
    assert route.conversation_container not in {
        "owner_canonical", "owner_primary", "family_shared", "owner_conversations",
    }


def test_family_provider_exposes_no_model_write_tools(monkeypatch, tmp_path):
    provider = configured_provider(monkeypatch, tmp_path)

    assert provider.get_tool_schemas() == []


def test_owner_save_and_forget_cannot_target_canonical_projections(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setenv("HERMES_MEMORY_AUDIENCE", "owner")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", CaptureClient)
    (tmp_path / "supermemory.json").write_text(json.dumps({
        "container_tag": "owner_primary",
        "routing_projection_enabled": True,
        "enable_custom_container_tags": True,
        "custom_containers": ["owner_canonical", "family_shared"],
        "owner_canonical_container": "owner_canonical",
        "owner_explicit_container": "owner_primary",
        "family_shared_container": "family_shared",
        "owner_conversation_container": "owner_conversations",
    }))
    provider = SupermemoryMemoryProvider()
    provider.initialize("owner-session", hermes_home=str(tmp_path), platform="api")

    for protected in ("owner_canonical", "family_shared"):
        saved = json.loads(provider.handle_tool_call(
            "supermemory_store", {"content": "synthetic", "container_tag": protected},
        ))
        forgotten = json.loads(provider.handle_tool_call(
            "supermemory_forget", {"id": "synthetic-id", "container_tag": protected},
        ))
        assert "error" in saved
        assert "error" in forgotten
    assert provider._client.add_calls == []
