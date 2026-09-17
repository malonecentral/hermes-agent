import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.run import GatewayRunner
from plugins.platforms.discord.adapter import DiscordAdapter


def test_voice_autojoin_config_prefers_platform_extra(monkeypatch):
    for key in (
        "HERMES_DISCORD_VOICE_AUTO_JOIN",
        "HERMES_DISCORD_VOICE_AUTO_JOIN_CHANNEL_ID",
        "HERMES_DISCORD_VOICE_AUTO_JOIN_CHANNEL_NAME",
        "HERMES_DISCORD_VOICE_AUTO_JOIN_TEXT_CHANNEL_ID",
        "HERMES_DISCORD_VOICE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    adapter = DiscordAdapter(PlatformConfig(
        enabled=True, token="test-token", extra={
            "voice_auto_join": True, "voice_auto_join_channel_id": "123",
            "voice_auto_join_channel_name": "General",
            "voice_auto_join_text_channel_id": "456", "voice_timeout_seconds": 0,
        },
    ))
    assert (adapter._voice_auto_join_enabled, adapter._voice_auto_join_channel_id) == (True, 123)
    assert (adapter._voice_auto_join_channel_name, adapter._voice_auto_join_text_channel_id) == ("General", 456)
    assert adapter._voice_timeout_limit() == 0


@pytest.mark.anyio
async def test_autojoin_precedes_a_slow_slash_sync(monkeypatch):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._client = SimpleNamespace(application_id=999, user=SimpleNamespace(id=999))
    order = []

    async def autojoin():
        order.append("autojoin")

    async def slow_sync():
        order.append("sync")
        await asyncio.Event().wait()

    monkeypatch.setattr(adapter, "_maybe_auto_join_voice_channel", autojoin)
    monkeypatch.setattr(adapter, "_get_discord_command_sync_policy", lambda: "safe")
    monkeypatch.setattr(adapter, "_desired_command_sync_fingerprint", lambda: "fingerprint")
    monkeypatch.setattr(adapter, "_command_sync_skip_reason", lambda *_: None)
    monkeypatch.setattr(adapter, "_record_command_sync_attempt", lambda *_: None)
    monkeypatch.setattr(adapter, "_safe_sync_slash_commands", slow_sync)
    task = asyncio.create_task(adapter._run_post_connect_initialization())
    for _ in range(20):
        if order == ["autojoin", "sync"]:
            break
        await asyncio.sleep(0)
    assert order == ["autojoin", "sync"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_voice_input_plays_quick_ack_before_dispatch(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._is_duplicate_voice_transcript = lambda *_: False
    order = []

    async def acknowledge(guild_id, phrase):
        assert (guild_id, phrase) == (7, "Working on it.")
        order.append("ack")

    adapter = SimpleNamespace(
        _voice_text_channels={7: 42},
        _voice_sources={},
        _owner_profile=None,
        _client=SimpleNamespace(get_channel=lambda _: None),
        play_ack_in_voice=acknowledge,
        handle_message=AsyncMock(side_effect=lambda _event: order.append("dispatch")),
    )
    monkeypatch.setenv("HERMES_DISCORD_VOICE_QUICK_ACK", "true")
    monkeypatch.setenv("HERMES_DISCORD_VOICE_QUICK_ACK_TEXT", "Working on it.")

    await runner._handle_voice_channel_input(7, 99, "status", adapter=adapter)

    assert order == ["ack", "dispatch"]