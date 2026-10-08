from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.system_prompt import build_system_prompt_parts
from agent.conversation_loop import _ollama_context_limit_error


def test_local_qwen_8192_window_is_valid_with_internal_tools_loaded():
    agent = SimpleNamespace(
        model="qwen3.5:4b",
        provider="local-qwen",
        tools=[{"function": {"name": "never-sent"}}],
        _ollama_num_ctx=8192,
    )
    assert _ollama_context_limit_error(agent, 3000) is None


def test_local_qwen_prompt_keeps_identity_and_memory_but_drops_bloat():
    store = MagicMock()
    store.format_for_system_prompt.side_effect = lambda target: {
        "memory": "CANONICAL MEMORY",
        "user": "OWNER PROFILE",
    }[target]
    agent = SimpleNamespace(
        model="qwen3.5:4b",
        provider="local-qwen",
        base_url="http://mcomen.malonecentral.com:11434/v1",
        context_compressor=SimpleNamespace(context_length=8192),
        _session_db=None,
        _memory_store=store,
        _memory_enabled=True,
        _user_profile_enabled=True,
    )

    with patch("run_agent.load_soul_md", return_value="JARVIS SOUL"):
        parts = build_system_prompt_parts(
            agent, system_message="IRRELEVANT CALLER CONTEXT"
        )

    prompt = "\n\n".join(parts.values())
    assert "JARVIS SOUL" in prompt
    assert "CANONICAL MEMORY" in prompt
    assert "OWNER PROFILE" in prompt
    assert "Local Qwen operating mode" in prompt
    assert "IRRELEVANT CALLER CONTEXT" not in prompt
    assert parts["context"] == ""
    assert len(prompt) < 12_000
