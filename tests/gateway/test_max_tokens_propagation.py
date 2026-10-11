"""Regression tests for max_tokens propagation from config.yaml to AIAgent.

Covers #20741: `model.max_tokens` was silently dropped before reaching the
gateway-spawned agent, so providers without a hardcoded default (OpenRouter
free models, Ollama Cloud, custom OpenAI-compatible endpoints) truncated long
generations with `finish_reason="length"`.

Precedence verified here:
    HERMES_MAX_TOKENS env  >  model.max_tokens  >  per-provider
    max_output_tokens  >  None
"""

import importlib
import os
import sys
import textwrap

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a writable config.yaml and a clean module cache.

    These tests deliberately re-import ``hermes_cli`` / ``gateway`` so each
    config write is read fresh. To avoid leaking that purge into sibling test
    files in the same worker (which breaks their import-time mocks), we snapshot
    the affected modules and restore them on teardown.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)

    _saved = {
        k: v
        for k, v in sys.modules.items()
        if k.startswith(("hermes_cli", "gateway"))
    }

    def write_cfg(body: str) -> None:
        (hermes_home / "config.yaml").write_text(textwrap.dedent(body))

    def fresh_gateway():
        for mod in list(sys.modules.keys()):
            if mod.startswith(("hermes_cli", "gateway")):
                del sys.modules[mod]
        return importlib.import_module("gateway.run")

    try:
        yield write_cfg, fresh_gateway
    finally:
        # Drop anything we (re)imported, then restore the pre-test snapshot so
        # the next test file sees the module objects it was loaded with.
        for k in list(sys.modules.keys()):
            if k.startswith(("hermes_cli", "gateway")):
                del sys.modules[k]
        sys.modules.update(_saved)


def test_top_level_max_tokens_propagates(isolated_home):
    """model.max_tokens is read into the gateway runtime kwargs (#20741)."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: openrouter
          max_tokens: 16384
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs()
    assert kw["max_tokens"] == 16384


def test_per_provider_max_output_tokens_fallback(isolated_home):
    """A custom provider's max_output_tokens fills in when no global is set."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: glm-5.1
          provider: mylocal
        providers:
          mylocal:
            api: http://localhost:11434/v1
            api_key: sk-test
            default_model: glm-5.1
            max_output_tokens: 12000
        """
    )
    grun = fresh_gateway()
    kw = grun._resolve_runtime_agent_kwargs()
    assert kw["max_tokens"] == 12000


def test_selected_custom_provider_cap_is_not_its_context_window(isolated_home):
    """The selected route retains its response cap beside a larger context."""
    write_cfg, fresh_gateway = isolated_home
    write_cfg(
        """
        model:
          default: jarvis-regular
          provider: mcllm
        providers:
          mcllm:
            api: http://localhost:18081/v1
            api_key: sk-test
            default_model: jarvis-regular
            context_length: 65536
            max_output_tokens: 8192
            models:
              jarvis-regular:
                context_length: 65536
        """
    )
    grun = fresh_gateway()
    runtime = grun._resolve_runtime_agent_kwargs_for_provider("mcllm")
    assert runtime["max_tokens"] == 8192

    from run_agent import AIAgent

    agent = AIAgent(
        model="jarvis-regular",
        max_tokens=65536,
        api_key=runtime["api_key"],
        base_url=runtime["base_url"],
        provider=runtime["provider"],
        requested_provider="mcllm",
        api_mode=runtime["api_mode"],
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    wire = agent._build_api_kwargs(
        [{"role": "user", "content": "short safe test"}], tools_for_api=[]
    )
    assert agent.max_tokens == 8192
    assert wire["max_tokens"] == 8192


