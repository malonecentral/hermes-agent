"""Regression tests for conversation loop fallback state management."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_defs(*names):
    """Helper: create minimal tool definitions for given names."""
    return [
        {
            "type": "function", "function": {
                "name": name,
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        for name in names
    ]


def _tool_call(name, call_id):
    """Helper: create a minimal tool call object."""
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _response(*, content, finish_reason, tool_calls=None):
    """Helper: create a minimal API response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


_REFINEMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "supermemory_search",
        "description": "Search only the staged memory provider.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _make_host_gate_agent(prefetch, responses, ordinary_tools=None):
    ordinary_tools = ordinary_tools or _tool_defs("web_search", "terminal")
    with (
        patch("run_agent.get_tool_definitions", return_value=ordinary_tools),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._cached_system_prompt = "You are helpful."
    # Preserve the worker's per-agent resolved registry snapshot even if plugin
    # discovery refreshes module-level registry aliases during construction.
    setattr(agent, "tools", ordinary_tools)
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {
        tool["function"]["name"] for tool in ordinary_tools
    }
    agent._memory_manager = MagicMock()
    agent._memory_manager._external_prefetch_timeout = 1.0
    agent._memory_manager.prefetch_all.return_value = prefetch
    agent._memory_manager.describe_recall.return_value = ""
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = responses
    return agent, ordinary_tools


def _run_host_gate(agent, *, input_mode="typed", mock_dispatch=True):
    gate = {
        "sentinel": '{"insufficient_evidence":true}',
        "refinement_schema": _REFINEMENT_SCHEMA,
    }
    persisted_snapshots = []

    def capture_persist(messages, *_args, **_kwargs):
        persisted_snapshots.append(copy.deepcopy(messages))

    dispatch = (
        patch("run_agent.handle_function_call", return_value="scoped result")
        if mock_dispatch else __import__("contextlib").nullcontext()
    )
    with dispatch, patch.object(
        agent, "_persist_session", side_effect=capture_persist
    ), patch.object(agent, "_save_trajectory"), patch.object(
        agent, "_cleanup_task_resources"
    ):
        result = agent.run_conversation(
            "Which fact applies?",
            host_staged_memory_gate=gate,
            trusted_request_context={"input_mode": input_mode},
        )
    return result, persisted_snapshots


def _sent_tools(agent):
    return [
        call.kwargs.get("tools", [])
        for call in agent.client.chat.completions.create.call_args_list
    ]


def _sent_messages(agent):
    return [
        call.kwargs.get("messages", [])
        for call in agent.client.chat.completions.create.call_args_list
    ]


@pytest.mark.parametrize("input_mode", ["typed", "voice"])
def test_host_staged_gate_incapability_prose_is_an_answer_not_control(input_mode):
    answer = "I cannot use tools in this response."
    agent, _ = _make_host_gate_agent(
        "irrelevant canonical prefetched evidence",
        [_response(content=answer, finish_reason="stop")],
    )

    result, _ = _run_host_gate(agent, input_mode=input_mode)

    assert result["final_response"] == answer
    assert result["api_calls"] == 1
    assert _sent_tools(agent) == [[]]


def test_host_staged_gate_sends_request_local_insufficiency_contract():
    agent, _ = _make_host_gate_agent(
        "irrelevant canonical prefetched evidence",
        [_response(content="Grounded answer.", finish_reason="stop")],
    )

    result, _ = _run_host_gate(agent)

    assert result["final_response"] == "Grounded answer."
    first_request = repr(_sent_messages(agent)[0])
    assert "return exactly" in first_request.lower()
    assert '{"insufficient_evidence":true}' in first_request


def test_host_staged_gate_keeps_sufficient_direct_answer():
    answer = "Molly's usual Hob Nob order is recorded as fish and chips."
    agent, _ = _make_host_gate_agent(
        "Molly's usual Hob Nob order: fish and chips.",
        [_response(content=answer, finish_reason="stop")],
    )

    result, _ = _run_host_gate(agent)

    assert result["final_response"] == answer
    assert result["api_calls"] == 1


def test_host_staged_gate_keeps_canonical_explicit_negative_answer():
    answer = "I don't have a usual Hob Nob order recorded in Family Shared Notes."
    agent, _ = _make_host_gate_agent(
        "Family Shared Notes explicitly records: Molly has no usual Hob Nob order.",
        [_response(content=answer, finish_reason="stop")],
    )

    result, _ = _run_host_gate(agent)

    assert result["final_response"] == answer
    assert result["api_calls"] == 1


def test_host_staged_gate_refines_then_synthesizes_without_tools():
    sentinel = '  {"insufficient_evidence":true}\n'
    agent, _ = _make_host_gate_agent(
        "canonical prefetched evidence",
        [
            _response(content=sentinel, finish_reason="stop"),
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("supermemory_search", "refine1")],
            ),
            _response(content="Grounded answer.", finish_reason="stop"),
        ],
    )

    result, persisted = _run_host_gate(agent)

    assert result["final_response"] == "Grounded answer."
    assert result["api_calls"] == 3
    assert _sent_tools(agent) == [[], [_REFINEMENT_SCHEMA], []]
    assert "insufficient_evidence" not in repr(result["messages"])
    assert "insufficient_evidence" not in repr(persisted)


def test_host_staged_gate_second_sentinel_opens_ordinary_fallback():
    sentinel = '{"insufficient_evidence":true}'
    agent, ordinary_tools = _make_host_gate_agent(
        "canonical prefetched evidence",
        [
            _response(content=sentinel, finish_reason="stop"),
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("supermemory_search", "refine1")],
            ),
            _response(content=sentinel, finish_reason="stop"),
            _response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_tool_call("web_search", "fallback1")],
            ),
            _response(content="Fallback answer.", finish_reason="stop"),
        ],
    )

    result, persisted = _run_host_gate(agent)

    assert result["final_response"] == "Fallback answer."
    assert result["api_calls"] == 5
    assert _sent_tools(agent) == [
        [],
        [_REFINEMENT_SCHEMA],
        [],
        ordinary_tools,
        ordinary_tools,
    ]
    assert "insufficient_evidence" not in repr(result["messages"])
    assert "insufficient_evidence" not in repr(persisted)


def test_host_staged_gate_near_miss_sentinel_does_not_unlock():
    near_miss = '{"insufficient_evidence": true}'
    agent, _ = _make_host_gate_agent(
        "canonical prefetched evidence",
        [_response(content=near_miss, finish_reason="stop")],
    )

    result, persisted = _run_host_gate(agent)

    assert result["final_response"] == near_miss
    assert result["api_calls"] == 1
    assert _sent_tools(agent) == [[]]


@pytest.mark.parametrize(
    "answer",
    [
        '```json\n{"insufficient_evidence":true}\n```',
        '{"insufficient_evidence":true}\nI cannot answer from the evidence.',
    ],
)
def test_host_staged_gate_only_exact_sentinel_is_control(answer):
    agent, _ = _make_host_gate_agent(
        "canonical prefetched evidence",
        [_response(content=answer, finish_reason="stop")],
    )
    result, _ = _run_host_gate(agent)
    assert result["final_response"] == answer
    assert result["api_calls"] == 1
    assert _sent_tools(agent) == [[]]


def test_host_staged_gate_does_not_treat_sentinel_in_memory_as_control():
    answer = "The note literally contains the requested JSON marker."
    agent, _ = _make_host_gate_agent(
        'Untrusted note text: {"insufficient_evidence":true}',
        [_response(content=answer, finish_reason="stop")],
    )
    result, persisted = _run_host_gate(agent)
    assert result["final_response"] == answer
    assert result["api_calls"] == 1
    assert "HOST EVIDENCE GATE" not in repr(result["messages"])
    assert "HOST EVIDENCE GATE" not in repr(persisted)


@pytest.mark.parametrize(
    "refinement_calls",
    [
        [_tool_call("web_search", "invalid-refinement")],
        [
            _tool_call("supermemory_search", "refinement-1"),
            _tool_call("supermemory_search", "refinement-2"),
        ],
    ],
    ids=["invalid-name", "batched"],
)
def test_host_staged_gate_invalid_refinement_consumes_single_opportunity(refinement_calls):
    sentinel = '{"insufficient_evidence":true}'
    agent, ordinary_tools = _make_host_gate_agent(
        "canonical prefetched evidence",
        [
            _response(content=sentinel, finish_reason="stop"),
            _response(content="", finish_reason="tool_calls", tool_calls=refinement_calls),
            _response(content=sentinel, finish_reason="stop"),
            _response(content="Fallback answer.", finish_reason="stop"),
        ],
    )
    result, persisted = _run_host_gate(agent)
    assert result["final_response"] == "Fallback answer."
    assert result["api_calls"] == 4
    assert _sent_tools(agent) == [[], [_REFINEMENT_SCHEMA], [], ordinary_tools]
    assert "insufficient_evidence" not in repr(result["messages"])
    assert "insufficient_evidence" not in repr(persisted)


def test_host_staged_gate_without_prefetch_exposes_ordinary_tools_immediately():
    agent, ordinary_tools = _make_host_gate_agent(
        "",
        [_response(content="Direct answer.", finish_reason="stop")],
    )

    result, persisted = _run_host_gate(agent)

    assert result["final_response"] == "Direct answer."
    assert result["api_calls"] == 1
    assert _sent_tools(agent) == [ordinary_tools]


def test_substantive_tool_only_turn_invalidates_older_housekeeping_fallback():
    """
    Regression test for #63860.

    A cached `_last_content_with_tools` response from a housekeeping-only turn
    must not survive a later substantive tool-only turn. When the model returns
    an empty response after the substantive tool turn, the system should enter
    the post-tool nudge path, not use the stale housekeeping fallback.

    Production impact: scheduled cron jobs could return early without
    completing their actual work (e.g., daily report job returning a
    housekeeping message instead of producing the report artifact).

    Test sequence:
    1. Content + todo (housekeeping) → sets fallback, marks as all-housekeeping
    2. Empty content + web_search (substantive) → should CLEAR old fallback
    3. Empty content, no tool calls → should enter post-tool nudge, not use old fallback
    4. Content "Recovered after nudge." → should be returned as final response

    Before the fix:
    - Step 2 would not clear the fallback state (no visible content)
    - Step 3 would incorrectly use the housekeeping fallback from step 1
    - API calls would stop at 3, never reaching the nudge response

    After the fix:
    - Step 2 classifies tools and clears the fallback because web_search is substantive
    - Step 3 enters the post-tool nudge path (no stale housekeeping fallback available)
    - Step 4 returns the nudge response as the final answer
    """
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("todo", "web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"todo", "web_search"}
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        # Turn 1: Content + housekeeping tool
        _response(
            content="I'll begin the work.",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("todo", "todo1")],
        ),
        # Turn 2: Empty content + substantive tool (should clear stale fallback)
        _response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("web_search", "search1")],
        ),
        # Turn 3: Empty response (should enter nudge path, not use stale fallback)
        _response(content="", finish_reason="stop"),
        # Turn 4: Nudge response
        _response(content="Recovered after nudge.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the full task")

    assert result["final_response"] == "Recovered after nudge.", (
        f"Expected nudge recovery response, got: {result['final_response']}. "
        f"This indicates the stale housekeeping fallback was incorrectly used."
    )
    assert result["api_calls"] == 4, (
        f"Expected 4 API calls (including nudge), got: {result['api_calls']}. "
        f"This indicates the conversation exited early without retrying."
    )
    assert result["turn_exit_reason"].startswith("text_response"), (
        f"Expected text_response exit, got: {result['turn_exit_reason']}. "
        f"This indicates the wrong fallback path was taken."
    )


def test_bare_tool_marker_is_not_reused_as_final_response():
    """
    Regression test for #78148.

    A provider/local template can emit a bare bracketed token (e.g. "[memory]")
    as assistant content alongside a tool call. That token is protocol
    scaffolding, not an answer. If it gets cached as `_last_content_with_tools`
    and the following turn is empty, the post-tool fallback replays it as the
    final response — and because it then enters the persisted transcript,
    later context compaction preserves it, letting the model repeat the
    marker in subsequent turns.

    Test sequence:
    1. Content "[memory]" + skill_manage (housekeeping) tool call → the bare
       marker must be discarded, not cached as a fallback.
    2. Empty content, no tool calls → enters the post-tool nudge path since
       no fallback is available.
    3. Content "Recovered after nudge." → returned as the final response.

    Before the fix:
    - Step 1 cached "[memory]" as `_last_content_with_tools`.
    - Step 2 reused it via the empty-response fallback, so the conversation
      never reached step 3 and "[memory]" leaked into the persisted history.

    After the fix:
    - Step 1 strips the bare marker before it is cached or persisted.
    - Step 2 has no fallback available and enters the nudge path instead.
    - Step 3 returns the nudge response as the final answer.
    """
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("skill_manage")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.valid_tool_names = {"skill_manage"}
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        # Turn 1: Bare "[memory]" marker + housekeeping tool call.
        _response(
            content="[memory]",
            finish_reason="tool_calls",
            tool_calls=[_tool_call("skill_manage", "skill1")],
        ),
        # Turn 2: Empty response (should enter nudge path, not reuse "[memory]").
        _response(content="", finish_reason="stop"),
        # Turn 3: Nudge response
        _response(content="Recovered after nudge.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the full task")

    assert result["final_response"] != "[memory]", (
        "The bare tool-call marker leaked through as the final response — "
        "it should have been discarded before caching/persistence."
    )
    assert result["final_response"] == "Recovered after nudge.", (
        f"Expected nudge recovery response, got: {result['final_response']}."
    )
    assert result["api_calls"] == 3, (
        f"Expected 3 API calls (including nudge), got: {result['api_calls']}."
    )


@pytest.mark.parametrize("input_mode", ["typed", "voice"])
def test_host_staged_gate_reopens_real_owner_and_family_profile_registries(input_mode):
    """The fallback wire gets each production profile's resolved registry."""
    from agent.request_context import bind_mcp_meta
    from gateway.run import _profile_runtime_scope
    from hermes_cli.config import load_config
    from tools.mcp_tool import discover_mcp_tools

    homes = {
        "owner": Path.home() / ".hermes",
        "family": Path.home() / ".hermes" / "profiles" / "jarvis-family",
    }
    if not all((home / "config.yaml").is_file() for home in homes.values()):
        pytest.skip("real Owner and Family profiles are not installed")

    agents = {}
    resolved = {}
    mcp_server_names = {}
    # Resolve Owner before Family discovery adds its MCP definitions to the
    # process registry. This is the same profile/config boundary used by a
    # multiplexed worker; schemas come from real registries, never fixtures.
    for audience, home in homes.items():
        with _profile_runtime_scope(home, prepared_secret_scope={}):
            if audience == "family":
                discover_mcp_tools()
            config = load_config()
            mcp_server_names[audience] = tuple(config.get("mcp_servers", {}))
            toolsets = [
                *config["platform_toolsets"]["discord"],
                *config.get("mcp_servers", {}),
            ]
            # Construct the same per-profile AIAgent snapshot as the gateway
            # worker. Only the provider client and requirement probes are
            # isolated; registry discovery/resolution remains production code.
            with (
                patch("run_agent.OpenAI"),
                patch("run_agent.check_toolset_requirements", return_value={}),
            ):
                profile_agent = AIAgent(
                    api_key="test-key",
                    base_url="https://openrouter.ai/api/v1/",
                    quiet_mode=True,
                    skip_context_files=True,
                    skip_memory=True,
                    enabled_toolsets=toolsets,
                    platform="discord",
                )
            agents[audience] = profile_agent
            resolved[audience] = copy.deepcopy(getattr(profile_agent, "tools"))

    assert resolved["owner"]
    owner_names = {schema["function"]["name"] for schema in resolved["owner"]}
    family_names = {schema["function"]["name"] for schema in resolved["family"]}
    assert "terminal" in owner_names
    assert "terminal" not in family_names
    assert not any(name.startswith("mcp__family_") for name in owner_names)
    assert any(name.startswith("mcp__family_") for name in family_names)
    family_notes = [
        schema for schema in resolved["family"]
        if schema["function"]["name"].startswith("mcp__family_shared_notes__")
    ]
    assert family_notes
    assert all(
        schema["function"]["parameters"].get("additionalProperties") is False
        for schema in family_notes
    )

    sentinel = '{"insufficient_evidence":true}'
    for audience, ordinary_tools in resolved.items():
        requester = {
            "person_id": "person-alice",
            "policy_generation": 7,
            "binding_generation": 3,
            "binding_ref": "binding-reference-alice-123",
        }
        mcp_meta = {}
        if audience == "family":
            mcp_meta = {
                name: {"jarvisRequester": requester}
                for name in mcp_server_names[audience]
                if name.startswith("family_")
            }
        with (
            _profile_runtime_scope(homes[audience], prepared_secret_scope={}),
            bind_mcp_meta(mcp_meta),
        ):
            agent = agents[audience]
            agent._cached_system_prompt = "You are helpful."
            agent._use_prompt_caching = False
            agent.compression_enabled = False
            agent.save_trajectories = False
            agent._memory_manager = MagicMock()
            agent._memory_manager._external_prefetch_timeout = 1.0
            agent._memory_manager.prefetch_all.return_value = (
                "irrelevant canonical prefetched evidence"
            )
            agent._memory_manager.describe_recall.return_value = ""
            agent.client = MagicMock()
            agent.client.chat.completions.create.side_effect = [
                _response(content=sentinel, finish_reason="stop"),
                _response(content=sentinel, finish_reason="stop"),
                _response(content="Bounded fallback answer.", finish_reason="stop"),
            ]

            result, _ = _run_host_gate(agent, input_mode=input_mode)

        assert result["final_response"] == "Bounded fallback answer."
        fallback_wire = _sent_tools(agent)[2]
        assert _sent_tools(agent)[:2] == [[], [_REFINEMENT_SCHEMA]]
        assert fallback_wire == getattr(agent, "tools")
        if audience == "family":
            assert mcp_meta
            assert repr(requester) not in repr(
                [_sent_messages(agent), _sent_tools(agent)]
            )
            wire_family_notes = [
                schema for schema in fallback_wire
                if schema["function"]["name"].startswith(
                    "mcp__family_shared_notes__"
                )
            ]
            assert wire_family_notes == family_notes

            # Exercise the registered handler through the real dispatcher.
            # Only the final transport object is replaced, so profile
            # registry lookup, schema, dispatch, and request-context binding
            # all remain production code while no family service is touched.
            from model_tools import handle_function_call
            from tools.mcp_tool import _servers

            server = _servers["family_shared_notes"]
            original_session = server.session
            transport = MagicMock()
            transport.call_tool = AsyncMock(
                return_value=SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")],
                    isError=False,
                    structuredContent=None,
                    _meta=None,
                )
            )
            server.session = transport
            try:
                with (
                    _profile_runtime_scope(
                        homes[audience], prepared_secret_scope={}
                    ),
                    bind_mcp_meta(mcp_meta),
                ):
                    family_config = load_config()
                    from tools.tool_search import validate_deferred_call_args

                    validation_error = validate_deferred_call_args(
                        "mcp__family_shared_notes__list_notes",
                        {"jarvisRequester": requester},
                    )
                    assert validation_error is not None
                    assert "jarvisRequester" in validation_error
                    transport.call_tool.assert_not_awaited()
                    dispatch_result = handle_function_call(
                        "mcp__family_shared_notes__list_notes",
                        {},
                        enabled_toolsets=[
                            *family_config["platform_toolsets"]["discord"],
                            *family_config.get("mcp_servers", {}),
                        ],
                    )
            finally:
                server.session = original_session
            assert "ok" in dispatch_result
            transport.call_tool.assert_awaited_once_with(
                "list_notes", arguments={}, meta={"jarvisRequester": requester}
            )

