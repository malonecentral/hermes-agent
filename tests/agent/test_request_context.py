import asyncio

import pytest
from unittest.mock import patch

from agent.request_context import (
    bind_mcp_meta,
    consume_cli_request_context,
    get_mcp_meta,
    resolve_request_mcp_meta,
)


def test_metadata_is_scoped_by_server_and_resets():
    assert get_mcp_meta("family_shared_notes") is None
    with bind_mcp_meta({"family_shared_notes": {"jarvisIdentityContext": "notes-token"}}):
        assert get_mcp_meta("family_shared_notes") == {"jarvisIdentityContext": "notes-token"}
        assert get_mcp_meta("family_home") is None
    assert get_mcp_meta("family_shared_notes") is None


def test_metadata_is_defensively_copied():
    supplied = {"family_shared_notes": {"jarvisIdentityContext": "notes-token"}}
    with bind_mcp_meta(supplied):
        supplied["family_shared_notes"]["jarvisIdentityContext"] = "mutated"
        observed = get_mcp_meta("family_shared_notes")
        assert observed == {"jarvisIdentityContext": "notes-token"}
        assert observed is not None
        observed["jarvisIdentityContext"] = "also-mutated"
        assert get_mcp_meta("family_shared_notes") == {"jarvisIdentityContext": "notes-token"}


def test_parallel_tasks_do_not_cross_metadata(monkeypatch):
    monkeypatch.setenv(
        "HERMES_REQUEST_MCP_META",
        '{"family_shared_notes":{"jarvisIdentityContext":"process-global"}}',
    )

    async def read(token):
        metadata = (
            {"family_shared_notes": {"jarvisIdentityContext": token}}
            if token is not None
            else None
        )
        with bind_mcp_meta(metadata):
            await asyncio.sleep(0)
            return get_mcp_meta("family_shared_notes")

    async def run_pair():
        return await asyncio.gather(read("request-local"), read(None))

    authenticated, anonymous = asyncio.run(run_pair())
    assert authenticated == {"jarvisIdentityContext": "request-local"}
    assert anonymous is None
    assert get_mcp_meta("family_shared_notes") is None


def test_rejects_non_json_metadata():
    with pytest.raises(ValueError):
        with bind_mcp_meta({"family_shared_notes": {"bad": object()}}):
            pass


def test_resolver_receives_trusted_ingress_and_merges_one_valid_result():
    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[
            None,
            {"mcp_meta": {"notes": {"identity": "opaque"}}},
            None,
        ],
    ) as invoke:
        resolved = resolve_request_mcp_meta(
            ingress={"kind": "api", "binding": "verified-by-adapter"},
            platform="api",
        )

    assert resolved == {"notes": {"identity": "opaque"}}
    invoke.assert_called_once_with(
        "resolve_request_context",
        ingress={"kind": "api", "binding": "verified-by-adapter"},
        platform="api",
    )


def test_resolver_rejects_invalid_or_conflicting_results():
    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        return_value=[
            {"mcp_meta": {"notes": {"identity": "one"}}},
            {"mcp_meta": {"notes": {"identity": "two"}}},
        ],
    ):
        with pytest.raises(ValueError, match="conflicting"):
            resolve_request_mcp_meta(ingress={"kind": "local-cli"}, platform="cli")


def test_cli_handoff_reaches_request_local_tool_context_and_is_consumed(monkeypatch):
    requester = {
        "person_id": "person-alice",
        "policy_generation": 1,
        "binding_generation": 1,
        "binding_ref": "binding-reference-123456789",
    }
    monkeypatch.setenv(
        "HERMES_REQUEST_MCP_META",
        __import__("json").dumps(
            {"family_home": {"jarvisRequester": requester}},
            separators=(",", ":"),
        ),
    )

    ingress = consume_cli_request_context()
    assert "HERMES_REQUEST_MCP_META" not in __import__("os").environ
    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]):
        resolved = resolve_request_mcp_meta(ingress=ingress, platform="cli")
    with bind_mcp_meta(resolved):
        assert get_mcp_meta("family_home") == {"jarvisRequester": requester}
        assert get_mcp_meta("family_calendar") is None
    assert get_mcp_meta("family_home") is None


def test_cli_handoff_absent_isolated_from_process_context(monkeypatch):
    monkeypatch.delenv("HERMES_REQUEST_MCP_META", raising=False)
    assert consume_cli_request_context() is None
    assert get_mcp_meta("family_home") is None


@pytest.mark.parametrize("raw", ["{", "[]", '{"family_home":NaN}'])
def test_cli_handoff_malformed_fails_closed_and_is_removed(monkeypatch, raw):
    monkeypatch.setenv("HERMES_REQUEST_MCP_META", raw)
    with pytest.raises((TypeError, ValueError)):
        consume_cli_request_context()
    assert "HERMES_REQUEST_MCP_META" not in __import__("os").environ


def test_cli_handoff_is_bounded(monkeypatch):
    monkeypatch.setenv("HERMES_REQUEST_MCP_META", '{"x":"' + "a" * 65536 + '"}')
    with pytest.raises(ValueError, match="size limit"):
        consume_cli_request_context()
    assert "HERMES_REQUEST_MCP_META" not in __import__("os").environ
