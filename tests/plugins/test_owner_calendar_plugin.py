import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from plugins.owner_calendar import _handle_read, _token_is_available, register


class Context:
    def __init__(self):
        self.tools = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)


def test_registers_one_narrow_typed_read_tool():
    context = Context()
    register(context)
    assert len(context.tools) == 1
    tool = context.tools[0]
    assert tool["name"] == "read_owner_calendar"
    assert tool["toolset"] == "owner_calendar"
    schema = tool["schema"]
    assert schema["parameters"]["required"] == ["date"]
    assert set(schema["parameters"]["properties"]) == {"date"}
    assert "America/Phoenix" in schema["description"]
    assert "date.calculate_date" not in schema["description"]
    assert "tomorrow" in schema["description"]


def test_handler_calls_app_owned_endpoint_without_shell_or_model_identity(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("t" * 32)
    token.chmod(0o600)
    monkeypatch.setattr("plugins.owner_calendar._TOKEN_PATH", token)
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return json.dumps({
                "timezone": "America/Phoenix", "date": "2026-09-18",
                "reply": "Your calendar is clear.", "events": [],
            }).encode()

    def opener(request, timeout):
        calls.append((request, timeout))
        return Response()

    result = json.loads(_handle_read({"date": "2026-09-18"}, opener=opener))

    assert result["success"] is True
    assert result["timezone"] == "America/Phoenix"
    request, timeout = calls[0]
    assert request.full_url.endswith("/v1/internal/calendar/read")
    assert json.loads(request.data) == {"date": "2026-09-18"}
    assert request.get_header("Authorization") == "Jarvis-Calendar " + "t" * 32
    assert timeout == 20
    assert b"identity" not in request.data and b"permission" not in request.data


def test_handler_passes_one_relative_day_expression_in_the_same_call(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("t" * 32)
    token.chmod(0o600)
    monkeypatch.setattr("plugins.owner_calendar._TOKEN_PATH", token)

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self, _limit):
            return json.dumps({"timezone": "America/Phoenix", "date": "2026-09-19",
                               "reply": "clear", "events": []}).encode()

    calls = []
    result = json.loads(_handle_read(
        {"date": "Saturday"},
        opener=lambda request, timeout: calls.append((request, timeout)) or Response(),
    ))
    assert result["success"] is True
    assert json.loads(calls[0][0].data) == {"date": "Saturday"}


@pytest.mark.parametrize("value", ["next Saturday", "this week", "Saturday and Sunday", "2026-09-18T00:00:00-07:00"])
def test_handler_rejects_unbounded_day_before_network(value):
    assert json.loads(_handle_read({"date": value}, opener=lambda *_args, **_kwargs: pytest.fail("network called"))) == {
        "success": False, "error": "date must name exactly one bounded America/Phoenix calendar day"
    }


def test_availability_requires_private_token_file(tmp_path, monkeypatch):
    token = tmp_path / "token"
    monkeypatch.setattr("plugins.owner_calendar._TOKEN_PATH", token)
    assert _token_is_available() is False
    token.write_text("t" * 32)
    token.chmod(0o644)
    assert _token_is_available() is False
    token.chmod(0o600)
    assert _token_is_available() is True


def test_owner_profile_config_discovers_calendar_tool_with_explicit_toolset(tmp_path):
    home = tmp_path / "owner"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - owner_calendar\ntoolsets:\n  - hermes-cli\n  - owner_calendar\n",
        encoding="utf-8",
    )
    token = home / "Library/Application Support/Jarvis/private-executor/calendar-read-token"
    token.parent.mkdir(parents=True)
    token.write_text("t" * 32, encoding="utf-8")
    token.chmod(0o600)
    script = """
from hermes_cli.plugins import discover_plugins
from model_tools import get_tool_definitions
discover_plugins(force=True)
tools = get_tool_definitions(
    enabled_toolsets=['hermes-cli', 'owner_calendar'],
    quiet_mode=True,
    skip_tool_search_assembly=True,
)
print(json.dumps(sorted(tool['function']['name'] for tool in tools)))
"""
    environment = dict(os.environ, HOME=str(home), HERMES_HOME=str(home))
    completed = subprocess.run(
        [sys.executable, "-c", "import json\n" + script],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "read_owner_calendar" in json.loads(completed.stdout.splitlines()[-1])
