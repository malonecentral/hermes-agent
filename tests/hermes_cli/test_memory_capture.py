import io
import json
from types import SimpleNamespace

import pytest

from hermes_cli.main import cmd_memory


class _Provider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.shutdown_called = False

    def initialize(self, *args, **kwargs):
        pass

    def capture_owner_app_turn(self, *args):
        if self.error:
            raise self.error
        return self.result

    def shutdown(self):
        self.shutdown_called = True


def _run(monkeypatch, capsys, provider):
    payload = {
        "session_id": "session-1",
        "request_id": "request-1",
        "user_content": "I prefer tea.",
        "assistant_content": "Noted.",
    }
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: provider)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    cmd_memory(SimpleNamespace(memory_command="capture-owner-turn"))
    return capsys.readouterr()


@pytest.mark.parametrize(("result", "status"), [(True, "delivered"), (False, "filtered")])
def test_capture_owner_turn_emits_terminal_receipts(monkeypatch, capsys, result, status):
    provider = _Provider(result=result)
    captured = _run(monkeypatch, capsys, provider)
    assert json.loads(captured.out) == {"status": status}
    assert provider.shutdown_called is True


def test_capture_owner_turn_unavailability_is_nonzero(monkeypatch, capsys):
    from plugins.memory.supermemory import OwnerAppCaptureUnavailable

    provider = _Provider(error=OwnerAppCaptureUnavailable("provider inactive"))
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, capsys, provider)
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().err) == {
        "status": "error", "error": "provider inactive"
    }
    assert provider.shutdown_called is True