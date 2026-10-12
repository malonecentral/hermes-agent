"""Delegation authority must come from each invocation, not saved shell state."""
import json
import os
import subprocess
from pathlib import Path

from tools.environments.base import BaseEnvironment

MARKER = "HERMES_DELEGATED_CHILD_CONTEXT"


class ShellFixture(BaseEnvironment):
    def _run_bash(self, *args, **kwargs):
        raise NotImplementedError

    def cleanup(self):
        pass


def invoke(tmp_path, incoming_child, snapshot_child, snapshot_extra=""):
    env = ShellFixture(cwd=str(tmp_path), timeout=10)
    snapshot = tmp_path / "session.sh"
    snapshot.write_text((f'export {MARKER}=1\n' if snapshot_child else "") + 'export USER_WORKFLOW_VALUE=preserved\n' + snapshot_extra)
    env._snapshot_path = str(snapshot)
    env._snapshot_ready = True
    # Only an isolated fixture environment is adjusted; never the running host.
    incoming = {name: os.environ[name] for name in ("PATH", "HOME", "TMPDIR") if name in os.environ}
    if incoming_child:
        incoming[MARKER] = "1"
    import sys
    command = subprocess.list2cmdline([sys.executable, "-c", "import os,json; from agent.delegation_context import is_delegated_child_process_context; print(json.dumps([is_delegated_child_process_context(),os.environ.get('USER_WORKFLOW_VALUE')]))"])
    incoming["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(["/bin/bash", "-c", env._wrap_command(command, str(tmp_path))], env=incoming, capture_output=True, text=True, check=True)
    value = json.loads(result.stdout.splitlines()[0])
    return value, snapshot.read_text()


def test_stale_child_snapshot_cannot_mark_owner(tmp_path):
    value, snapshot = invoke(tmp_path, False, True)
    assert value == [False, "preserved"]
    assert MARKER not in snapshot


def test_real_child_keeps_guard_despite_parent_snapshot(tmp_path):
    value, snapshot = invoke(tmp_path, True, False)
    assert value == [True, "preserved"]
    assert MARKER not in snapshot


def test_real_child_keeps_guard_with_legacy_child_snapshot(tmp_path):
    value, snapshot = invoke(tmp_path, True, True)
    assert value == [True, "preserved"]
    assert MARKER not in snapshot


def test_legacy_exported_helpers_cannot_grant_owner_child_identity(tmp_path):
    prefix = f"_HERMES_RUNTIME_PASSTHROUGH_{MARKER}"
    extra = f'export {prefix}_PRESENT=x\nexport {prefix}_VALUE=1\n'
    value, snapshot = invoke(tmp_path, False, False, extra)
    assert value == [False, "preserved"]
    assert "_HERMES_RUNTIME_PASSTHROUGH_" not in snapshot


def test_legacy_exported_helpers_cannot_remove_real_child_guard(tmp_path):
    prefix = f"_HERMES_RUNTIME_PASSTHROUGH_{MARKER}"
    extra = f'export {prefix}_PRESENT=""\nexport {prefix}_VALUE=""\n'
    value, snapshot = invoke(tmp_path, True, False, extra)
    assert value == [True, "preserved"]
    assert "_HERMES_RUNTIME_PASSTHROUGH_" not in snapshot

