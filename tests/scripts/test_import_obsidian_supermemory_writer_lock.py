from __future__ import annotations

import fcntl
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture()
def importer():
    spec = importlib.util.spec_from_file_location("writer_lock_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _hold(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_reset_execute_abbreviation_is_rejected_before_dispatch_or_provider_creation(
        importer, tmp_path, monkeypatch):
    effects = []
    source = tmp_path / "source"
    private = tmp_path / "private"
    source.mkdir()
    private.mkdir(mode=0o700)
    monkeypatch.setattr(importer, "reset_owner_canonical",
                        lambda *args, **kwargs: effects.append("dispatch"))
    monkeypatch.setattr(importer, "_client", lambda *args, **kwargs: effects.append("client"))

    with pytest.raises(SystemExit):
        importer._phase2_main([
            "reset-owner-canonical", "--exec", "--confirm", "opaque-token",
            "--source-root", str(source), "--private-root", str(private),
        ])

    assert effects == []


def test_reset_execute_lock_failure_precedes_provider_creation(importer, tmp_path, monkeypatch):
    lock = tmp_path / "canonical.lock"
    monkeypatch.setattr(importer, "LOCK", lock)
    provider_created = False

    def forbidden_client(*args, **kwargs):
        nonlocal provider_created
        provider_created = True
        raise AssertionError("provider must not be created while writer lock is held")

    monkeypatch.setattr(importer, "_client", forbidden_client)
    handle = _hold(lock)
    try:
        with pytest.raises(SystemExit, match="canonical writer already running"):
            importer._phase2_main([
                "reset-owner-canonical", "--source-root", str(tmp_path / "missing-source"),
                "--private-root", str(tmp_path / "missing-private"), "--execute",
                "--confirm", "opaque-token",
            ])
    finally:
        handle.close()
    assert provider_created is False


@pytest.mark.parametrize("command", [
    "reset-owner-canonical", "rebuild-owner-canonical", "sync-canonical", "reconcile",
])
def test_mutating_phase2_commands_share_nonblocking_writer_lock(
        importer, tmp_path, monkeypatch, command):
    lock = tmp_path / "canonical.lock"
    monkeypatch.setattr(importer, "LOCK", lock)
    entered = False

    def forbidden_dispatch(argv):
        nonlocal entered
        entered = True
        raise AssertionError("mutating dispatch must not start while writer lock is held")

    monkeypatch.setattr(importer, "_phase2_main_unlocked", forbidden_dispatch)
    argv = [command]
    if command == "reset-owner-canonical":
        argv.append("--execute")
    handle = _hold(lock)
    try:
        with pytest.raises(SystemExit, match="canonical writer already running"):
            importer._phase2_main(argv)
    finally:
        handle.close()
    assert entered is False


@pytest.mark.parametrize("argv", [
    ["dry-run"], ["plan"], ["verify-only"], ["reset-owner-canonical"],
])
def test_read_only_phase2_commands_do_not_create_or_touch_lock(
        importer, tmp_path, monkeypatch, argv):
    lock = tmp_path / "canonical.lock"
    monkeypatch.setattr(importer, "LOCK", lock)
    entered = []
    monkeypatch.setattr(importer, "_phase2_main_unlocked", lambda actual: entered.append(actual))

    importer._phase2_main(argv)

    assert entered == [argv]
    assert not lock.exists()

    lock.write_text("sentinel", encoding="utf-8")
    before = lock.stat().st_mtime_ns
    importer._phase2_main(argv)
    assert entered == [argv, argv]
    assert lock.read_text(encoding="utf-8") == "sentinel"
    assert lock.stat().st_mtime_ns == before
