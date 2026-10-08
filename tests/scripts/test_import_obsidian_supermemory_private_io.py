from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("private_io_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_private_json_round_trip_is_exact_and_private(importer, tmp_path):
    root = tmp_path / "private"
    digest = importer.private_json_write(root, "journals/txn.json", {"z": 1, "a": "é"})
    path = root / "journals/txn.json"
    expected = b'{"a":"\xc3\xa9","z":1}\n'
    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert importer.private_json_read(root, "journals/txn.json", expected_sha256=digest) == {"a": "é", "z": 1}


@pytest.mark.parametrize("child", ["/absolute.json", "../escape.json", "a/../../escape.json", "./x.json", "a\\x.json", ""])
def test_private_json_rejects_absolute_and_noncanonical_children(importer, tmp_path, child):
    with pytest.raises(importer.PrivateArtifactError):
        importer.private_json_write(tmp_path / "private", child, {})


def test_private_root_symlink_is_rejected(importer, tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    root = tmp_path / "private"
    root.symlink_to(real, target_is_directory=True)
    with pytest.raises(importer.PrivateArtifactError):
        importer.private_json_write(root, "x.json", {})


def test_private_file_and_parent_symlinks_are_rejected(importer, tmp_path):
    root = tmp_path / "private"
    importer.private_json_write(root, "x.json", {"safe": True})
    outside = tmp_path / "outside"
    outside.write_text("{}")
    (root / "x.json").unlink()
    (root / "x.json").symlink_to(outside)
    with pytest.raises(importer.PrivateArtifactError):
        importer.private_json_read(root, "x.json")
    (root / "x.json").unlink()
    (root / "linked").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(importer.PrivateArtifactError):
        importer.private_json_write(root, "linked/escape.json", {})
    assert not (tmp_path / "escape.json").exists()


def test_private_temp_symlink_is_never_followed(importer, tmp_path, monkeypatch):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    temp = root / ".x.json.fixed.tmp"
    temp.symlink_to(outside)
    monkeypatch.setattr(importer.secrets, "token_hex", lambda size: "fixed")
    with pytest.raises(importer.PrivateArtifactError, match="allocate"):
        importer.private_json_write(root, "x.json", {"unsafe": True})
    assert outside.read_text() == "unchanged"
    assert temp.is_symlink()


@pytest.mark.parametrize("kind", ["root", "directory", "file"])
def test_private_json_rejects_wrong_modes(importer, tmp_path, kind):
    root = tmp_path / "private"
    importer.private_json_write(root, "nested/x.json", {})
    target = {"root": root, "directory": root / "nested", "file": root / "nested/x.json"}[kind]
    target.chmod(0o755 if kind != "file" else 0o644)
    with pytest.raises(importer.PrivateArtifactError, match="mode"):
        importer.private_json_read(root, "nested/x.json")


def test_private_json_rejects_nonregular_and_stale_hash(importer, tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    (root / "not-file.json").mkdir(mode=0o700)
    with pytest.raises(importer.PrivateArtifactError, match="regular"):
        importer.private_json_read(root, "not-file.json")
    importer.private_json_write(root, "x.json", {"value": 1})
    with pytest.raises(importer.PrivateArtifactError, match="SHA-256"):
        importer.private_json_read(root, "x.json", expected_sha256="0" * 64)


def test_private_json_is_bounded(importer, tmp_path):
    root = tmp_path / "private"
    importer.private_json_write(root, "x.json", {"value": "long"})
    with pytest.raises(importer.PrivateArtifactError, match="maximum"):
        importer.private_json_read(root, "x.json", max_bytes=4)


@pytest.mark.parametrize("fault", ["file_fsync", "replace", "directory_fsync"])
def test_private_json_write_faults_leave_no_temp_and_never_follow_target(importer, tmp_path, monkeypatch, fault):
    root = tmp_path / "private"
    importer.private_json_write(root, "x.json", {"old": True})
    old = (root / "x.json").read_bytes()
    original_fsync = os.fsync
    original_replace = os.replace
    calls = 0

    def fsync(fd):
        nonlocal calls
        calls += 1
        if (fault == "file_fsync" and calls == 1) or (fault == "directory_fsync" and calls == 2):
            raise OSError("injected fsync")
        return original_fsync(fd)

    def replace(*args, **kwargs):
        if fault == "replace":
            raise OSError("injected replace")
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(importer.PrivateArtifactError):
        importer.private_json_write(root, "x.json", {"new": True})
    assert not [p for p in root.iterdir() if p.name != "x.json"]
    if fault != "directory_fsync":
        assert (root / "x.json").read_bytes() == old


def test_interrupted_write_cleans_temp_and_preserves_old_file(importer, tmp_path, monkeypatch):
    root = tmp_path / "private"
    importer.private_json_write(root, "x.json", {"old": True})
    old = (root / "x.json").read_bytes()
    monkeypatch.setattr(os, "write", lambda fd, data: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(importer.PrivateArtifactError):
        importer.private_json_write(root, "x.json", {"new": True})
    assert (root / "x.json").read_bytes() == old
    assert not [p for p in root.iterdir() if p.name != "x.json"]


def test_private_json_read_detects_path_replacement_after_open(importer, tmp_path, monkeypatch):
    root = tmp_path / "private"
    importer.private_json_write(root, "x.json", {"safe": True})
    original_fstat = os.fstat
    replaced = False

    def fstat(fd):
        nonlocal replaced
        result = original_fstat(fd)
        if stat.S_ISREG(result.st_mode) and not replaced:
            replaced = True
            replacement = root / "replacement"
            replacement.write_text(json.dumps({"attacker": True}))
            replacement.chmod(0o600)
            os.replace(replacement, root / "x.json")
        return result

    monkeypatch.setattr(os, "fstat", fstat)
    with pytest.raises(importer.PrivateArtifactError, match="identity changed"):
        importer.private_json_read(root, "x.json")


@pytest.mark.parametrize("raw", [
    b'{"a":1,"a":2}', b'{"outer":{"a":1,"a":2}}',
    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}',
    b'{"a":1} trailing', b'[]', b'1', b'null',
])
def test_private_json_read_rejects_non_strict_json(importer, tmp_path, raw):
    root = tmp_path / "private"
    importer.private_json_write(root, "x.json", {})
    path = root / "x.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    with pytest.raises(importer.PrivateArtifactError, match="invalid JSON"):
        importer.private_json_read(root, "x.json")


def test_private_json_write_rejects_nonfinite_without_artifact(importer, tmp_path):
    root = tmp_path / "private"
    with pytest.raises(importer.PrivateArtifactError, match="canonical JSON"):
        importer.private_json_write(root, "x.json", {"value": float("nan")})
    assert not root.exists()


@pytest.mark.parametrize("capability", ["O_NOFOLLOW", "O_DIRECTORY", "dir_fd", "replace"])
def test_private_io_fails_closed_when_capability_is_missing(importer, tmp_path, monkeypatch, capability):
    root = tmp_path / "private"
    if capability in {"O_NOFOLLOW", "O_DIRECTORY"}:
        monkeypatch.setattr(os, capability, 0)
    elif capability == "dir_fd":
        monkeypatch.setattr(os, "supports_dir_fd", set())
    else:
        monkeypatch.setattr(importer.inspect, "signature", lambda function: importer.inspect.Signature())
    with pytest.raises(importer.PrivateArtifactError, match="capabil"):
        importer.private_json_write(root, "x.json", {})
    assert not root.exists()


def test_private_root_creation_fsyncs_opened_parent(importer, tmp_path, monkeypatch):
    root = tmp_path / "private"
    original_fsync = os.fsync
    directory_fsyncs = []

    def recording_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_fsyncs.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    importer.private_json_write(root, "x.json", {})
    # Capability probe, root creation parent, and final replacement parent.
    assert len(directory_fsyncs) >= 3


def test_symlinked_absolute_ancestor_is_rejected(importer, tmp_path):
    trusted = tmp_path / "trusted"
    outside = tmp_path / "outside"
    trusted.mkdir(mode=0o700)
    outside.mkdir(mode=0o700)
    (trusted / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(importer.PrivateArtifactError):
        importer.private_json_write(trusted / "link" / "private", "x.json", {})
    assert not (outside / "private").exists()


def test_replaced_absolute_ancestor_is_detected(importer, tmp_path, monkeypatch):
    ancestor = tmp_path / "trusted"
    ancestor.mkdir(mode=0o700)
    original_open = os.open
    replaced = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "trusted" and dir_fd is not None and not replaced:
            replaced = True
            ancestor.rename(tmp_path / "displaced")
            ancestor.mkdir(mode=0o700)
        return fd

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(importer.PrivateArtifactError, match="ancestry changed"):
        importer._open_private_root(ancestor / "private", create=True)
    assert not (ancestor / "private").exists()


def test_private_root_replacement_during_creation_is_rejected(importer, tmp_path, monkeypatch):
    root = tmp_path / "private"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    original_mkdir = os.mkdir

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        result = original_mkdir(path, mode, dir_fd=dir_fd)
        if path == "private" and dir_fd is not None:
            os.rmdir(path, dir_fd=dir_fd)
            os.symlink(outside, path, target_is_directory=True, dir_fd=dir_fd)
        return result

    monkeypatch.setattr(os, "mkdir", racing_mkdir)
    with pytest.raises(importer.PrivateArtifactError):
        importer.private_json_write(root, "x.json", {})
    assert not (outside / "x.json").exists()


def test_relative_private_root_is_rejected_without_escape(importer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(importer.PrivateArtifactError, match="absolute"):
        importer.private_json_write(Path("private"), "x.json", {})
    assert not (tmp_path / "private").exists()
