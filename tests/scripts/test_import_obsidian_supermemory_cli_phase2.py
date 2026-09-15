from __future__ import annotations

import hashlib
import copy
import importlib.util
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture
def importer():
    spec = importlib.util.spec_from_file_location("phase2_cli_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _source(root: Path) -> None:
    (root / "Skills").mkdir(parents=True)
    (root / "Skills/a.md").write_text("distinctive alpha bravo charlie delta echo foxtrot golf hotel\n")


class StatefulDocuments:
    def __init__(self, rows=()):
        self.rows = {row["id"]: copy.deepcopy(row) for row in rows}
        self.calls = []
        self.next_id = 1

    def list(self, **kwargs):
        container = kwargs["container_tags"][0]
        self.calls.append(("list", container))
        rows = [copy.deepcopy(row) for row in self.rows.values()
                if row["container_tags"] == [container]]
        return {"memories": rows, "pagination": {"current_page": 1,
                "total_pages": 1 if rows else 0, "total_items": len(rows)}}

    def get(self, ident, timeout):
        self.calls.append(("get", ident))
        return copy.deepcopy(self.rows[ident])

    def add(self, **kwargs):
        self.calls.append(("add", kwargs["custom_id"]))
        ident = f"new-{self.next_id}"; self.next_id += 1
        self.rows[ident] = {"id": ident, "custom_id": kwargs["custom_id"],
            "status": "done", "task_type": kwargs["task_type"],
            "content": kwargs["content"], "metadata": copy.deepcopy(kwargs["metadata"]),
            "container_tags": [kwargs["container_tag"]]}
        return {"id": ident, "status": "queued"}

    def delete(self, ident, timeout):
        self.calls.append(("delete", ident)); self.rows.pop(ident)
        return {"id": ident}


class StatefulClient:
    def __init__(self, rows=()):
        self.documents = StatefulDocuments(rows)


def _remote(importer, doc, ident):
    return {"id": ident, "custom_id": doc["custom_id"], "status": "done",
            "task_type": "superrag", "content": doc["content"],
            "metadata": importer.metadata(doc),
            "container_tags": [importer.container_for_doc(doc)]}


def _run_command(importer, monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *argv])
    importer.main()
    return json.loads(capsys.readouterr().out)


def test_phase2_dry_run_never_constructs_client_or_writes(importer, tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"
    private = tmp_path / "private"
    _source(source)
    monkeypatch.setattr(importer, "_client", lambda _: (_ for _ in ()).throw(AssertionError("client")))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "dry-run", "--source-root", str(source),
                                     "--private-root", str(private)])
    importer.main()
    value = json.loads(capsys.readouterr().out)
    assert value["provider_client_created"] is value["filesystem_mutated"] is False
    assert value["actions"] == {"add": 1, "replace": 0, "delete": 0}
    assert not private.exists()


def test_topology_fresh_empty_owner_destination_plans_only_adds(
        importer, tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"; _source(source)
    private = tmp_path / "private"; private.mkdir(mode=0o700)
    client = StatefulClient()
    monkeypatch.setattr(importer, "_client", lambda _: client)

    result = _run_command(importer, monkeypatch, capsys, [
        "plan", "--transaction-id", "fresh", "--source-root", str(source),
        "--private-root", str(private), "--owner-canonical-container", "canonical_v2",
        "--owner-explicit-container", "explicit_v2",
    ])

    assert result["actions"] == {"add": 1}
    assert result["containers"] == ["canonical_v2", "family_shared"]
    assert client.documents.calls == [("list", "canonical_v2"), ("list", "family_shared")]
    assert "owner_primary" not in json.dumps(result)


def test_topology_fresh_rebuild_binds_writes_manifest_and_receipts(
        importer, tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"; _source(source)
    private = tmp_path / "private"; private.mkdir(mode=0o700)
    importer.generate_receipt_key(private / "readiness/receipt-hmac.key")
    plist = tmp_path / "agent.plist"; plist.write_bytes(plistlib.dumps({"Label": "fresh"}))
    client = StatefulClient()
    monkeypatch.setattr(importer, "_client", lambda _: client)
    _install_search(importer, monkeypatch, client)
    common = [
        "--source-root", str(source), "--private-root", str(private),
        "--owner-canonical-container", "canonical_v2",
        "--owner-explicit-container", "explicit_v2",
    ]
    planned = _run_command(importer, monkeypatch, capsys, [
        "plan", "--transaction-id", "fresh_execute", *common,
    ])
    result = _run_command(importer, monkeypatch, capsys, [
        "reconcile", "--execute", "--transaction-id", "fresh_execute",
        "--confirm", "fresh_execute", "--confirm-plan", planned["plan_digest"],
        "--plist", str(plist), "--first-install", *common,
    ])

    assert result["readiness"] is True
    writes = [row for row in client.documents.calls if row[0] in {"add", "delete"}]
    assert writes == [("add", next(iter(client.documents.rows.values()))["custom_id"])]
    assert next(iter(client.documents.rows.values()))["container_tags"] == ["canonical_v2"]
    plan = importer.private_json_read(private, "transactions/fresh_execute/plan.json")
    assert plan["records"][0]["source_container"] is None
    assert plan["records"][0]["expected_pre_identity"] is None
    triple = [importer.private_json_read(private, child)
              for _, child in importer.PUBLICATION_ARTIFACTS]
    assert len({row["generation"] for row in triple}) == 1
    assert triple[2]["documents"][0]["container"] == "canonical_v2"
    assert "owner_primary" not in json.dumps(triple)


def test_topology_collision_fails_before_scan_or_provider(importer, monkeypatch):
    effects = []
    monkeypatch.setattr(importer, "_scan_at", lambda _: effects.append("scan"))
    monkeypatch.setattr(importer, "_client", lambda _: effects.append("client"))
    with pytest.raises(SystemExit, match="collide"):
        importer._phase2_main([
            "plan", "--transaction-id", "collision", "--source-root", "/tmp/source",
            "--private-root", "/tmp/private", "--owner-canonical-container", "same",
            "--owner-explicit-container", "same",
        ])
    assert effects == []


def test_key_commands_are_explicit_and_rotation_invalidates_old_key(importer, tmp_path, monkeypatch, capsys):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "key-init", "--private-root", str(root)])
    importer.main()
    first = (root / "readiness/receipt-hmac.key").read_bytes()
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "key-rotate", "--private-root", str(root)])
    importer.main()
    second = (root / "readiness/receipt-hmac.key").read_bytes()
    value = json.loads(capsys.readouterr().out)
    assert first != second and value["reverification_required"] is True
    assert (root / "readiness/receipt-hmac.key").stat().st_mode & 0o777 == 0o600


def test_secure_install_hash_backup_plist_and_no_reload(importer, tmp_path):
    tracked = tmp_path / "tracked.py"
    tracked.write_bytes(b"#!/usr/bin/python3\nprint('tracked')\n")
    stable_dir = tmp_path / "stable"
    stable_dir.mkdir(mode=0o700)
    stable = stable_dir / "importer.py"
    stable.write_bytes(b"old\n")
    stable.chmod(0o700)
    interpreter = Path(sys.executable).resolve()
    plist = tmp_path / "agent.plist"
    args = [str(interpreter), str(stable), "verify-only", "--source-root", "/tmp/source",
            "--private-root", "/tmp/private"]
    plist.write_bytes(plistlib.dumps({"ProgramArguments": args}))
    digest = hashlib.sha256(tracked.read_bytes()).hexdigest()
    result = importer.secure_install(tracked, stable, tmp_path / "backups", digest, plist, interpreter)
    assert stable.read_bytes() == tracked.read_bytes()
    assert stable.stat().st_mode & 0o777 == 0o700
    assert result == {"installed_sha256": digest, "backup_created": True,
                      "launch_agent_reloaded": False}
    backups = list((tmp_path / "backups").iterdir())
    assert len(backups) == 1 and backups[0].read_bytes() == b"old\n"
    assert backups[0].stat().st_mode & 0o777 == 0o600


def test_install_rejects_hash_and_interactive_or_wrong_target_plist(importer, tmp_path):
    tracked = tmp_path / "tracked.py"
    tracked.write_bytes(b"new")
    interpreter = Path(sys.executable).resolve()
    stable_dir = tmp_path / "stable"
    stable_dir.mkdir(mode=0o700)
    stable = stable_dir / "importer.py"
    plist = tmp_path / "agent.plist"
    plist.write_bytes(plistlib.dumps({"ProgramArguments": [str(interpreter), str(stable), "reconcile"]}))
    with pytest.raises(importer.PrivateArtifactError, match="SHA-256"):
        importer.secure_install(tracked, stable, tmp_path / "backup", "0" * 64, plist, interpreter)
    with pytest.raises(importer.PrivateArtifactError, match="noninteractive"):
        importer.secure_install(tracked, stable, tmp_path / "backup", hashlib.sha256(b"new").hexdigest(),
                                plist, interpreter)
    assert not stable.exists()


def test_bound_plan_uses_exact_provider_identity(importer):
    doc = importer.item("Skills/a.md", b"distinctive alpha bravo charlie delta echo foxtrot")
    remote = {"id": "old", "custom_id": doc["custom_id"], "status": "done",
              "task_type": "superrag", "metadata": importer.metadata(doc),
              "container_tags": [importer.CONTAINER]}
    current = {doc["relative_path"]: doc}
    classification = importer.classify_inventory(current, {
        importer.CONTAINER: (remote,), importer.FAMILY_CONTAINER: (),
    })
    assert importer.reconciliation_plan(classification) == ()
    changed = importer.item("Skills/a.md", b"distinctive changed bravo charlie delta echo foxtrot")
    plan = importer.reconciliation_plan(importer.classify_inventory(
        {changed["relative_path"]: changed},
        {importer.CONTAINER: (remote,), importer.FAMILY_CONTAINER: ()},
    ))
    bound = importer.transaction_plan_from_inventory(
        {changed["relative_path"]: changed},
        {importer.CONTAINER: (remote,), importer.FAMILY_CONTAINER: ()}, plan)
    assert bound[0]["action"] == "replace"
    assert bound[0]["expected_pre_identity"]["document_id"] == "old"
    assert bound[0]["expected_post_identity"]["document_id"] is None


def test_release_backup_excludes_key_contents(importer, tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    importer.generate_receipt_key(root / "readiness/receipt-hmac.key")
    source = tmp_path / "legacy.json"
    source.write_text('{"legacy":true}\n')
    value = importer.create_release_backup(root, "tx", manifest_path=source, first_install=True)
    metadata = importer.private_json_read(root, f'{value["path"]}/metadata.json')
    assert metadata["key_contents_backed_up"] is False
    assert "key_id" in metadata["receipt_key"]
    assert not list((root / value["path"]).glob("*key*"))


@pytest.mark.parametrize("argv", [[], ["--dry-run"], ["unknown"], ["--verify-only"]])
def test_default_unknown_and_legacy_flags_fail_before_any_effect(importer, monkeypatch, argv):
    effects = []
    monkeypatch.setattr(importer, "legacy_main", lambda: effects.append("legacy"))
    monkeypatch.setattr(importer, "_client", lambda _: effects.append("client"))
    monkeypatch.setattr(importer, "_scan_at", lambda _: effects.append("scan"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *argv])
    with pytest.raises(SystemExit) as exc:
        importer.main()
    assert exc.value.code != 0
    assert effects == []


def test_transaction_id_is_rejected_by_argparse_before_scan_or_client(importer, monkeypatch):
    effects = []
    monkeypatch.setattr(importer, "_scan_at", lambda _: effects.append("scan"))
    monkeypatch.setattr(importer, "_client", lambda _: effects.append("client"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "plan", "--transaction-id", "../bad",
                                     "--source-root", "/tmp/source", "--private-root", "/tmp/private"])
    with pytest.raises(SystemExit):
        importer.main()
    assert effects == []


def test_backup_requires_each_artifact_or_explicit_first_install(importer, tmp_path):
    root = tmp_path / "private"; root.mkdir(mode=0o700)
    importer.generate_receipt_key(root / "readiness/receipt-hmac.key")
    with pytest.raises(importer.PrivateArtifactError, match="unspecified"):
        importer.create_release_backup(root, "tx", first_install=False)
    value = importer.create_release_backup(root, "first", first_install=True)
    receipt = importer.private_json_read(root, f'{value["path"]}/metadata.json')
    assert set(receipt["artifacts"]) == {"importer", "legacy-manifest", "launch-agent.plist"}
    assert all(row == {"status": "absent", "reason": "approved-first-install"}
               for row in receipt["artifacts"].values())


def test_plan_digest_drift_fails_before_backup_or_executor(importer, tmp_path, monkeypatch, capsys):
    first = importer.item("Skills/a.md", b"distinctive alpha bravo charlie delta echo foxtrot")
    second = importer.item("Skills/a.md", b"distinctive changed bravo charlie delta echo foxtrot")
    current = {first["relative_path"]: first}
    monkeypatch.setattr(importer, "_scan_at", lambda _: (list(current.values()), dict(current)))
    monkeypatch.setattr(importer, "_client", lambda _: object())
    monkeypatch.setattr(importer, "provider_inventory", lambda _: {
        importer.CONTAINER: (), importer.FAMILY_CONTAINER: (),
    })
    source_root = tmp_path / "source"; source_root.mkdir()
    private_root = tmp_path / "private"; private_root.mkdir(mode=0o700)
    argv = ["plan", "--transaction-id", "tx", "--source-root", str(source_root),
            "--private-root", str(private_root)]
    importer._phase2_main(argv)
    digest = json.loads(capsys.readouterr().out)["plan_digest"]
    current[first["relative_path"]] = second
    effects = []
    monkeypatch.setattr(importer, "create_release_backup", lambda *a, **k: effects.append("backup"))
    monkeypatch.setattr(importer.DurableReconciliationExecutor, "execute",
                        lambda *a, **k: effects.append("execute"))
    plist = tmp_path / "agent.plist"; plist.write_bytes(plistlib.dumps({}))
    with pytest.raises(SystemExit, match="plan digest mismatch"):
        importer._phase2_main([
            "reconcile", "--execute", "--transaction-id", "tx", "--confirm", "tx",
            "--confirm-plan", digest, "--source-root", str(tmp_path / "source"),
            "--private-root", str(tmp_path / "private"), "--plist", str(plist),
        ])
    assert effects == []


def test_stateful_cli_two_adds_then_deep_noop_publication(importer, tmp_path, monkeypatch, capsys):
    source = tmp_path / "source"; _source(source)
    (source / "Skills/b.md").write_text(
        "distinctive india juliet kilo lima mike november oscar papa\n")
    private = tmp_path / "private"; private.mkdir(mode=0o700)
    importer.generate_receipt_key(private / "readiness/receipt-hmac.key")
    plist = tmp_path / "agent.plist"; plist.write_bytes(plistlib.dumps({"Label": "test"}))
    client = StatefulClient()
    searches = []

    def search(_client, query, **kwargs):
        searches.append((query, kwargs["container_tag"], copy.deepcopy(kwargs["filters"])))
        rel = kwargs["filters"]["AND"][1]["value"]
        row = next(row for row in client.documents.rows.values()
                   if row["metadata"]["relative_path"] == rel)
        metadata = row["metadata"]
        parent = {"id": row["id"], "status": "done", "metadata": metadata,
                  "custom_id": row["custom_id"], "task_type": "superrag",
                  "container_tags": row["container_tags"]}
        return {"results": [{"id": f'chunk-{row["id"]}', "chunk": row["content"],
                              "metadata": metadata, "documents": [parent],
                              "document_id": row["id"]}], "total": 1}

    monkeypatch.setattr(importer, "_client", lambda _: client)
    monkeypatch.setattr(importer, "search_documents_v4", search)
    common = ["--source-root", str(source), "--private-root", str(private)]
    planned = _run_command(importer, monkeypatch, capsys,
                           ["plan", "--transaction-id", "adds", *common])
    assert planned["actions"] == {"add": 2}
    result = _run_command(importer, monkeypatch, capsys, [
        "reconcile", "--execute", "--transaction-id", "adds", "--confirm", "adds",
        "--confirm-plan", planned["plan_digest"], "--plist", str(plist),
        "--first-install", *common])
    assert result["readiness"] and result["behavioral_receipt_written"] is False
    assert [call[0] for call in client.documents.calls].count("add") == 2
    triple = [importer.private_json_read(private, child)
              for _, child in importer.PUBLICATION_ARTIFACTS]
    assert len({row["generation"] for row in triple}) == 1
    key = importer.read_receipt_key(private / "readiness/receipt-hmac.key")
    _, current = importer._scan_at(source)
    assert importer.validate_readiness_receipts(
        triple[0], triple[1], key=key,
        current_fingerprint=importer.source_fingerprint_from_documents(current),
        now=importer.datetime.now(importer.timezone.utc))

    client.documents.calls.clear(); searches.clear()
    planned = _run_command(importer, monkeypatch, capsys,
                           ["plan", "--transaction-id", "noop", *common])
    assert planned["change_count"] == 0 and planned["plan_digest"]
    result = _run_command(importer, monkeypatch, capsys, [
        "reconcile", "--execute", "--transaction-id", "noop", "--confirm", "noop",
        "--confirm-plan", planned["plan_digest"], "--plist", str(plist), *common])
    assert result["readiness"] and result["actions"] == {}
    assert searches and {call[0] for call in client.documents.calls} == {"list", "get"}
    assert not [call for call in client.documents.calls if call[0] in {"add", "delete"}]
    assert (private / "transactions/noop/snapshot.json").is_file()
    assert (private / "transactions/noop/forward-journal.json").is_file()
    assert not (private / "behavioral-readiness.json").exists()


def _write_docs(root: Path, rows: dict[str, str]) -> None:
    if root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path != root:
                path.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in rows.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip("\n") + "\n")


def _docs(importer, rows: dict[str, str]):
    return {path: importer.item(path, (text.rstrip("\n") + "\n").encode())
            for path, text in rows.items()}


def _install_search(importer, monkeypatch, client, *, fail=False):
    def search(_client, query, **kwargs):
        if fail:
            raise RuntimeError("vector unavailable secret-needle")
        relative = kwargs["filters"]["AND"][1]["value"]
        row = next(row for row in client.documents.rows.values()
                   if row["metadata"]["relative_path"] == relative)
        return {"results": [{"id": f'chunk-{row["id"]}', "chunk": row["content"],
                              "metadata": copy.deepcopy(row["metadata"]),
                              "documents": [copy.deepcopy(row)],
                              "document_id": row["id"]}], "total": 1}
    monkeypatch.setattr(importer, "search_documents_v4", search)


def _setup_cli(importer, tmp_path, monkeypatch, remote_rows):
    source = tmp_path / "source"
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    importer.generate_receipt_key(private / "readiness/receipt-hmac.key")
    plist = tmp_path / "agent.plist"
    plist.write_bytes(plistlib.dumps({"Label": "acceptance"}))
    client = StatefulClient(remote_rows)
    monkeypatch.setattr(importer, "_client", lambda _: client)
    _install_search(importer, monkeypatch, client)
    return source, private, plist, client


def _execute_cli(importer, monkeypatch, capsys, source, private, plist, tx, *, first=False):
    common = ["--source-root", str(source), "--private-root", str(private)]
    planned = _run_command(importer, monkeypatch, capsys,
                           ["plan", "--transaction-id", tx, *common])
    argv = ["reconcile", "--execute", "--transaction-id", tx, "--confirm", tx,
            "--confirm-plan", planned["plan_digest"], "--plist", str(plist), *common]
    if first:
        argv.append("--first-install")
    result = _run_command(importer, monkeypatch, capsys, argv)
    return planned, result


def _semantic_inventory(rows):
    return sorted((row["custom_id"], row["content"],
                   json.dumps(row["metadata"], sort_keys=True), tuple(row["container_tags"]))
                  for row in rows.values())


def _assert_transaction_and_publication(importer, private, tx, result, expected_actions):
    assert result["actions"] == expected_actions
    assert result["readiness"] is True
    assert result["behavioral_receipt_written"] is False
    plan = importer.private_json_read(private, f"transactions/{tx}/plan.json")
    journal = importer.private_json_read(private, f"transactions/{tx}/forward-journal.json")
    snapshot = importer.private_json_read(private, f"transactions/{tx}/snapshot.json")
    assert plan["transaction_id"] == journal["transaction_id"] == snapshot["transaction_id"] == tx
    assert all(row["stage"] == "done" for row in journal["records"])
    assert (private / plan["backup"]["path"] / "metadata.json").is_file()
    triple = [importer.private_json_read(private, child) for _, child in importer.PUBLICATION_ARTIFACTS]
    assert len({row["generation"] for row in triple}) == 1
    assert triple[0]["generation"].startswith(f"{tx}:")
    assert not (private / "behavioral-readiness.json").exists()


def test_cli_content_change_executes_replace_and_publishes_exact_inventory(
        importer, tmp_path, monkeypatch, capsys):
    old = _docs(importer, {"Skills/private-name.md": "old distinctive alpha bravo charlie delta echo"})
    new_rows = {"Skills/private-name.md": "new distinctive alpha bravo charlie delta echo"}
    source, private, plist, client = _setup_cli(
        importer, tmp_path, monkeypatch, [_remote(importer, next(iter(old.values())), "old")])
    _write_docs(source, new_rows)
    client.documents.calls.clear()
    _, result = _execute_cli(importer, monkeypatch, capsys, source, private, plist, "replace", first=True)
    _assert_transaction_and_publication(importer, private, "replace", result, {"replace": 1})
    assert [row[0] for row in client.documents.calls if row[0] in {"add", "delete"}] == ["delete", "add"]
    assert _semantic_inventory(client.documents.rows) == _semantic_inventory(
        {"expected": _remote(importer, next(iter(_docs(importer, new_rows).values())), "ignored")})
    assert "private-name" not in capsys.readouterr().err


def test_cli_delete_removes_only_bound_document_and_publishes_empty_inventory(
        importer, tmp_path, monkeypatch, capsys):
    doc = next(iter(_docs(importer, {"Skills/deleted-secret.md": "distinctive delete alpha bravo charlie delta"}).values()))
    keeper = next(iter(_docs(importer, {"Skills/keeper.md": "distinctive keeper alpha bravo charlie delta"}).values()))
    source, private, plist, client = _setup_cli(importer, tmp_path, monkeypatch,
        [_remote(importer, doc, "delete-me"), _remote(importer, keeper, "keeper")])
    _write_docs(source, {"Skills/keeper.md": "distinctive keeper alpha bravo charlie delta"})
    _, result = _execute_cli(importer, monkeypatch, capsys, source, private, plist, "delete", first=True)
    _assert_transaction_and_publication(importer, private, "delete", result, {"delete": 1})
    assert len(client.documents.rows) == 1
    assert [call for call in client.documents.calls if call[0] == "delete"] == [("delete", "delete-me")]


def test_cli_rename_is_old_delete_plus_new_add(importer, tmp_path, monkeypatch, capsys):
    raw = "distinctive rename alpha bravo charlie delta"
    old = next(iter(_docs(importer, {"Skills/old-secret.md": raw}).values()))
    new_rows = {"Skills/new-secret.md": raw}
    source, private, plist, client = _setup_cli(importer, tmp_path, monkeypatch,
                                                [_remote(importer, old, "old-name")])
    _write_docs(source, new_rows)
    _, result = _execute_cli(importer, monkeypatch, capsys, source, private, plist, "rename", first=True)
    _assert_transaction_and_publication(importer, private, "rename", result, {"add": 1, "delete": 1})
    assert [call[0] for call in client.documents.calls if call[0] in {"add", "delete"}] == ["add", "delete"]
    only = next(iter(client.documents.rows.values()))
    assert only["metadata"]["relative_path"] == "Skills/new-secret.md"


@pytest.mark.parametrize("direction", ["owner-to-family", "family-to-owner"])
def test_cli_move_crosses_exact_owner_and_family_containers(
        importer, tmp_path, monkeypatch, capsys, direction):
    owner = "Skills/moved-secret.md"
    family = "Jarvis/Family Shared/moved-secret.md"
    old_path, new_path = (owner, family) if direction == "owner-to-family" else (family, owner)
    raw = "distinctive move alpha bravo charlie delta"
    old = next(iter(_docs(importer, {old_path: raw}).values()))
    source, private, plist, client = _setup_cli(importer, tmp_path, monkeypatch,
                                                [_remote(importer, old, "old-move")])
    _write_docs(source, {new_path: raw})
    tx = direction.replace("-", "_")
    _, result = _execute_cli(importer, monkeypatch, capsys, source, private, plist, tx, first=True)
    _assert_transaction_and_publication(importer, private, tx, result, {"add": 1, "delete": 1})
    only = next(iter(client.documents.rows.values()))
    expected = importer.FAMILY_CONTAINER if direction == "owner-to-family" else importer.CONTAINER
    assert only["container_tags"] == [expected]
    assert {call[0] for call in client.documents.calls} >= {"list", "get", "add", "delete"}


def test_cli_failed_transaction_resumes_same_persisted_plan_and_publishes(
        importer, tmp_path, monkeypatch, capsys):
    old = next(iter(_docs(importer, {"Skills/resume-secret.md": "distinctive old alpha bravo charlie delta"}).values()))
    source, private, plist, client = _setup_cli(importer, tmp_path, monkeypatch,
                                                [_remote(importer, old, "resume-old")])
    _write_docs(source, {"Skills/resume-secret.md": "distinctive new alpha bravo charlie delta"})
    common = ["--source-root", str(source), "--private-root", str(private)]
    planned = _run_command(importer, monkeypatch, capsys, ["plan", "--transaction-id", "resume", *common])
    original = importer.DurableReconciliationExecutor
    injected = {"raised": False}

    class FailingOnce(original):
        def __init__(self, *args, **kwargs):
            def fault(point):
                if point == "after_delete" and not injected["raised"]:
                    injected["raised"] = True
                    raise RuntimeError("injected-private-detail")
            super().__init__(*args, fault_injector=fault, **kwargs)

    monkeypatch.setattr(importer, "DurableReconciliationExecutor", FailingOnce)
    argv = ["reconcile", "--execute", "--transaction-id", "resume", "--confirm", "resume",
            "--confirm-plan", planned["plan_digest"], "--plist", str(plist), "--first-install", *common]
    with pytest.raises(SystemExit) as failure:
        _run_command(importer, monkeypatch, capsys, argv)
    assert failure.value.code == 1
    captured = capsys.readouterr()
    failure_output = json.loads(captured.out)
    assert captured.err == "" and "injected-private-detail" not in captured.out
    assert failure_output["readiness"] is False and failure_output["resume"]
    monkeypatch.setattr(importer, "DurableReconciliationExecutor", original)
    result = _run_command(importer, monkeypatch, capsys, argv)
    _assert_transaction_and_publication(importer, private, "resume", result, {"replace": 1})
    assert len([call for call in client.documents.calls if call[0] == "delete"]) == 1
    assert len([call for call in client.documents.calls if call[0] == "add"]) == 1


@pytest.mark.parametrize("scenario", ["add", "delete", "replace", "move", "mixed"])
def test_cli_rollback_restores_exact_pre_state_for_every_action_shape(
        importer, tmp_path, monkeypatch, capsys, scenario):
    bodies = {
        "add": ({}, {"Skills/add.md": "distinctive add alpha bravo charlie delta"}),
        "delete": ({"Skills/delete.md": "distinctive delete alpha bravo charlie delta",
                    "Skills/keeper.md": "distinctive keeper alpha bravo charlie delta"},
                   {"Skills/keeper.md": "distinctive keeper alpha bravo charlie delta"}),
        "replace": ({"Skills/replace.md": "distinctive old alpha bravo charlie delta"},
                    {"Skills/replace.md": "distinctive new alpha bravo charlie delta"}),
        "move": ({"Skills/move.md": "distinctive move alpha bravo charlie delta"},
                 {"Jarvis/Family Shared/move.md": "distinctive move alpha bravo charlie delta"}),
        "mixed": ({"Skills/delete.md": "distinctive delete alpha bravo charlie delta",
                   "Skills/replace.md": "distinctive old alpha bravo charlie delta",
                   "Skills/move.md": "distinctive move alpha bravo charlie delta"},
                  {"Skills/add.md": "distinctive add alpha bravo charlie delta",
                   "Skills/replace.md": "distinctive new alpha bravo charlie delta",
                   "Jarvis/Family Shared/move.md": "distinctive move alpha bravo charlie delta"}),
    }
    pre_rows, current_rows = bodies[scenario]
    pre_docs = _docs(importer, pre_rows)
    remotes = [_remote(importer, doc, f"pre-{index}") for index, doc in enumerate(pre_docs.values())]
    source, private, plist, client = _setup_cli(importer, tmp_path, monkeypatch, remotes)
    before = _semantic_inventory(client.documents.rows)
    _write_docs(source, current_rows)
    _, forward = _execute_cli(importer, monkeypatch, capsys, source, private, plist,
                              f"forward_{scenario}", first=True)
    assert forward["readiness"] is True
    mutator_count = len([call for call in client.documents.calls if call[0] in {"add", "delete"}])
    rollback = _run_command(importer, monkeypatch, capsys, [
        "rollback", "--private-root", str(private), "--transaction-id", f"rollback_{scenario}",
        "--forward-transaction-id", f"forward_{scenario}",
        "--confirm", f"ROLLBACK:forward_{scenario}"])
    assert rollback == {"mode": "rollback", "complete": True,
                        "transaction_id": f"rollback_{scenario}", "readiness": False}
    assert _semantic_inventory(client.documents.rows) == before
    assert len([call for call in client.documents.calls if call[0] in {"add", "delete"}]) > mutator_count
    journal = importer.private_json_read(private,
        f"transactions/rollback_{scenario}/rollback-journal.json")
    assert all(row["stage"] == "done" for row in journal["records"])


def _tree_snapshot(root: Path):
    result = {}
    for path in [root, *sorted(root.rglob("*"))]:
        info = path.lstat()
        relative = path.relative_to(root).as_posix() if path != root else "."
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        result[relative] = (info.st_ino, info.st_mode, info.st_mtime_ns, info.st_size, digest)
    return result


def _seed_interrupted_publication(importer, private: Path, *, tampered=False, partial=False):
    doc = importer.item("Skills/a.md", b"distinctive alpha bravo charlie delta echo foxtrot golf hotel\n")
    current = {doc["relative_path"]: doc}
    metadata = importer.metadata(doc)
    inventory = [{"relative_path": doc["relative_path"], "custom_id": doc["custom_id"],
                  "document_id": "existing", "container": importer.container_for_doc(doc),
                  "sha256": doc["sha256"], "bytes": doc["bytes"], "status": "done",
                  "index_schema_version": 4, "task_type": "superrag",
                  "provenance": "canonical_obsidian"}]
    observations = {doc["relative_path"]: [{
        "id": "chunk", "chunk": "distinctive alpha bravo charlie delta echo foxtrot golf hotel",
        "metadata": metadata, "document_id": "existing",
        "documents": [{"id": "existing", "status": "done", "metadata": metadata,
                       "custom_id": doc["custom_id"], "task_type": "superrag",
                       "container_tags": [importer.container_for_doc(doc)]}],
    }]}
    importer.generate_receipt_key(private / "readiness/receipt-hmac.key")
    with pytest.raises(RuntimeError, match="interrupt"):
        importer.publish_readiness_pair(
            private, current, inventory, observations, "interrupted:generation",
            importer.datetime.now(importer.timezone.utc),
            fault_injector=lambda point: (_ for _ in ()).throw(RuntimeError("interrupt"))
            if point == "after_prepared_marker" else None,
        )
    if partial:
        (private / "readiness/storage-reconciliation.json").write_text('{"partial":true}\n')
    if tampered:
        marker = json.loads((private / importer.PUBLICATION_MARKER).read_text())
        marker["generation"] = "tampered:private-detail"
        (private / importer.PUBLICATION_MARKER).write_text(json.dumps(marker))


@pytest.mark.parametrize("tampered,partial", [(False, False), (True, False), (False, True)])
def test_verify_only_interrupted_publication_is_zero_write_before_provider_or_scan(
        importer, tmp_path, monkeypatch, capsys, tampered, partial):
    source = tmp_path / "source"; _source(source)
    private = tmp_path / "private"; private.mkdir(mode=0o700)
    _seed_interrupted_publication(importer, private, tampered=tampered, partial=partial)
    effects = []
    monkeypatch.setattr(importer, "recover_publication", lambda *a, **k: effects.append("recover"))
    monkeypatch.setattr(importer, "_scan_at", lambda *a, **k: effects.append("scan"))
    monkeypatch.setattr(importer, "_client", lambda *a, **k: effects.append("client"))
    before = _tree_snapshot(tmp_path)
    with pytest.raises(SystemExit) as failure:
        importer._phase2_main(["verify-only", "--source-root", str(source),
                               "--private-root", str(private)])
    assert failure.value.code == 1
    output = json.loads(capsys.readouterr().out)
    assert output == {"mode": "verify-only", "publication_recovery_required": True,
                      "filesystem_mutated": False, "backend_mutated": False}
    assert effects == []
    assert _tree_snapshot(tmp_path) == before
    assert "private-detail" not in json.dumps(output)


def test_verify_only_subprocess_interrupted_publication_is_zero_write(tmp_path, importer):
    source = tmp_path / "source"; _source(source)
    private = tmp_path / "private"; private.mkdir(mode=0o700)
    _seed_interrupted_publication(importer, private, partial=True)
    before = {"private": _tree_snapshot(private), "source": _tree_snapshot(source)}
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "verify-only", "--source-root", str(source),
         "--private-root", str(private), "--base-url", "http://127.0.0.1:9"],
        text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 1 and result.stderr == ""
    assert json.loads(result.stdout)["publication_recovery_required"] is True
    assert {"private": _tree_snapshot(private), "source": _tree_snapshot(source)} == before


def test_verify_only_without_marker_keeps_deep_proof_read_only(
        importer, tmp_path, monkeypatch, capsys):
    rows = {"Skills/a.md": "distinctive alpha bravo charlie delta echo foxtrot"}
    doc = next(iter(_docs(importer, rows).values()))
    source, private, plist, client = _setup_cli(
        importer, tmp_path, monkeypatch, [_remote(importer, doc, "existing")])
    _write_docs(source, rows)
    _execute_cli(importer, monkeypatch, capsys, source, private, plist, "seed", first=True)
    assert not (private / importer.PUBLICATION_MARKER).exists()
    client.documents.calls.clear()
    before = {"private": _tree_snapshot(private), "source": _tree_snapshot(source)}
    result = _run_command(importer, monkeypatch, capsys, [
        "verify-only", "--source-root", str(source), "--private-root", str(private)])
    assert result["verification_complete"] is True
    assert result["storage_verified_count"] == result["vector_verified_count"] == 1
    assert {call[0] for call in client.documents.calls} == {"list", "get"}
    assert {"private": _tree_snapshot(private), "source": _tree_snapshot(source)} == before


def test_deep_noop_allows_only_transaction_backup_and_atomic_publication_files(
        importer, tmp_path, monkeypatch, capsys):
    rows = {"Skills/a.md": "distinctive alpha bravo charlie delta echo foxtrot"}
    doc = next(iter(_docs(importer, rows).values()))
    source, private, plist, client = _setup_cli(importer, tmp_path, monkeypatch,
                                                [_remote(importer, doc, "existing")])
    _write_docs(source, rows)
    # Seed a valid old triple, then prove no-op republishes a current triple without collateral writes.
    _execute_cli(importer, monkeypatch, capsys, source, private, plist, "seed", first=True)
    before = _tree_snapshot(tmp_path)
    client.documents.calls.clear()
    planned, result = _execute_cli(importer, monkeypatch, capsys, source, private, plist, "noop")
    after = _tree_snapshot(tmp_path)
    assert planned["change_count"] == 0 and result["actions"] == {}
    assert {call[0] for call in client.documents.calls} == {"list", "get"}
    changed = {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}
    allowed_exact = {"private", "private/transactions", "private/transactions/noop",
                     "private/backups", "private/readiness", "private/manifest",
                     "private/publication",
                     *[f"private/{child}" for _, child in importer.PUBLICATION_ARTIFACTS]}
    unexpected = {path for path in changed if path not in allowed_exact
                  and not path.startswith("private/transactions/noop/")
                  and not path.startswith("private/backups/")}
    assert not unexpected, sorted(unexpected)
    assert not any("__pycache__" in path or path.endswith(".lock") for path in after)
    assert before["source"] == after["source"]
    for path in before:
        if not path.startswith(("private/transactions", "private/backups", "private/readiness",
                                "private/manifest", "private")):
            assert before[path] == after[path]
    triple = [importer.private_json_read(private, child) for _, child in importer.PUBLICATION_ARTIFACTS]
    assert len({row["generation"] for row in triple}) == 1
    assert triple[0]["generation"].startswith("noop:")


def test_deep_noop_vector_failure_preserves_old_publication_triple_and_exits_nonzero(
        importer, tmp_path, monkeypatch, capsys):
    rows = {"Skills/a.md": "distinctive alpha bravo charlie delta echo foxtrot"}
    doc = next(iter(_docs(importer, rows).values()))
    source, private, plist, client = _setup_cli(importer, tmp_path, monkeypatch,
                                                [_remote(importer, doc, "existing")])
    _write_docs(source, rows)
    _execute_cli(importer, monkeypatch, capsys, source, private, plist, "seed", first=True)
    old = {child: (private / child).read_bytes() for _, child in importer.PUBLICATION_ARTIFACTS}
    _install_search(importer, monkeypatch, client, fail=True)
    common = ["--source-root", str(source), "--private-root", str(private)]
    planned = _run_command(importer, monkeypatch, capsys, ["plan", "--transaction-id", "noopfail", *common])
    argv = ["reconcile", "--execute", "--transaction-id", "noopfail", "--confirm", "noopfail",
            "--confirm-plan", planned["plan_digest"], "--plist", str(plist), *common]
    with pytest.raises(SystemExit) as failure:
        _run_command(importer, monkeypatch, capsys, argv)
    assert failure.value.code == 1
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert captured.err == "" and "secret-needle" not in captured.out
    assert output["readiness"] is False
    assert all((private / child).read_bytes() == value for child, value in old.items())
    assert not (private / "behavioral-readiness.json").exists()
