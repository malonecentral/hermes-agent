from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("obsidian_receipts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _docs(importer):
    return {doc["relative_path"]: doc for doc in (
        importer.item("Skills/A.md", b"alpha cedar telescope marmalade compass current distinctive body content for readiness proof"),
        importer.item("Jarvis/Family Shared/B.md", b"---\nevent_date: 2026-09-12\n---\nbravo quartz violin nebula saffron current distinctive body content for readiness proof"),
    )}


def _inventory(importer, docs):
    return [{"relative_path": path, "custom_id": doc["custom_id"], "document_id": f"doc-{index}",
             "container": importer.container_for_doc(doc), "sha256": doc["sha256"], "bytes": doc["bytes"],
             "status": "done", "index_schema_version": 4, "task_type": "superrag",
             "provenance": "canonical_obsidian"}
            for index, (path, doc) in enumerate(sorted(docs.items()))]


def _observations(importer, docs):
    result = {}
    for index, (path, doc) in enumerate(sorted(docs.items())):
        metadata = importer.metadata(doc)
        body = " ".join(doc["content"].split()[-11:])
        parent = {"id": f"doc-{index}", "status": "done", "metadata": metadata}
        result_row = {"id": f"chunk-{index}", "chunk": body,
                      "metadata": metadata, "documents": [parent]}
        if index % 2:
            parent.update(customId=doc["custom_id"], taskType="superrag",
                          containerTags=[importer.container_for_doc(doc)])
            result_row["documentId"] = f"doc-{index}"
        else:
            parent.update(custom_id=doc["custom_id"], task_type="superrag",
                          container_tags=[importer.container_for_doc(doc)])
            result_row["document_id"] = f"doc-{index}"
        result[path] = [result_row]
    return result


def _proofs(importer, tmp_path):
    docs = _docs(importer); inventory = _inventory(importer, docs); observations = _observations(importer, docs)
    key_path = tmp_path / "private/readiness/receipt-hmac.key"
    importer.generate_receipt_key(key_path)
    key = importer.read_receipt_key(key_path)
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    fingerprint = importer.source_fingerprint_from_documents(docs)
    storage = importer.build_storage_reconciliation_receipt(docs, inventory, generation="generation-1", generated_at=now, key=key)
    vector = importer.build_vector_readiness_receipt(docs, observations, generation="generation-1", generated_at=now, key=key)
    return docs, inventory, observations, key_path, key, fingerprint, storage, vector, now


def test_hmac_receipts_are_private_aggregate_and_deterministic(importer, tmp_path):
    docs, inventory, observations, _, key, fingerprint, storage, vector, now = _proofs(importer, tmp_path)
    again = importer.build_vector_readiness_receipt(docs, observations, generation="generation-1", generated_at=now, key=key)
    assert importer.canonical_bytes(vector) == importer.canonical_bytes(again)
    assert importer.validate_readiness_receipts(storage, vector, key=key, current_fingerprint=fingerprint, now=now)
    encoded = json.dumps([storage, vector])
    assert all(path not in encoded and doc["custom_id"] not in encoded and doc["content"] not in encoded
               for path, doc in docs.items())
    assert key.hex() not in encoded
    assert "documents" not in storage and "documents" not in vector


def test_public_reseal_and_wrong_missing_replaced_keys_fail(importer, tmp_path):
    _, _, _, key_path, key, fingerprint, storage, vector, now = _proofs(importer, tmp_path)
    forged = copy.deepcopy(vector); forged["verified_document_count"] = 999
    unsigned = dict(forged); unsigned_auth = dict(unsigned["authentication"]); unsigned_auth.pop("tag")
    unsigned["authentication"] = unsigned_auth
    forged["authentication"]["tag"] = importer.canonical_digest(unsigned)
    assert not importer.validate_readiness_receipts(storage, forged, key=key, current_fingerprint=fingerprint, now=now)
    assert not importer.validate_readiness_receipts(storage, vector, key=b"x" * 32, current_fingerprint=fingerprint, now=now)
    key_path.unlink()
    assert importer.read_receipt_key(key_path) is None
    key_path.write_bytes(b"y" * 32); key_path.chmod(0o600)
    assert importer.read_receipt_key(key_path) != key


@pytest.mark.parametrize("boundary", [
    "after_stage_storage", "after_stage_vector", "after_stage_manifest",
    "after_prepared_marker", "after_recover_storage", "after_recover_vector",
    "after_recover_manifest", "after_committed_marker", "after_marker_cleanup",
])
def test_publication_recovers_after_abrupt_process_death(importer, tmp_path, boundary):
    docs, inventory, observations, _, _, _, _, _, now = _proofs(importer, tmp_path)
    root = tmp_path / "private"
    importer.publish_readiness_pair(root, docs, inventory, observations, "old:generation", now)
    children = [child for _, child in importer.PUBLICATION_ARTIFACTS]
    old = [importer.private_json_read(root, child) for child in children]
    payload = json.dumps({"root": str(root), "docs": docs, "inventory": inventory,
                          "observations": observations, "now": now.isoformat(),
                          "boundary": boundary})
    code = f'''\
import importlib.util, json, os, sys
from datetime import datetime
spec=importlib.util.spec_from_file_location("crash_importer", {str(SCRIPT)!r})
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p=json.loads(sys.stdin.read())
def die(point):
    if point == p["boundary"]: os._exit(91)
m.publish_readiness_pair(m.Path(p["root"]), p["docs"], p["inventory"],
    p["observations"], "new:generation", datetime.fromisoformat(p["now"]),
    fault_injector=die)
'''
    result = subprocess.run([sys.executable, "-c", code], input=payload, text=True,
                            cwd=SCRIPT.parents[1], check=False)
    assert result.returncode == 91
    importer.recover_publication(root)
    after = [importer.private_json_read(root, child) for child in children]
    generations = {row["generation"] for row in after}
    assert generations in ({"old:generation"}, {"new:generation"})
    assert after == old or generations == {"new:generation"}
    key = importer.read_receipt_key(root / "readiness/receipt-hmac.key")
    assert importer.validate_readiness_receipts(
        after[0], after[1], key=key,
        current_fingerprint=importer.source_fingerprint_from_documents(docs), now=now)
    assert after[2]["generation"] == after[0]["generation"] == after[1]["generation"]
    assert not (root / importer.PUBLICATION_MARKER).exists()


def test_key_file_security_generation_and_rotation_are_explicit(importer, tmp_path):
    key_path = tmp_path / "private/readiness/receipt-hmac.key"
    assert importer.read_receipt_key(key_path) is None and not key_path.exists()
    first_id = importer.generate_receipt_key(key_path); first = key_path.read_bytes()
    assert len(first) >= 32 and key_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        importer.generate_receipt_key(key_path)
    second_id = importer.generate_receipt_key(key_path, rotate=True)
    assert first_id != second_id and first != key_path.read_bytes()
    key_path.chmod(0o644); assert importer.read_receipt_key(key_path) is None
    key_path.unlink(); target = tmp_path / "target"; target.write_bytes(b"z" * 32); target.chmod(0o600)
    key_path.symlink_to(target); assert importer.read_receipt_key(key_path) is None


@pytest.mark.parametrize("mutation", ["stale", "common", "missing", "duplicate", "container", "parent", "custom_id",
                                      "chunk_id", "document_id", "task", "status", "hash", "bytes", "metadata"])
def test_vector_builder_rejects_non_document_proofs(importer, tmp_path, mutation):
    docs = _docs(importer); observations = _observations(importer, docs)
    path = sorted(docs)[0]
    if mutation == "stale": observations[path][0]["chunk"] = "stale unrelated distinctive words from old canonical body"
    elif mutation == "common": observations[path][0]["chunk"] = "the and is of"
    elif mutation == "missing": observations.pop(path)
    elif mutation == "duplicate": observations[path].append(copy.deepcopy(observations[path][0]))
    elif mutation == "container": observations[path][0]["documents"][0]["container_tags"] = ["wrong"]
    elif mutation == "parent": observations[path][0]["documents"] = []
    elif mutation == "custom_id": observations[path][0]["documents"][0]["custom_id"] = "wrong"
    elif mutation == "chunk_id": observations[path][0].pop("id")
    elif mutation == "document_id": observations[path][0].pop("document_id")
    elif mutation == "task": observations[path][0]["documents"][0].pop("task_type")
    elif mutation == "status": observations[path][0]["documents"][0].pop("status")
    elif mutation == "hash": observations[path][0]["metadata"]["content_sha256"] = "0" * 64
    elif mutation == "bytes": observations[path][0]["metadata"]["content_bytes"] += 1
    else: observations[path][0]["metadata"].pop("canonical_root")
    key_path = tmp_path / "key"; importer.generate_receipt_key(key_path); key = importer.read_receipt_key(key_path)
    with pytest.raises(ValueError):
        importer.build_vector_readiness_receipt(docs, observations, generation="g", generated_at=datetime.now(timezone.utc), key=key)


def test_arbitrary_digest_cannot_be_supplied_to_builder(importer, tmp_path):
    docs = _docs(importer); observations = _observations(importer, docs)
    observations[sorted(docs)[0]][0]["chunk_digest"] = "f" * 64
    key_path = tmp_path / "key"; importer.generate_receipt_key(key_path); key = importer.read_receipt_key(key_path)
    receipt = importer.build_vector_readiness_receipt(docs, observations, generation="g", generated_at=datetime.now(timezone.utc), key=key)
    assert "f" * 64 not in json.dumps(receipt)


def test_same_boilerplate_chunk_cannot_prove_two_documents(importer, tmp_path):
    common = b"standard repeated boilerplate language appears identically in every canonical document"
    docs = {doc["relative_path"]: doc for doc in (
        importer.item("Skills/A.md", common), importer.item("Skills/B.md", common))}
    observations = _observations(importer, docs)
    key_path = tmp_path / "key"; importer.generate_receipt_key(key_path)
    with pytest.raises(ValueError, match="corpus-unique"):
        importer.build_vector_readiness_receipt(docs, observations, generation="g",
            generated_at=datetime.now(timezone.utc), key=importer.read_receipt_key(key_path))


def test_unique_chunk_proves_only_its_current_document(importer, tmp_path):
    docs = _docs(importer); observations = _observations(importer, docs)
    first, second = sorted(docs)
    observations[first][0]["chunk"] = observations[second][0]["chunk"]
    key_path = tmp_path / "key"; importer.generate_receipt_key(key_path)
    with pytest.raises(ValueError):
        importer.build_vector_readiness_receipt(docs, observations, generation="g",
            generated_at=datetime.now(timezone.utc), key=importer.read_receipt_key(key_path))


def test_small_document_without_unique_run_fails_closed(importer, tmp_path):
    doc = importer.item("Skills/Tiny.md", b"tiny")
    docs = {doc["relative_path"]: doc}; observations = _observations(importer, docs)
    key_path = tmp_path / "key"; importer.generate_receipt_key(key_path)
    with pytest.raises(ValueError, match="corpus-unique"):
        importer.build_vector_readiness_receipt(docs, observations, generation="g",
            generated_at=datetime.now(timezone.utc), key=importer.read_receipt_key(key_path))


@pytest.mark.parametrize("mutation", ["conflicting_custom", "conflicting_document", "aggregate", "duplicate_chunk",
                                      "malformed_chunk", "malformed_parent"])
def test_vector_contract_rejects_alias_aggregate_and_duplicate_shapes(importer, tmp_path, mutation):
    docs = _docs(importer); observations = _observations(importer, docs); paths = sorted(docs)
    result = observations[paths[0]][0]
    if mutation == "conflicting_custom": result["documents"][0]["customId"] = "wrong"
    elif mutation == "conflicting_document": result["documentId"] = "wrong"
    elif mutation == "aggregate": result["isAggregated"] = True
    elif mutation == "duplicate_chunk": observations[paths[1]][0]["id"] = result["id"]
    elif mutation == "malformed_chunk": result["id"] = "bad chunk id"
    else: result["documents"][0]["id"] = "bad/parent"
    key_path = tmp_path / "key"; importer.generate_receipt_key(key_path)
    with pytest.raises(ValueError):
        importer.build_vector_readiness_receipt(docs, observations, generation="g",
            generated_at=datetime.now(timezone.utc), key=importer.read_receipt_key(key_path))


def test_runtime_and_importer_use_same_eligible_source_projection(importer, tmp_path):
    source = tmp_path / "source"; (source / "Skills").mkdir(parents=True); (source / "Jarvis/Family Shared").mkdir(parents=True)
    (source / "Skills/A.md").write_bytes(b"alpha cedar telescope marmalade compass current distinctive body content for readiness proof")
    (source / "Jarvis/Family Shared/B.md").write_bytes(b"---\nevent_date: 2026-09-12\n---\nbravo quartz violin nebula saffron current distinctive body content for readiness proof")
    template = source / "Skills/Recipe.md"
    sensitive = source / "Jarvis/Sensitive.md"
    template.write_bytes(b"---\ntype: recipe-template\n---\nexcluded template")
    sensitive.write_bytes(b"password=do-not-leak excluded sensitive document")

    old_root = importer.ROOT
    try:
        importer.ROOT = source
        scanned = importer.canonical_documents()
    finally:
        importer.ROOT = old_root
    projection = importer.project_canonical_documents(scanned)
    docs = projection["documents"]
    assert set(docs) == {"Skills/A.md", "Jarvis/Family Shared/B.md"}
    assert projection["excluded"] == (
        {"relative_path": "Jarvis/Sensitive.md", "reason": "sensitive_content"},
        {"relative_path": "Skills/Recipe.md", "reason": "template"},
    )
    assert importer.scan_canonical_source(source) == importer.source_fingerprint_from_documents(docs)

    inventory = _inventory(importer, docs); observations = _observations(importer, docs)
    key_path = tmp_path / "private/readiness/receipt-hmac.key"
    importer.generate_receipt_key(key_path); key = importer.read_receipt_key(key_path)
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    storage = importer.build_storage_reconciliation_receipt(
        docs, inventory, generation="generation-1", generated_at=now, key=key)
    vector = importer.build_vector_readiness_receipt(
        docs, observations, generation="generation-1", generated_at=now, key=key)
    encoded_receipts = json.dumps([storage, vector])
    assert "do-not-leak" not in encoded_receipts
    assert "password=" not in encoded_receipts
    assert all(path not in encoded_receipts for path in projection["documents"])
    receipt_dir = tmp_path / "private/readiness"
    for name, value in (("storage-reconciliation.json", storage), ("vector-readiness.json", vector)):
        path = receipt_dir / name; path.write_bytes(importer.canonical_bytes(value)); path.chmod(0o600)
    assert importer.readiness_from_paths(receipt_dir / "storage-reconciliation.json", receipt_dir / "vector-readiness.json", key_path=key_path, source_root=source, now=now)

    template.write_bytes(b"ordinary eligible source after template marker removed")
    assert not importer.readiness_from_paths(receipt_dir / "storage-reconciliation.json", receipt_dir / "vector-readiness.json", key_path=key_path, source_root=source, now=now)
    template.write_bytes(b"---\ntype: recipe-template\n---\nexcluded template")
    sensitive.write_bytes(b"safe source after sensitive marker removed")
    assert not importer.readiness_from_paths(receipt_dir / "storage-reconciliation.json", receipt_dir / "vector-readiness.json", key_path=key_path, source_root=source, now=now)
    sensitive.write_bytes(b"password=changed-but-still-excluded")
    assert importer.readiness_from_paths(receipt_dir / "storage-reconciliation.json", receipt_dir / "vector-readiness.json", key_path=key_path, source_root=source, now=now)


def test_receipt_write_requires_explicit_key_init_and_writes_no_secrets(importer, tmp_path):
    docs = _docs(importer); inventory = _inventory(importer, docs); observations = _observations(importer, docs)
    now = datetime.now(timezone.utc); private = tmp_path / "private"
    with pytest.raises(importer.ReconciliationRequired):
        importer.write_verified_readiness_receipts(private, docs, inventory, observations, generation="g", generated_at=now)
    assert not (private / "readiness/receipt-hmac.key").exists()
    importer.write_verified_readiness_receipts(private, docs, inventory, observations, generation="g", generated_at=now, initialize_key=True)
    key = (private / "readiness/receipt-hmac.key").read_bytes()
    for name in ("storage-reconciliation.json", "vector-readiness.json"):
        payload = (private / "readiness" / name).read_bytes()
        assert key not in payload and (private / "readiness" / name).stat().st_mode & 0o777 == 0o600
