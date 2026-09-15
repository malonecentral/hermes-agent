from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("obsidian_supermemory_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_exact_acl_path_and_immutable_identity(importer):
    shared = importer.item("Jarvis/Family Shared/Food.md", b"hello")
    assert shared["visibility"] == "family_shared"
    assert shared["identity_scope"] == shared["canonical_root"] == "owner"
    assert importer.container_for_doc(shared) == "family_shared"
    assert importer.container_for_doc(importer.item("Skills/Dennis.md", b"hello")) == "owner_primary"
    for similar in ("Jarvis/family shared/Food.md", "Jarvis/Family Sharedness/Food.md"):
        assert importer.item(similar, b"hello")["visibility"] == "owner_private"
    for unapproved in ("Family Shared/Food.md", "JarvisX/Family Shared/Food.md"):
        with pytest.raises(ValueError, match="canonical path"):
            importer.item(unapproved, b"hello")


@pytest.mark.parametrize("path", ["../x.md", "Jarvis/../x.md", "/x.md", "Jarvis\\Family Shared\\x.md", "./x.md", "x.txt"])
def test_path_spoof_and_traversal_rejected(importer, path):
    with pytest.raises(ValueError, match="canonical path"):
        importer.item(path, b"hello")


def test_event_date_alias_scalar_and_list_normalization(importer):
    for frontmatter in ("event_date: 2024-02-29", "eventDate: [2024-02-29]", "event_date: [2024-02-29, 2024-02-29]"):
        doc = importer.item("x.md", f"---\n{frontmatter}\n---\nbody".encode())
        assert doc["event_date"] == "2024-02-29"


@pytest.mark.parametrize("frontmatter", [
    "event_date: 2023-02-29", "event_date: yesterday", "eventDate: [2024-01-01, 2024-01-02]",
    "event_date: 2024-01-01\neventDate: 2024-01-02", "event_date: [2024-01-01",
])
def test_event_date_rejects_malformed_impossible_or_conflicting(importer, frontmatter):
    with pytest.raises(ValueError):
        importer.item("private/path.md", f"---\n{frontmatter}\n---\nbody".encode())


def test_replacement_delete_boundary_records_backend_reality(importer):
    class Documents:
        def delete(self, document_id, timeout):
            assert document_id == "old-id"
    client = type("Client", (), {"documents": Documents()})()
    doc = importer.item("x.md", b"new")
    row = importer.update(client, doc, {"document_id": "old-id"})
    assert row["replacement_stage"] == "deleted"
    assert row["document_id"] == ""
    assert row["replaced_document_id"] == "old-id"


def test_replacement_delete_failure_does_not_claim_deleted(importer):
    class Documents:
        def delete(self, document_id, timeout):
            raise RuntimeError("offline")
    client = type("Client", (), {"documents": Documents()})()
    with pytest.raises(RuntimeError):
        importer.update(client, importer.item("x.md", b"new"), {"document_id": "old-id"})


class _Result:
    def __init__(self, ident="new-id", status="queued"):
        self.id, self.status = ident, status


def _recovery_client(*, get_status=None, add_error=None, listed=None, list_error=None):
    calls = []
    class Documents:
        def list(self, **kwargs):
            calls.append(("list", kwargs.get("filters")))
            if list_error: raise list_error
            return {"memories": listed or [], "pagination": {"current_page": 1, "total_pages": 1}}
        def get(self, ident, timeout):
            calls.append(("get", ident))
            if isinstance(get_status, Exception): raise get_status
            return _Result(ident, get_status)
        def delete(self, ident, timeout): calls.append(("delete", ident))
        def add(self, **kwargs):
            calls.append(("add", kwargs["custom_id"]))
            if add_error: raise add_error
            return _Result()
    return type("Client", (), {"documents": Documents()})(), calls


@pytest.mark.parametrize("stage", ["deleted", "add_failed"])
def test_restart_deleted_or_idless_add_failed_resubmits_and_checkpoints(importer, stage):
    client, calls = _recovery_client()
    checkpoints = []
    doc = importer.item("x.md", b"new")
    row = importer.recover_submission(client, doc, {
        "relative_path": "x.md", "sha256": doc["sha256"], "replacement_stage": stage,
    }, lambda value: checkpoints.append(dict(value)))
    assert [call[0] for call in calls] == ["list", "add"]
    assert row["replacement_stage"] == "add_submitted"
    assert checkpoints[-1]["document_id"] == "new-id"


@pytest.mark.parametrize("stage", ["add_submitted", "pending"])
def test_restart_known_submission_polls_before_resubmit(importer, stage):
    client, calls = _recovery_client(get_status="processing")
    checkpoints = []
    doc = importer.item("x.md", b"new")
    row = importer.recover_submission(client, doc, {
        "relative_path": "x.md", "sha256": doc["sha256"], "document_id": "known",
        "replacement_stage": stage,
    }, lambda value: checkpoints.append(dict(value)))
    assert calls == [("get", "known")]
    assert row["replacement_stage"] == "pending"
    assert checkpoints[-1]["final_status"] == "processing"


def test_restart_failed_known_submission_deletes_then_resubmits(importer):
    client, calls = _recovery_client(get_status="failed")
    checkpoints = []
    doc = importer.item("x.md", b"new")
    row = importer.recover_submission(client, doc, {
        "relative_path": "x.md", "sha256": doc["sha256"], "document_id": "known",
        "replacement_stage": "add_failed",
    }, lambda value: checkpoints.append(dict(value)))
    assert [call[0] for call in calls] == ["get", "delete", "list", "add"]
    assert [value["replacement_stage"] for value in checkpoints] == ["deleted", "add_submitted"]
    assert row["document_id"] == "new-id"


def test_restart_done_submission_is_idempotent(importer):
    client, calls = _recovery_client(get_status="done")
    checkpoints = []
    doc = importer.item("x.md", b"new")
    row = importer.recover_submission(client, doc, {
        "relative_path": "x.md", "sha256": doc["sha256"], "document_id": "known",
        "replacement_stage": "pending",
    }, lambda value: checkpoints.append(dict(value)))
    assert calls == [("get", "known")]
    assert row["replacement_stage"] == row["final_status"] == "done"
    assert checkpoints == [row]


def test_add_fault_checkpoints_add_failed_for_restart(importer):
    client, calls = _recovery_client(add_error=RuntimeError("fault"))
    checkpoints = []
    doc = importer.item("x.md", b"new")
    with pytest.raises(RuntimeError, match="fault"):
        importer.recover_submission(client, doc, {}, lambda value: checkpoints.append(dict(value)))
    assert [call[0] for call in calls] == ["list", "add"]
    assert checkpoints[-1]["replacement_stage"] == "reconcile_required"
    assert checkpoints[-1]["final_status"] == "unknown"


def test_restart_after_lost_response_reconciles_without_resubmit(importer):
    doc = importer.item("x.md", b"new")
    remote = {"id": "accepted-id", "custom_id": doc["custom_id"], "status": "done",
              "metadata": importer.metadata(doc)}
    client, calls = _recovery_client(listed=[remote])
    row = importer.recover_submission(client, doc, {"replacement_stage": "reconcile_required"}, lambda value: None)
    assert [call[0] for call in calls] == ["list"]
    assert row["document_id"] == "accepted-id"
    assert row["final_status"] == row["replacement_stage"] == "done"


def test_idless_recovery_fails_closed_when_lookup_unavailable(importer):
    client, calls = _recovery_client(list_error=RuntimeError("backend unavailable"))
    doc = importer.item("x.md", b"new")
    checkpoints = []
    with pytest.raises(RuntimeError, match="backend unavailable"):
        importer.recover_submission(client, doc, {"replacement_stage": "reconcile_required"},
                                    lambda value: checkpoints.append(dict(value)))
    assert [call[0] for call in calls] == ["list"]
    assert checkpoints[-1]["replacement_stage"] == "reconcile_required"


def test_backend_metadata_mismatch_requires_operator_reconciliation(importer):
    doc = importer.item("Jarvis/Family Shared/x.md", b"new")
    wrong = importer.metadata(doc) | {"visibility": "owner_private"}
    remote = {"id": "wrong", "custom_id": doc["custom_id"], "status": "done", "metadata": wrong}
    client, calls = _recovery_client(listed=[remote])
    with pytest.raises(importer.ReconciliationRequired, match="metadata"):
        importer.recover_submission(client, doc, {"replacement_stage": "reconcile_required"}, lambda value: None)
    assert [call[0] for call in calls] == ["list"]


def _remote(importer, doc, **overrides):
    value = {"id": "backend-id", "custom_id": doc["custom_id"], "status": "done",
             "metadata": importer.metadata(doc)}
    value.update(overrides)
    return value


def test_verify_only_is_read_only_and_reads_manifest_without_lock(importer, monkeypatch, capsys, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"documents":[]}\n')
    before = manifest.stat()
    lock = tmp_path / "sync.lock"
    monkeypatch.setattr(importer, "OUT", manifest)
    monkeypatch.setattr(importer, "LOCK", lock)
    private = importer.item("Skills/Ada.md", b"private")
    shared = importer.item("Jarvis/Family Shared/Food.md", b"shared")
    calls = []

    class Documents:
        def list(self, **kwargs):
            calls.append(("list", kwargs["page"]))
            doc = private if "Skills/Ada.md" in str(kwargs["filters"]) else shared
            return {"memories": [_remote(importer, doc)],
                    "pagination": {"current_page": 1, "total_pages": 1}}
        def add(self, **kwargs): raise AssertionError("verify-only called add")
        def delete(self, *args, **kwargs): raise AssertionError("verify-only called delete")
        def update(self, *args, **kwargs): raise AssertionError("verify-only called update")

    monkeypatch.setattr(importer, "canonical_documents", lambda: [private, shared])

    monkeypatch.setattr(importer, "atomic_receipt", lambda payload: (_ for _ in ()).throw(AssertionError("manifest write")))
    monkeypatch.setattr(importer, "Supermemory", lambda **kwargs: type("Client", (), {"documents": Documents()})())
    monkeypatch.setattr(importer, "api_key", lambda: "key")
    monkeypatch.setattr(importer, "completion_readiness", lambda client, current: {
        "reconciliation_complete": True,
    })
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--verify-only"])

    importer.main()

    result = json.loads(capsys.readouterr().out)
    assert result == {"schema_version": 4, "scanned": 2, "eligible": 2,
                      "owner_private_count": 1, "family_shared_count": 1, "manifest_count": 0,
                      "backend_verified_count": 2, "failure_count": 0,
                      "failed_paths": [], "verification_complete": True,
                      "reconciliation_complete": True,
                      "verify_only": True, "filesystem_mutated": False,
                      "backend_mutated": False}
    assert calls == [("list", 1), ("list", 1)]
    after = manifest.stat()
    assert manifest.read_bytes() == b'{"documents":[]}\n'
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)
    assert not lock.exists()


def test_verify_only_exits_nonzero_when_any_document_is_incomplete(importer, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(importer, "OUT", tmp_path / "manifest.json")
    doc = importer.item("x.md", b"content")
    client, calls = _recovery_client(listed=[])
    monkeypatch.setattr(importer, "canonical_documents", lambda: [doc])
    monkeypatch.setattr(importer, "Supermemory", lambda **kwargs: client)
    monkeypatch.setattr(importer, "api_key", lambda: "key")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--verify-only"])
    with pytest.raises(SystemExit) as exc:
        importer.main()
    assert exc.value.code == 1
    assert [call[0] for call in calls] and all(call[0] == "list" for call in calls)
    result = json.loads(capsys.readouterr().out)
    assert result["failed_paths"] == ["x.md"]
    assert result["verification_complete"] is False


def test_lookup_backend_document_accepts_provider_empty_page_one_of_zero(importer):
    doc = importer.item("x.md", b"content")
    calls = []

    class Documents:
        def list(self, **kwargs):
            calls.append(kwargs["page"])
            return {"memories": [],
                    "pagination": {"current_page": 1.0, "total_pages": 0.0}}

    client = type("Client", (), {"documents": Documents()})()
    assert importer.lookup_backend_document(client, doc) is None
    assert calls == [1]


def test_lookup_backend_document_rejects_zero_pages_with_rows(importer):
    doc = importer.item("x.md", b"content")

    class Documents:
        def list(self, **kwargs):
            return {"memories": [_remote(importer, doc)],
                    "pagination": {"current_page": 1, "total_pages": 0}}

    client = type("Client", (), {"documents": Documents()})()
    with pytest.raises(importer.ReconciliationRequired, match="pagination"):
        importer.lookup_backend_document(client, doc)


def test_verify_only_exhausts_pagination_and_rejects_duplicates(importer):
    doc = importer.item("x.md", b"content")
    pages = {
        1: {"memories": [_remote(importer, doc)], "pagination": {"current_page": 1, "total_pages": 2}},
        2: {"memories": [_remote(importer, doc, id="duplicate")], "pagination": {"current_page": 2, "total_pages": 2}},
    }
    calls = []
    client = type("Client", (), {"documents": type("Documents", (), {
        "list": lambda self, **kwargs: calls.append(kwargs["page"]) or pages[kwargs["page"]]
    })()})()
    result = importer.verify_backend(client, {doc["relative_path"]: doc})
    assert calls == [1, 2]
    assert result["backend_verified_count"] == 0
    assert result["failed_paths"] == ["x.md"]


@pytest.mark.parametrize("change", [
    {"custom_id": "wrong"},
    {"status": "processing"},
    {"metadata": {}},
])
def test_verify_only_reports_path_only_for_backend_mismatch(importer, change):
    doc = importer.item("Skills/secret-name.md", b"content")
    remote = _remote(importer, doc, **change)
    client, _ = _recovery_client(listed=[remote])
    result = importer.verify_backend(client, {doc["relative_path"]: doc})
    assert result["verification_complete"] is False
    assert result["failure_count"] == 1
    assert result["failed_paths"] == ["Skills/secret-name.md"]
    assert set(result) == {"backend_verified_count", "failure_count", "failed_paths", "verification_complete"}


def test_search_readiness_requires_result_and_sole_parent_v4_metadata_convergence(importer):
    doc = importer.item("Jarvis/Family Shared/x.md", b"content")
    expected = importer.metadata(doc)
    parent = type("Parent", (), {"metadata": expected})()
    result = type("Result", (), {"metadata": expected, "documents": [parent]})()
    assert importer.validate_search_result_metadata(result, doc) is True


@pytest.mark.parametrize("result_metadata,parents", [
    (None, [{}]),
    ({}, [{}]),
    ({"index_schema_version": 4}, []),
    ({"index_schema_version": 4}, [{}, {}]),
])
def test_search_readiness_rejects_missing_partial_or_ambiguous_metadata(
        importer, result_metadata, parents):
    doc = importer.item("x.md", b"content")
    result = {"metadata": result_metadata,
              "documents": [{"metadata": value} for value in parents]}
    with pytest.raises(importer.ReconciliationRequired, match="search"):
        importer.validate_search_result_metadata(result, doc)


def test_search_readiness_rejects_result_parent_disagreement(importer):
    doc = importer.item("x.md", b"content")
    expected = importer.metadata(doc)
    wrong = expected | {"visibility": "family_shared"}
    result = {"metadata": expected, "documents": [{"metadata": wrong}]}
    with pytest.raises(importer.ReconciliationRequired, match="search"):
        importer.validate_search_result_metadata(result, doc)


def test_search_readiness_summary_fails_closed_on_missing_results(importer):
    first = importer.item("a.md", b"a")
    second = importer.item("b.md", b"b")
    expected = importer.metadata(first)
    valid = {"metadata": expected, "documents": [{"metadata": expected}]}
    result = importer.verify_search_readiness(
        {"a.md": [valid], "b.md": []}, {"a.md": first, "b.md": second})
    assert result == {
        "search_verified_count": 1, "search_failure_count": 1,
        "search_failed_paths": ["b.md"], "search_readiness_complete": False,
    }


def _legacy_remote(importer, doc, ident="legacy-id"):
    legacy = importer.metadata(doc)
    for key in ("index_schema_version", "visibility", "identity_scope", "canonical_root"):
        legacy.pop(key, None)
    return {"id": ident, "custom_id": doc["custom_id"], "status": "done",
            "content": doc["content"], "metadata": legacy}


def _previous_row(importer, doc, **overrides):
    row = importer.record(doc, _Result("legacy-id", "done"), "legacy-id")
    row.update({"final_status": "done", "replacement_stage": "done"})
    row.update(overrides)
    return row


def test_schema_v4_backfill_plans_all_verified_legacy_records(importer):
    private = [importer.item(f"Skills/Private-{index}.md", f"private-{index}".encode())
               for index in range(133)]
    shared = [importer.item(f"Jarvis/Family Shared/Shared-{index}.md", f"shared-{index}".encode())
              for index in range(104)]
    docs = {doc["relative_path"]: doc for doc in private + shared}
    previous = {rel: _previous_row(importer, doc) for rel, doc in docs.items()}
    remotes = {rel: _legacy_remote(importer, doc, f"legacy-{index}")
               for index, (rel, doc) in enumerate(docs.items())}

    class Documents:
        def list(self, **kwargs):
            assert kwargs["include_content"] is True
            container = kwargs["container_tags"][0]
            page = kwargs["page"]
            values = [remote for rel, remote in remotes.items()
                      if importer.container_for_doc(docs[rel]) == container]
            total_pages = (len(values) + 99) // 100
            return {"memories": values[(page - 1) * 100:page * 100],
                    "pagination": {"current_page": page, "total_pages": total_pages}}

    plan = importer.build_schema_v4_backfill_plan(
        type("Client", (), {"documents": Documents()})(), docs, previous,
        expected_eligible=237, expected_owner_private=133, expected_family_shared=104)
    assert set(plan["replacements"]) == set(docs)
    assert plan["already_v4"] == 0
    assert len(plan["replacements"]) == 237
    assert all(row["replacement_stage"] == "verified_legacy" for row in plan["replacements"].values())


def test_schema_v4_backfill_resume_accepts_only_matching_atomic_checkpoint(importer):
    doc = importer.item("x.md", b"content")
    current = {"x.md": doc}
    checkpoint = _previous_row(importer, doc, document_id="", replacement_stage="deleted",
                               replaced_document_id="legacy-id")
    client, calls = _recovery_client(listed=[])
    plan = importer.build_schema_v4_backfill_plan(
        client, current, {"x.md": checkpoint},
        expected_eligible=1, expected_owner_private=1, expected_family_shared=0)
    assert calls == []
    assert plan["replacements"]["x.md"]["replacement_stage"] == "deleted"


@pytest.mark.parametrize("counts", [(2, 1, 0), (1, 0, 1), (1, 1, 1)])
def test_schema_v4_backfill_fails_closed_on_count_drift(importer, counts):
    doc = importer.item("x.md", b"content")
    with pytest.raises(importer.ReconciliationRequired, match="count"):
        importer.build_schema_v4_backfill_plan(
            object(), {"x.md": doc}, {"x.md": _previous_row(importer, doc)},
            expected_eligible=counts[0], expected_owner_private=counts[1],
            expected_family_shared=counts[2])


def test_schema_v4_backfill_rejects_unexpected_removed_paths(importer):
    doc = importer.item("x.md", b"content")
    removed = dict(_previous_row(importer, doc), relative_path="removed.md")
    with pytest.raises(importer.ReconciliationRequired, match="path set"):
        importer.build_schema_v4_backfill_plan(
            object(), {"x.md": doc}, {"x.md": _previous_row(importer, doc), "removed.md": removed},
            expected_eligible=1, expected_owner_private=1, expected_family_shared=0)


@pytest.mark.parametrize("change, message", [
    ({"content": "wrong"}, "content"),
    ({"status": "processing"}, "terminal"),
    ({"metadata": {"source": "other"}}, "identity"),
])
def test_schema_v4_backfill_rejects_unverified_legacy_record(importer, change, message):
    doc = importer.item("x.md", b"content")
    remote = _legacy_remote(importer, doc) | change
    class Documents:
        def list(self, **kwargs):
            rows = [remote] if kwargs["container_tags"] == [importer.CONTAINER] else []
            return {"memories": rows, "pagination": {
                "current_page": 1, "total_pages": 1 if rows else 0}}
    client = type("Client", (), {"documents": Documents()})()
    with pytest.raises(importer.ReconciliationRequired, match=message):
        importer.build_schema_v4_backfill_plan(
            client, {"x.md": doc}, {"x.md": _previous_row(importer, doc)},
            expected_eligible=1, expected_owner_private=1, expected_family_shared=0)


@pytest.mark.parametrize("suffix", ["", " ", "\t\r\n", " \n\n"])
def test_schema_v4_backfill_accepts_only_eof_whitespace_drift(importer, suffix):
    doc = importer.item("x.md", b"content")
    remote = _legacy_remote(importer, doc) | {"content": doc["content"] + suffix}
    importer.validate_legacy_backend_document(remote, doc)


@pytest.mark.parametrize("content", ["Xcontent", "con tent", "content\nX", "content \nX"])
def test_schema_v4_backfill_rejects_non_eof_content_drift(importer, content):
    doc = importer.item("x.md", b"content")
    remote = _legacy_remote(importer, doc) | {"content": content}
    with pytest.raises(importer.ReconciliationRequired, match="content"):
        importer.validate_legacy_backend_document(remote, doc)


@pytest.mark.parametrize("key,value", [("content_sha256", "bad"), ("content_bytes", 999)])
def test_schema_v4_backfill_requires_exact_canonical_byte_metadata(importer, key, value):
    doc = importer.item("x.md", b"content")
    remote = _legacy_remote(importer, doc)
    remote["metadata"][key] = value
    with pytest.raises(importer.ReconciliationRequired, match="identity"):
        importer.validate_legacy_backend_document(remote, doc)


def test_logical_v3_snapshot_is_private_hashed_and_exact(importer, tmp_path):
    doc = importer.item("x.md", b"content")
    remote = _legacy_remote(importer, doc) | {"_inventory_container": importer.CONTAINER}
    path = tmp_path / "private" / "logical-v3.json"
    plan = {"already_v4": 0, "replacements": {"x.md": {}},
            "inventory": {doc["custom_id"]: remote}}
    receipt = importer.create_logical_v3_snapshot(path, {"x.md": doc}, plan,
        expected_eligible=1, expected_owner_private=1, expected_family_shared=0)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert receipt["snapshot_format"] == "logical_backend_returned_v3"
    assert receipt["byte_original"] is False and receipt["verified"] is True
    loaded = importer.load_logical_v3_snapshot(path, expected_eligible=1,
        expected_owner_private=1, expected_family_shared=0)
    assert loaded["records"][0]["content"] == remote["content"]
    assert loaded["records"][0]["metadata"] == remote["metadata"]


def _write_snapshot(importer, tmp_path, doc):
    legacy = _legacy_remote(importer, doc) | {
        "_inventory_container": importer.container_for_doc(doc),
    }
    path = tmp_path / "logical-v3.json"
    importer.create_logical_v3_snapshot(path, {doc["relative_path"]: doc},
        {"already_v4": 0, "replacements": {doc["relative_path"]: {}},
         "inventory": {doc["custom_id"]: legacy}},
        expected_eligible=1, expected_owner_private=1, expected_family_shared=0)
    return path, legacy


def test_schema_v3_inverse_deletes_only_verified_v4_and_restores_exact_snapshot(importer, tmp_path):
    doc = importer.item("x.md", b"content"); path, legacy = _write_snapshot(importer, tmp_path, doc)
    v4 = _remote(importer, doc, content=doc["content"])
    calls = []; replacement = dict(legacy, id="replacement-id")
    class Documents:
        def list(self, **kwargs):
            calls.append(("list", kwargs["container_tags"]))
            if kwargs["container_tags"] != [importer.CONTAINER]:
                return {"memories": [], "pagination": {"current_page": 1, "total_pages": 0}}
            row = replacement if any(c[0] == "add" for c in calls) else v4
            return {"memories": [row], "pagination": {"current_page": 1, "total_pages": 1}}
        def delete(self, ident, timeout): calls.append(("delete", ident))
        def add(self, **kwargs):
            calls.append(("add", kwargs["custom_id"]))
            assert kwargs["content"] == legacy["content"] and kwargs["metadata"] == legacy["metadata"]
            return _Result("replacement-id", "queued")
        def get(self, ident, timeout): calls.append(("get", ident)); return replacement
    receipt = importer.execute_schema_v3_rollback(type("Client", (), {"documents": Documents()})(),
        {"x.md": doc}, path, expected_eligible=1, expected_owner_private=1, expected_family_shared=0)
    assert receipt["complete"] is True and receipt["restored_count"] == 1
    assert ("delete", "backend-id") in calls
    checkpoint = json.loads(path.with_suffix(".json.rollback.json").read_text())
    assert checkpoint["records"][0]["replacement_document_id"] == "replacement-id"
    assert checkpoint["records"][0]["stage"] == "done"


def test_schema_v3_delete_fault_is_fail_closed_and_resumable(importer, tmp_path):
    doc = importer.item("x.md", b"content"); path, _ = _write_snapshot(importer, tmp_path, doc)
    v4 = _remote(importer, doc, content=doc["content"])
    class Documents:
        def list(self, **kwargs):
            rows = [v4] if kwargs["container_tags"] == [importer.CONTAINER] else []
            return {"memories": rows, "pagination": {
                "current_page": 1, "total_pages": 1 if rows else 0}}
        def delete(self, ident, timeout): raise RuntimeError("injected delete fault")
        def add(self, **kwargs): raise AssertionError("add after failed delete")
    with pytest.raises(RuntimeError, match="injected"):
        importer.execute_schema_v3_rollback(type("Client", (), {"documents": Documents()})(),
            {"x.md": doc}, path, expected_eligible=1, expected_owner_private=1, expected_family_shared=0)
    checkpoint = json.loads(path.with_suffix(".json.rollback.json").read_text())
    assert checkpoint["records"][0]["stage"] == "verified_v4"
    assert checkpoint["records"][0]["replacement_document_id"] == ""


def test_schema_v4_backfill_container_isolation_never_touches_conversations(importer):
    assert importer.CONTAINER == "owner_primary"
    assert importer.FAMILY_CONTAINER == "family_shared"
    assert importer.CONVERSATION_CONTAINER == "owner_conversations"
    assert len({importer.CONTAINER, importer.FAMILY_CONTAINER, importer.CONVERSATION_CONTAINER}) == 3


def test_schema_v4_backfill_cli_requires_all_explicit_counts(importer, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--backfill-schema-v4"])
    with pytest.raises(SystemExit):
        importer.parse_args()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--backfill-schema-v4",
                                      "--snapshot", "/private/logical-v3.json",
                                      "--expected-eligible", "237",
                                      "--expected-owner-private", "133",
                                      "--expected-family-shared", "104"])
    args = importer.parse_args()
    assert (args.expected_eligible, args.expected_owner_private,
            args.expected_family_shared) == (237, 133, 104)


def test_dry_run_reads_only_source_and_manifest_and_returns_path_plan(importer, monkeypatch, capsys, tmp_path):
    manifest = tmp_path / "manifest.json"
    old = importer.item("Skills/Old.md", b"old")
    manifest.write_text(json.dumps({"documents": [importer.record(old, _Result("id", "done"), "id")]}))
    manifest.chmod(0o600)
    before = manifest.stat(); before_bytes = manifest.read_bytes()
    new = importer.item("Skills/New.md", b"new")
    lock = tmp_path / "sync.lock"
    monkeypatch.setattr(importer, "OUT", manifest)
    monkeypatch.setattr(importer, "LOCK", lock)
    monkeypatch.setattr(importer, "canonical_documents", lambda: [new])
    monkeypatch.setattr(importer, "Supermemory", lambda **kwargs: (_ for _ in ()).throw(AssertionError("provider constructed")))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--dry-run"])
    importer.main()
    result = json.loads(capsys.readouterr().out)
    assert result["plan"] == [
        {"action": "add", "relative_path": "Skills/New.md"},
        {"action": "delete", "relative_path": "Skills/Old.md"},
    ]
    after = manifest.stat()
    assert before_bytes == manifest.read_bytes()
    assert (before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns)
    assert not lock.exists()


def test_read_only_modes_reject_output_file(importer, monkeypatch, tmp_path):
    for mode in ("--dry-run", "--verify-only"):
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), mode, "--output", str(tmp_path / "out.json")])
        with pytest.raises(SystemExit):
            importer.parse_args()
    assert not (tmp_path / "out.json").exists()


def test_schema_v4_backfill_cli_is_mutually_exclusive_with_read_only_modes(importer, monkeypatch):
    for read_only in ("--dry-run", "--verify-only"):
        monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--backfill-schema-v4", read_only,
                                          "--snapshot", "/private/logical-v3.json",
                                          "--expected-eligible", "237",
                                          "--expected-owner-private", "133",
                                          "--expected-family-shared", "104"])
        with pytest.raises(SystemExit):
            importer.parse_args()
