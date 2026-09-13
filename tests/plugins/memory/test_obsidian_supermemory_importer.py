from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path.home() / ".hermes/scripts/import-obsidian-supermemory.py"


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
    for similar in ("Family Shared/Food.md", "Jarvis/family shared/Food.md", "Jarvis/Family Sharedness/Food.md", "JarvisX/Family Shared/Food.md"):
        assert importer.item(similar, b"hello")["visibility"] == "owner_private"


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
            calls.append(("list", kwargs["filters"]))
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
