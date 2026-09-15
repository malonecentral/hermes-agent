from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("rollback_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class LostResponse(RuntimeError):
    pass


NO_RESPONSE = object()


class Documents:
    def __init__(self, rows=()):
        self.rows = {r["id"]: copy.deepcopy(r) for r in rows}
        self.calls = []
        self.next_id = 1
        self.add_fault = self.delete_fault = None
        self.list_fault = self.get_fault = None
        self.get_response = NO_RESPONSE
        self.delete_mutates = True

    def list(self, **kwargs):
        container = kwargs["container_tags"][0]
        self.calls.append(("list", container))
        if self.list_fault:
            fault, self.list_fault = self.list_fault, None
            raise fault
        rows = [copy.deepcopy(r) for r in self.rows.values() if r["container_tags"] == [container]]
        return {"memories": rows, "pagination": {"current_page": 1, "total_pages": 1 if rows else 0,
                                                    "total_items": len(rows)}}

    def get(self, ident, timeout):
        self.calls.append(("get", ident))
        if self.get_fault:
            fault, self.get_fault = self.get_fault, None
            raise fault
        if self.get_response is not NO_RESPONSE:
            response = self.get_response
            return response() if callable(response) else response
        if ident not in self.rows:
            raise KeyError(ident)
        return copy.deepcopy(self.rows[ident])

    def add(self, **kwargs):
        self.calls.append(("add", kwargs["custom_id"])); ident = f"restored-{self.next_id}"; self.next_id += 1
        self.rows[ident] = {"id": ident, "custom_id": kwargs["custom_id"], "status": "done",
            "task_type": kwargs["task_type"], "content": kwargs["content"],
            "metadata": copy.deepcopy(kwargs["metadata"]), "container_tags": [kwargs["container_tag"]]}
        if self.add_fault:
            fault, self.add_fault = self.add_fault, None
            raise fault
        return {"id": ident, "status": "queued"}

    def delete(self, ident, timeout):
        self.calls.append(("delete", ident))
        if self.delete_mutates:
            self.rows.pop(ident, None)
        if self.delete_fault:
            fault, self.delete_fault = self.delete_fault, None
            raise fault
        return {"id": ident}


class Client:
    def __init__(self, rows=()): self.documents = Documents(rows)


def remote(i, doc, ident, *, container=None):
    return {"id": ident, "custom_id": doc["custom_id"], "status": "done", "task_type": "superrag",
            "content": doc["content"], "metadata": copy.deepcopy(i.metadata(doc)),
            "container_tags": [container or i.container_for_doc(doc)]}


def ident(row):
    return {"custom_id": row["custom_id"], "container": row["container_tags"][0],
            "sha256": row["metadata"]["content_sha256"], "bytes": row["metadata"]["content_bytes"],
            "document_id": row["id"]}


def plan_row(action, doc, old=None):
    target = None if action == "delete" else ("family_shared" if doc["visibility"] == "family_shared" else "owner_primary")
    post = None if action == "delete" else {"custom_id": doc["custom_id"], "container": target,
        "sha256": doc["sha256"], "bytes": doc["bytes"], "document_id": None}
    expected = old if action == "delete" else post
    return {"action": action, "relative_path": doc["relative_path"], "custom_id": doc["custom_id"],
        "source_container": old["container"] if old else None, "target_container": target,
        "expected_sha256": expected["sha256"], "expected_bytes": expected["bytes"],
        "expected_pre_identity": old, "expected_post_identity": post}


def forward(i, tmp_path, client, current, plan):
    return i.DurableReconciliationExecutor(client, tmp_path / "private", sleep=lambda _: None,
                                             poll_attempts=2).execute(current, plan, "forward-txn")


def rollback(i, tmp_path, client, plan, **kwargs):
    return i.DurableRollbackExecutor(client, tmp_path / "private", sleep=lambda _: None,
                                      poll_attempts=2, **kwargs).execute(plan, "forward-txn", "rollback-txn")


def test_add_delete_replace_move_and_mixed_restore_exact_pre_state(importer, tmp_path):
    add = importer.item("Skills/add.md", b"add")
    deleted = importer.item("Skills/deleted.md", b"deleted")
    old = importer.item("Skills/replaced.md", b"old")
    new = importer.item("Skills/replaced.md", b"new")
    moved_old = importer.item("Skills/moved.md", b"move")
    moved_new = importer.item("Jarvis/Family Shared/moved.md", b"move")
    rows = [remote(importer, deleted, "deleted"), remote(importer, old, "old"), remote(importer, moved_old, "moved")]
    before = copy.deepcopy(rows)
    client = Client(rows)
    plan = [plan_row("add", add), plan_row("delete", deleted, ident(rows[0])),
            plan_row("replace", new, ident(rows[1])),
            plan_row("delete", moved_old, ident(rows[2])), plan_row("add", moved_new)]
    forward(importer, tmp_path, client, {d["relative_path"]: d for d in (add, new, moved_new)}, plan)
    state = rollback(importer, tmp_path, client, plan)
    assert state["complete"] and state["verified_count"] == 3
    logical = sorted((r["custom_id"], r["container_tags"][0], r["content"], r["metadata"])
                     for r in client.documents.rows.values())
    expected = sorted((r["custom_id"], r["container_tags"][0], r["content"], r["metadata"]) for r in before)
    assert logical == expected
    journal = importer.private_json_read(tmp_path / "private", "journals/rollback.json")
    assert journal["forward_transaction_id"] == "forward-txn"
    assert journal["forward_plan_digest"] == importer.transaction_plan_digest(plan)
    assert journal["snapshot_digest"] == state["snapshot_digest"]
    assert len(journal["verification_digest"]) == 64


def test_restore_uses_snapshot_without_current_source_and_records_new_identity(importer, tmp_path):
    doc = importer.item("Skills/deleted.md", b"provider logical")
    row = remote(importer, doc, "old")
    client = Client([row]); plan = [plan_row("delete", doc, ident(row))]
    forward(importer, tmp_path, client, {}, plan)
    state = rollback(importer, tmp_path, client, plan)
    assert state["records"][0]["restored_document_id"].startswith("restored-")
    assert state["records"][0]["restored_document_id"] != "old"


@pytest.mark.parametrize("operation,mutates", [("add", True), ("delete", True), ("delete", False)])
def test_lost_mutation_responses_resume_from_inventory_truth(importer, tmp_path, operation, mutates):
    doc = importer.item("Skills/a.md", b"body")
    if operation == "add":
        row = remote(importer, doc, "old"); client = Client([row]); plan = [plan_row("delete", doc, ident(row))]
    else:
        client = Client(); plan = [plan_row("add", doc)]
    forward(importer, tmp_path, client, {} if operation == "add" else {doc["relative_path"]: doc}, plan)
    if operation == "add": client.documents.add_fault = LostResponse("lost")
    else:
        client.documents.delete_fault = LostResponse("lost"); client.documents.delete_mutates = mutates
    with pytest.raises(importer.ReconciliationRequired): rollback(importer, tmp_path, client, plan)
    client.documents.delete_mutates = True
    state = rollback(importer, tmp_path, client, plan)
    assert state["complete"]
    if operation == "add": assert len([c for c in client.documents.calls if c[0] == "add"]) == 1
    elif mutates: assert len([c for c in client.documents.calls if c[0] == "delete"]) == 1
    else: assert len([c for c in client.documents.calls if c[0] == "delete"]) == 2


def test_wrong_current_replacement_refuses_without_mutation(importer, tmp_path):
    old = importer.item("Skills/a.md", b"old"); new = importer.item("Skills/a.md", b"new")
    row = remote(importer, old, "old"); client = Client([row]); plan = [plan_row("replace", new, ident(row))]
    forward(importer, tmp_path, client, {new["relative_path"]: new}, plan)
    replacement = next(iter(client.documents.rows.values())); replacement["content"] = "wrong"
    before = list(client.documents.calls)
    with pytest.raises(importer.ReconciliationRequired): rollback(importer, tmp_path, client, plan)
    assert not [c for c in client.documents.calls[len(before):] if c[0] in {"add", "delete"}]


class DumpResponse:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def model_dump(self):
        if self.error:
            raise self.error
        return self.value


@pytest.mark.parametrize("action", ["remove", "restore"])
@pytest.mark.parametrize("phase", ["initial", "resume"])
@pytest.mark.parametrize("response", [
    "secret response private-id", ["secret response", "private-id"], None,
    DumpResponse(error=RuntimeError("secret SDK error private-id")),
    DumpResponse("secret non-dict private-id"),
    DumpResponse({"id": "private-id", "content": "secret malformed"}),
])
def test_malformed_rollback_hydration_is_sanitized_durable_and_retryable(
        importer, tmp_path, action, phase, response):
    doc = importer.item("Skills/a.md", b"body")
    if action == "remove":
        client = Client(); plan = [plan_row("add", doc)]; current = {doc["relative_path"]: doc}
    else:
        old = remote(importer, doc, "old")
        client = Client([old]); plan = [plan_row("delete", doc, ident(old))]; current = {}
    forward(importer, tmp_path, client, current, plan)
    if phase == "resume":
        if action == "restore":
            client.documents.add_fault = LostResponse("lost")
            with pytest.raises(importer.ReconciliationRequired):
                rollback(importer, tmp_path, client, plan)
        else:
            def stop(point):
                if point == "before_get":
                    raise RuntimeError("stop")
            with pytest.raises(RuntimeError, match="stop"):
                rollback(importer, tmp_path, client, plan, fault_injector=stop)
    before_mutations = len([call for call in client.documents.calls if call[0] in {"add", "delete"}])
    client.documents.get_response = response
    with pytest.raises(importer.ReconciliationRequired) as caught:
        rollback(importer, tmp_path, client, plan)
    assert str(caught.value) == "rollback hydration failed"
    assert all(text not in str(caught.value) for text in ("secret", "private-id", "SDK error"))
    journal = importer.private_json_read(tmp_path / "private", "journals/rollback.json")
    assert journal["records"][0]["stage"] == "reconcile_required"
    assert journal["records"][0]["history"][-1] == "reconcile_required"
    expected_new_mutations = int(action == "restore" and phase == "initial")
    assert (len([call for call in client.documents.calls if call[0] in {"add", "delete"}])
            == before_mutations + expected_new_mutations)
    client.documents.get_response = NO_RESPONSE
    assert rollback(importer, tmp_path, client, plan)["complete"]


def test_tampered_snapshot_and_forged_or_reordered_journal_fail_closed(importer, tmp_path):
    doc = importer.item("Skills/a.md", b"body"); client = Client(); plan = [plan_row("add", doc)]
    forward(importer, tmp_path, client, {doc["relative_path"]: doc}, plan)
    snapshot = importer.private_json_read(tmp_path / "private", "snapshots/forward.json")
    snapshot["expected_pre_inventory"] = [{"forged": True}]
    importer.private_json_write(tmp_path / "private", "snapshots/forward.json", snapshot)
    with pytest.raises(importer.PrivateArtifactError): rollback(importer, tmp_path, client, plan)


def test_tampered_reordered_or_stale_rollback_journal_is_rejected(importer, tmp_path):
    docs = [importer.item(f"Skills/{name}.md", name.encode()) for name in ("a", "b")]
    client = Client(); plan = [plan_row("add", doc) for doc in docs]
    forward(importer, tmp_path, client, {doc["relative_path"]: doc for doc in docs}, plan)
    fired = False
    def stop(point):
        nonlocal fired
        if point == "before_delete" and not fired:
            fired = True
            raise RuntimeError("stop")
    with pytest.raises(RuntimeError): rollback(importer, tmp_path, client, plan, fault_injector=stop)
    root = tmp_path / "private"
    original = importer.private_json_read(root, "journals/rollback.json")
    for mutation in ("reordered", "stale", "forged"):
        journal = copy.deepcopy(original)
        if mutation == "reordered": journal["records"].reverse()
        elif mutation == "stale": journal["forward_transaction_id"] = "other-forward"
        else: journal["records"][0]["stage"] = "done"
        importer.private_json_write(root, "journals/rollback.json", journal)
        with pytest.raises(importer.TransactionJournalError): rollback(importer, tmp_path, client, plan)
    importer.private_json_write(root, "journals/rollback.json", original)


@pytest.mark.parametrize("mutation", [
    "identity_and_digest", "remove", "add", "reorder", "container", "hash", "bytes",
])
def test_self_consistent_forged_post_inventory_is_rejected_before_provider_access(
        importer, tmp_path, mutation):
    docs = [importer.item(f"Skills/{name}.md", name.encode()) for name in ("a", "b")]
    client = Client(); plan = [plan_row("add", doc) for doc in docs]
    forward(importer, tmp_path, client, {doc["relative_path"]: doc for doc in docs}, plan)
    fired = False
    def stop(point):
        nonlocal fired
        if point == "before_delete" and not fired:
            fired = True
            raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        rollback(importer, tmp_path, client, plan, fault_injector=stop)
    root = tmp_path / "private"
    journal = importer.private_json_read(root, "journals/rollback.json")
    rows = journal["expected_post_inventory"]
    if mutation == "identity_and_digest": rows[0]["custom_id"] = "forged"
    elif mutation == "remove": rows.pop()
    elif mutation == "add": rows.append(copy.deepcopy(rows[0]) | {"custom_id": "forged"})
    elif mutation == "reorder": rows.reverse()
    elif mutation == "container": rows[0]["container"] = "family_shared"
    elif mutation == "hash": rows[0]["sha256"] = "f" * 64
    else: rows[0]["bytes"] += 1
    journal["post_inventory_digest"] = importer._canonical_digest(rows)
    importer.private_json_write(root, "journals/rollback.json", journal)
    before = list(client.documents.calls)
    with pytest.raises(importer.TransactionJournalError):
        rollback(importer, tmp_path, client, plan)
    assert client.documents.calls == before


@pytest.mark.parametrize("failure", ["timeout", "malformed"])
@pytest.mark.parametrize("phase,fault_kind", [
    ("create", "list"), ("create", "get"), ("resume", "list"),
    ("resume", "get"), ("final", "list"), ("final", "get"),
])
def test_rollback_provider_read_failures_are_safe_and_retryable(
        importer, tmp_path, phase, fault_kind, failure):
    doc = importer.item("Skills/a.md", b"body")
    client = Client(); plan = [plan_row("add", doc)]
    forward(importer, tmp_path, client, {doc["relative_path"]: doc}, plan)
    original_list = client.documents.list
    original_get = client.documents.get
    calls = {"list": 0, "get": 0}

    def failing_list(**kwargs):
        calls["list"] += 1
        if ((phase == "create" and calls["list"] == 1)
                or (phase == "resume" and calls["list"] == 1)
                or (phase == "final" and calls["list"] == 5)):
            if failure == "timeout":
                raise TimeoutError("https://secret.invalid/doc/private-id?token=secret")
            return {"memories": "private content", "pagination": {}}
        return original_list(**kwargs)

    def failing_get(ident, timeout):
        calls["get"] += 1
        target = 1 if phase in {"create", "resume"} else 3
        if calls["get"] == target:
            if failure == "timeout":
                raise RuntimeError("https://secret.invalid/doc/private-id?content=secret")
            return {"content": "private content", "id": "private-id"}
        return original_get(ident, timeout)

    if phase == "resume":
        fired = False
        def stop(point):
            nonlocal fired
            if point == "before_delete" and not fired:
                fired = True
                raise RuntimeError("stop")
        with pytest.raises(RuntimeError):
            rollback(importer, tmp_path, client, plan, fault_injector=stop)
    setattr(client.documents, fault_kind, failing_list if fault_kind == "list" else failing_get)
    before_mutations = len([call for call in client.documents.calls if call[0] in {"add", "delete"}])
    with pytest.raises(importer.ReconciliationRequired) as caught:
        rollback(importer, tmp_path, client, plan)
    assert "secret" not in str(caught.value) and "private-id" not in str(caught.value)
    setattr(client.documents, fault_kind, original_list if fault_kind == "list" else original_get)
    assert rollback(importer, tmp_path, client, plan)["complete"]
    mutation_calls = [call for call in client.documents.calls if call[0] in {"add", "delete"}]
    assert len(mutation_calls) <= before_mutations + 1


@pytest.mark.parametrize("point", ["before_rollback_journal_write", "after_rollback_journal_write",
    "before_inventory", "after_inventory", "before_get", "after_get", "before_delete", "after_delete",
    "before_add", "after_add"])
def test_every_rollback_boundary_crash_resumes(importer, tmp_path, point):
    doc = importer.item("Skills/a.md", b"body")
    if point in {"before_add", "after_add"}:
        old = remote(importer, doc, "old"); client = Client([old])
        plan = [plan_row("delete", doc, ident(old))]; current = {}
    else:
        client = Client(); plan = [plan_row("add", doc)]; current = {doc["relative_path"]: doc}
    forward(importer, tmp_path, client, current, plan)
    fired = False
    def fault(actual):
        nonlocal fired
        if actual == point and not fired: fired = True; raise RuntimeError("crash")
    try: rollback(importer, tmp_path, client, plan, fault_injector=fault)
    except (RuntimeError, importer.ReconciliationRequired): pass
    assert fired
    assert rollback(importer, tmp_path, client, plan)["complete"]
