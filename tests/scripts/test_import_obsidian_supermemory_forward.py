from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
from supermemory.types.document_get_response import DocumentGetResponse

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("forward_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class LostResponse(RuntimeError):
    pass


class FakeDocuments:
    """Production-shaped, stateful in-memory documents provider."""

    def __init__(self, importer, rows=()):
        self.i = importer
        self.rows = {row["id"]: copy.deepcopy(row) for row in rows}
        self.calls = []
        self.next_id = 1
        self.add_fault = self.delete_fault = self.get_fault = None
        self.delete_mutates = True
        self.statuses = {}

    def list(self, **kwargs):
        container = kwargs["container_tags"][0]
        self.calls.append(("list", container))
        rows = [copy.deepcopy(r) for r in self.rows.values() if r["container_tags"] == [container]]
        return {"memories": rows, "pagination": {"current_page": 1, "total_pages": 1 if rows else 0,
                                                    "total_items": len(rows)}}

    def get(self, ident, timeout):
        self.calls.append(("get", ident))
        if self.get_fault:
            fault, self.get_fault = self.get_fault, None
            raise fault
        if ident not in self.rows:
            raise KeyError(ident)
        row = copy.deepcopy(self.rows[ident])
        sequence = self.statuses.get(ident)
        if sequence:
            value = sequence.pop(0)
            if isinstance(value, Exception):
                raise value
            if value == "malformed":
                row.pop("status", None)
            else:
                row["status"] = value
        return row

    def add(self, **kwargs):
        self.calls.append(("add", kwargs["custom_id"]))
        ident = f"new-{self.next_id}"
        self.next_id += 1
        row = {"id": ident, "custom_id": kwargs["custom_id"], "status": "done",
               "task_type": kwargs["task_type"],
               "content": kwargs["content"], "metadata": copy.deepcopy(kwargs["metadata"]),
               "container_tags": [kwargs["container_tag"]]}
        self.rows[ident] = row
        if self.add_fault:
            fault, self.add_fault = self.add_fault, None
            if isinstance(fault, tuple):
                self.statuses[ident] = list(fault)
            else:
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


class FakeClient:
    def __init__(self, importer, rows=()):
        self.documents = FakeDocuments(importer, rows)


def remote(importer, doc, ident, *, content=None, container=None, status="done", metadata=None):
    return {"id": ident, "custom_id": doc["custom_id"], "status": status,
            "task_type": "superrag",
            "content": doc["content"] if content is None else content,
            "metadata": copy.deepcopy(importer.metadata(doc) if metadata is None else metadata),
            "container_tags": [container or importer.container_for_doc(doc)]}


def sdk_get_response(importer, doc, ident):
    """Build the same SDK model returned by documents.get in supermemory 3.50.0."""
    row = remote(importer, doc, ident)
    return DocumentGetResponse.model_validate({
        **row,
        "customId": row["custom_id"],
        "taskType": row["task_type"],
        "containerTags": row["container_tags"],
        "createdAt": "2026-09-15T00:00:00Z",
        "updatedAt": "2026-09-15T00:00:00Z",
        "dreamingStatus": "done",
        "raw": None,
        "type": "text",
    })


def identity(importer, row):
    meta = row["metadata"]
    return {"custom_id": row["custom_id"], "container": row["container_tags"][0],
            "sha256": meta["content_sha256"], "bytes": meta["content_bytes"],
            "document_id": row["id"]}


def plan_row(action, doc, old=None, old_doc=None):
    container = None if action == "delete" else (
        "family_shared" if doc["visibility"] == "family_shared" else "owner_primary")
    post = None if action == "delete" else {
        "custom_id": doc["custom_id"], "container": container,
        "sha256": doc["sha256"], "bytes": doc["bytes"], "document_id": None,
    }
    expected = old if action == "delete" else post
    return {"action": action, "relative_path": doc["relative_path"],
            "custom_id": doc["custom_id"],
            "source_container": old["container"] if old else None,
            "target_container": container,
            "expected_sha256": expected["sha256"], "expected_bytes": expected["bytes"],
            "expected_pre_identity": old, "expected_post_identity": post}


def run(importer, tmp_path, docs, plan, client, **kwargs):
    root = tmp_path / "private"
    executor = importer.DurableReconciliationExecutor(client, root, sleep=lambda _: None,
                                                       poll_attempts=3, **kwargs)
    return executor.execute({d["relative_path"]: d for d in docs}, plan, "txn")


def test_mixed_add_delete_replace_scope_move_snapshots_before_mutation(importer, tmp_path):
    add_doc = importer.item("Skills/add.md", b"add")
    delete_doc = importer.item("Skills/delete.md", b"delete")
    old_replace = importer.item("Skills/replace.md", b"old")
    replace_doc = importer.item("Skills/replace.md", b"new")
    old_move = importer.item("Skills/move.md", b"move")
    move_doc = importer.item("Jarvis/Family Shared/move.md", b"move")
    old_rows = [remote(importer, delete_doc, "delete"), remote(importer, old_replace, "replace"),
                remote(importer, old_move, "move")]
    client = FakeClient(importer, old_rows)
    events = []
    original_delete, original_add = client.documents.delete, client.documents.add
    client.documents.delete = lambda *a, **k: (events.append("mutation"), original_delete(*a, **k))[1]
    client.documents.add = lambda **k: (events.append("mutation"), original_add(**k))[1]
    plan = [plan_row("add", add_doc),
            plan_row("delete", delete_doc, identity(importer, old_rows[0]), delete_doc),
            plan_row("replace", replace_doc, identity(importer, old_rows[1]), old_replace),
            # A scope move is represented by deleting the old stable identity and adding the new one.
            plan_row("delete", old_move, identity(importer, old_rows[2]), old_move),
            plan_row("add", move_doc)]
    state = run(importer, tmp_path, [add_doc, replace_doc, move_doc], plan, client,
                fault_injector=lambda point: events.append(point))
    assert state["complete"] and state["verified_count"] == 3
    assert events.index("snapshot_verified") < events.index("mutation")
    snapshot = importer.private_json_read(tmp_path / "private", "transactions/txn/snapshot.json",
                                          expected_sha256=state["snapshot_digest"])
    assert {r["backend_identity"] for r in snapshot["records"]} == {"delete", "replace", "move"}
    assert snapshot["count"] == 3


def test_content_accepts_only_narrow_terminal_whitespace_equivalence(importer, tmp_path):
    doc = importer.item("Skills/a.md", b"body")
    good = remote(importer, doc, "a", content=doc["content"] + " \t\r\n")
    client = FakeClient(importer, [good])
    state = run(importer, tmp_path, [doc], [], client)
    assert state["complete"]
    for index, bad in enumerate((" " + doc["content"], doc["content"].replace("body", "bo dy"),
                                 doc["content"] + "x")):
        root = tmp_path / f"bad-{index}"
        root.mkdir(mode=0o700)
        with pytest.raises(importer.ReconciliationRequired):
            run(importer, root, [doc], [], FakeClient(importer, [remote(importer, doc, "a", content=bad)]))


def test_installed_sdk_provider_object_normalizes_duplicate_sdk_aliases(importer, tmp_path):
    doc = importer.item("Jarvis/Family Shared/a.md", b"body")
    row = remote(importer, doc, "a")
    client = FakeClient(importer, [row])
    client.documents.get = lambda ident, timeout: sdk_get_response(importer, doc, ident)

    state = run(importer, tmp_path, [doc], [], client)

    assert state["complete"] and state["verified_count"] == 1


@pytest.mark.parametrize("invalid", [
    object(),
    type("BadDump", (), {"model_dump": lambda self: []})(),
    type("ExplodingDump", (), {"model_dump": lambda self: (_ for _ in ()).throw(ValueError())})(),
])
def test_provider_object_sanitizer_rejects_invalid_objects(importer, invalid):
    with pytest.raises(importer.ReconciliationRequired, match="provider object is malformed"):
        importer.normalize_provider_object(invalid)


@pytest.mark.parametrize("canonical,alias", [
    ("custom_id", "customId"),
    ("task_type", "taskType"),
    ("container_tags", "containerTags"),
    ("container_tag", "containerTag"),
])
@pytest.mark.parametrize("left,right", [
    ("canonical", "different"),
    (1, True),
    ([1], [True]),
    ({"nested": [1]}, {"nested": [True]}),
    ({1: "value"}, {True: "value"}),
])
def test_provider_object_sanitizer_rejects_conflicting_aliases(
        importer, canonical, alias, left, right):
    with pytest.raises(importer.ReconciliationRequired, match="ambiguous provider response aliases"):
        importer.normalize_provider_object({canonical: left, alias: right})


@pytest.mark.parametrize("canonical,alias,value", [
    ("custom_id", "customId", "shared-id"),
    ("task_type", "taskType", "superrag"),
    ("container_tags", "containerTags", ["family"]),
    ("container_tag", "containerTag", {"scope": ("family", 1)}),
])
def test_provider_object_sanitizer_collapses_exact_duplicate_aliases(
        importer, canonical, alias, value):
    assert importer.normalize_provider_object({canonical: value, alias: value}) == {canonical: value}


def test_immediate_delete_revalidation_rejects_stale_identity_without_mutation(importer, tmp_path):
    doc = importer.item("Skills/a.md", b"old")
    row = remote(importer, doc, "a")
    client = FakeClient(importer, [row])
    changed = False
    def fault(point):
        nonlocal changed
        if point == "journal_written" and not changed:
            client.documents.rows["a"]["metadata"]["content_sha256"] = "f" * 64
            changed = True
    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, [], [plan_row("delete", doc, identity(importer, row), doc)], client,
            fault_injector=fault)
    assert not [c for c in client.documents.calls if c[0] == "delete"]


@pytest.mark.parametrize("operation", ["add", "delete"])
def test_lost_provider_response_is_reconciled_without_duplicate_or_wrong_action(importer, tmp_path, operation):
    doc = importer.item("Skills/a.md", b"body")
    if operation == "add":
        client = FakeClient(importer)
        client.documents.add_fault = LostResponse("lost")
        state = run(importer, tmp_path, [doc], [plan_row("add", doc)], client)
        assert len([c for c in client.documents.calls if c[0] == "add"]) == 1
    else:
        row = remote(importer, doc, "old")
        client = FakeClient(importer, [row])
        client.documents.delete_fault = LostResponse("lost")
        state = run(importer, tmp_path, [], [plan_row("delete", doc, identity(importer, row), doc)], client)
        assert len([c for c in client.documents.calls if c[0] == "delete"]) == 1
        assert not [c for c in client.documents.calls if c[0] == "add"]
    assert state["complete"]


@pytest.mark.parametrize("outcome", ["duplicate", "wrong_container", "malformed"])
def test_add_identity_outcomes_fail_closed(importer, tmp_path, outcome):
    doc = importer.item("Skills/a.md", b"body")
    client = FakeClient(importer)
    original = client.documents.add
    def add(**kwargs):
        result = original(**kwargs)
        row = client.documents.rows[result["id"]]
        if outcome == "duplicate":
            duplicate = copy.deepcopy(row); duplicate["id"] = "duplicate"; client.documents.rows["duplicate"] = duplicate
        elif outcome == "wrong_container":
            row["container_tags"] = [importer.FAMILY_CONTAINER]
        else:
            row["metadata"] = "bad"
        return result
    client.documents.add = add
    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, [doc], [plan_row("add", doc)], client)


@pytest.mark.parametrize("sequence", [
    ("processing", "done"), (RuntimeError("poll"),), ("processing", "processing", "processing"),
    ("failed",), ("malformed",),
], ids=["processing-done", "exception", "timeout", "failed", "malformed"])
def test_poll_outcomes(importer, tmp_path, sequence):
    doc = importer.item("Skills/a.md", b"body")
    client = FakeClient(importer)
    client.documents.add_fault = sequence
    if sequence == ("processing", "done"):
        assert run(importer, tmp_path, [doc], [plan_row("add", doc)], client)["complete"]
    else:
        with pytest.raises(importer.ReconciliationRequired):
            run(importer, tmp_path, [doc], [plan_row("add", doc)], client)


@pytest.mark.parametrize("point", [
    "before_snapshot_write", "after_snapshot_write", "snapshot_verified",
    "before_journal_write", "after_journal_write", "journal_written",
    "before_inventory", "after_inventory", "before_get", "after_get",
    "before_add", "after_add", "before_delete", "after_delete",
])
def test_crash_at_every_artifact_and_provider_boundary_resumes_deterministically(importer, tmp_path, point):
    old = importer.item("Skills/old.md", b"old")
    new = importer.item("Skills/old.md", b"new")
    add = importer.item("Skills/add.md", b"add")
    row = remote(importer, old, "old")
    client = FakeClient(importer, [row])
    plan = [plan_row("replace", new, identity(importer, row), old), plan_row("add", add)]
    fired = False
    def fault(actual):
        nonlocal fired
        if actual == point and not fired:
            fired = True
            raise RuntimeError("crash")
    try:
        run(importer, tmp_path, [new, add], plan, client, fault_injector=fault)
    except (RuntimeError, importer.PrivateArtifactError, importer.ReconciliationRequired):
        pass
    assert fired, f"fault boundary was never reached: {point}"
    state = run(importer, tmp_path, [new, add], plan, client)
    assert state["complete"]
    assert len([c for c in client.documents.calls if c == ("add", new["custom_id"])]) == 1
    assert len([c for c in client.documents.calls if c == ("add", add["custom_id"])]) == 1


def test_artifact_write_fault_occurs_at_snapshot_boundary_not_root_setup(importer, tmp_path, monkeypatch):
    doc = importer.item("Skills/a.md", b"body")
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    original = importer.private_json_write
    calls = []
    def fail(root_arg, child, payload):
        calls.append(child)
        raise importer.PrivateArtifactError("injected")
    monkeypatch.setattr(importer, "private_json_write", fail)
    with pytest.raises(importer.PrivateArtifactError, match="injected"):
        importer.DurableReconciliationExecutor(FakeClient(importer), root).execute(
            {doc["relative_path"]: doc}, [plan_row("add", doc)], "txn")
    assert calls == ["transactions/txn/snapshot.json"]
    monkeypatch.setattr(importer, "private_json_write", original)


@pytest.mark.parametrize("corruption", [
    "orphan", "missing", "duplicate", "wrong_container", "stale_hash", "stale_bytes",
    "non_done", "malformed", "old_identity", "content",
])
def test_final_exhaustive_inventory_rejects_every_inconsistency(importer, tmp_path, corruption):
    doc = importer.item("Skills/a.md", b"body")
    row = remote(importer, doc, "a")
    client = FakeClient(importer, [row])
    if corruption == "orphan":
        orphan_doc = importer.item("Skills/orphan.md", b"x")
        client.documents.rows["orphan"] = remote(importer, orphan_doc, "orphan")
    elif corruption == "missing": client.documents.rows.clear()
    elif corruption == "duplicate":
        client.documents.rows["duplicate"] = remote(importer, doc, "duplicate")
    elif corruption == "wrong_container": client.documents.rows["a"]["container_tags"] = [importer.FAMILY_CONTAINER]
    elif corruption == "stale_hash": client.documents.rows["a"]["metadata"]["content_sha256"] = "f" * 64
    elif corruption == "stale_bytes": client.documents.rows["a"]["metadata"]["content_bytes"] += 1
    elif corruption == "non_done": client.documents.rows["a"]["status"] = "processing"
    elif corruption == "malformed": client.documents.rows["a"]["metadata"] = "bad"
    elif corruption == "old_identity": client.documents.rows["a"]["custom_id"] = "old"
    else: client.documents.rows["a"]["content"] += "x"
    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, [doc], [], client)


def test_final_journal_binds_proof_and_terminal_journal_alone_never_succeeds(importer, tmp_path):
    doc = importer.item("Skills/a.md", b"body")
    client = FakeClient(importer)
    state = run(importer, tmp_path, [doc], [plan_row("add", doc)], client)
    assert len(state["verification_digest"]) == 64
    assert state["verified_count"] == 1
    assert sum(state["verified_container_counts"].values()) == 1
    client.documents.rows.clear()
    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, [doc], [plan_row("add", doc)], client)


def test_partial_multi_record_resume_is_ordered_and_deterministic(importer, tmp_path):
    docs = [importer.item(f"Skills/{name}.md", name.encode()) for name in ("a", "b", "c")]
    client = FakeClient(importer)
    add_count = 0
    original = client.documents.add
    def add(**kwargs):
        nonlocal add_count
        add_count += 1
        if add_count == 2:
            raise RuntimeError("pre-call failure")
        return original(**kwargs)
    client.documents.add = add
    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, docs, [plan_row("add", d) for d in docs], client)
    client.documents.add = original
    # Explicit reconciliation-required is resumable by exhaustive identity lookup.
    state = run(importer, tmp_path, docs, [plan_row("add", d) for d in docs], client)
    assert state["complete"]
    adds = [c[1] for c in client.documents.calls if c[0] == "add"]
    assert adds == [d["custom_id"] for d in docs]


def test_executor_durably_visits_all_forward_transition_stages(importer, tmp_path):
    old = importer.item("Skills/a.md", b"old")
    new = importer.item("Skills/a.md", b"new")
    row = remote(importer, old, "old")
    client = FakeClient(importer, [row])
    original_add = client.documents.add

    def processing_add(**kwargs):
        result = original_add(**kwargs)
        client.documents.statuses[result["id"]] = ["processing", "done"]
        return result

    client.documents.add = processing_add
    seen = []

    def capture(point):
        if point == "journal_written":
            journal = importer.private_json_read(tmp_path / "private", "transactions/txn/forward-journal.json")
            seen.extend(record["stage"] for record in journal["records"])

    state = run(importer, tmp_path, [new],
                [plan_row("replace", new, identity(importer, row), old)], client,
                fault_injector=capture)
    assert state["complete"]
    assert {"existing", "delete_verified", "delete_submitted", "deleted",
            "add_submitted", "processing", "done"} <= set(seen)


@pytest.mark.parametrize("task_fields", [
    {}, {"task_type": "other"}, {"taskType": "other"},
    {"task_type": "superrag", "taskType": "other"},
])
def test_wrong_missing_or_conflicting_task_type_fails_before_mutation(
        importer, tmp_path, task_fields):
    doc = importer.item("Skills/task.md", b"old")
    row = remote(importer, doc, "task")
    row.pop("task_type")
    row.update(task_fields)
    client = FakeClient(importer, [row])
    with pytest.raises(importer.ReconciliationRequired, match="task type|aliases"):
        run(importer, tmp_path, [],
            [plan_row("delete", doc, identity(importer, row), doc)], client)
    assert not [call for call in client.documents.calls if call[0] == "delete"]


def test_camel_task_type_alias_is_accepted(importer, tmp_path):
    doc = importer.item("Skills/task.md", b"old")
    row = remote(importer, doc, "task")
    row["taskType"] = row.pop("task_type")
    state = run(importer, tmp_path, [],
                [plan_row("delete", doc, identity(importer, row), doc)],
                FakeClient(importer, [row]))
    assert state["complete"]


@pytest.mark.parametrize("checkpoint_stage", ["delete_verified", "delete_submitted"])
def test_identity_change_after_each_predelete_checkpoint_blocks_delete(
        importer, tmp_path, checkpoint_stage):
    doc = importer.item("Skills/checkpoint.md", b"old")
    row = remote(importer, doc, "checkpoint")
    client = FakeClient(importer, [row])
    changed = False

    def fault(point):
        nonlocal changed
        if point != "journal_written" or changed:
            return
        journal = importer.private_json_read(tmp_path / "private", "transactions/txn/forward-journal.json")
        if journal["records"][0]["stage"] == checkpoint_stage:
            client.documents.rows["checkpoint"]["metadata"]["content_sha256"] = "f" * 64
            changed = True

    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, [],
            [plan_row("delete", doc, identity(importer, row), doc)], client,
            fault_injector=fault)
    assert changed
    assert not [call for call in client.documents.calls if call[0] == "delete"]


def test_lost_delete_without_mutation_resumes_delete_from_reconcile_required(
        importer, tmp_path):
    doc = importer.item("Skills/resume.md", b"old")
    row = remote(importer, doc, "resume")
    client = FakeClient(importer, [row])
    client.documents.delete_mutates = False
    client.documents.delete_fault = LostResponse("lost-before-mutation")
    plan = [plan_row("delete", doc, identity(importer, row), doc)]
    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, [], plan, client)
    journal = importer.private_json_read(tmp_path / "private", "transactions/txn/forward-journal.json")
    assert journal["records"][0]["stage"] == "reconcile_required"
    client.documents.delete_mutates = True
    state = run(importer, tmp_path, [], plan, client)
    assert state["complete"]
    assert len([call for call in client.documents.calls if call[0] == "delete"]) == 2


def test_replace_never_adds_while_old_identity_is_ambiguous(importer, tmp_path):
    old = importer.item("Skills/replace-safe.md", b"old")
    new = importer.item("Skills/replace-safe.md", b"new")
    row = remote(importer, old, "old")
    client = FakeClient(importer, [row])
    duplicate = copy.deepcopy(row)
    duplicate["id"] = "duplicate"
    client.documents.rows["duplicate"] = duplicate
    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, [new],
            [plan_row("replace", new, identity(importer, row), old)], client)
    assert not [call for call in client.documents.calls if call[0] == "add"]


def test_forged_terminal_journal_cannot_succeed_without_provider_proof(
        importer, tmp_path):
    doc = importer.item("Skills/forged.md", b"new")
    client = FakeClient(importer)
    plan = [plan_row("add", doc)]
    fired = False

    def stop(point):
        nonlocal fired
        if point == "before_add" and not fired:
            fired = True
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        run(importer, tmp_path, [doc], plan, client, fault_injector=stop)
    journal = importer.private_json_read(tmp_path / "private", "transactions/txn/forward-journal.json")
    journal["records"][0]["stage"] = "done"
    journal["records"][0]["history"] = ["existing", "add_submitted", "done"]
    importer.private_json_write(tmp_path / "private", "transactions/txn/forward-journal.json", journal)
    before = list(client.documents.calls)
    with pytest.raises(importer.ReconciliationRequired):
        run(importer, tmp_path, [doc], plan, client)
    assert not [call for call in client.documents.calls[len(before):]
                if call[0] in {"add", "delete"}]
