from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("journal_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _identity(custom_id, container, sha256, byte_count, document_id):
    return {
        "custom_id": custom_id, "container": container, "sha256": sha256,
        "bytes": byte_count, "document_id": document_id,
    }


def _plans(importer):
    a = importer.item("a.md", b"a")
    b = importer.item("Skills/b.md", b"bb")
    c = importer.item("Jarvis/Family Shared/c.md", b"ccc")
    return (
        {
            "action": "add", "relative_path": a["relative_path"], "custom_id": a["custom_id"],
            "source_container": None, "target_container": importer.container_for_doc(a),
            "expected_sha256": a["sha256"], "expected_bytes": a["bytes"],
            "expected_pre_identity": None,
            "expected_post_identity": _identity(a["custom_id"], importer.container_for_doc(a), a["sha256"], a["bytes"], None),
        },
        {
            "action": "replace", "relative_path": b["relative_path"], "custom_id": b["custom_id"],
            "source_container": importer.CONTAINER, "target_container": importer.CONTAINER,
            "expected_sha256": b["sha256"], "expected_bytes": b["bytes"],
            "expected_pre_identity": _identity(b["custom_id"], importer.CONTAINER, "1" * 64, 1, "old-b"),
            "expected_post_identity": _identity(b["custom_id"], importer.CONTAINER, b["sha256"], b["bytes"], None),
        },
        {
            "action": "delete", "relative_path": c["relative_path"], "custom_id": c["custom_id"],
            "source_container": importer.FAMILY_CONTAINER, "target_container": None,
            "expected_sha256": c["sha256"], "expected_bytes": c["bytes"],
            "expected_pre_identity": _identity(c["custom_id"], importer.FAMILY_CONTAINER, c["sha256"], c["bytes"], "old-c"),
            "expected_post_identity": None,
        },
    )


def test_journal_is_immutable_and_bound_to_exact_ordered_plan(importer):
    plans = _plans(importer)
    journal = importer.new_transaction_journal("txn-001", plans, snapshot_digest="a" * 64)
    assert isinstance(journal, MappingProxyType)
    assert isinstance(journal["records"], tuple)
    assert journal["transaction_id"] == "txn-001"
    assert journal["plan_digest"] == importer.transaction_plan_digest(plans)
    assert tuple(record["order"] for record in journal["records"]) == (0, 1, 2)
    assert all(record["history"] == ("existing",) and record["stage"] == "existing" for record in journal["records"])
    importer.validate_transaction_journal(journal, plans, expected_transaction_id="txn-001", expected_snapshot_digest="a" * 64)
    with pytest.raises(TypeError):
        journal["transaction_id"] = "forged"


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "substituted"])
def test_journal_rejects_plan_record_set_or_order_drift(importer, mutation):
    plans = list(_plans(importer))
    journal = importer.new_transaction_journal("txn-001", plans)
    changed = copy.deepcopy(plans)
    if mutation == "missing":
        changed.pop()
    elif mutation == "extra":
        changed.append(copy.deepcopy(changed[-1]))
        changed[-1]["relative_path"] = "extra.md"
        changed[-1]["custom_id"] = importer.stable_custom_id_from_path("extra.md")
        changed[-1]["expected_pre_identity"]["custom_id"] = changed[-1]["custom_id"]
        changed[-1]["expected_pre_identity"]["document_id"] = "old-extra"
    elif mutation == "reordered":
        changed[0], changed[1] = changed[1], changed[0]
    else:
        changed[0]["expected_bytes"] += 1
        changed[0]["expected_post_identity"]["bytes"] += 1
    with pytest.raises(importer.TransactionJournalError, match="plan"):
        importer.validate_transaction_journal(journal, changed, expected_transaction_id="txn-001")


def test_journal_rejects_stale_transaction_plan_and_snapshot(importer):
    plans = _plans(importer)
    journal = importer.new_transaction_journal("txn-001", plans, snapshot_digest="a" * 64)
    with pytest.raises(importer.TransactionJournalError, match="transaction"):
        importer.validate_transaction_journal(journal, plans, expected_transaction_id="txn-002")
    with pytest.raises(importer.TransactionJournalError, match="snapshot"):
        importer.validate_transaction_journal(journal, plans, expected_transaction_id="txn-001", expected_snapshot_digest="b" * 64)


@pytest.mark.parametrize("action,progression", [
    ("add", ("add_submitted", "processing", "done")),
    ("replace", ("delete_verified", "delete_submitted", "deleted", "add_submitted", "processing", "done")),
    ("delete", ("delete_verified", "delete_submitted", "deleted", "done")),
])
def test_action_specific_progressions_return_new_valid_journals(importer, action, progression):
    plans = _plans(importer)
    index = {row["action"]: n for n, row in enumerate(plans)}[action]
    journal = importer.new_transaction_journal("txn-001", plans)
    original = journal
    for stage in progression:
        journal = importer.advance_transaction_journal(journal, plans, index, stage, expected_transaction_id="txn-001")
        importer.validate_transaction_journal(journal, plans, expected_transaction_id="txn-001")
    assert journal["records"][index]["history"] == ("existing",) + progression
    assert original["records"][index]["stage"] == "existing"


@pytest.mark.parametrize("action,stage", [
    ("add", "deleted"), ("add", "done"), ("replace", "add_submitted"),
    ("delete", "processing"), ("delete", "done"),
])
def test_journal_rejects_impossible_skips(importer, action, stage):
    plans = _plans(importer)
    index = {row["action"]: n for n, row in enumerate(plans)}[action]
    journal = importer.new_transaction_journal("txn-001", plans)
    with pytest.raises(importer.TransactionJournalError, match="transition"):
        importer.advance_transaction_journal(journal, plans, index, stage, expected_transaction_id="txn-001")


def test_journal_rejects_backtrack_unknown_stage_and_forged_done(importer):
    plans = _plans(importer)
    journal = importer.new_transaction_journal("txn-001", plans)
    journal = importer.advance_transaction_journal(journal, plans, 0, "add_submitted", expected_transaction_id="txn-001")
    with pytest.raises(importer.TransactionJournalError, match="transition"):
        importer.advance_transaction_journal(journal, plans, 0, "existing", expected_transaction_id="txn-001")
    with pytest.raises(importer.TransactionJournalError, match="stage"):
        importer.advance_transaction_journal(journal, plans, 0, "invented", expected_transaction_id="txn-001")
    forged = importer.thaw_transaction_journal(journal)
    forged["records"][0]["stage"] = "done"
    forged["records"][0]["history"] = ["existing", "done"]
    with pytest.raises(importer.TransactionJournalError, match="transition"):
        importer.validate_transaction_journal(forged, plans, expected_transaction_id="txn-001")


def test_failure_and_reconciliation_states_are_explicit(importer):
    plans = _plans(importer)
    journal = importer.new_transaction_journal("txn-001", plans)
    journal = importer.advance_transaction_journal(journal, plans, 1, "delete_verified", expected_transaction_id="txn-001")
    journal = importer.advance_transaction_journal(journal, plans, 1, "failed", expected_transaction_id="txn-001")
    journal = importer.advance_transaction_journal(journal, plans, 1, "reconcile_required", expected_transaction_id="txn-001")
    assert journal["records"][1]["stage"] == "reconcile_required"


@pytest.mark.parametrize("duplicate", ["path", "custom_id", "pre_document_id"])
def test_plan_rejects_duplicate_identities(importer, duplicate):
    plans = list(copy.deepcopy(_plans(importer)))
    if duplicate == "path":
        plans[1] = copy.deepcopy(plans[0])
    elif duplicate == "custom_id":
        plans[1] = copy.deepcopy(plans[0])
    else:
        plans[2]["expected_pre_identity"]["document_id"] = plans[1]["expected_pre_identity"]["document_id"]
    with pytest.raises(importer.TransactionJournalError, match="duplicate"):
        importer.new_transaction_journal("txn-001", plans)


def test_plan_rejects_action_identity_inconsistency_and_extra_fields(importer):
    plans = list(copy.deepcopy(_plans(importer)))
    plans[0]["unexpected"] = True
    with pytest.raises(importer.TransactionJournalError, match="fields"):
        importer.new_transaction_journal("txn-001", plans)
    plans = list(copy.deepcopy(_plans(importer)))
    plans[0]["expected_pre_identity"] = plans[1]["expected_pre_identity"]
    with pytest.raises(importer.TransactionJournalError, match="identity"):
        importer.new_transaction_journal("txn-001", plans)


@pytest.mark.parametrize("field,bad", [
    ("expected_bytes", True), ("expected_bytes", 1.0), ("expected_bytes", -1),
    ("expected_sha256", "A" * 64), ("expected_sha256", "a" * 63),
    ("expected_sha256", "g" * 64),
])
def test_plan_rejects_inexact_scalars_and_digests(importer, field, bad):
    plans = list(copy.deepcopy(_plans(importer)))
    plans[0][field] = bad
    if field == "expected_bytes":
        plans[0]["expected_post_identity"]["bytes"] = bad
    with pytest.raises(importer.TransactionJournalError):
        importer.new_transaction_journal("txn-001", plans)


@pytest.mark.parametrize("field,bad", [
    ("bytes", True), ("bytes", 1.0), ("bytes", -1),
    ("sha256", "A" * 64), ("sha256", "a" * 8), ("sha256", "z" * 64),
])
def test_identity_rejects_inexact_scalars_and_digests(importer, field, bad):
    plans = list(copy.deepcopy(_plans(importer)))
    plans[0]["expected_post_identity"][field] = bad
    with pytest.raises(importer.TransactionJournalError):
        importer.new_transaction_journal("txn-001", plans)


@pytest.mark.parametrize("transaction_id", [True, 1, "", " txn", "txn/escape", "x" * 129])
def test_transaction_id_is_constrained(importer, transaction_id):
    with pytest.raises(importer.TransactionJournalError, match="transaction id"):
        importer.new_transaction_journal(transaction_id, _plans(importer))


@pytest.mark.parametrize("field,bad", [
    ("journal_schema_version", True), ("plan_digest", "A" * 64),
    ("plan_digest", "a" * 63), ("plan_digest", "g" * 64),
    ("snapshot_digest", "A" * 64), ("snapshot_digest", 1),
])
def test_journal_rejects_inexact_header_scalars(importer, field, bad):
    plans = _plans(importer)
    journal = importer.thaw_transaction_journal(importer.new_transaction_journal("txn-001", plans))
    journal[field] = bad
    with pytest.raises(importer.TransactionJournalError):
        importer.validate_transaction_journal(journal, plans, expected_transaction_id="txn-001")


@pytest.mark.parametrize("bad", [True, 0.0, -1])
def test_journal_order_is_exact_nonnegative_integer(importer, bad):
    plans = _plans(importer)
    journal = importer.thaw_transaction_journal(importer.new_transaction_journal("txn-001", plans))
    journal["records"][0]["order"] = bad
    with pytest.raises(importer.TransactionJournalError, match="record mismatch"):
        importer.validate_transaction_journal(journal, plans, expected_transaction_id="txn-001")


@pytest.mark.parametrize("location", ["record", "nested_identity"])
def test_journal_rejects_bool_alias_for_integer(importer, location):
    plans = _plans(importer)
    journal = importer.thaw_transaction_journal(importer.new_transaction_journal("txn-001", plans))
    if location == "record":
        journal["records"][0]["expected_bytes"] = True
    else:
        journal["records"][0]["expected_post_identity"]["bytes"] = True
    with pytest.raises(importer.TransactionJournalError, match="record mismatch"):
        importer.validate_transaction_journal(journal, plans, expected_transaction_id="txn-001")


@pytest.mark.parametrize("digest", [True, 1, "A" * 64, "a" * 63, "g" * 64])
def test_caller_snapshot_digest_is_exact_lowercase_sha256(importer, digest):
    plans = _plans(importer)
    with pytest.raises(importer.TransactionJournalError, match="snapshot digest"):
        importer.new_transaction_journal("txn-001", plans, snapshot_digest=digest)
    journal = importer.new_transaction_journal("txn-001", plans)
    with pytest.raises(importer.TransactionJournalError, match="expected snapshot digest"):
        importer.validate_transaction_journal(
            journal, plans, expected_transaction_id="txn-001", expected_snapshot_digest=digest,
        )
