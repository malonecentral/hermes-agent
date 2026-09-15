from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("owner_rebuild_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    module.configure_destinations("owner_canonical", "owner_explicit")
    return module


class Documents:
    def __init__(self, importer, rows=(), *, fail_add=False):
        self.importer = importer
        self.rows = {row["id"]: copy.deepcopy(row) for row in rows}
        self.calls = []
        self.fail_add = fail_add

    def list(self, **kwargs):
        container = kwargs["container_tags"][0]
        page = kwargs["page"]
        self.calls.append(("list", container, page))
        rows = [copy.deepcopy(row) for row in self.rows.values()
                if row["container_tags"] == [container]]
        pages = [rows[index:index + 2] for index in range(0, len(rows), 2)]
        if not pages:
            return {"memories": [], "pagination": {"current_page": 1, "total_pages": 0,
                                                     "total_items": 0}}
        return {"memories": pages[page - 1],
                "pagination": {"current_page": page, "total_pages": len(pages),
                               "total_items": len(rows)}}

    def add(self, **kwargs):
        self.calls.append(("add", kwargs["container_tag"], kwargs["custom_id"]))
        if self.fail_add:
            raise RuntimeError("add failed")
        ident = f"new-{len(self.rows) + 1}"
        self.rows[ident] = {
            "id": ident, "custom_id": kwargs["custom_id"], "status": "done",
            "task_type": kwargs["task_type"], "content": kwargs["content"],
            "metadata": copy.deepcopy(kwargs["metadata"]),
            "container_tags": [kwargs["container_tag"]],
        }
        return {"id": ident, "status": "queued"}

    def get(self, ident, timeout):
        self.calls.append(("get", ident))
        return copy.deepcopy(self.rows[ident])

    def delete(self, *args, **kwargs):
        raise AssertionError("delete must not be called")


class Client:
    def __init__(self, importer, rows=(), **kwargs):
        self.documents = Documents(importer, rows, **kwargs)


def owner_docs(importer):
    return {doc["relative_path"]: doc for doc in (
        importer.item("Skills/a.md", b"a"),
        importer.item("Skills/b.md", b"bb"),
        importer.item("Skills/c.md", b"ccc"),
    )}


def test_command_defaults_to_isolated_owner_canonical_destination(importer, tmp_path):
    args = importer._phase2_parser().parse_args([
        "rebuild-owner-canonical", "--source-root", str(tmp_path),
        "--private-root", str(tmp_path / "private"), "--expected-source-count", "136",
        "--expected-source-fingerprint", "0" * 64, "--confirm", "token",
    ])
    assert args.rebuild_owner_canonical_container == "owner_canonical"
    assert args.rebuild_owner_explicit_container == "owner_explicit"


def remote(importer, doc, ident):
    return {"id": ident, "custom_id": doc["custom_id"], "status": "done",
            "task_type": "superrag", "content": doc["content"],
            "metadata": importer.metadata(doc),
            "container_tags": [importer.CONTAINER]}


def invoke(importer, tmp_path, client, docs, **overrides):
    fingerprint = importer.source_fingerprint_from_documents(
        docs, owner_canonical_container=importer.CONTAINER)
    values = {"expected_count": len(docs), "expected_fingerprint": fingerprint["digest"],
              "confirm": f"REBUILD-OWNER-CANONICAL:{len(docs)}:{fingerprint['digest']}"}
    values.update(overrides)
    return importer.rebuild_owner_canonical(client, docs, tmp_path / "private",
                                            sleep=lambda _: None, poll_attempts=2, **values)


def test_empty_destination_adds_polls_verifies_and_writes_owner_only_proof(importer, tmp_path):
    docs = owner_docs(importer)
    client = Client(importer)
    result = invoke(importer, tmp_path, client, docs)

    assert result["readiness"] is True and result["verified_count"] == 3
    assert [call[:2] for call in client.documents.calls if call[0] == "list"] == [
        ("list", importer.CONTAINER), ("list", importer.CONTAINER), ("list", importer.CONTAINER)]
    assert not any(importer.FAMILY_CONTAINER in call for call in client.documents.calls)
    assert not any("owner_primary" in call for call in client.documents.calls)
    manifest = importer.private_json_read(tmp_path / "private", "readiness/owner-canonical-manifest.json")
    proof = importer.private_json_read(tmp_path / "private", "readiness/owner-canonical-readiness.json")
    assert manifest["documents"] == sorted(manifest["documents"], key=lambda row: row["relative_path"])
    assert proof["source_fingerprint"] == result["source_fingerprint"]
    assert not (tmp_path / "private/readiness/storage-reconciliation.json").exists()
    assert not (tmp_path / "private/readiness/vector-readiness.json").exists()


def test_nonempty_destination_refuses_before_mutation(importer, tmp_path):
    docs = owner_docs(importer)
    client = Client(importer, [remote(importer, next(iter(docs.values())), "existing")])
    with pytest.raises(importer.ReconciliationRequired, match="not empty"):
        invoke(importer, tmp_path, client, docs)
    assert not any(call[0] == "add" for call in client.documents.calls)


@pytest.mark.parametrize("field,value", [("expected_count", 2), ("expected_fingerprint", "0" * 64),
                                           ("confirm", "wrong")])
def test_source_count_fingerprint_and_confirmation_drift_refuse(importer, tmp_path, field, value):
    docs = owner_docs(importer)
    client = Client(importer)
    with pytest.raises((SystemExit, importer.ReconciliationRequired)):
        invoke(importer, tmp_path, client, docs, **{field: value})
    assert client.documents.calls == []


def test_add_failure_reports_disposable_reset_without_other_container_calls(importer, tmp_path):
    docs = owner_docs(importer)
    client = Client(importer, fail_add=True)
    with pytest.raises(importer.OwnerCanonicalRebuildFailed, match="reset-owner-canonical"):
        invoke(importer, tmp_path, client, docs)
    assert not any(importer.FAMILY_CONTAINER in call for call in client.documents.calls)
    assert not any(call[0] == "delete" for call in client.documents.calls)


def test_poll_embedded_id_must_exactly_match_add_returned_id(importer, tmp_path):
    docs = owner_docs(importer)
    client = Client(importer)
    original_get = client.documents.get

    def mismatched_get(ident, timeout):
        row = original_get(ident, timeout)
        row["id"] = f"wrong-{ident}"
        return row

    client.documents.get = mismatched_get
    with pytest.raises(importer.OwnerCanonicalRebuildFailed) as failure:
        invoke(importer, tmp_path, client, docs)
    assert "mismatched document id" in str(failure.value.__cause__)
    assert not (tmp_path / "private/readiness/owner-canonical-readiness.json").exists()


def test_inventory_id_must_exactly_match_final_hydrated_id(importer, tmp_path):
    docs = owner_docs(importer)
    client = Client(importer)
    original_get = client.documents.get
    get_count = 0

    def mismatch_only_after_all_add_polls(ident, timeout):
        nonlocal get_count
        get_count += 1
        row = original_get(ident, timeout)
        if get_count > len(docs):
            row["id"] = f"wrong-{ident}"
        return row

    client.documents.get = mismatch_only_after_all_add_polls
    with pytest.raises(importer.OwnerCanonicalRebuildFailed) as failure:
        invoke(importer, tmp_path, client, docs)
    assert "mismatched document id" in str(failure.value.__cause__)
    assert not (tmp_path / "private/readiness/owner-canonical-readiness.json").exists()
