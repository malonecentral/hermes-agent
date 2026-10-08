from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("owner_reset_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    module.configure_destinations("owner_canonical", "owner_explicit")
    return module


def owner_docs(importer):
    return {doc["relative_path"]: doc for doc in (
        importer.item("Skills/a.md", b"a"), importer.item("Skills/b.md", b"bb"),
    )}


def remote(importer, doc, ident):
    return {"id": ident, "custom_id": doc["custom_id"], "status": "done",
            "task_type": "superrag", "content": doc["content"],
            "metadata": importer.metadata(doc), "container_tags": ["owner_canonical"]}


class Documents:
    def __init__(self, importer, rows, private_root):
        self.importer = importer
        self.rows = {row["id"]: copy.deepcopy(row) for row in rows}
        self.private_root = private_root
        self.calls = []

    def list(self, **kwargs):
        assert kwargs["container_tags"] == ["owner_canonical"]
        self.calls.append(("list", kwargs["container_tags"][0]))
        rows = [copy.deepcopy(row) for row in self.rows.values()]
        return {"memories": rows, "pagination": {"current_page": 1,
                "total_pages": int(bool(rows)), "total_items": len(rows)}}

    def get(self, ident, **kwargs):
        self.calls.append(("get", ident))
        return copy.deepcopy(self.rows[ident])

    def delete(self, ident, **kwargs):
        journals = list((self.private_root / "owner-canonical-resets").glob("*.json"))
        assert len(journals) == 1
        journal = json.loads(journals[0].read_text())
        assert journal["records"] and any(
            record["snapshot"]["id"] == ident and record["stage"] == "delete_intent"
            for record in journal["records"])
        self.calls.append(("delete", ident))
        del self.rows[ident]


class Client:
    def __init__(self, importer, rows, private_root):
        self.documents = Documents(importer, rows, private_root)


def preview(importer, client, docs, private_root):
    return importer.reset_owner_canonical(
        client, docs, private_root, base_url="https://example.invalid")


def test_preview_is_read_only_owner_only_and_token_hides_provider_ids(importer, tmp_path):
    docs = owner_docs(importer)
    rows = [remote(importer, doc, f"private-provider-id-{index}")
            for index, doc in enumerate(docs.values())]
    client = Client(importer, rows, tmp_path / "private")

    result = preview(importer, client, docs, tmp_path / "private")

    assert result["inventory_count"] == len(rows)
    assert result["source_count"] == len(docs)
    assert result["filesystem_mutated"] is result["backend_mutated"] is False
    assert set(result) == {"container", "inventory_count", "source_count", "source_fingerprint",
                           "confirmation_token", "readiness", "filesystem_mutated",
                           "backend_mutated"}
    assert not any(row["id"] in json.dumps(result) for row in rows)
    assert not (tmp_path / "private").exists()
    assert {call[1] for call in client.documents.calls if call[0] == "list"} == {"owner_canonical"}


def test_execute_revalidates_source_and_inventory_before_delete(importer, tmp_path):
    docs = owner_docs(importer)
    rows = [remote(importer, doc, f"id-{index}") for index, doc in enumerate(docs.values())]
    private_root = tmp_path / "private"
    client = Client(importer, rows, private_root)
    token = preview(importer, client, docs, private_root)["confirmation_token"]
    changed = dict(docs)
    changed.pop(next(iter(changed)))

    with pytest.raises(importer.ReconciliationRequired, match="source changed"):
        importer.reset_owner_canonical(
            client, docs, private_root, base_url="https://example.invalid", execute=True,
            confirm=token, rescan=lambda: changed)
    assert not any(call[0] == "delete" for call in client.documents.calls)

    client.documents.rows["unexpected"] = copy.deepcopy(rows[0])
    client.documents.rows["unexpected"]["id"] = "unexpected"
    with pytest.raises(importer.ReconciliationRequired):
        importer.reset_owner_canonical(
            client, docs, private_root, base_url="https://example.invalid", execute=True,
            confirm=token, rescan=lambda: docs)
    assert not any(call[0] == "delete" for call in client.documents.calls)


@pytest.mark.parametrize("mutation", ["container", "metadata", "unknown"])
def test_unknown_ambiguous_or_noncanonical_rows_fail_closed(importer, tmp_path, mutation):
    docs = owner_docs(importer)
    row = remote(importer, next(iter(docs.values())), "do-not-delete")
    if mutation == "container":
        row["container_tags"] = ["owner_primary"]
    elif mutation == "metadata":
        row["metadata"]["identity_scope"] = "family"
    else:
        row["future_provider_field"] = True
    client = Client(importer, [row], tmp_path / "private")

    with pytest.raises(importer.ReconciliationRequired):
        preview(importer, client, docs, tmp_path / "private")
    assert not any(call[0] == "delete" for call in client.documents.calls)


def test_execute_journals_before_delete_and_verifies_exact_empty_inventory(importer, tmp_path):
    docs = owner_docs(importer)
    rows = [remote(importer, doc, f"id-{index}") for index, doc in enumerate(docs.values())]
    private_root = tmp_path / "private"
    client = Client(importer, rows, private_root)
    token = preview(importer, client, docs, private_root)["confirmation_token"]

    result = importer.reset_owner_canonical(
        client, docs, private_root, base_url="https://example.invalid", execute=True,
        confirm=token, rescan=lambda: docs)

    assert result["verified_empty"] is True
    assert result["deleted_count"] == len(rows)
    assert client.documents.rows == {}
    journal = importer.private_json_read(
        private_root, f"owner-canonical-resets/{result['journal_id']}.json")
    assert journal["complete"] is journal["verified_empty"] is True
    assert {record["stage"] for record in journal["records"]} == {"verified_absent"}
    assert client.documents.calls[-1] == ("list", "owner_canonical")


def test_reset_cli_masks_provider_errors_and_identifiers(importer, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "reset-owner-canonical"])

    def fail(_argv):
        raise RuntimeError("provider failed for secret-id-123")

    monkeypatch.setattr(importer, "_phase2_main", fail)
    with pytest.raises(SystemExit) as caught:
        importer.main()
    captured = capsys.readouterr()
    assert "secret-id-123" not in str(caught.value)
    assert "secret-id-123" not in captured.out + captured.err
    assert str(caught.value) == (
        "Owner canonical reset stopped; preview again before retrying. "
        "If execution began, inspect the private reset journals.")
