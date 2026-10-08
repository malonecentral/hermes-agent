from __future__ import annotations

import hashlib
import importlib.util
import itertools
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/import-obsidian-supermemory.py"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("pure_obsidian_importer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _row(importer, doc, *, ident="provider-row", status="done", metadata=None):
    return {
        "id": ident,
        "customId": doc["custom_id"],
        "status": status,
        "metadata": importer.metadata(doc) if metadata is None else metadata,
    }


def _orphan(importer, path="old.md", *, ident="old-provider-id", custom_id=None,
            container=None, metadata=None, **claims):
    doc = importer.item(path, b"old bytes")
    row = _row(importer, doc, ident=ident, metadata=metadata)
    row["customId"] = custom_id or doc["custom_id"]
    if container is not None:
        row["_inventory_container"] = container
    row.update(claims)
    return row


def test_projection_uses_shared_policy_exact_bytes_and_path_only_exclusions(importer):
    raw = b"body\r\n"
    doc = importer.item("Skills/Example.md", raw)
    sensitive = importer.item("Secret.md", b"api_key=redacted-fixture")
    template = importer.item("Skills/Note Template.md", b"template")
    projected = importer.project_canonical_documents([template, sensitive, doc])
    assert doc["sha256"] == hashlib.sha256(raw).hexdigest()
    assert doc["bytes"] == len(raw)
    assert doc["custom_id"] == importer.stable_custom_id_from_path("Skills/Example.md")
    assert tuple(projected["documents"]) == ("Skills/Example.md",)
    assert projected["excluded"] == (
        {"relative_path": "Secret.md", "reason": "sensitive_content"},
        {"relative_path": "Skills/Note Template.md", "reason": "template"},
    )
    assert "redacted-fixture" not in repr(projected["excluded"])


@pytest.mark.parametrize("path", ["People/Ada.md", "Other/x.md", "Jarvis/Family Shared/"])
def test_projection_rejects_unapproved_paths_before_content_projection(importer, path):
    with pytest.raises(ValueError, match="canonical path"):
        importer.item(path, b"private")


def test_inventory_exhausts_sdk_350_shapes_and_never_mutates(importer):
    doc = importer.item("x.md", b"x")
    calls = []

    class Documents:
        def list(self, **kwargs):
            calls.append((kwargs["container_tags"][0], kwargs["page"]))
            container = kwargs["container_tags"][0]
            if container == importer.CONTAINER:
                if kwargs["page"] == 1:
                    return {"memories": [_row(importer, doc)], "pagination": {
                        "currentPage": 1.0, "totalPages": 2.0, "totalItems": 2.0}}
                return {"memories": [{"id": "noncanonical", "customId": "other", "status": "done", "metadata": {}}],
                        "pagination": {"currentPage": 2.0, "totalPages": 2.0, "totalItems": 2.0}}
            return {"memories": [], "pagination": {
                "current_page": 1.0, "total_pages": 0.0, "total_items": 0.0}}

        def add(self, **kwargs):
            raise AssertionError("mutation called")

        def delete(self, *args, **kwargs):
            raise AssertionError("mutation called")

        def update(self, *args, **kwargs):
            raise AssertionError("mutation called")

    inventory = importer.provider_inventory(type("Client", (), {"documents": Documents()})())
    assert calls == [("family_shared", 1), ("owner_primary", 1), ("owner_primary", 2)]
    assert len(inventory["owner_primary"]) == 2


@pytest.mark.parametrize("pages", [
    {1: {"memories": [], "pagination": {"currentPage": 1.5, "totalPages": 1.5}}},
    {1: {"memories": [{}], "pagination": {"currentPage": 1, "totalPages": 0}}},
    {1: {"memories": [{}], "pagination": {"currentPage": 1, "totalPages": 2}},
     2: {"memories": [{}], "pagination": {"currentPage": 2, "totalPages": 3}}},
    {1: {"memories": [], "pagination": {"currentPage": 1, "totalPages": 2}}},
    {1: {"memories": [], "pagination": {"current_page": 1, "currentPage": 2,
                                           "total_page": 1, "totalPages": 1}}},
])
def test_inventory_rejects_unprovable_pagination(importer, pages):
    class Documents:
        def list(self, **kwargs):
            return pages[kwargs["page"]]
    with pytest.raises(importer.ReconciliationRequired, match="pagination|aliases"):
        importer.provider_inventory(type("Client", (), {"documents": Documents()})())


def _explicit_memory(*, custom_id=None, metadata=None, **fields):
    row = {
        "id": "explicit-provider-row",
        "createdAt": "2026-09-15T00:00:00.000Z",
        "custom_id": custom_id,
        "metadata": {"sm_source": "hermes", "target": "user", "type": "explicit_memory"}
        if metadata is None else metadata,
        "status": "done",
        "type": "text",
        "updatedAt": "2026-09-15T00:00:01.000Z",
        "containerTags": ["owner_primary"],
    }
    row.update(fields)
    return row


def test_present_null_sdk_custom_id_excludes_exact_production_three_and_keeps_two_adds(importer):
    from supermemory.types.document_list_response import Memory

    first = importer.item("Skills/Missing One.md", b"one")
    second = importer.item("Jarvis/Family Shared/Missing Two.md", b"two")
    current = {doc["relative_path"]: doc for doc in (first, second)}
    inventory = {
        "owner_primary": tuple(
            Memory.model_validate(
                _explicit_memory(custom_id=None, id=f"explicit-{index}")
            ).model_dump()
            for index in range(3)
        ),
        "family_shared": (),
    }

    classification = importer.classify_inventory(current, inventory)

    assert all("custom_id" in row and "customId" not in row
               for row in inventory["owner_primary"])

    assert classification["malformed"] == ()
    assert classification["untrusted_orphans"] == ()
    assert importer.reconciliation_plan(classification) == (
        {"action": "add", "relative_path": "Jarvis/Family Shared/Missing Two.md",
         "reason": "missing"},
        {"action": "add", "relative_path": "Skills/Missing One.md", "reason": "missing"},
    )


@pytest.mark.parametrize("custom_id", [
    "value", 0, 1, False, True, 0.0, 1.5, [], [None], {}, {"id": None}, (), (None,), object(),
], ids=lambda value: type(value).__name__ + "-" + repr(value))
def test_non_null_custom_id_types_remain_inventory_and_require_operator_review(
        importer, custom_id):
    classification = importer.classify_inventory({}, {
        "owner_primary": (_explicit_memory(custom_id=custom_id),), "family_shared": (),
    })
    assert classification["malformed"] == ("*",)
    assert importer.reconciliation_plan(classification) == (
        {"action": "operator_review", "relative_path": "*", "reason": "malformed"},
    )


@pytest.mark.parametrize("row", [
    {key: value for key, value in _explicit_memory().items() if key != "custom_id"},
    {key: value for key, value in _explicit_memory().items() if key != "custom_id"}
    | {"customId": None},
    _explicit_memory() | {"customId": None},
    _explicit_memory() | {"customId": "conflict"},
])
def test_missing_or_aliased_custom_id_remains_inventory_and_requires_operator_review(importer, row):
    classification = importer.classify_inventory({}, {
        "owner_primary": (row,), "family_shared": (),
    })
    assert classification["malformed"] == ("*",)
    assert importer.reconciliation_plan(classification) == (
        {"action": "operator_review", "relative_path": "*", "reason": "malformed"},
    )


@pytest.mark.parametrize("canonical_marker", [
    {"relative_path": "Skills/Forged.md"},
    {"canonical_path": "Skills/Forged.md"},
    {"index_schema_version": 4},
    {"source": "obsidian", "authority": "canonical"},
    {"schema_version": "4"},
    {"entity_type": "person"},
    {"relativePath": "Skills/Forged.md"},
    {"canonicalPath": "Skills/Forged.md"},
    {"identityScope": "owner"},
    {"canonicalRoot": "owner"},
    {"contentSha256": "0" * 64},
])
def test_explicit_memory_claim_cannot_hide_canonical_metadata(importer, canonical_marker):
    row = _explicit_memory(metadata={
        "sm_source": "hermes", "target": "user", "type": "explicit_memory",
        **canonical_marker,
    })
    classification = importer.classify_inventory({}, {
        "owner_primary": (row,), "family_shared": (),
    })
    assert classification["malformed"] == ("*",)
    assert importer.reconciliation_plan(classification) == (
        {"action": "operator_review", "relative_path": "*", "reason": "malformed"},
    )


@pytest.mark.parametrize("row", [
    _explicit_memory(type="pdf"),
    _explicit_memory(status="unknown"),
    _explicit_memory(containerTags=["unknown"]),
    _explicit_memory(mystery="unknown-provider-field"),
    _explicit_memory() | {"content": "unexpected"},
    _explicit_memory(metadata={"sm_source": "hermes", "target": "user",
                               "type": "explicit_memory", "mystery": True}),
    _explicit_memory(metadata={"sm_source": "hermes", "target": "user", "type": "unknown"}),
    {"id": "unknown", "customId": None, "metadata": None, "status": "done", "type": "text"},
])
def test_unknown_or_malformed_noncanonical_rows_remain_fail_closed(importer, row):
    classification = importer.classify_inventory({}, {
        "owner_primary": (row,), "family_shared": (),
    })
    assert classification["malformed"] == ("*",)
    assert importer.reconciliation_plan(classification)[0]["action"] == "operator_review"


def test_canonical_custom_id_alias_on_explicit_claim_remains_fail_closed(importer):
    row = _explicit_memory(customId="obsidian-" + "a" * 64)
    classification = importer.classify_inventory({}, {
        "owner_primary": (row,), "family_shared": (),
    })
    assert classification["malformed"] == ("*",)
    assert importer.reconciliation_plan(classification) == (
        {"action": "operator_review", "relative_path": "*", "reason": "malformed"},
    )


def test_explicit_memory_exclusion_is_owner_only_and_canonical_rows_are_unchanged(importer):
    owner = importer.item("Skills/Owner.md", b"owner")
    family = importer.item("Jarvis/Family Shared/Family.md", b"family")
    current = {doc["relative_path"]: doc for doc in (owner, family)}
    inventory = {
        "owner_primary": (_row(importer, owner, ident="owner-canonical"),),
        "family_shared": (
            _row(importer, family, ident="family-canonical"),
            _explicit_memory(id="family-forgery", containerTags=["family_shared"]),
        ),
    }
    classification = importer.classify_inventory(current, inventory)
    assert classification["expected"] == (
        "Jarvis/Family Shared/Family.md", "Skills/Owner.md",
    )
    assert classification["missing"] == ()
    assert classification["malformed"] == ("*",)
    assert importer.reconciliation_plan(classification) == (
        {"action": "operator_review", "relative_path": "*", "reason": "malformed"},
    )


def test_only_exact_production_explicit_memory_shapes_are_excluded(importer):
    camel_user = _explicit_memory(id="user")
    camel_memory = _explicit_memory(
        id="memory",
        metadata={"sm_source": "hermes", "target": "memory", "type": "explicit_memory"},
    )
    snake = _explicit_memory(id="snake")
    snake["container_tags"] = snake.pop("containerTags")
    snake["created_at"] = snake.pop("createdAt")
    snake["updated_at"] = snake.pop("updatedAt")
    classification = importer.classify_inventory({}, {
        "owner_primary": (camel_user, camel_memory, snake), "family_shared": (),
    })
    assert classification["malformed"] == ()
    assert importer.reconciliation_plan(classification) == ()


def test_mixed_explicit_and_canonical_signals_and_family_scope_fail_closed(importer):
    mixed = _explicit_memory(metadata={
        "sm_source": "hermes", "target": "user", "type": "explicit_memory",
        "schema_version": "4", "entity_type": "person",
        "canonical_path": "Jarvis/Family Shared/Person.md",
    })
    family = _explicit_memory(id="family", containerTags=["family_shared"])
    classification = importer.classify_inventory({}, {
        "owner_primary": (mixed,), "family_shared": (family,),
    })
    assert classification["malformed"] == ("*",)
    assert importer.reconciliation_plan(classification) == (
        {"action": "operator_review", "relative_path": "*", "reason": "malformed"},
    )


def test_classification_and_plan_cover_moves_staleness_duplicates_orphans_and_status(importer):
    clean = importer.item("clean.md", b"clean")
    missing = importer.item("missing.md", b"missing")
    moved = importer.item("Jarvis/Family Shared/Moved.md", b"moved")
    stale = importer.item("Skills/Stale.md", b"stale")
    pending = importer.item("pending.md", b"pending")
    failed = importer.item("failed.md", b"failed")
    current = {doc["relative_path"]: doc for doc in (clean, missing, moved, stale, pending, failed)}
    stale_metadata = importer.metadata(stale) | {"content_sha256": "stale", "content_bytes": 999}
    orphan = _orphan(importer, "deleted.md", ident="orphan-row")
    inventory = {
        "family_shared": (),
        "owner_primary": (
            _row(importer, clean, ident="clean"), _row(importer, clean, ident="duplicate"),
            _row(importer, moved, ident="moved"),
            _row(importer, stale, ident="stale", metadata=stale_metadata),
            _row(importer, pending, ident="pending", status="indexing"),
            _row(importer, failed, ident="failed", status="failed"),
            orphan, {"customId": "obsidian-" + "e" * 64, "status": "done", "metadata": {}},
        ),
    }
    classified = importer.classify_inventory(current, inventory)
    assert classified["missing"] == ("missing.md",)
    assert classified["duplicate_expected_custom_ids"] == ()
    assert classified["identity_collision"] == ("*",)
    assert classified["wrong_container"] == ("Jarvis/Family Shared/Moved.md",)
    assert classified["stale_hash"] == classified["stale_byte_count"] == ("Skills/Stale.md",)
    assert classified["pending"] == ("pending.md",)
    assert classified["failed"] == ("failed.md",)
    assert classified["trusted_orphans"] == ("deleted.md",)
    assert classified["untrusted_orphans"] == ("*",)
    assert classified["malformed"] == ("*",)
    plan = importer.reconciliation_plan(classified)
    assert {row["action"] for row in plan} == {"add", "replace", "delete", "operator_review"}
    assert list(plan) == sorted(plan, key=lambda row: (row["relative_path"], row["action"], row["reason"]))
    assert all(set(row) == {"action", "relative_path", "reason"}
               and "provider-row" not in repr(row) for row in plan)


def test_rename_and_scope_move_are_old_delete_plus_new_insert(importer):
    old_path = "old.md"
    new_doc = importer.item("Jarvis/Family Shared/new.md", b"same bytes")
    inventory = {
        "family_shared": (),
        "owner_primary": (_orphan(importer, old_path),),
    }
    classification = importer.classify_inventory({new_doc["relative_path"]: new_doc}, inventory)
    assert importer.reconciliation_plan(classification) == (
        {"action": "add", "relative_path": "Jarvis/Family Shared/new.md", "reason": "missing"},
        {"action": "delete", "relative_path": "old.md", "reason": "trusted_orphan"},
    )


@pytest.mark.parametrize("make_row,source_container", [
    (lambda i: _orphan(i, "old.md", custom_id="obsidian-" + "a" * 64,
                       metadata={"relative_path": "../private.md"}), "owner_primary"),
    (lambda i: _orphan(i, "old.md", custom_id="obsidian-" + "a" * 64), "owner_primary"),
    (lambda i: _orphan(i, "old.md", container="family_shared"), "owner_primary"),
    (lambda i: _orphan(i, "old.md", containerTags=["family_shared"]), "owner_primary"),
])
def test_untrusted_orphan_identity_never_emits_delete(importer, make_row, source_container):
    classification = importer.classify_inventory({}, {
        "family_shared": (), "owner_primary": (), source_container: (make_row(importer),),
    })
    plan = importer.reconciliation_plan(classification)
    assert classification["trusted_orphans"] == ()
    assert classification["untrusted_orphans"] == ("*",)
    assert not [entry for entry in plan if entry["action"] == "delete"]
    assert {entry["relative_path"] for entry in plan} == {"*"}


def test_wrong_root_orphan_never_emits_delete_or_discloses_path(importer):
    row = _orphan(importer, "old.md")
    row["metadata"] = row["metadata"] | {"relative_path": "Other/private.md",
                                          "canonical_path": "Other/private.md"}
    classification = importer.classify_inventory({}, {
        "owner_primary": (row,), "family_shared": (),
    })
    plan = importer.reconciliation_plan(classification)
    assert classification["trusted_orphans"] == ()
    assert all("Other/private.md" not in repr(value) for value in (classification, plan))
    assert all(entry["action"] != "delete" for entry in plan)


@pytest.mark.parametrize("cross_container", [False, True])
def test_duplicate_orphan_same_or_cross_container_never_emits_delete(importer, cross_container):
    first = _orphan(importer, "old.md", ident="first")
    second = _orphan(importer, "old.md", ident="second")
    inventory = {"owner_primary": (first, second), "family_shared": ()}
    if cross_container:
        shared_path = "Jarvis/Family Shared/old.md"
        first = _orphan(importer, shared_path, ident="first")
        second = _orphan(importer, shared_path, ident="second")
        inventory = {"owner_primary": (first,), "family_shared": (second,)}
    classification = importer.classify_inventory({}, inventory)
    assert classification["trusted_orphans"] == ()
    assert classification["untrusted_orphans"] == ("*",)
    assert all(entry["action"] != "delete" for entry in importer.reconciliation_plan(classification))


def test_ambiguous_aliases_and_untrusted_paths_are_privacy_safe(importer):
    ambiguous = _orphan(importer, "Skills/secret.md")
    ambiguous["custom_id"] = "obsidian-" + "c" * 64
    traversal = _orphan(importer, "old.md")
    traversal["metadata"] = traversal["metadata"] | {"relative_path": "../secret.md"}
    classification = importer.classify_inventory({}, {
        "owner_primary": (ambiguous, traversal), "family_shared": (),
    })
    plan = importer.reconciliation_plan(classification)
    assert all("secret" not in repr(value) for value in (classification, plan))
    assert plan == ({"action": "operator_review", "relative_path": "*",
                     "reason": "identity_collision,malformed,untrusted_orphans"},)


def _assert_collision_review_only(importer, rows_by_container):
    classification = importer.classify_inventory({}, rows_by_container)
    plan = importer.reconciliation_plan(classification)
    assert classification["trusted_orphans"] == ()
    assert classification["identity_collision"] == ("*",)
    assert plan
    assert {entry["action"] for entry in plan} == {"operator_review"}
    assert {entry["relative_path"] for entry in plan} == {"*"}
    assert "identity_collision" in plan[0]["reason"]
    return classification, plan


@pytest.mark.parametrize("case", [
    "same_path_different_custom_id",
    "same_custom_id_different_path",
    "same_backend_id_different_path",
    "malformed_peer_sharing_id",
    "alias_conflict_peer",
    "same_container_repeat",
    "cross_container_repeat",
])
def test_global_raw_identity_collisions_never_emit_execution_plans(importer, case):
    first = _orphan(importer, "old-a.md", ident="backend-a")
    second = _orphan(importer, "old-b.md", ident="backend-b")
    if case == "same_path_different_custom_id":
        second["metadata"] = first["metadata"]
    elif case == "same_custom_id_different_path":
        second["customId"] = first["customId"]
    elif case == "same_backend_id_different_path":
        second["id"] = first["id"]
    elif case == "malformed_peer_sharing_id":
        second = {"id": first["id"], "customId": None, "metadata": None}
    elif case == "alias_conflict_peer":
        second["custom_id"] = "obsidian-" + "f" * 64
        second["customId"] = first["customId"]
    elif case in {"same_container_repeat", "cross_container_repeat"}:
        second = dict(first)

    if case == "cross_container_repeat":
        inventories = [
            {"owner_primary": (first,), "family_shared": (second,)},
            {"family_shared": (second,), "owner_primary": (first,)},
        ]
    else:
        inventories = [
            {"owner_primary": ordering, "family_shared": ()}
            for ordering in ((first, second), (second, first))
        ]
    results = [_assert_collision_review_only(importer, inventory) for inventory in inventories]
    assert results[0] == results[1]


def test_global_raw_identity_collision_taint_is_transitive_and_order_independent(importer):
    first = _orphan(importer, "old-a.md", ident="backend-a")
    bridge = _orphan(importer, "old-a.md", ident="backend-b")
    third = _orphan(importer, "old-c.md", ident="backend-c")
    third["customId"] = bridge["customId"]
    results = []
    for ordering in itertools.permutations((first, bridge, third)):
        results.append(_assert_collision_review_only(importer, {
            "owner_primary": ordering, "family_shared": (),
        }))
    assert all(result == results[0] for result in results)


def test_collision_tainted_expected_record_never_emits_replace(importer):
    expected = importer.item("Skills/current.md", b"current")
    stale = _row(importer, expected, ident="shared-id",
                 metadata=importer.metadata(expected) | {"content_sha256": "stale"})
    malformed_peer = {"id": "shared-id", "customId": None, "metadata": None}
    classification = importer.classify_inventory({expected["relative_path"]: expected}, {
        "owner_primary": (stale, malformed_peer), "family_shared": (),
    })
    plan = importer.reconciliation_plan(classification)
    assert classification["identity_collision"] == ("*",)
    assert classification["stale_hash"] == ()
    assert {entry["action"] for entry in plan} == {"operator_review"}


def test_repeated_provider_pages_are_included_in_global_collision_accounting(importer):
    repeated = _orphan(importer, "old.md")

    class Documents:
        def list(self, **kwargs):
            if kwargs["container_tags"] == ["family_shared"]:
                return {"memories": [], "pagination": {
                    "currentPage": 1, "totalPages": 0, "totalItems": 0}}
            return {"memories": [repeated], "pagination": {
                "currentPage": kwargs["page"], "totalPages": 2, "totalItems": 2}}

    client = type("Client", (), {"documents": Documents()})()
    inventory = importer.provider_inventory(client)
    _assert_collision_review_only(importer, inventory)
