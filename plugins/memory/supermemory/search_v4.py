"""Small SDK adapter for document-only v4 search; no audience policy here."""

from typing import Any

from .canonical_policy import (
    CANONICAL_CONTAINERS,
    canonical_scope_from_path,
    canonical_visibility_from_path,
    stable_custom_id_from_path,
)

_MISSING = object()


def _summary_custom_id(result: Any) -> Any:
    parents = field(result, "documents")
    if not isinstance(parents, list) or len(parents) != 1:
        return None
    return field(parents[0], "custom_id", field(parents[0], "customId"))


def _explicit_container_claim_agrees(value: Any, container_tag: str) -> bool:
    """Validate one optional raw scalar/list container claim without coercion."""
    claims = []
    for name, plural in (
        ("container_tags", True), ("containerTags", True),
        ("container_tag", False), ("containerTag", False),
    ):
        claim = field(value, name, _MISSING)
        if claim is not _MISSING:
            claims.append((claim, plural))
    if not claims:
        return True
    # More than one spelling/shape is ambiguous even when the values happen
    # to agree.  Raw v4 responses support one scalar or one list field.
    if len(claims) != 1:
        return False
    claim, plural = claims[0]
    if plural:
        return type(claim) is list and claim == [container_tag]
    return type(claim) is str and claim == container_tag


def _summary_proves_identity(result: Any, container_tag: str) -> bool:
    """Return true only when the associated parent summary is a full proof."""
    if not canonical_chunk_preauthorized(result, container_tag):
        return False
    metadata = field(result, "metadata")
    parents = field(result, "documents")
    if not isinstance(metadata, dict) or not isinstance(parents, list) or len(parents) != 1:
        return False
    parent = parents[0]
    relative_path = metadata["relative_path"]
    expected = stable_custom_id_from_path(relative_path)
    custom_id = _summary_custom_id(result)
    tags = field(parent, "container_tags", field(parent, "containerTags"))
    tag = field(parent, "container_tag", field(parent, "containerTag"))
    return bool(custom_id == expected and (tags == [container_tag] or tag == container_tag))


def canonical_chunk_preauthorized(
    result: Any, container_tag: str, *, canonical_scope: str | None = None,
) -> bool:
    """Reject malformed canonical identity before fetching its parent."""
    metadata = field(result, "metadata")
    policy_container = canonical_scope or container_tag
    if not isinstance(metadata, dict) or policy_container not in CANONICAL_CONTAINERS:
        return False
    parents = field(result, "documents")
    if (not _explicit_container_claim_agrees(result, container_tag)
            or (isinstance(parents, list) and len(parents) == 1
                and not _explicit_container_claim_agrees(parents[0], container_tag))):
        return False
    scope = canonical_scope_from_path(metadata.get("relative_path"))
    return bool(
        scope == policy_container
        and metadata.get("index_schema_version") == 4
        and not isinstance(metadata.get("index_schema_version"), bool)
        and metadata.get("source") == "obsidian"
        and metadata.get("authority") == "canonical"
        and metadata.get("identity_scope") == "owner"
        and metadata.get("canonical_root") == "owner"
        and metadata.get("visibility") == canonical_visibility_from_path(
            metadata.get("relative_path")
        )
    )


def field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def search_documents_v4(client, query: str, *, container_tag: str, limit: int,
                        filters: dict | None = None, timeout: float | None = None,
                        search_mode: str = "documents"):
    if search_mode != "documents":
        raise ValueError("canonical document search requires documents mode")
    kwargs = {"q": query, "container_tag": container_tag}
    if filters:
        kwargs["filters"] = filters
    kwargs.update(limit=limit, search_mode=search_mode, rerank=False,
                  rewrite_query=False, aggregate=False, include={"documents": True})
    if timeout is not None:
        kwargs["timeout"] = max(0.001, timeout)
    return client.search.memories(**kwargs)


def normalize_document_chunk(result, container_tag: str, hydrated_parent,
                             *, allow_summary_proof: bool = False,
                             canonical_scope: str | None = None) -> dict | None:
    """Accept only chunks whose hydrated parent proves identity and scope."""
    text = field(result, "chunk")
    parents = field(result, "documents")
    metadata = field(result, "metadata")
    policy_container = canonical_scope or container_tag
    if (not canonical_chunk_preauthorized(
            result, container_tag, canonical_scope=policy_container)
            or not isinstance(text, str) or not text.strip()
            or field(result, "is_aggregated", field(result, "isAggregated", False))
            or not isinstance(parents, list) or len(parents) != 1
            or not isinstance(metadata, dict)
            or metadata != field(parents[0], "metadata")):
        return None
    parent = parents[0]
    parent_id = field(parent, "id")
    chunk_id = field(result, "id")
    if not isinstance(parent_id, str) or not parent_id or not isinstance(chunk_id, str) or not chunk_id:
        return None
    relative_path = metadata.get("relative_path")
    source_container = canonical_scope_from_path(relative_path) or ""
    expected_custom_id = stable_custom_id_from_path(relative_path) or ""
    proof_parent = hydrated_parent
    if proof_parent is None and allow_summary_proof and _summary_proves_identity(result, container_tag):
        proof_parent = parent
    hydrated_id = field(proof_parent, "id")
    custom_id = field(proof_parent, "custom_id", field(proof_parent, "customId", ""))
    if (metadata.get("source") != "obsidian" or metadata.get("authority") != "canonical"
            or parent_id not in {hydrated_id, custom_id}
            or source_container != policy_container or not expected_custom_id
            or custom_id != expected_custom_id):
        return None
    document_id = field(result, "document_id", field(result, "documentId"))
    if document_id and document_id != parent_id:
        return None
    hydrated_tags = field(proof_parent, "container_tags",
                          field(proof_parent, "containerTags"))
    hydrated_tag = field(proof_parent, "container_tag", field(proof_parent, "containerTag"))
    if hydrated_tags != [container_tag] and hydrated_tag != container_tag:
        return None
    for value in (result, parent):
        tags = field(value, "container_tags", field(value, "containerTags"))
        tag = field(value, "container_tag", field(value, "containerTag"))
        if tags is not None and tags != [container_tag]:
            return None
        if tag is not None and tag != container_tag:
            return None
    updated = field(result, "updated_at", field(result, "updatedAt"))
    parent_updated = field(parent, "updated_at", field(parent, "updatedAt"))
    if updated and parent_updated and updated != parent_updated:
        return None
    identity = [f"{key}: {metadata[key]}" for key in (
        "canonical_path", "entity_type", "entity_name", "venue_name", "branch", "schema_version",
    ) if metadata.get(key) not in (None, "")]
    if identity:
        text = "[canonical-identity]\n" + "\n".join(identity) + "\n[/canonical-identity]\n\n" + text
    return {"id": chunk_id, "memory": text, "metadata": metadata,
            "similarity": field(result, "similarity"), "updated_at": updated or parent_updated,
            "_source_container": container_tag, "_canonical_scope": source_container,
            "_parent_document_id": parent_id,
            "_source_custom_id": custom_id}
