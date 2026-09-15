#!/Users/dennis/.hermes/hermes-agent/venv/bin/python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import inspect
import json
import os
import plistlib
import re
import secrets
import stat
import sys
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from supermemory import Supermemory
from plugins.memory.supermemory.canonical_policy import (
    CANONICAL_CONTAINERS,
    FAMILY_CONTAINER,
    OWNER_CONTAINER,
    SENSITIVE_CONTENT,
    canonical_scope_from_path,
    canonical_entity_type,
    canonical_exclusion_reason,
    canonical_frontmatter_values,
    canonical_visibility_from_path,
    stable_custom_id_from_path,
)
from plugins.memory.supermemory.search_v4 import normalize_document_chunk, search_documents_v4
from plugins.memory.supermemory.readiness_receipts import (
    build_storage_reconciliation_receipt,
    build_vector_readiness_receipt,
    canonical_bytes,
    canonical_digest,
    generate_receipt_key,
    read_receipt_key,
    readiness_from_paths,
    scan_canonical_source,
    source_fingerprint_from_documents,
    validate_readiness_receipts,
)

ROOT = Path('/Users/dennis/Documents/ObsidianVault/Personal/Hermes').resolve()
ENV = Path('/Users/dennis/.hermes/.env')
OUT = Path('/Users/dennis/.hermes/obsidian-supermemory-import.json')
LOCK = Path('/Users/dennis/.hermes/obsidian-supermemory-sync.lock')
BASE_URL = 'http://127.0.0.1:6767'
CONTAINER = OWNER_CONTAINER
CONVERSATION_CONTAINER = 'owner_conversations'
OWNER_EXPLICIT_CONTAINER = OWNER_CONTAINER
CANONICAL_CONTAINERS = frozenset({CONTAINER, FAMILY_CONTAINER})
INDEX_SCHEMA_VERSION = 4

TERMINAL = {'done', 'failed', 'error'}


class ReconciliationRequired(RuntimeError):
    """Backend identity could not be proven; mutation must stop."""


class PrivateArtifactError(RuntimeError):
    """A private artifact path or file failed a fail-closed safety check."""


class TransactionJournalError(ValueError):
    """A transaction journal is malformed or does not match its canonical plan."""

def normalized_relative_path(rel: str) -> Path:
    if canonical_scope_from_path(rel) is None:
        raise ValueError("invalid canonical path")
    return Path(rel)

def visibility_for_path(rel: str) -> str:
    normalized_relative_path(rel)
    visibility = canonical_visibility_from_path(rel)
    assert visibility is not None
    return visibility


def container_for_doc(doc: dict[str, Any]) -> str:
    scope = canonical_scope_from_path(doc.get('relative_path'))
    if scope is None or doc.get('visibility') != canonical_visibility_from_path(doc.get('relative_path')):
        raise ValueError('invalid canonical projection')
    return CONTAINER if scope == OWNER_CONTAINER else FAMILY_CONTAINER


def destination_for_path(relative_path: Any) -> str | None:
    """Map a policy-derived logical scope to its configured exact destination."""
    scope = canonical_scope_from_path(relative_path)
    if scope == OWNER_CONTAINER:
        return CONTAINER
    if scope == FAMILY_CONTAINER:
        return FAMILY_CONTAINER
    return None


def configure_destinations(owner_canonical: str, owner_explicit: str) -> None:
    """Bind validated destinations before inventory or provider construction."""
    global CONTAINER, OWNER_EXPLICIT_CONTAINER, CANONICAL_CONTAINERS
    valid = re.compile(r'[A-Za-z0-9][A-Za-z0-9_]{0,127}').fullmatch
    if not valid(owner_canonical) or not valid(owner_explicit):
        raise SystemExit('Owner memory destination is invalid')
    protected = (owner_canonical, owner_explicit, FAMILY_CONTAINER, CONVERSATION_CONTAINER)
    topology_enabled = (owner_canonical, owner_explicit) != (OWNER_CONTAINER, OWNER_CONTAINER)
    if topology_enabled and (len(set(protected)) != len(protected)
            or any(tag.startswith('requester_conversations_') for tag in protected)):
        raise SystemExit('Owner memory destinations collide with protected topology')
    CONTAINER = owner_canonical
    OWNER_EXPLICIT_CONTAINER = owner_explicit
    CANONICAL_CONTAINERS = frozenset({CONTAINER, FAMILY_CONTAINER})


def container_for_row(row: dict[str, Any]) -> str:
    # Pre-isolation manifests implicitly placed every canonical record in Owner.
    return str(row.get('container') or CONTAINER)

_frontmatter_values = canonical_frontmatter_values

def trusted_event_date(fields: dict[str, list[str]]) -> str:
    normalized = []
    for key in ("event_date", "eventDate"):
        for value in fields.get(key, []):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value): raise ValueError("invalid event date")
            try: normalized.append(date.fromisoformat(value).isoformat())
            except ValueError: raise ValueError("invalid event date") from None
    if len(set(normalized)) > 1: raise ValueError("conflicting event dates")
    return normalized[0] if normalized else ""



def api_key() -> str:
    match = re.search(r'^SUPERMEMORY_API_KEY=(.+)$', ENV.read_text(encoding='utf-8'), re.MULTILINE)
    if not match:
        raise RuntimeError('SUPERMEMORY_API_KEY is not configured')
    return match.group(1).strip().strip('"').strip("'")


def canonical_documents() -> list[dict[str, Any]]:
    if not ROOT.is_dir() or not os.access(ROOT, os.R_OK | os.X_OK):
        raise RuntimeError(f'Canonical root is unavailable: {ROOT}')
    documents: list[dict[str, Any]] = []

    def scan_error(exc: OSError) -> None:
        raise exc

    root_fd = -1
    try:
        root_fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_identity = (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino)
        expected_directories: dict[str, tuple[int, int]] = {'.': root_identity}
        visited_directories: set[str] = set()
        for directory, names, files, directory_fd in os.fwalk(
                '.', topdown=True, onerror=scan_error, follow_symlinks=False, dir_fd=root_fd):
            directory_key = Path(directory).as_posix()
            directory_stat = os.fstat(directory_fd)
            identity = (directory_stat.st_dev, directory_stat.st_ino)
            if expected_directories.get(directory_key) != identity:
                raise OSError(f'Canonical directory identity changed during scan: {directory_key}')
            visited_directories.add(directory_key)
            kept_names: list[str] = []
            for name in sorted(names):
                if name == '.obsidian' or name.startswith('.'):
                    continue
                if directory_key == '.' and name not in {'Jarvis', 'Skills'}:
                    continue
                mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise OSError(f'Canonical directory symlink is not allowed: {Path(directory) / name}')
                if not stat.S_ISDIR(mode):
                    raise OSError(f'Canonical directory entry is not a directory: {Path(directory) / name}')
                child_key = (Path(directory) / name).as_posix()
                child_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                expected_directories[child_key] = (child_stat.st_dev, child_stat.st_ino)
                kept_names.append(name)
            names[:] = kept_names

            relative_directory = Path() if directory_key == '.' else Path(directory)
            for name in sorted(files):
                if name.startswith('.') or not name.endswith('.md'):
                    continue
                relative_path = (relative_directory / name).as_posix()
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                try:
                    file_stat = os.fstat(fd)
                    if not stat.S_ISREG(file_stat.st_mode):
                        raise OSError(f'Canonical Markdown entry is not a regular file: {relative_path}')
                    with os.fdopen(fd, 'rb', closefd=False) as handle:
                        raw = handle.read()
                finally:
                    os.close(fd)
                documents.append(item(relative_path, raw))
        if visited_directories != set(expected_directories):
            raise OSError('Canonical directory traversal was incomplete')
        final_root_stat = os.fstat(root_fd)
        if (final_root_stat.st_dev, final_root_stat.st_ino) != root_identity:
            raise OSError('Canonical root identity changed during scan')
        return sorted(documents, key=lambda doc: doc['relative_path'])
    except OSError as exc:
        raise RuntimeError(f'Canonical root scan failed: {type(exc).__name__}') from None
    finally:
        if root_fd >= 0:
            os.close(root_fd)


def item(rel: str, raw: bytes) -> dict[str, Any]:
    path = normalized_relative_path(rel); source = raw.decode("utf-8")
    parsed = _frontmatter_values(source); fields = {k: v[0] for k, v in parsed.items() if len(v) == 1}
    entity_type = canonical_entity_type(rel, fields)
    entity_name = fields.get("restaurant") or fields.get("category") or path.stem
    branch = path.stem.split(" - ", 1)[1] if entity_type == "restaurant" and " - " in path.stem else ""
    identity = {"canonical_path": rel, "entity_type": entity_type, "entity_name": entity_name, "schema_version": fields.get("schema_version", ""), "venue_name": fields.get("restaurant", "") if entity_type == "restaurant" else "", "branch": fields.get("location", "") or branch}
    governed = {k: fields[k] for k in ("fact_subject", "fact_key", "fact_value", "person_id", "entity_id") if fields.get(k)}
    event_date = trusted_event_date(parsed); parsed_date = date.fromisoformat(event_date) if event_date else None
    identity_lines = ["[canonical-identity]"] + [f"{k}: {v}" for k, v in identity.items() if v] + ["[/canonical-identity]"]
    return {"relative_path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "custom_id": stable_custom_id_from_path(rel), "bytes": len(raw), "content": "\n".join(identity_lines) + "\n\n" + source, "identity": identity, "governed_identity": governed, "event_date": event_date, "event_date_ordinal": parsed_date.toordinal() if parsed_date else None, "event_year": parsed_date.year if parsed_date else None, "event_month": parsed_date.month if parsed_date else None, "visibility": visibility_for_path(rel), "identity_scope": "owner", "canonical_root": "owner", "is_template": entity_type.endswith("-template") or path.stem.lower().endswith("template")}


def metadata(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        'source': 'obsidian',
        'authority': 'canonical',
        'relative_path': doc['relative_path'],
        'content_sha256': doc['sha256'],
        'content_bytes': doc['bytes'],
        'index_schema_version': INDEX_SCHEMA_VERSION,
        'visibility': doc['visibility'],
        'identity_scope': 'owner',
        'canonical_root': 'owner',
        **{key: value for key, value in doc['identity'].items() if value},
        **doc['governed_identity'],
        **{key: doc[key] for key in (
            'event_date', 'event_date_ordinal', 'event_year', 'event_month',
        ) if doc.get(key) is not None and doc.get(key) != ''},
    }


def record(doc: dict[str, Any], result: Any, document_id: str = '') -> dict[str, Any]:
    return {
        'relative_path': doc['relative_path'],
        'sha256': doc['sha256'],
        'custom_id': doc['custom_id'],
        'document_id': document_id or str(getattr(result, 'id', '') or ''),
        'submission_status': str(getattr(result, 'status', '') or ''),
        'bytes': doc['bytes'],
        'index_schema_version': INDEX_SCHEMA_VERSION,
        'container': container_for_doc(doc),
    }


def add(client: Supermemory, doc: dict[str, Any]) -> dict[str, Any]:
    result = client.documents.add(
        content=doc['content'], container_tag=container_for_doc(doc),
        custom_id=doc['custom_id'], task_type='superrag',
        metadata=metadata(doc), timeout=30.0,
    )
    return record(doc, result)


def update(client: Supermemory, doc: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    document_id = str(old.get('document_id') or '')
    if not document_id:
        return add(client, doc)
    # Supermemory's update endpoint appends a new indexed version to the
    # existing document. That leaves stale chunks searchable after canonical
    # edits. Replace the document instead so Obsidian remains authoritative.
    try:
        client.documents.delete(document_id, timeout=30.0)
    except Exception as exc:
        if not is_not_found(exc):
            raise
    row = record(doc, object())
    row.update({'document_id': '', 'submission_status': '',
                'replacement_stage': 'deleted', 'replaced_document_id': document_id})
    return row


def status_name(obj: Any) -> str:
    value = _field(obj, 'status', '')
    return str(getattr(value, 'value', value) or '').lower()


def is_not_found(exc: Exception) -> bool:
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
    return status == 404


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def lookup_backend_document(client: Supermemory, doc: dict[str, Any]) -> Any | None:
    """Exhaustively resolve one stable custom_id in its provider identity scope."""
    matches = []
    page = 1
    while True:
        response = client.documents.list(
            container_tags=[container_for_doc(doc)], include_content=True, limit=100, page=page,
            filters={'AND': [
                {'key': 'source', 'value': 'obsidian'},
                {'key': 'relative_path', 'value': doc['relative_path']},
            ]}, timeout=30.0,
        )
        rows = list(_field(response, 'memories', []) or [])
        matches.extend(row for row in rows if str(_field(row, 'custom_id', '') or '') == doc['custom_id'])
        pagination = _field(response, 'pagination')
        if pagination is None:
            raise ReconciliationRequired('backend pagination metadata is unavailable')
        try:
            current = int(_field(pagination, 'current_page'))
            total = int(_field(pagination, 'total_pages'))
        except (TypeError, ValueError):
            raise ReconciliationRequired('backend pagination metadata is invalid') from None
        # Supermemory represents an empty filtered result as page 1 of 0.
        # That is an exhaustive empty set only when the first page has no rows.
        if total == 0 and current == page == 1 and not rows:
            break
        if current != page or current < 1 or total < current:
            raise ReconciliationRequired('backend pagination could not prove identity')
        if current >= total:
            break
        if not rows or total > 10000:
            raise ReconciliationRequired('backend pagination could not prove identity')
        page = current + 1
    if len(matches) > 1:
        raise ReconciliationRequired('duplicate backend custom_id requires operator reconciliation')
    return matches[0] if matches else None


def validate_backend_document(remote: Any, doc: dict[str, Any]) -> dict[str, Any]:
    """Validate terminal state plus schema-v4 authority and ACL metadata."""
    if remote is None:
        raise ReconciliationRequired('stable custom_id is absent from backend')
    if str(_field(remote, 'custom_id', '') or '') != doc['custom_id']:
        raise ReconciliationRequired('backend stable custom_id does not match canonical projection')
    metadata_value = _field(remote, 'metadata', {})
    if not isinstance(metadata_value, dict):
        raise ReconciliationRequired('backend metadata is unavailable')
    expected = metadata(doc)
    required = ('source', 'authority', 'relative_path', 'content_sha256',
                'index_schema_version', 'visibility', 'identity_scope', 'canonical_root')
    if any(metadata_value.get(key) != expected[key] for key in required):
        raise ReconciliationRequired('backend metadata does not match canonical projection')
    tags = _field(remote, 'container_tags', _field(remote, 'containerTags'))
    if tags is not None and tags != [container_for_doc(doc)]:
        raise ReconciliationRequired('backend canonical container does not match visibility')
    actual_container = _field(remote, '_inventory_container')
    if actual_container is not None and actual_container != container_for_doc(doc):
        raise ReconciliationRequired('backend canonical container does not match visibility')
    state = status_name(remote)
    if state != 'done':
        raise ReconciliationRequired(f'backend document is not terminal done: {state or "unknown"}')
    ident = str(_field(remote, 'id', '') or '')
    if not ident:
        raise ReconciliationRequired('backend document id is unavailable')
    return record(doc, remote, ident) | {'submission_status': 'done', 'final_status': 'done',
                                         'replacement_stage': 'done'}


def validate_search_result_metadata(result: Any, doc: dict[str, Any]) -> bool:
    """Prove indexed search metadata converges with its sole parent and source."""
    result_metadata = _field(result, 'metadata')
    parents = list(_field(result, 'documents', []) or [])
    if not isinstance(result_metadata, dict) or len(parents) != 1:
        raise ReconciliationRequired('search result metadata or sole parent is unavailable')
    parent_metadata = _field(parents[0], 'metadata')
    if not isinstance(parent_metadata, dict):
        raise ReconciliationRequired('search parent metadata is unavailable')
    expected = metadata(doc)
    if (result_metadata != parent_metadata
            or any(result_metadata.get(key) != value for key, value in expected.items())):
        raise ReconciliationRequired('search result and parent metadata do not converge')
    return True


def verify_search_readiness(
        results_by_path: dict[str, list[Any]],
        current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Verify exactly one converged indexed result for every canonical source."""
    verified = 0
    failed_paths: list[str] = []
    for rel in sorted(current):
        results = results_by_path.get(rel, [])
        try:
            if len(results) != 1:
                raise ReconciliationRequired('search result is absent or ambiguous')
            validate_search_result_metadata(results[0], current[rel])
        except Exception:
            failed_paths.append(rel)
        else:
            verified += 1
    return {
        'search_verified_count': verified,
        'search_failure_count': len(failed_paths),
        'search_failed_paths': failed_paths,
        'search_readiness_complete': verified == len(current) and not failed_paths,
    }


def validate_legacy_backend_document(remote: Any, doc: dict[str, Any]) -> dict[str, Any]:
    """Prove a record is the exact canonical content with only v4 ACL drift."""
    if remote is None:
        raise ReconciliationRequired('stable custom_id is absent from backend')
    if str(_field(remote, 'custom_id', '') or '') != doc['custom_id']:
        raise ReconciliationRequired('backend stable identity does not match canonical projection')
    if status_name(remote) != 'done':
        raise ReconciliationRequired('backend document is not terminal done')
    ident = str(_field(remote, 'id', '') or '')
    if not ident:
        raise ReconciliationRequired('backend document id is unavailable')
    remote_content = _field(remote, 'content')
    if not isinstance(remote_content, str) or remote_content.rstrip(" \t\r\n") != doc['content'].rstrip(" \t\r\n"):
        raise ReconciliationRequired('backend content does not match canonical projection')
    actual = _field(remote, 'metadata', {})
    if not isinstance(actual, dict):
        raise ReconciliationRequired('backend identity metadata is unavailable')
    expected = metadata(doc)
    identity_keys = ('source', 'authority', 'relative_path', 'content_sha256', 'content_bytes')
    if any(actual.get(key) != expected[key] for key in identity_keys):
        raise ReconciliationRequired('backend identity metadata does not match canonical projection')
    acl_keys = ('index_schema_version', 'visibility', 'identity_scope', 'canonical_root')
    if all(actual.get(key) == expected[key] for key in acl_keys):
        return validate_backend_document(remote, doc)
    return record(doc, remote, ident) | {
        'submission_status': 'done', 'final_status': 'done',
        'replacement_stage': 'verified_legacy',
    }


def schema_v4_backend_inventory(client: Supermemory, expected_custom_ids: set[str]) -> dict[str, Any]:
    """Exhaust both canonical containers and reject duplicate stable IDs across them."""
    found: dict[str, Any] = {}
    if len({CONTAINER, FAMILY_CONTAINER, CONVERSATION_CONTAINER}) != 3:
        raise ReconciliationRequired('canonical and conversation containers are not isolated')
    for container in (CONTAINER, FAMILY_CONTAINER):
        page = 1
        while True:
            response = client.documents.list(container_tags=[container], include_content=True,
                                             limit=100, page=page, timeout=30.0)
            rows = list(_field(response, 'memories', []) or [])
            for remote in rows:
                custom_id = str(_field(remote, 'custom_id', '') or '')
                if custom_id not in expected_custom_ids:
                    continue
                if custom_id in found:
                    raise ReconciliationRequired('duplicate backend custom_id requires operator reconciliation')
                remote = dict(remote) if isinstance(remote, dict) else remote.model_dump()
                remote['_inventory_container'] = container
                found[custom_id] = remote
            pagination = _field(response, 'pagination')
            if pagination is None:
                raise ReconciliationRequired('backend pagination metadata is unavailable')
            try:
                current = int(_field(pagination, 'current_page'))
                total = int(_field(pagination, 'total_pages'))
            except (TypeError, ValueError):
                raise ReconciliationRequired('backend pagination metadata is invalid') from None
            if total == 0 and current == page == 1 and not rows:
                break
            if current != page or current < 1 or total < current or total > 10000:
                raise ReconciliationRequired('backend pagination could not prove identity')
            if current >= total:
                break
            if not rows:
                raise ReconciliationRequired('backend pagination could not prove identity')
            page += 1
    return found


def _alias(obj: Any, snake: str, camel: str, default: Any = None) -> Any:
    """Read one SDK/wire alias, rejecting disagreeing duplicate representations."""
    marker = object()
    left = _field(obj, snake, marker)
    right = _field(obj, camel, marker)
    if left is not marker and right is not marker and left != right:
        raise ReconciliationRequired('ambiguous provider response aliases')
    return left if left is not marker else right if right is not marker else default


def _require_superrag_task(obj: Any) -> None:
    """Require the SDK's exact task type, rejecting missing/conflicting aliases."""
    if _alias(obj, 'task_type', 'taskType') != 'superrag':
        raise ReconciliationRequired('provider task type is not exactly superrag')


def _page_integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReconciliationRequired('provider pagination metadata is invalid')
    if isinstance(value, float) and (not value.is_integer() or not value < float('inf')):
        raise ReconciliationRequired('provider pagination metadata is invalid')
    return int(value)


def provider_inventory(client: Supermemory) -> dict[str, tuple[dict[str, Any], ...]]:
    """Exhaustively list each canonical container without filtering or mutation."""
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for container in sorted(CANONICAL_CONTAINERS):
        collected: list[dict[str, Any]] = []
        requested = 1
        seen: set[int] = set()
        declared_total: int | None = None
        declared_items: int | None = None
        while True:
            response = client.documents.list(
                container_tags=[container], include_content=False, limit=100,
                page=requested, timeout=30.0,
            )
            rows = _field(response, 'memories')
            if not isinstance(rows, (list, tuple)):
                raise ReconciliationRequired('provider inventory rows are malformed')
            pagination = _field(response, 'pagination')
            if pagination is None:
                raise ReconciliationRequired('provider pagination metadata is unavailable')
            current = _page_integer(_alias(pagination, 'current_page', 'currentPage'))
            total = _page_integer(_alias(pagination, 'total_pages', 'totalPages'))
            total_items = _page_integer(_alias(pagination, 'total_items', 'totalItems'))
            if declared_total is not None and total != declared_total:
                raise ReconciliationRequired('provider pagination is inconsistent')
            if declared_items is not None and total_items != declared_items:
                raise ReconciliationRequired('provider pagination is inconsistent')
            declared_total = total
            declared_items = total_items
            if total == 0:
                if current == requested == 1 and total_items == 0 and not rows:
                    break
                raise ReconciliationRequired('provider pagination could not prove inventory')
            if current != requested or current in seen or current < 1 or total < current or total > 10000:
                raise ReconciliationRequired('provider pagination could not prove inventory')
            seen.add(current)
            if current < total and not rows:
                raise ReconciliationRequired('provider pagination has a missing page')
            for raw in rows:
                if not isinstance(raw, dict) and not hasattr(raw, 'model_dump'):
                    raise ReconciliationRequired('provider inventory row is malformed')
                row = dict(raw) if isinstance(raw, dict) else raw.model_dump()
                row['_inventory_container'] = container
                collected.append(row)
            if current == total:
                break
            requested = current + 1
        if declared_total and seen != set(range(1, declared_total + 1)):
            raise ReconciliationRequired('provider pagination has missing pages')
        if declared_total and declared_items != len(collected):
            raise ReconciliationRequired('provider pagination item count is inconsistent')
        result[container] = tuple(collected)
    return result


def _explicit_container_claim_agrees(row: dict[str, Any], container: str) -> bool:
    """Accept at most one exact provider container claim and reject alias ambiguity."""
    claims: list[tuple[Any, bool]] = []
    marker = object()
    for name, plural in (
        ('container_tags', True), ('containerTags', True),
        ('container_tag', False), ('containerTag', False),
    ):
        value = _field(row, name, marker)
        if value is not marker:
            claims.append((value, plural))
    if not claims:
        return True
    if len(claims) != 1:
        return False
    value, plural = claims[0]
    return (type(value) is list and value == [container]) if plural else (
        type(value) is str and value == container
    )


def _trusted_orphan_path(row: dict[str, Any], source_container: str) -> str | None:
    """Return a delete-safe path only when the row independently proves identity."""
    try:
        custom_id = _alias(row, 'custom_id', 'customId')
        metadata_value = _field(row, 'metadata')
        ident = _field(row, 'id')
        inventory_container = _field(row, '_inventory_container', source_container)
        state = status_name(row)
        if (not isinstance(custom_id, str) or not isinstance(ident, str) or not ident
                or not isinstance(metadata_value, dict) or source_container not in CANONICAL_CONTAINERS
                or inventory_container != source_container or state != 'done'):
            return None
        rel = metadata_value.get('relative_path')
        derived_container = destination_for_path(rel)
        expected_visibility = canonical_visibility_from_path(rel)
        if (derived_container is None or derived_container != source_container
                or custom_id != stable_custom_id_from_path(rel)):
            return None
        if not _explicit_container_claim_agrees(row, derived_container):
            return None
        sha256 = metadata_value.get('content_sha256')
        byte_count = metadata_value.get('content_bytes')
        required = {
            'source': 'obsidian', 'authority': 'canonical',
            'relative_path': rel, 'index_schema_version': INDEX_SCHEMA_VERSION,
            'visibility': expected_visibility, 'identity_scope': 'owner',
            'canonical_root': 'owner', 'canonical_path': rel,
        }
        if (any(metadata_value.get(key) != value for key, value in required.items())
                or not isinstance(sha256, str) or re.fullmatch(r'[0-9a-f]{64}', sha256) is None
                or isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0):
            return None
        return rel
    except ReconciliationRequired:
        return None


def _raw_identity_tokens(row: Any) -> tuple[set[tuple[str, str]], bool]:
    """Extract every usable raw identity claim without trusting normalization."""
    if not isinstance(row, dict):
        return set(), True
    claims: dict[str, set[str]] = {'path': set(), 'custom_id': set(), 'backend_id': set()}
    for name in ('custom_id', 'customId'):
        value = row.get(name)
        if isinstance(value, str) and value:
            claims['custom_id'].add(value)
    for name in ('id', 'document_id', 'documentId'):
        value = row.get(name)
        if isinstance(value, str) and value:
            claims['backend_id'].add(value)
    metadata_value = row.get('metadata')
    if isinstance(metadata_value, dict):
        for name in ('relative_path', 'relativePath', 'canonical_path', 'canonicalPath'):
            value = metadata_value.get(name)
            if isinstance(value, str) and value:
                claims['path'].add(value)
    ambiguous = any(len(values) > 1 for values in claims.values())
    return {(kind, value) for kind, values in claims.items() for value in values}, ambiguous


_EXPLICIT_MEMORY_METADATA = frozenset({'sm_source', 'target', 'type'})
# These are the only harmless fields returned by the document-list endpoint for
# a Hermes explicit write. ``model_dump()`` uses the snake-case SDK field names;
# wire-format camel aliases remain recognized only so they can fail closed.
_EXPLICIT_MEMORY_ROW_FIELDS = frozenset({
    'id', 'created_at', 'createdAt', 'custom_id', 'customId', 'metadata',
    'status', 'type', 'updated_at', 'updatedAt', 'container_tags',
    'containerTags', 'container_tag', 'containerTag', '_inventory_container',
    'connection_id', 'filepath', 'summary', 'title', 'content', 'url',
})
# Keep this exhaustive with every schema, identity, provenance, path, ACL, and
# content field consumed or emitted by canonical metadata parsing. The exact
# metadata allowlist below is the future-safe boundary; this list documents the
# canonical-like signals that must never be mistaken for provider operations.
_CANONICAL_METADATA_MARKERS = frozenset({
    'source', 'authority', 'relative_path', 'relativePath', 'canonical_path',
    'canonicalPath', 'index_schema_version', 'indexSchemaVersion',
    'schema_version', 'schemaVersion', 'entity_type', 'entityType',
    'entity_name', 'entityName', 'venue_name', 'venueName', 'branch',
    'fact_subject', 'factSubject', 'fact_key', 'factKey', 'fact_value',
    'factValue', 'person_id', 'personId', 'entity_id', 'entityId',
    'event_date', 'eventDate', 'event_date_ordinal', 'eventDateOrdinal',
    'event_year', 'eventYear', 'event_month', 'eventMonth', 'visibility',
    'identity_scope', 'identityScope', 'canonical_root', 'canonicalRoot',
    'content_sha256', 'contentSha256', 'content_bytes', 'contentBytes',
})


def _is_noncanonical_explicit_memory(row: Any, source_container: str) -> bool:
    """Exclude only the exact, documented Hermes explicit-write envelope."""
    if not isinstance(row, dict) or source_container != OWNER_EXPLICIT_CONTAINER:
        return False
    if not set(row).issubset(_EXPLICIT_MEMORY_ROW_FIELDS):
        return False
    # provider_inventory normalizes SDK models with model_dump(), whose exact
    # custom-ID key is custom_id. Absence, a wire alias, duplicate aliases, or
    # any value other than the singleton None must remain in closed inventory.
    if 'custom_id' not in row or 'customId' in row or row['custom_id'] is not None:
        return False
    if any(sum(name in row for name in aliases) > 1 for aliases in (
        ('created_at', 'createdAt'),
        ('updated_at', 'updatedAt'),
        ('container_tags', 'containerTags', 'container_tag', 'containerTag'),
    )):
        return False
    try:
        metadata_value = _field(row, 'metadata')
        ident = _field(row, 'id')
        created_at = _alias(row, 'created_at', 'createdAt')
        updated_at = _alias(row, 'updated_at', 'updatedAt')
        inventory_container = _field(row, '_inventory_container', source_container)
        if (
            not isinstance(ident, str) or not ident
            or not isinstance(created_at, str) or not created_at
            or not isinstance(updated_at, str) or not updated_at
            or any(row.get(name) is not None for name in (
                'connection_id', 'filepath', 'summary', 'title', 'content', 'url'))
            or _field(row, 'type') != 'text'
            or status_name(row) != 'done'
            or not isinstance(metadata_value, dict)
            or set(metadata_value) != _EXPLICIT_MEMORY_METADATA
            or bool(_CANONICAL_METADATA_MARKERS & metadata_value.keys())
            or metadata_value.get('type') != 'explicit_memory'
            or metadata_value.get('sm_source') != 'hermes'
            or metadata_value.get('target') not in {'memory', 'user'}
            or inventory_container != source_container
            or not _explicit_container_claim_agrees(row, source_container)
        ):
            return False
    except ReconciliationRequired:
        return False
    return True


def _claims_explicit_memory(row: Any) -> bool:
    """Identify any row carrying a Hermes explicit-write signal."""
    if not isinstance(row, dict):
        return False
    metadata_value = row.get('metadata')
    return isinstance(metadata_value, dict) and (
        metadata_value.get('type') == 'explicit_memory'
        or metadata_value.get('sm_source') == 'hermes'
    )


def _globally_tainted_rows(rows: list[tuple[str, Any]]) -> tuple[set[int], set[tuple[str, str]]]:
    """Taint complete raw-identity components containing a collision or ambiguity."""
    tokens_by_row: list[set[tuple[str, str]]] = []
    ambiguous_rows: set[int] = set()
    owners: dict[tuple[str, str], list[int]] = {}
    for index, (_, row) in enumerate(rows):
        tokens, ambiguous = _raw_identity_tokens(row)
        tokens_by_row.append(tokens)
        if ambiguous:
            ambiguous_rows.add(index)
        for token in tokens:
            owners.setdefault(token, []).append(index)

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    collision_rows: set[int] = set()
    for indexes in owners.values():
        if len(indexes) > 1:
            collision_rows.update(indexes)
            for index in indexes[1:]:
                union(indexes[0], index)
    tainted_roots = {find(index) for index in collision_rows | ambiguous_rows}
    tainted_rows = {index for index in range(len(rows)) if find(index) in tainted_roots}
    tainted_tokens = set().union(*(tokens_by_row[index] for index in tainted_rows)) if tainted_rows else set()
    return tainted_rows, tainted_tokens


def classify_inventory(
        current: dict[str, dict[str, Any]],
        inventory: dict[str, tuple[dict[str, Any], ...]]) -> dict[str, tuple[str, ...]]:
    """Classify the closed canonical/provider sets using path-only diagnostics."""
    categories: dict[str, set[str]] = {name: set() for name in (
        'expected', 'missing', 'duplicate_expected_custom_ids', 'wrong_container',
        'trusted_orphans', 'untrusted_orphans', 'pending', 'failed', 'malformed',
        'stale_hash', 'stale_byte_count', 'status', 'identity_collision',
    )}
    expected_by_id = {doc['custom_id']: doc for doc in current.values()}
    matches: dict[str, list[dict[str, Any]]] = {key: [] for key in expected_by_id}
    orphan_candidates: list[tuple[str | None, dict[str, Any], str | None]] = []
    raw_rows = [
        (container, raw)
        for container in sorted(inventory)
        for raw in inventory[container]
        if not _is_noncanonical_explicit_memory(raw, container)
    ]
    tainted_rows, tainted_tokens = _globally_tainted_rows(raw_rows)
    if tainted_rows:
        categories['identity_collision'].add('*')
    for row_index, (container, raw) in enumerate(raw_rows):
        if not isinstance(raw, dict):
            categories['malformed'].add('*')
            continue
        if _claims_explicit_memory(raw):
            # Any explicit-like row reaching the canonical inventory failed the
            # exact exclusion contract and therefore requires operator review.
            categories['malformed'].add('*')
            continue
        row = dict(raw)
        row['_identity_tainted'] = row_index in tainted_rows
        try:
            custom_id = _alias(row, 'custom_id', 'customId')
            metadata_value = _field(row, 'metadata')
            ident = _field(row, 'id')
            if (not isinstance(custom_id, str) or not isinstance(ident, str) or not ident
                    or not isinstance(metadata_value, dict)):
                raise ValueError
        except (ValueError, ReconciliationRequired):
            categories['malformed'].add('*')
            if str(_field(row, 'custom_id', _field(row, 'customId', ''))).startswith('obsidian-'):
                categories['untrusted_orphans'].add('*')
            continue
        row.setdefault('_inventory_container', container)
        if custom_id in matches:
            matches[custom_id].append(row)
        elif custom_id.startswith('obsidian-'):
            candidate_path = metadata_value.get('relative_path')
            if (canonical_scope_from_path(candidate_path) is None
                    or custom_id != stable_custom_id_from_path(candidate_path)):
                candidate_path = None
            orphan_candidates.append(
                (_trusted_orphan_path(row, container), row, candidate_path)
            )

    for path, row, candidate_path in orphan_candidates:
        if (path is not None and candidate_path is not None
                and not row['_identity_tainted']):
            categories['trusted_orphans'].add(path)
        else:
            categories['untrusted_orphans'].add('*')

    for custom_id, doc in expected_by_id.items():
        rel = doc['relative_path']
        rows = matches[custom_id]
        if not rows:
            if ('custom_id', custom_id) in tainted_tokens:
                categories['expected'].add(rel)
                continue
            categories['missing'].add(rel)
            continue
        categories['expected'].add(rel)
        if any(row['_identity_tainted'] for row in rows):
            continue
        if len(rows) > 1:
            categories['duplicate_expected_custom_ids'].add(rel)
        for row in rows:
            if row['_inventory_container'] != container_for_doc(doc):
                categories['wrong_container'].add(rel)
            actual = row['metadata']
            if actual.get('content_sha256') != doc['sha256']:
                categories['stale_hash'].add(rel)
            if actual.get('content_bytes') != doc['bytes']:
                categories['stale_byte_count'].add(rel)
            state = status_name(row)
            if state != 'done':
                categories['status'].add(rel)
            if state in {'queued', 'extracting', 'chunking', 'embedding', 'indexing', 'processing'}:
                categories['pending'].add(rel)
            elif state in {'failed', 'error'}:
                categories['failed'].add(rel)
            elif state != 'done':
                categories['malformed'].add(rel)
    return {name: tuple(sorted(paths)) for name, paths in categories.items()}


def reconciliation_plan(classification: dict[str, tuple[str, ...]]) -> tuple[dict[str, str], ...]:
    """Return a deterministic, path-only plan; this function cannot mutate."""
    replace_reasons = ('duplicate_expected_custom_ids', 'wrong_container', 'pending',
                       'failed', 'malformed', 'stale_hash', 'stale_byte_count', 'status')
    reasons_by_path: dict[str, set[str]] = {}
    for reason in replace_reasons:
        for path in classification.get(reason, ()):
            if path != '*':
                reasons_by_path.setdefault(path, set()).add(reason)
    plan = [
        {'action': 'add', 'relative_path': path, 'reason': 'missing'}
        for path in classification.get('missing', ())
    ]
    plan.extend(
        {'action': 'replace', 'relative_path': path, 'reason': ','.join(sorted(reasons))}
        for path, reasons in reasons_by_path.items() if path not in classification.get('missing', ())
    )
    plan.extend(
        {'action': 'delete', 'relative_path': path, 'reason': 'trusted_orphan'}
        for path in classification.get('trusted_orphans', ())
    )
    review_reasons = tuple(sorted(
        reason for reason in ('identity_collision', 'malformed', 'untrusted_orphans')
        if '*' in classification.get(reason, ())
    ))
    if review_reasons:
        plan.append({'action': 'operator_review', 'relative_path': '*',
                     'reason': ','.join(review_reasons)})
    return tuple(sorted(plan, key=lambda row: (row['relative_path'], row['action'], row['reason'])))


def build_schema_v4_backfill_plan(
        client: Supermemory, current: dict[str, dict[str, Any]],
        previous: dict[str, dict[str, Any]], *, expected_eligible: int,
        expected_owner_private: int, expected_family_shared: int) -> dict[str, Any]:
    """Exhaustively verify the closed replacement set before any mutation."""
    actual_counts = (
        len(current),
        sum(doc['visibility'] == 'owner_private' for doc in current.values()),
        sum(doc['visibility'] == 'family_shared' for doc in current.values()),
    )
    if actual_counts != (expected_eligible, expected_owner_private, expected_family_shared):
        raise ReconciliationRequired('operator-confirmed eligible/visibility count drift')
    if expected_owner_private + expected_family_shared != expected_eligible:
        raise ReconciliationRequired('operator-confirmed counts are inconsistent')
    if set(previous) != set(current):
        raise ReconciliationRequired('manifest/canonical path set drift')

    replacements: dict[str, dict[str, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}
    resumable = {'deleted', 'add_submitted', 'pending', 'add_failed', 'reconcile_required'}
    needs_backend = {
        current[rel]['custom_id'] for rel in current
        if str(previous[rel].get('replacement_stage') or '') not in resumable
    }
    inventory = schema_v4_backend_inventory(client, needs_backend) if needs_backend else {}
    for rel in sorted(current):
        doc = current[rel]
        old = previous[rel]
        stage = str(old.get('replacement_stage') or '')
        if stage in resumable:
            if (old.get('relative_path') != rel or old.get('sha256') != doc['sha256']
                    or old.get('custom_id') != doc['custom_id']
                    or old.get('index_schema_version') != INDEX_SCHEMA_VERSION):
                raise ReconciliationRequired('recovery checkpoint identity mismatch')
            replacements[rel] = dict(old)
            continue
        row = validate_legacy_backend_document(inventory.get(doc['custom_id']), doc)
        if row['replacement_stage'] == 'done':
            completed[rel] = row
        else:
            replacements[rel] = row
    return {'replacements': replacements, 'completed': completed, 'inventory': inventory,
            'already_v4': len(completed), 'eligible': len(current)}


TRANSACTION_PLAN_FIELDS = frozenset({
    'action', 'relative_path', 'custom_id', 'source_container', 'target_container',
    'expected_sha256', 'expected_bytes', 'expected_pre_identity', 'expected_post_identity',
})
TRANSACTION_IDENTITY_FIELDS = frozenset({'custom_id', 'container', 'sha256', 'bytes', 'document_id'})
TRANSACTION_STAGES = frozenset({
    'existing', 'delete_verified', 'delete_submitted', 'deleted', 'add_submitted',
    'processing', 'done', 'failed', 'reconcile_required',
})
TRANSACTION_TRANSITIONS = {
    'add': {
        'existing': {'add_submitted', 'failed', 'reconcile_required'},
        'add_submitted': {'processing', 'done', 'failed', 'reconcile_required'},
        'processing': {'done', 'failed', 'reconcile_required'},
        'failed': {'reconcile_required'},
        'reconcile_required': {'add_submitted'},
    },
    'replace': {
        'existing': {'delete_verified', 'failed', 'reconcile_required'},
        'delete_verified': {'delete_submitted', 'failed', 'reconcile_required'},
        'delete_submitted': {'deleted', 'failed', 'reconcile_required'},
        'deleted': {'add_submitted', 'failed', 'reconcile_required'},
        'add_submitted': {'processing', 'done', 'failed', 'reconcile_required'},
        'processing': {'done', 'failed', 'reconcile_required'},
        'failed': {'reconcile_required'},
        'reconcile_required': {'delete_verified', 'deleted', 'add_submitted'},
    },
    'delete': {
        'existing': {'delete_verified', 'failed', 'reconcile_required'},
        'delete_verified': {'delete_submitted', 'failed', 'reconcile_required'},
        'delete_submitted': {'deleted', 'failed', 'reconcile_required'},
        'deleted': {'done', 'failed', 'reconcile_required'},
        'failed': {'reconcile_required'},
        'reconcile_required': {'delete_verified', 'deleted'},
    },
}


def _plain_json(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float aliases."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def thaw_transaction_journal(journal: Any) -> dict[str, Any]:
    value = _plain_json(journal)
    if not isinstance(value, dict):
        raise TransactionJournalError('transaction journal must be an object')
    return value


def _canonical_digest(value: Any) -> str:
    try:
        data = json.dumps(_plain_json(value), ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError):
        raise TransactionJournalError('transaction plan is not canonical JSON') from None
    return hashlib.sha256(data).hexdigest()


_SHA256_RE = re.compile(r'[0-9a-f]{64}')
_TRANSACTION_ID_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')


def _is_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _validate_transaction_id(value: Any, *, label: str = 'transaction id') -> None:
    if type(value) is not str or _TRANSACTION_ID_RE.fullmatch(value) is None:
        raise TransactionJournalError(f'{label} is invalid')


def _validate_transaction_identity(identity: Any, record: dict[str, Any], label: str) -> None:
    if not isinstance(identity, dict) or set(identity) != TRANSACTION_IDENTITY_FIELDS:
        raise TransactionJournalError(f'{label} identity fields are invalid')
    if (type(identity['custom_id']) is not str or identity['custom_id'] != record['custom_id']
            or type(identity['container']) is not str
            or identity['container'] not in CANONICAL_CONTAINERS
            or not _is_sha256(identity['sha256'])
            or type(identity['bytes']) is not int
            or identity['bytes'] < 0
            or (identity['document_id'] is not None
                and (type(identity['document_id']) is not str
                     or not identity['document_id']
                     or identity['document_id'].strip() != identity['document_id']))):
        raise TransactionJournalError(f'{label} identity is invalid')


def _validated_transaction_plan(plan_records: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(plan_records, (tuple, list)):
        raise TransactionJournalError('transaction plan records are invalid')
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    custom_ids: set[str] = set()
    document_ids: set[str] = set()
    for raw in plan_records:
        raw = _plain_json(raw)
        if not isinstance(raw, dict) or set(raw) != TRANSACTION_PLAN_FIELDS:
            raise TransactionJournalError('transaction plan record fields are invalid')
        record = dict(raw)
        action = record['action']
        rel = record['relative_path']
        custom_id = record['custom_id']
        if type(action) is not str or action not in TRANSACTION_TRANSITIONS:
            raise TransactionJournalError('transaction plan action is invalid')
        if (type(rel) is not str or canonical_scope_from_path(rel) is None
                or type(custom_id) is not str or custom_id != stable_custom_id_from_path(rel)
                or not _is_sha256(record['expected_sha256'])
                or type(record['expected_bytes']) is not int or record['expected_bytes'] < 0
                or (record['source_container'] is not None
                    and (type(record['source_container']) is not str
                         or record['source_container'] not in CANONICAL_CONTAINERS))
                or (record['target_container'] is not None
                    and (type(record['target_container']) is not str
                         or record['target_container'] not in CANONICAL_CONTAINERS))):
            raise TransactionJournalError('transaction plan record identity is invalid')
        pre, post = record['expected_pre_identity'], record['expected_post_identity']
        if pre is not None:
            _validate_transaction_identity(pre, record, 'pre')
        if post is not None:
            _validate_transaction_identity(post, record, 'post')
        if ((action == 'add' and (pre is not None or post is None
                                 or record['source_container'] is not None
                                 or record['target_container'] != post['container']))
                or (action == 'delete' and (pre is None or post is not None
                                            or record['source_container'] != pre['container']
                                            or record['target_container'] is not None))
                or (action == 'replace' and (pre is None or post is None
                                             or record['source_container'] != pre['container']
                                             or record['target_container'] != post['container']))):
            raise TransactionJournalError('transaction action identity is inconsistent')
        expected_identity = post if action in {'add', 'replace'} else pre
        if (expected_identity['sha256'] != record['expected_sha256']
                or expected_identity['bytes'] != record['expected_bytes']):
            raise TransactionJournalError('transaction expected identity is inconsistent')
        if rel in paths or custom_id in custom_ids:
            raise TransactionJournalError('duplicate transaction plan identity')
        paths.add(rel)
        custom_ids.add(custom_id)
        for identity in (pre, post):
            document_id = identity and identity['document_id']
            if document_id:
                if document_id in document_ids:
                    raise TransactionJournalError('duplicate transaction document identity')
                document_ids.add(document_id)
        records.append(record)
    return tuple(records)


def transaction_plan_digest(plan_records: Any) -> str:
    """Hash the exact order and complete expected identities of a pure plan."""
    return _canonical_digest(_validated_transaction_plan(plan_records))


def new_transaction_journal(transaction_id: str, plan_records: Any, *,
                            snapshot_digest: str | None = None) -> MappingProxyType:
    records = _validated_transaction_plan(plan_records)
    _validate_transaction_id(transaction_id)
    if snapshot_digest is not None and not _is_sha256(snapshot_digest):
        raise TransactionJournalError('snapshot digest is invalid')
    journal = {
        'journal_schema_version': 1,
        'transaction_id': transaction_id,
        'plan_digest': _canonical_digest(records),
        'snapshot_digest': snapshot_digest,
        'records': [dict(record, order=index, stage='existing', history=['existing'])
                    for index, record in enumerate(records)],
    }
    return _freeze_json(journal)


def validate_transaction_journal(journal: Any, plan_records: Any, *,
                                 expected_transaction_id: str,
                                 expected_snapshot_digest: str | None = None) -> MappingProxyType:
    plan = _validated_transaction_plan(plan_records)
    _validate_transaction_id(expected_transaction_id, label='expected transaction id')
    if expected_snapshot_digest is not None and not _is_sha256(expected_snapshot_digest):
        raise TransactionJournalError('expected snapshot digest is invalid')
    value = thaw_transaction_journal(journal)
    fields = {'journal_schema_version', 'transaction_id', 'plan_digest', 'snapshot_digest', 'records'}
    if (set(value) != fields or type(value['journal_schema_version']) is not int
            or value['journal_schema_version'] != 1):
        raise TransactionJournalError('transaction journal fields are invalid')
    _validate_transaction_id(value['transaction_id'])
    if value['transaction_id'] != expected_transaction_id:
        raise TransactionJournalError('transaction id is stale or mismatched')
    digest = _canonical_digest(plan)
    if not _is_sha256(value['plan_digest']) or value['plan_digest'] != digest:
        raise TransactionJournalError('transaction journal plan digest mismatch')
    if ((value['snapshot_digest'] is not None and not _is_sha256(value['snapshot_digest']))
            or value['snapshot_digest'] != expected_snapshot_digest):
        raise TransactionJournalError('transaction snapshot digest mismatch')
    rows = value['records']
    if not isinstance(rows, list) or len(rows) != len(plan):
        raise TransactionJournalError('transaction journal plan record count mismatch')
    journal_extra = {'order', 'stage', 'history'}
    for index, (row, expected) in enumerate(zip(rows, plan, strict=True)):
        if not isinstance(row, dict) or set(row) != TRANSACTION_PLAN_FIELDS | journal_extra:
            raise TransactionJournalError('transaction journal record fields are invalid')
        if (type(row['order']) is not int or row['order'] < 0 or row['order'] != index
                or not _exact_json_equal(
                    {key: row[key] for key in TRANSACTION_PLAN_FIELDS}, expected
                )):
            raise TransactionJournalError('transaction journal plan record mismatch')
        stage, history = row['stage'], row['history']
        if (type(stage) is not str or stage not in TRANSACTION_STAGES
                or not isinstance(history, list) or not history or history[-1] != stage):
            raise TransactionJournalError('transaction journal stage is invalid')
        if (history[0] != 'existing'
                or any(type(item) is not str or item not in TRANSACTION_STAGES for item in history)):
            raise TransactionJournalError('transaction journal stage history is invalid')
        transitions = TRANSACTION_TRANSITIONS[expected['action']]
        if any(right not in transitions.get(left, set()) for left, right in zip(history, history[1:])):
            raise TransactionJournalError('transaction journal transition is invalid')
    return _freeze_json(value)


def advance_transaction_journal(journal: Any, plan_records: Any, record_index: int,
                                stage: str, *, expected_transaction_id: str,
                                expected_snapshot_digest: str | None = None) -> MappingProxyType:
    validated = validate_transaction_journal(
        journal, plan_records, expected_transaction_id=expected_transaction_id,
        expected_snapshot_digest=expected_snapshot_digest,
    )
    if stage not in TRANSACTION_STAGES:
        raise TransactionJournalError('transaction journal stage is unknown')
    if (isinstance(record_index, bool) or not isinstance(record_index, int)
            or not 0 <= record_index < len(validated['records'])):
        raise TransactionJournalError('transaction journal record index is invalid')
    value = thaw_transaction_journal(validated)
    row = value['records'][record_index]
    if stage not in TRANSACTION_TRANSITIONS[row['action']].get(row['stage'], set()):
        raise TransactionJournalError('transaction journal transition is invalid')
    row['stage'] = stage
    row['history'].append(stage)
    return validate_transaction_journal(
        value, plan_records, expected_transaction_id=expected_transaction_id,
        expected_snapshot_digest=expected_snapshot_digest,
    )


def _private_child_parts(child: str) -> tuple[str, ...]:
    if (not isinstance(child, str) or not child or '\x00' in child or '\\' in child
            or child.startswith('/') or Path(child).is_absolute()):
        raise PrivateArtifactError('private artifact child path is not relative')
    parts = tuple(child.split('/'))
    if any(part in {'', '.', '..'} for part in parts) or '/'.join(parts) != child:
        raise PrivateArtifactError('private artifact child path is not canonical')
    return parts


def _verify_private_stat(value: os.stat_result, *, directory: bool, label: str) -> None:
    expected_mode = 0o700 if directory else 0o600
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(value.st_mode):
        raise PrivateArtifactError(f'{label} is not a {"directory" if directory else "regular file"}')
    if value.st_uid != os.getuid():
        raise PrivateArtifactError(f'{label} is not owned by the current user')
    if stat.S_IMODE(value.st_mode) != expected_mode:
        raise PrivateArtifactError(f'{label} has wrong mode; expected {expected_mode:04o}')


def _require_private_io_capabilities() -> None:
    """Fail before private I/O if descriptor-confined durability is unavailable."""
    if any(type(getattr(os, name, None)) is not int or getattr(os, name) == 0
           for name in ('O_DIRECTORY', 'O_NOFOLLOW')):
        raise PrivateArtifactError('platform lacks required private I/O capabilities')
    if any(not callable(getattr(os, name, None))
           for name in ('open', 'mkdir', 'stat', 'unlink', 'replace', 'fsync', 'fstat')):
        raise PrivateArtifactError('platform lacks required private I/O capabilities')
    supports_dir_fd = getattr(os, 'supports_dir_fd', None)
    supports_follow_symlinks = getattr(os, 'supports_follow_symlinks', None)
    if not isinstance(supports_dir_fd, set) or any(function not in supports_dir_fd
           for function in (os.open, os.mkdir, os.stat, os.unlink)):
        raise PrivateArtifactError('platform lacks required dir_fd capabilities')
    if (not isinstance(supports_follow_symlinks, set)
            or os.stat not in supports_follow_symlinks):
        raise PrivateArtifactError('platform lacks no-follow stat capability')
    try:
        parameters = inspect.signature(os.replace).parameters
    except (TypeError, ValueError):
        raise PrivateArtifactError('platform lacks confined replace capability') from None
    if not {'src_dir_fd', 'dst_dir_fd'} <= set(parameters):
        raise PrivateArtifactError('platform lacks confined replace capability')
    probe_fd = -1
    try:
        probe_fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        os.fsync(probe_fd)
    except OSError:
        raise PrivateArtifactError('platform lacks directory fsync capability') from None
    finally:
        if probe_fd >= 0:
            os.close(probe_fd)


def _verify_trusted_ancestor(value: os.stat_result, *, label: str) -> None:
    """Accept safe root-owned/sticky ancestors or non-writable user-owned ones."""
    if not stat.S_ISDIR(value.st_mode):
        raise PrivateArtifactError(f'{label} is not a directory')
    mode = stat.S_IMODE(value.st_mode)
    if value.st_uid == 0 and (not mode & 0o022 or mode & stat.S_ISVTX):
        return
    if value.st_uid != os.getuid() or mode & 0o022:
        raise PrivateArtifactError(f'{label} has unsafe ownership or mode')


def _open_private_root(root: Path, *, create: bool) -> int:
    root = Path(root)
    if not root.is_absolute() or root == Path('/') or any(part in {'.', '..'} for part in root.parts[1:]):
        raise PrivateArtifactError('private root must be a canonical absolute path')
    retained: list[int] = []
    try:
        current = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        retained.append(current)
        _verify_trusted_ancestor(os.fstat(current), label='trusted anchor')
        parts = root.parts[1:]
        for index, name in enumerate(parts):
            final = index == len(parts) - 1
            try:
                before = os.stat(name, dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                if not (final and create):
                    raise
                os.mkdir(name, 0o700, dir_fd=current)
                os.fsync(current)
                before = os.stat(name, dir_fd=current, follow_symlinks=False)
            if final:
                _verify_private_stat(before, directory=True, label='private root')
            else:
                _verify_trusted_ancestor(before, label='private root ancestor')
            child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                               dir_fd=current)
            retained.append(child_fd)
            after = os.fstat(child_fd)
            if final:
                _verify_private_stat(after, directory=True, label='private root')
            else:
                _verify_trusted_ancestor(after, label='private root ancestor')
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise PrivateArtifactError('private root ancestry changed during open')
            for parent_fd, descendant_fd, component in zip(retained, retained[1:], parts):
                linked = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                opened = os.fstat(descendant_fd)
                if (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino):
                    raise PrivateArtifactError('private root ancestry changed during traversal')
            current = child_fd
        return os.dup(retained[-1])
    except OSError as exc:
        raise PrivateArtifactError(f'private root is unavailable: {type(exc).__name__}') from None
    finally:
        for fd in reversed(retained):
            os.close(fd)


def _open_private_parent(root_fd: int, parts: tuple[str, ...], *, create: bool) -> int:
    current = os.dup(root_fd)
    child_fd = -1
    try:
        for name in parts[:-1]:
            child_fd = -1
            created = False
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=current)
                    created = True
                    os.fsync(current)
                except FileExistsError:
                    pass
            before = os.stat(name, dir_fd=current, follow_symlinks=False)
            if not created:
                _verify_private_stat(before, directory=True, label='private artifact directory')
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            child_fd = os.open(name, flags, dir_fd=current)
            if created:
                os.fchmod(child_fd, 0o700)
            after = os.fstat(child_fd)
            _verify_private_stat(after, directory=True, label='private artifact directory')
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise PrivateArtifactError('private artifact directory identity changed')
            os.close(current)
            current = child_fd
            child_fd = -1
        return current
    except Exception:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(current)
        raise


def _read_confined_regular(path: Path, *, label: str) -> bytes:
    """Read an absolute regular file through a no-follow descriptor walk."""
    _require_private_io_capabilities()
    path = Path(path)
    if not path.is_absolute() or path == Path('/') or any(part in {'.', '..'} for part in path.parts[1:]):
        raise PrivateArtifactError(f'{label} must be a canonical absolute path')
    current = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in path.parts[1:-1]:
            before = os.stat(name, dir_fd=current, follow_symlinks=False)
            _verify_trusted_ancestor(before, label=f'{label} ancestor')
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            after = os.fstat(child)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                os.close(child)
                raise PrivateArtifactError(f'{label} ancestry changed during open')
            os.close(current)
            current = child
        before = os.stat(path.name, dir_fd=current, follow_symlinks=False)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
        try:
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise PrivateArtifactError(f'{label} identity changed during open')
            if not stat.S_ISREG(after.st_mode) or after.st_uid != os.getuid():
                raise PrivateArtifactError(f'{label} is not a trusted regular file')
            chunks = []
            while True:
                block = os.read(fd, 65536)
                if not block:
                    break
                chunks.append(block)
            linked = os.stat(path.name, dir_fd=current, follow_symlinks=False)
            if (linked.st_dev, linked.st_ino) != (after.st_dev, after.st_ino):
                raise PrivateArtifactError(f'{label} changed during read')
            return b''.join(chunks)
        finally:
            os.close(fd)
    except OSError as exc:
        raise PrivateArtifactError(f'{label} is unavailable: {type(exc).__name__}') from None
    finally:
        os.close(current)


def _json_bytes(payload: Any) -> bytes:
    try:
        return (json.dumps(_plain_json(payload), ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise PrivateArtifactError(f'private artifact is not canonical JSON: {type(exc).__name__}') from None


def private_json_write(root: Path, child: str, payload: Any) -> str:
    """Crash-durably replace one mode-0600 JSON artifact below a private root."""
    _require_private_io_capabilities()
    parts = _private_child_parts(child)
    data = _json_bytes(payload)
    root_fd = parent_fd = temp_fd = -1
    temp_name = ''
    try:
        root_fd = _open_private_root(root, create=True)
        parent_fd = _open_private_parent(root_fd, parts, create=True)
        leaf = parts[-1]
        try:
            existing = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            _verify_private_stat(existing, directory=False, label='private artifact')
        for _ in range(128):
            candidate = f'.{leaf}.{secrets.token_hex(16)}.tmp'
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                temp_fd = os.open(candidate, flags, 0o600, dir_fd=parent_fd)
                temp_name = candidate
                break
            except FileExistsError:
                continue
        else:
            raise PrivateArtifactError('could not allocate a private temporary file')
        _verify_private_stat(os.fstat(temp_fd), directory=False, label='private temporary file')
        view = memoryview(data)
        while view:
            written = os.write(temp_fd, view)
            if written <= 0:
                raise OSError('short private artifact write')
            view = view[written:]
        os.fsync(temp_fd)
        os.close(temp_fd)
        temp_fd = -1
        os.replace(temp_name, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temp_name = ''
        os.fsync(parent_fd)
        return hashlib.sha256(data).hexdigest()
    except PrivateArtifactError:
        raise
    except OSError as exc:
        raise PrivateArtifactError(f'private artifact write failed: {type(exc).__name__}') from None
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if temp_name and parent_fd >= 0:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        if parent_fd >= 0:
            os.close(parent_fd)
        if root_fd >= 0:
            os.close(root_fd)


def private_json_read(root: Path, child: str, *, max_bytes: int = 8 * 1024 * 1024,
                      expected_sha256: str | None = None) -> Any:
    """Read bounded JSON through verified directory/file descriptors only."""
    _require_private_io_capabilities()
    parts = _private_child_parts(child)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise PrivateArtifactError('private artifact maximum size is invalid')
    if expected_sha256 is not None and not _is_sha256(expected_sha256):
        raise PrivateArtifactError('expected SHA-256 is invalid')
    root_fd = parent_fd = file_fd = -1
    try:
        root_fd = _open_private_root(root, create=False)
        parent_fd = _open_private_parent(root_fd, parts, create=False)
        leaf = parts[-1]
        before = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        _verify_private_stat(before, directory=False, label='private artifact')
        file_fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        opened = os.fstat(file_fd)
        _verify_private_stat(opened, directory=False, label='private artifact')
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise PrivateArtifactError('private artifact identity changed during open')
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b''.join(chunks)
        if len(data) > max_bytes:
            raise PrivateArtifactError('private artifact exceeds maximum size')
        after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if ((after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
                or after.st_size != opened.st_size or after.st_mtime_ns != opened.st_mtime_ns):
            raise PrivateArtifactError('private artifact identity changed during read')
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise PrivateArtifactError('private artifact SHA-256 mismatch')
        try:
            def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, item in pairs:
                    if key in result:
                        raise ValueError('duplicate JSON object key')
                    result[key] = item
                return result

            value = json.loads(
                data, object_pairs_hook=unique_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f'non-finite JSON number: {token}')
                ),
            )
            if not isinstance(value, dict):
                raise ValueError('private state must be an object')
            return value
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise PrivateArtifactError('private artifact contains invalid JSON') from None
    except PrivateArtifactError:
        raise
    except OSError as exc:
        raise PrivateArtifactError(f'private artifact read failed: {type(exc).__name__}') from None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
        if root_fd >= 0:
            os.close(root_fd)


def provider_identity(remote: Any, container: str | None = None) -> dict[str, Any]:
    """Extract a complete, delete-safe provider identity."""
    row = dict(remote) if isinstance(remote, dict) else remote.model_dump()
    meta = row.get('metadata'); custom_id = _alias(row, 'custom_id', 'customId')
    ident = row.get('id'); tags = _alias(row, 'container_tags', 'containerTags')
    actual_container = container or row.get('_inventory_container')
    if actual_container is None and type(tags) is list and len(tags) == 1: actual_container = tags[0]
    sha256 = meta.get('content_sha256') if isinstance(meta, dict) else None
    byte_count = meta.get('content_bytes') if isinstance(meta, dict) else None
    if (type(custom_id) is not str or not custom_id or type(ident) is not str or not ident
            or actual_container not in CANONICAL_CONTAINERS or not _is_sha256(sha256)
            or type(byte_count) is not int or byte_count < 0):
        raise ReconciliationRequired('provider identity is incomplete')
    if not isinstance(meta, dict):
        raise ReconciliationRequired('provider identity metadata is incomplete')
    return {'custom_id': custom_id, 'container': actual_container, 'sha256': sha256,
            'bytes': byte_count, 'document_id': ident}


class DurableReconciliationExecutor:
    """Execute canonical plans with a strictly validated durable journal.

    Supermemory does not expose compare-and-delete.  The final hydration and
    delete are deliberately adjacent (no callback, checkpoint, inventory, or
    other provider call between them), minimizing but not eliminating the
    provider-side TOCTOU window.
    """
    def __init__(self, client: Supermemory, private_root: Path,
                 sleep: Callable[[float], None] = time.sleep, poll_limit: int = 360,
                 fault_injector: Callable[[str], None] | None = None,
                 poll_attempts: int | None = None):
        self.client, self.private_root = client, Path(private_root)
        self.sleep = sleep
        self.poll_limit = poll_attempts if poll_attempts is not None else poll_limit
        self.fault_injector = fault_injector
        self.snapshot_path = self.private_root / 'transactions/unbound/snapshot.json'
        self.journal_path = self.private_root / 'transactions/unbound/forward-journal.json'
        self._snapshot_child = ''
        self._journal_child = ''
        self._plan: tuple[dict[str, Any], ...] = ()
        self._transaction_id = ''
        self._snapshot_digest: str | None = None

    def _fault(self, point: str) -> None:
        if self.fault_injector:
            self.fault_injector(point)

    def _checkpoint(self, journal: Any) -> dict[str, Any]:
        validated = validate_transaction_journal(
            journal, self._plan, expected_transaction_id=self._transaction_id,
            expected_snapshot_digest=self._snapshot_digest,
        )
        state = thaw_transaction_journal(validated)
        self._fault('before_journal_write')
        digest = private_json_write(self.private_root, self._journal_child, state)
        self._fault('after_journal_write')
        persisted = private_json_read(
            self.private_root, self._journal_child, expected_sha256=digest,
        )
        validate_transaction_journal(
            persisted, self._plan, expected_transaction_id=self._transaction_id,
            expected_snapshot_digest=self._snapshot_digest,
        )
        self._fault('journal_written')
        return state

    def _advance(self, state: dict[str, Any], index: int, stage: str) -> dict[str, Any]:
        advanced = advance_transaction_journal(
            state, self._plan, index, stage,
            expected_transaction_id=self._transaction_id,
            expected_snapshot_digest=self._snapshot_digest,
        )
        return self._checkpoint(advanced)

    def _inventory(self) -> dict[str, tuple[dict[str, Any], ...]]:
        self._fault('before_inventory')
        value = provider_inventory(self.client)
        for rows in value.values():
            for row in rows:
                _require_superrag_task(row)
        self._fault('after_inventory')
        return value

    def _matches(self, custom_id: str) -> list[dict[str, Any]]:
        found = []
        for container, rows in self._inventory().items():
            for raw in rows:
                if _alias(raw, 'custom_id', 'customId') == custom_id:
                    row = dict(raw)
                    row['_inventory_container'] = container
                    found.append(row)
        return found

    def _hydrate(self, identity: dict[str, Any]) -> dict[str, Any]:
        self._fault('before_get')
        try:
            remote = self.client.documents.get(identity['document_id'], timeout=15.0)
        except Exception:
            raise ReconciliationRequired('provider hydration failed') from None
        self._fault('after_get')
        return self._validate_installed(remote, identity)

    def _validate_installed(self, remote: Any, identity: dict[str, Any]) -> dict[str, Any]:
        row = dict(remote) if isinstance(remote, dict) else remote.model_dump()
        row['_inventory_container'] = identity['container']
        _require_superrag_task(row)
        if not _exact_json_equal(provider_identity(row, identity['container']), identity):
            raise ReconciliationRequired('installed identity changed')
        meta, content = row.get('metadata'), row.get('content')
        if not isinstance(meta, dict):
            raise ReconciliationRequired('installed provider metadata is invalid')
        rel = meta.get('relative_path')
        required = {
            'source': 'obsidian', 'authority': 'canonical', 'relative_path': rel,
            'index_schema_version': INDEX_SCHEMA_VERSION,
            'visibility': canonical_visibility_from_path(rel),
            'identity_scope': 'owner', 'canonical_root': 'owner',
        }
        if (type(content) is not str
                or destination_for_path(rel) != identity['container']
                or stable_custom_id_from_path(rel) != identity['custom_id']
                or status_name(row) != 'done'
                or any(meta.get(key) != value for key, value in required.items())
                or not _explicit_container_claim_agrees(row, identity['container'])):
            raise ReconciliationRequired('installed provider object is invalid')
        return row

    def _validate_post(self, remote: Any, doc: dict[str, Any], container: str,
                       expected: dict[str, Any]) -> dict[str, Any]:
        row = dict(remote) if isinstance(remote, dict) else remote.model_dump()
        row['_inventory_container'] = container
        _require_superrag_task(row)
        actual = provider_identity(row, container)
        expected_actual = dict(expected)
        expected_actual['document_id'] = actual['document_id']
        if not _exact_json_equal(actual, expected_actual):
            raise ReconciliationRequired('installed post identity changed')
        validate_backend_document(row, doc)
        content = row.get('content')
        if (type(content) is not str
                or content.rstrip(' \t\r\n') != doc['content'].rstrip(' \t\r\n')):
            raise ReconciliationRequired('provider returned content is inconsistent')
        return row

    def _snapshot(self, records: list[dict[str, Any]],
                  pre_inventory: list[dict[str, Any]]) -> str:
        payload = {
            'snapshot_schema_version': 1, 'transaction_id': self._transaction_id,
            'plan_digest': transaction_plan_digest(self._plan),
            'label': 'provider-returned logical snapshot; not raw source bytes',
            'count': len(records), 'records': records,
            'expected_pre_inventory': pre_inventory,
            'expected_pre_inventory_digest': _canonical_digest(pre_inventory),
            'container_counts': {c: sum(r['container'] == c for r in records)
                                 for c in sorted(CANONICAL_CONTAINERS)},
        }
        self._fault('before_snapshot_write')
        digest = private_json_write(self.private_root, self._snapshot_child, payload)
        self._fault('after_snapshot_write')
        if not _exact_json_equal(private_json_read(
                self.private_root, self._snapshot_child, expected_sha256=digest), payload):
            raise ReconciliationRequired('snapshot read-back mismatch')
        self._fault('snapshot_verified')
        return digest

    def _add(self, state: dict[str, Any], index: int, doc: dict[str, Any]) -> dict[str, Any]:
        record = state['records'][index]
        expected = record['expected_post_identity']
        matches = self._matches(record['custom_id'])
        if len(matches) > 1:
            raise ReconciliationRequired('ambiguous duplicate stable identity')
        if not matches:
            self._fault('before_add')
            try:
                self.client.documents.add(
                    content=doc['content'], container_tag=record['target_container'],
                    custom_id=record['custom_id'], task_type='superrag',
                    metadata=metadata(doc), timeout=30.0,
                )
            except Exception:
                matches = self._matches(record['custom_id'])
                if len(matches) != 1:
                    return self._advance(state, index, 'reconcile_required')
                state = self._advance(state, index, 'add_submitted')
            else:
                self._fault('after_add')
                state = self._advance(state, index, 'add_submitted')
        elif state['records'][index]['stage'] in {'existing', 'deleted', 'reconcile_required'}:
            state = self._advance(state, index, 'add_submitted')
        for _ in range(self.poll_limit):
            matches = self._matches(record['custom_id'])
            if len(matches) == 1:
                row = matches[0]
                ident = str(row.get('id') or '')
                self._fault('before_get')
                try:
                    hydrated = self.client.documents.get(ident, timeout=15.0)
                except Exception:
                    return self._advance(state, index, 'reconcile_required')
                self._fault('after_get')
                status = status_name(hydrated)
                if status in {'failed', 'error'}:
                    return self._advance(state, index, 'reconcile_required')
                if status == 'done':
                    self._validate_post(hydrated, doc, record['target_container'], expected)
                    return self._advance(state, index, 'done')
                if status == 'processing':
                    if state['records'][index]['stage'] != 'processing':
                        state = self._advance(state, index, 'processing')
                else:
                    return self._advance(state, index, 'reconcile_required')
            self.sleep(5)
        return self._advance(state, index, 'reconcile_required')

    def _verify(self, current: dict[str, dict[str, Any]]) -> dict[str, Any]:
        inventory = self._inventory()
        classification = classify_inventory(current, inventory)
        blockers = {key: value for key, value in classification.items()
                    if key != 'expected' and value}
        rows = [row for container in sorted(inventory) for row in inventory[container]]
        if (blockers or classification['expected'] != tuple(sorted(current))
                or len(rows) != len(current)):
            raise ReconciliationRequired('exhaustive post-state proof failed')
        proofs = []
        post_by_id = {r['custom_id']: r['expected_post_identity'] for r in self._plan
                      if r['expected_post_identity'] is not None}
        for rel, doc in sorted(current.items()):
            matches = [(container, row) for container in sorted(inventory)
                       for row in inventory[container]
                       if _alias(row, 'custom_id', 'customId') == doc['custom_id']]
            if len(matches) != 1:
                raise ReconciliationRequired('post-state identity count is not one')
            container, row = matches[0]
            remote = self.client.documents.get(str(row.get('id')), timeout=15.0)
            expected = post_by_id.get(doc['custom_id'], {
                'custom_id': doc['custom_id'], 'container': container,
                'sha256': doc['sha256'], 'bytes': doc['bytes'], 'document_id': None,
            })
            validated = self._validate_post(remote, doc, container, expected)
            proofs.append(provider_identity(validated, container))
        return {'digest': _canonical_digest(proofs), 'count': len(proofs),
                'container_counts': {c: sum(p['container'] == c for p in proofs)
                                     for c in sorted(CANONICAL_CONTAINERS)}}

    def _validate_current(self, current: dict[str, dict[str, Any]]) -> None:
        for record in self._plan:
            doc = current.get(record['relative_path'])
            if record['action'] == 'delete':
                if doc is not None:
                    raise ReconciliationRequired('delete target remains in canonical projection')
                continue
            if not isinstance(doc, dict):
                raise ReconciliationRequired('add target is absent from canonical projection')
            expected = record['expected_post_identity']
            if (doc.get('custom_id') != record['custom_id']
                    or doc.get('sha256') != record['expected_sha256']
                    or doc.get('bytes') != record['expected_bytes']
                    or container_for_doc(doc) != record['target_container']
                    or expected['sha256'] != doc['sha256']
                    or expected['bytes'] != doc['bytes']):
                raise ReconciliationRequired('canonical projection does not match transaction plan')

    def execute(self, current: dict[str, dict[str, Any]], plan: Any,
                transaction_id: str = 'forward') -> dict[str, Any]:
        self._plan = _validated_transaction_plan(plan)
        self._transaction_id = transaction_id
        _validate_transaction_id(transaction_id)
        prefix = f'transactions/{transaction_id}'
        self._snapshot_child = f'{prefix}/snapshot.json'
        self._journal_child = f'{prefix}/forward-journal.json'
        self.snapshot_path = self.private_root / self._snapshot_child
        self.journal_path = self.private_root / self._journal_child
        self._validate_current(current)
        journal_exists = self.journal_path.is_file()
        if not journal_exists:
            # Capture the complete provider-returned logical pre-state before any
            # mutation.  Rollback must prove the whole two-container inventory,
            # not merely reconstruct state from the old manifest.
            pre_inventory = []
            for container, rows in self._inventory().items():
                for listed in rows:
                    identity = provider_identity(listed, container)
                    remote = self._hydrate(identity)
                    pre_inventory.append({
                        'content': remote['content'], 'metadata': remote['metadata'],
                        'custom_id': identity['custom_id'],
                        'backend_identity': identity['document_id'],
                        'container': container, 'sha256': identity['sha256'],
                        'bytes': identity['bytes'], 'status': 'done',
                        'task_type': 'superrag',
                    })
            pre_inventory.sort(key=lambda row: (row['container'], row['custom_id'],
                                                row['backend_identity']))
            snapshots = []
            for order, record in enumerate(self._plan):
                identity = record['expected_pre_identity']
                if identity is None:
                    continue
                remote = self._hydrate(identity)
                snapshots.append({
                    'order': order, 'relative_path': record['relative_path'],
                    'content': remote['content'], 'metadata': remote['metadata'],
                    'custom_id': identity['custom_id'],
                    'backend_identity': identity['document_id'],
                    'container': identity['container'], 'sha256': identity['sha256'],
                    'bytes': identity['bytes'], 'task_type': 'superrag',
                })
            self._snapshot_digest = self._snapshot(snapshots, pre_inventory)
            state = self._checkpoint(new_transaction_journal(
                transaction_id, self._plan, snapshot_digest=self._snapshot_digest,
            ))
        else:
            raw = private_json_read(self.private_root, self._journal_child)
            snapshot_digest = raw.get('snapshot_digest') if isinstance(raw, dict) else None
            if not _is_sha256(snapshot_digest):
                raise TransactionJournalError('transaction snapshot digest mismatch')
            self._snapshot_digest = snapshot_digest
            validate_transaction_journal(
                raw, self._plan, expected_transaction_id=transaction_id,
                expected_snapshot_digest=snapshot_digest,
            )
            private_json_read(
                self.private_root, self._snapshot_child, expected_sha256=snapshot_digest,
            )
            state = thaw_transaction_journal(validate_transaction_journal(
                raw, self._plan, expected_transaction_id=transaction_id,
                expected_snapshot_digest=snapshot_digest,
            ))

        for index in range(len(state['records'])):
            record = state['records'][index]
            stage, action = record['stage'], record['action']
            if stage == 'done':
                continue
            destructive_reconcile = (
                stage == 'reconcile_required' and 'deleted' not in record['history']
            )
            if action in {'replace', 'delete'} and (
                    stage in {'existing', 'delete_verified', 'delete_submitted'}
                    or destructive_reconcile):
                identity = record['expected_pre_identity']
                matches = self._matches(record['custom_id'])
                exact = []
                malformed = False
                for match in matches:
                    try:
                        if _exact_json_equal(
                                provider_identity(match, match.get('_inventory_container')),
                                identity):
                            exact.append(match)
                        else:
                            malformed = True
                    except ReconciliationRequired:
                        malformed = True
                if not matches:
                    state = self._advance(state, index, 'deleted')
                elif len(matches) != 1 or len(exact) != 1 or malformed:
                    if stage != 'reconcile_required':
                        state = self._advance(state, index, 'reconcile_required')
                    raise ReconciliationRequired('destructive identity is ambiguous')
                else:
                    if stage in {'existing', 'reconcile_required'}:
                        state = self._advance(state, index, 'delete_verified')
                    if state['records'][index]['stage'] == 'delete_verified':
                        state = self._advance(state, index, 'delete_submitted')
                    self._fault('before_delete')
                    self._fault('before_delete_validation')
                    # Final provider operation before delete: no local callback or
                    # durable checkpoint is permitted in this critical section.
                    self._validate_installed(
                        self.client.documents.get(identity['document_id'], timeout=15.0), identity,
                    )
                    try:
                        self.client.documents.delete(identity['document_id'], timeout=30.0)
                    except Exception:
                        matches = self._matches(record['custom_id'])
                        if not matches:
                            state = self._advance(state, index, 'deleted')
                        else:
                            state = self._advance(state, index, 'reconcile_required')
                            raise ReconciliationRequired('delete response is unknown') from None
                    else:
                        self._fault('after_delete')
                        state = self._advance(state, index, 'deleted')
            if action == 'delete':
                if state['records'][index]['stage'] == 'deleted':
                    state = self._advance(state, index, 'done')
                continue
            state = self._add(state, index, current[record['relative_path']])
            if state['records'][index]['stage'] == 'reconcile_required':
                raise ReconciliationRequired('add outcome requires reconciliation')

        if any(record['stage'] != 'done' for record in state['records']):
            raise ReconciliationRequired('transaction is not terminal')
        proof = self._verify(current)
        return state | {'complete': True, 'verification_digest': proof['digest'],
                        'verified_count': proof['count'],
                        'verified_container_counts': proof['container_counts']}

ROLLBACK_STAGES = frozenset({
    'existing', 'remove_verified', 'remove_submitted', 'removed', 'restore_submitted',
    'processing', 'done', 'reconcile_required',
})
ROLLBACK_TRANSITIONS = {
    'remove': {
        'existing': {'remove_verified', 'removed', 'reconcile_required'},
        'remove_verified': {'remove_submitted', 'reconcile_required'},
        'remove_submitted': {'removed', 'reconcile_required'},
        'reconcile_required': {'remove_verified', 'removed', 'reconcile_required'},
        'removed': {'done'}, 'done': set(),
    },
    'restore': {
        'existing': {'restore_submitted', 'reconcile_required'},
        'restore_submitted': {'processing', 'done', 'reconcile_required'},
        'processing': {'done', 'reconcile_required'},
        'reconcile_required': {'restore_submitted', 'processing', 'done', 'reconcile_required'},
        'done': set(),
    },
    'replace': {
        'existing': {'remove_verified', 'removed', 'reconcile_required'},
        'remove_verified': {'remove_submitted', 'reconcile_required'},
        'remove_submitted': {'removed', 'reconcile_required'},
        'removed': {'restore_submitted', 'reconcile_required'},
        'restore_submitted': {'processing', 'done', 'reconcile_required'},
        'processing': {'done', 'reconcile_required'},
        'reconcile_required': {'remove_verified', 'removed', 'restore_submitted',
                               'processing', 'done', 'reconcile_required'},
        'done': set(),
    },
}


def _snapshot_logical_record(row: Any) -> dict[str, Any]:
    """Validate and return a privacy-sensitive provider logical snapshot row."""
    if not isinstance(row, dict) or set(row) != {
            'content', 'metadata', 'custom_id', 'backend_identity', 'container',
            'sha256', 'bytes', 'status', 'task_type'}:
        raise TransactionJournalError('rollback snapshot record fields are invalid')
    content, metadata_value = row['content'], row['metadata']
    if (type(content) is not str or not isinstance(metadata_value, dict)
            or type(row['custom_id']) is not str or not row['custom_id']
            or type(row['backend_identity']) is not str or not row['backend_identity']
            or row['container'] not in CANONICAL_CONTAINERS
            or not _is_sha256(row['sha256']) or type(row['bytes']) is not int
            or row['bytes'] < 0 or row['status'] != 'done' or row['task_type'] != 'superrag'
            or metadata_value.get('content_sha256') != row['sha256']
            or metadata_value.get('content_bytes') != row['bytes']):
        raise TransactionJournalError('rollback snapshot record is invalid')
    marker = '[/canonical-identity]\n\n'
    if content.count(marker) != 1:
        raise TransactionJournalError('rollback snapshot content envelope is invalid')
    logical_content = content.split(marker, 1)[1]
    candidates = (logical_content, logical_content.rstrip(' \t\r\n'))
    if not any(hashlib.sha256(value.encode('utf-8')).hexdigest() == row['sha256']
               and len(value.encode('utf-8')) == row['bytes'] for value in candidates):
        raise TransactionJournalError('rollback snapshot content digest is inconsistent')
    rel = metadata_value.get('relative_path')
    required = {'source': 'obsidian', 'authority': 'canonical',
                'index_schema_version': INDEX_SCHEMA_VERSION, 'identity_scope': 'owner',
                'canonical_root': 'owner'}
    if (destination_for_path(rel) != row['container']
            or stable_custom_id_from_path(rel) != row['custom_id']
            or metadata_value.get('visibility') != canonical_visibility_from_path(rel)
            or any(metadata_value.get(key) != value for key, value in required.items())):
        raise TransactionJournalError('rollback snapshot metadata is unsafe')
    return _plain_json(row)


def _rollback_plan(forward_plan: Any, snapshot: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    plan = _validated_transaction_plan(forward_plan)
    snapshots = snapshot.get('records')
    if not isinstance(snapshots, list):
        raise TransactionJournalError('rollback snapshot records are invalid')
    by_order: dict[int, dict[str, Any]] = {}
    for row in snapshots:
        if not isinstance(row, dict) or type(row.get('order')) is not int:
            raise TransactionJournalError('rollback mutation snapshot is invalid')
        logical = {key: value for key, value in row.items() if key != 'order' and key != 'relative_path'}
        logical['status'] = 'done'
        validated = _snapshot_logical_record(logical)
        if row['order'] in by_order:
            raise TransactionJournalError('duplicate rollback snapshot order')
        by_order[row['order']] = validated
    inverse = []
    for reverse_order, (forward_order, record) in enumerate(reversed(tuple(enumerate(plan)))):
        action = {'add': 'remove', 'delete': 'restore', 'replace': 'replace'}[record['action']]
        old = by_order.get(forward_order)
        if action in {'restore', 'replace'} and old is None:
            raise TransactionJournalError('rollback restore snapshot is missing')
        inverse.append({'order': reverse_order, 'forward_order': forward_order,
                        'action': action, 'custom_id': record['custom_id'],
                        'expected_forward_identity': record['expected_post_identity'],
                        'restore_snapshot': old})
    return tuple(inverse)


def _project_forward_post_inventory(
        forward_plan: Any, expected_pre_inventory: Any) -> tuple[dict[str, Any], ...]:
    """Purely project the logical forward post-state from immutable inputs."""
    plan = _validated_transaction_plan(forward_plan)
    if not isinstance(expected_pre_inventory, (list, tuple)):
        raise TransactionJournalError('forward pre-inventory is invalid')
    projected: dict[str, dict[str, Any]] = {}
    for raw in expected_pre_inventory:
        row = _snapshot_logical_record(raw)
        if row['custom_id'] in projected:
            raise TransactionJournalError('forward pre-inventory contains duplicate identity')
        projected[row['custom_id']] = {
            'custom_id': row['custom_id'], 'container': row['container'],
            'sha256': row['sha256'], 'bytes': row['bytes'],
            'document_id': row['backend_identity'],
        }
    for record in plan:
        if record['action'] == 'delete':
            projected.pop(record['custom_id'], None)
        else:
            # Provider-assigned ids are deliberately not authority.  Forward
            # execution validates this same canonical identity with the actual
            # id substituted only for the provider read being checked.
            projected[record['custom_id']] = _plain_json(record['expected_post_identity'])
    rows = tuple(sorted(projected.values(), key=lambda row: (
        row['container'], row['custom_id'], row['document_id'] or '',
    )))
    if len({(row['container'], row['custom_id']) for row in rows}) != len(rows):
        raise TransactionJournalError('projected forward post-inventory is ambiguous')
    return rows


def validate_rollback_journal(journal: Any, rollback_plan: Any, *, rollback_transaction_id: str,
                              forward_transaction_id: str, forward_plan_digest: str,
                              forward_journal_digest: str, snapshot_digest: str,
                              pre_inventory_digest: str, post_inventory_digest: str,
                              expected_post_inventory: Any) -> MappingProxyType:
    _validate_transaction_id(rollback_transaction_id, label='rollback transaction id')
    _validate_transaction_id(forward_transaction_id, label='forward transaction id')
    for value in (forward_plan_digest, forward_journal_digest, snapshot_digest,
                  pre_inventory_digest, post_inventory_digest):
        if not _is_sha256(value):
            raise TransactionJournalError('rollback binding digest is invalid')
    if (not isinstance(expected_post_inventory, (list, tuple))
            or _canonical_digest(expected_post_inventory) != post_inventory_digest):
        raise TransactionJournalError('rollback expected post-inventory is invalid')
    value = thaw_transaction_journal(journal)
    fields = {'journal_schema_version', 'rollback_transaction_id', 'forward_transaction_id',
              'forward_plan_digest', 'forward_journal_digest', 'snapshot_digest',
              'pre_inventory_digest', 'post_inventory_digest', 'rollback_plan_digest',
              'expected_post_inventory', 'records', 'verification_digest', 'verified_count',
              'verified_container_counts'}
    if set(value) != fields or value['journal_schema_version'] != 1:
        raise TransactionJournalError('rollback journal fields are invalid')
    expected_header = {
        'rollback_transaction_id': rollback_transaction_id,
        'forward_transaction_id': forward_transaction_id,
        'forward_plan_digest': forward_plan_digest, 'forward_journal_digest': forward_journal_digest,
        'snapshot_digest': snapshot_digest, 'pre_inventory_digest': pre_inventory_digest,
        'post_inventory_digest': post_inventory_digest,
        'expected_post_inventory': _plain_json(expected_post_inventory),
        'rollback_plan_digest': _canonical_digest(rollback_plan),
    }
    if any(value.get(key) != expected for key, expected in expected_header.items()):
        raise TransactionJournalError('rollback journal binding mismatch')
    rows = value['records']
    if not isinstance(rows, list) or len(rows) != len(rollback_plan):
        raise TransactionJournalError('rollback journal record count mismatch')
    runtime = {'stage', 'history', 'restored_document_id'}
    for expected, row in zip(rollback_plan, rows, strict=True):
        if (not isinstance(row, dict) or set(row) != set(expected) | runtime
                or not _exact_json_equal({key: row[key] for key in expected}, expected)
                or (row['restored_document_id'] is not None
                    and (type(row['restored_document_id']) is not str or not row['restored_document_id']))):
            raise TransactionJournalError('rollback journal record mismatch')
        history = row['history']; stage = row['stage']
        if (not isinstance(history, list) or not history or history[0] != 'existing'
                or history[-1] != stage or any(item not in ROLLBACK_STAGES for item in history)
                or any(right not in ROLLBACK_TRANSITIONS[row['action']].get(left, set())
                       for left, right in zip(history, history[1:]))):
            raise TransactionJournalError('rollback journal transition is invalid')
    complete = all(row['stage'] == 'done' for row in rows)
    if complete:
        if (not _is_sha256(value['verification_digest'])
                or type(value['verified_count']) is not int or value['verified_count'] < 0
                or not isinstance(value['verified_container_counts'], dict)
                or set(value['verified_container_counts']) != CANONICAL_CONTAINERS
                or any(type(count) is not int or count < 0
                       for count in value['verified_container_counts'].values())):
            raise TransactionJournalError('rollback terminal verification is invalid')
    elif any(value[key] is not None for key in ('verification_digest', 'verified_count',
                                                'verified_container_counts')):
        raise TransactionJournalError('nonterminal rollback journal has verification proof')
    return _freeze_json(value)


class DurableRollbackExecutor:
    """Resume an inverse transaction from provider truth and prove exact pre-state."""
    def __init__(self, client: Supermemory, private_root: Path,
                 sleep: Callable[[float], None] = time.sleep, poll_limit: int = 360,
                 fault_injector: Callable[[str], None] | None = None,
                 poll_attempts: int | None = None):
        self.client, self.private_root = client, Path(private_root)
        self.sleep = sleep
        self.poll_limit = poll_attempts if poll_attempts is not None else poll_limit
        self.fault_injector = fault_injector
        self.bindings: dict[str, Any] = {}
        self.plan: tuple[dict[str, Any], ...] = ()
        self.expected_pre: list[dict[str, Any]] = []
        self.expected_post: dict[str, dict[str, Any]] = {}
        self._rollback_child = ''

    def _fault(self, point: str) -> None:
        if self.fault_injector:
            self.fault_injector(point)

    def _inventory(self) -> dict[str, tuple[dict[str, Any], ...]]:
        self._fault('before_inventory')
        try:
            result = provider_inventory(self.client)
            for rows in result.values():
                for row in rows: _require_superrag_task(row)
        except Exception:
            raise ReconciliationRequired('rollback provider inventory is unavailable') from None
        self._fault('after_inventory')
        return result

    def _matches(self, custom_id: str) -> list[dict[str, Any]]:
        result = []
        for container, rows in self._inventory().items():
            for raw in rows:
                if _alias(raw, 'custom_id', 'customId') == custom_id:
                    row = dict(raw); row['_inventory_container'] = container; result.append(row)
        return result

    def _write(self, state: dict[str, Any]) -> dict[str, Any]:
        validate_rollback_journal(state, self.plan, **self.bindings)
        self._fault('before_rollback_journal_write')
        digest = private_json_write(self.private_root, self._rollback_child, state)
        self._fault('after_rollback_journal_write')
        persisted = private_json_read(self.private_root, self._rollback_child, expected_sha256=digest)
        validate_rollback_journal(persisted, self.plan, **self.bindings)
        return persisted

    def _advance(self, state: dict[str, Any], index: int, stage: str,
                 restored_document_id: str | None = None) -> dict[str, Any]:
        state = _plain_json(state); row = state['records'][index]
        if stage not in ROLLBACK_TRANSITIONS[row['action']].get(row['stage'], set()):
            raise TransactionJournalError('rollback journal transition is invalid')
        row['stage'] = stage; row['history'].append(stage)
        if restored_document_id is not None: row['restored_document_id'] = restored_document_id
        if all(record['stage'] == 'done' for record in state['records']):
            proof = self._proof(self.expected_pre)
            state['verification_digest'] = proof['digest']
            state['verified_count'] = proof['count']
            state['verified_container_counts'] = proof['container_counts']
        return self._write(state)

    def _hydrated(self, match: dict[str, Any]) -> dict[str, Any]:
        ident = str(match.get('id') or '')
        self._fault('before_get')
        try:
            remote = self.client.documents.get(ident, timeout=15.0)
            if isinstance(remote, dict):
                row = dict(remote)
            else:
                model_dump = getattr(remote, 'model_dump', None)
                if not callable(model_dump):
                    raise TypeError
                row = model_dump()
                if not isinstance(row, dict):
                    raise TypeError
                row = dict(row)
            for snake, camel in (('custom_id', 'customId'),
                                  ('container_tags', 'containerTags'),
                                  ('task_type', 'taskType')):
                value = _alias(row, snake, camel)
                row[snake] = value
                row.pop(camel, None)
            row['_inventory_container'] = match['_inventory_container']
            if (type(row.get('id')) is not str or not row['id']
                    or type(row.get('custom_id')) is not str or not row['custom_id']
                    or status_name(row) not in {'processing', 'done'}
                    or row.get('task_type') != 'superrag'
                    or not isinstance(row.get('metadata'), dict)
                    or type(row.get('content')) is not str
                    or type(row.get('container_tags')) is not list
                    or len(row['container_tags']) != 1
                    or row['container_tags'][0] not in CANONICAL_CONTAINERS):
                raise ValueError
            _require_superrag_task(row)
            self._fault('after_get')
            return row
        except Exception:
            raise ReconciliationRequired('rollback hydration failed') from None

    def _logical_matches(self, row: dict[str, Any], snapshot: dict[str, Any],
                         *, require_backend_id: bool) -> bool:
        try:
            actual = provider_identity(row, row['_inventory_container'])
        except ReconciliationRequired:
            return False
        expected = {'custom_id': snapshot['custom_id'], 'container': snapshot['container'],
                    'sha256': snapshot['sha256'], 'bytes': snapshot['bytes'],
                    'document_id': snapshot['backend_identity'] if require_backend_id
                                   else actual['document_id']}
        return (_exact_json_equal(actual, expected) and status_name(row) == 'done'
                and row.get('content') == snapshot['content']
                and _exact_json_equal(row.get('metadata'), snapshot['metadata']))

    def _forward_matches(self, row: dict[str, Any], expected: dict[str, Any]) -> bool:
        actual = provider_identity(row, row['_inventory_container'])
        anchored = dict(expected)
        if anchored['document_id'] is None:
            anchored['document_id'] = actual['document_id']
        if not _exact_json_equal(actual, anchored) or status_name(row) != 'done': return False
        meta, content = row.get('metadata'), row.get('content')
        marker = '[/canonical-identity]\n\n'
        if not isinstance(meta, dict) or type(content) is not str or content.count(marker) != 1:
            return False
        rel = meta.get('relative_path')
        required = {'source': 'obsidian', 'authority': 'canonical',
                    'index_schema_version': INDEX_SCHEMA_VERSION,
                    'visibility': canonical_visibility_from_path(rel),
                    'identity_scope': 'owner', 'canonical_root': 'owner',
                    'content_sha256': expected['sha256'], 'content_bytes': expected['bytes']}
        if (destination_for_path(rel) != expected['container']
                or stable_custom_id_from_path(rel) != expected['custom_id']
                or any(meta.get(key) != value for key, value in required.items())):
            return False
        logical_content = content.split(marker, 1)[1]
        return (hashlib.sha256(logical_content.encode()).hexdigest() == expected['sha256']
                and len(logical_content.encode()) == expected['bytes'])

    def _remove(self, state: dict[str, Any], index: int) -> dict[str, Any]:
        record = state['records'][index]
        expected = self.expected_post.get(record['custom_id'])
        matches = self._matches(record['custom_id'])
        if not matches:
            if record['stage'] in {'existing', 'reconcile_required'}:
                return self._advance(state, index, 'removed')
            return self._advance(state, index, 'removed')
        if len(matches) != 1 or expected is None:
            raise ReconciliationRequired('rollback delete identity is ambiguous')
        hydrated = self._hydrated(matches[0])
        if not self._forward_matches(hydrated, expected):
            raise ReconciliationRequired('rollback delete target changed')
        if record['stage'] in {'existing', 'reconcile_required'}:
            state = self._advance(state, index, 'remove_verified')
        if state['records'][index]['stage'] == 'remove_verified':
            state = self._advance(state, index, 'remove_submitted')
        # Immediate fresh hydration is the final provider call before delete.
        fresh = self._hydrated(matches[0])
        if not self._forward_matches(fresh, expected):
            self._advance(state, index, 'reconcile_required')
            raise ReconciliationRequired('rollback delete target changed')
        self._fault('before_delete')
        try: self.client.documents.delete(str(matches[0]['id']), timeout=30.0)
        except Exception:
            state = self._advance(state, index, 'reconcile_required')
            raise ReconciliationRequired('rollback delete response is unknown') from None
        self._fault('after_delete')
        state = self._advance(state, index, 'removed')
        if self._matches(record['custom_id']):
            state = self._advance(state, index, 'reconcile_required')
            raise ReconciliationRequired('rollback delete absence is unproven')
        return state

    def _restore(self, state: dict[str, Any], index: int) -> dict[str, Any]:
        record = state['records'][index]; snapshot = record['restore_snapshot']
        matches = self._matches(record['custom_id'])
        if len(matches) > 1:
            raise ReconciliationRequired('duplicate rollback restore identity')
        if matches:
            hydrated = self._hydrated(matches[0])
            if not self._logical_matches(hydrated, snapshot, require_backend_id=False):
                raise ReconciliationRequired('rollback restore identity is occupied')
            if record['stage'] == 'existing':
                state = self._advance(state, index, 'restore_submitted')
            return self._advance(state, index, 'done', str(matches[0]['id']))
        self._fault('before_add')
        try:
            self.client.documents.add(content=snapshot['content'], container_tag=snapshot['container'],
                custom_id=snapshot['custom_id'], task_type=snapshot['task_type'],
                metadata=snapshot['metadata'], timeout=30.0)
        except Exception:
            state = self._advance(state, index, 'reconcile_required')
            raise ReconciliationRequired('rollback add response is unknown') from None
        self._fault('after_add')
        state = self._advance(state, index, 'restore_submitted')
        for _ in range(self.poll_limit):
            matches = self._matches(record['custom_id'])
            if len(matches) == 1:
                try:
                    hydrated = self._hydrated(matches[0])
                except ReconciliationRequired:
                    self._advance(state, index, 'reconcile_required')
                    raise
                status = status_name(hydrated)
                if status == 'done' and self._logical_matches(hydrated, snapshot, require_backend_id=False):
                    return self._advance(state, index, 'done', str(matches[0]['id']))
                if status == 'processing':
                    if state['records'][index]['stage'] != 'processing':
                        state = self._advance(state, index, 'processing')
                else:
                    state = self._advance(state, index, 'reconcile_required')
                    raise ReconciliationRequired('rollback restore failed validation')
            elif len(matches) > 1:
                state = self._advance(state, index, 'reconcile_required')
                raise ReconciliationRequired('duplicate rollback restore identity')
            self.sleep(5)
        state = self._advance(state, index, 'reconcile_required')
        raise ReconciliationRequired('rollback restore polling timed out')

    def _proof(self, expected_pre: list[dict[str, Any]]) -> dict[str, Any]:
        inventory = self._inventory()
        actual_rows = [(container, row) for container in sorted(inventory)
                       for row in inventory[container]]
        if len(actual_rows) != len(expected_pre):
            raise ReconciliationRequired('exact rollback pre-state count mismatch')
        expected_by_key = {(row['container'], row['custom_id']): row for row in expected_pre}
        if len(expected_by_key) != len(expected_pre):
            raise TransactionJournalError('rollback pre-state contains duplicate identity')
        proofs = []
        for container, listed in actual_rows:
            key = (container, _alias(listed, 'custom_id', 'customId'))
            snapshot = expected_by_key.get(key)
            if snapshot is None:
                raise ReconciliationRequired('rollback left an unexpected object')
            match = dict(listed)
            match['_inventory_container'] = container
            hydrated = self._hydrated(match)
            if not self._logical_matches(hydrated, snapshot, require_backend_id=False):
                raise ReconciliationRequired('restored object differs from exact pre-state')
            proofs.append({'custom_id': snapshot['custom_id'], 'container': container,
                           'sha256': snapshot['sha256'], 'bytes': snapshot['bytes'],
                           'status': 'done', 'task_type': 'superrag',
                           'metadata_digest': _canonical_digest(snapshot['metadata']),
                           'content_digest': hashlib.sha256(snapshot['content'].encode()).hexdigest()})
        proofs.sort(key=lambda row: (row['container'], row['custom_id']))
        return {'digest': _canonical_digest(proofs), 'count': len(proofs),
                'container_counts': {container: sum(p['container'] == container for p in proofs)
                                     for container in sorted(CANONICAL_CONTAINERS)}}

    def execute(self, forward_plan: Any, forward_transaction_id: str,
                rollback_transaction_id: str = 'rollback') -> dict[str, Any]:
        forward_plan = _validated_transaction_plan(forward_plan)
        _validate_transaction_id(forward_transaction_id)
        _validate_transaction_id(rollback_transaction_id)
        forward_prefix = f'transactions/{forward_transaction_id}'
        self._rollback_child = f'transactions/{rollback_transaction_id}/rollback-journal.json'
        forward_raw = private_json_read(self.private_root, f'{forward_prefix}/forward-journal.json')
        snapshot_digest = forward_raw.get('snapshot_digest')
        if not _is_sha256(snapshot_digest): raise TransactionJournalError('forward snapshot binding is invalid')
        forward = validate_transaction_journal(forward_raw, forward_plan,
            expected_transaction_id=forward_transaction_id, expected_snapshot_digest=snapshot_digest)
        if any(row['stage'] != 'done' for row in forward['records']):
            raise ReconciliationRequired('forward transaction is not complete')
        snapshot = private_json_read(self.private_root, f'{forward_prefix}/snapshot.json',
                                     expected_sha256=snapshot_digest)
        if (snapshot.get('transaction_id') != forward_transaction_id
                or snapshot.get('plan_digest') != transaction_plan_digest(forward_plan)):
            raise TransactionJournalError('forward snapshot binding mismatch')
        expected_pre = snapshot.get('expected_pre_inventory')
        if (not isinstance(expected_pre, list)
                or snapshot.get('expected_pre_inventory_digest') != _canonical_digest(expected_pre)):
            raise TransactionJournalError('forward pre-inventory snapshot is invalid')
        expected_pre = [_snapshot_logical_record(row) for row in expected_pre]
        anchored_post = _project_forward_post_inventory(forward_plan, expected_pre)
        self.expected_pre = expected_pre
        self.plan = _rollback_plan(forward_plan, snapshot)
        forward_digest = _canonical_digest(forward_raw)
        journal_path = self.private_root / self._rollback_child
        if journal_path.is_file():
            # Once inverse mutations begin the forward post-state intentionally no
            # longer exists. Re-derive it from the validated forward plan and bound
            # pre-snapshot before trusting any rollback journal field or provider read.
            state = private_json_read(self.private_root, self._rollback_child)
            self.bindings = dict(rollback_transaction_id=rollback_transaction_id,
                forward_transaction_id=forward_transaction_id,
                forward_plan_digest=transaction_plan_digest(forward_plan),
                forward_journal_digest=forward_digest, snapshot_digest=snapshot_digest,
                pre_inventory_digest=_canonical_digest(expected_pre),
                post_inventory_digest=_canonical_digest(anchored_post),
                expected_post_inventory=anchored_post)
            state = thaw_transaction_journal(validate_rollback_journal(
                state, self.plan, **self.bindings))
        else:
            # Derive and prove the exact logical forward post-state before creating
            # the rollback journal. Provider-assigned ids are never trust anchors.
            post_inventory = self._inventory()
            expected_ids = {row['custom_id']: row for row in anchored_post}
            expected_keys = set(expected_ids)
            listed_matches = []
            for container, rows in post_inventory.items():
                for listed in rows:
                    cid = _alias(listed, 'custom_id', 'customId')
                    if cid not in expected_keys:
                        raise ReconciliationRequired('forward post-state has collateral objects')
                    match = dict(listed)
                    match['_inventory_container'] = container
                    listed_matches.append((cid, match))
            if (len(listed_matches) != len(expected_keys)
                    or {cid for cid, _ in listed_matches} != expected_keys):
                raise ReconciliationRequired('forward post-state is incomplete')
            post_rows = list(anchored_post)
            self.bindings = dict(rollback_transaction_id=rollback_transaction_id,
                forward_transaction_id=forward_transaction_id,
                forward_plan_digest=transaction_plan_digest(forward_plan),
                forward_journal_digest=forward_digest, snapshot_digest=snapshot_digest,
                pre_inventory_digest=_canonical_digest(expected_pre),
                post_inventory_digest=_canonical_digest(post_rows),
                expected_post_inventory=post_rows)
            state = dict(journal_schema_version=1, **self.bindings,
                rollback_plan_digest=_canonical_digest(self.plan),
                records=[dict(row, stage='existing', history=['existing'], restored_document_id=None)
                         for row in self.plan], verification_digest=None, verified_count=None,
                verified_container_counts=None)
            state = self._write(state)
            record_indexes = {row['custom_id']: index
                              for index, row in enumerate(state['records'])}
            for cid, match in listed_matches:
                try:
                    hydrated = self._hydrated(match)
                    expected_forward = expected_ids[cid]
                    if expected_forward['document_id'] is None:
                        if not self._forward_matches(hydrated, expected_forward):
                            raise ReconciliationRequired('rollback hydration failed')
                    else:
                        snap = next(row for row in expected_pre if row['custom_id'] == cid)
                        if not self._logical_matches(hydrated, snap, require_backend_id=True):
                            raise ReconciliationRequired('rollback hydration failed')
                except ReconciliationRequired:
                    index = record_indexes.get(cid, 0)
                    state = self._advance(state, index, 'reconcile_required')
                    raise ReconciliationRequired('rollback hydration failed') from None
        self.expected_post = {row['custom_id']: row
                              for row in self.bindings['expected_post_inventory']}
        for index, row in enumerate(state['records']):
            if row['stage'] == 'done': continue
            try:
                if row['action'] in {'remove', 'replace'} and row['stage'] not in {
                        'removed', 'restore_submitted', 'processing'}:
                    state = self._remove(state, index)
                if row['action'] == 'remove':
                    if state['records'][index]['stage'] == 'removed':
                        state = self._advance(state, index, 'done')
                else:
                    state = self._restore(state, index)
            except ReconciliationRequired:
                current_stage = state['records'][index]['stage']
                if ('reconcile_required' in ROLLBACK_TRANSITIONS[row['action']].get(
                        current_stage, set())):
                    state = self._advance(state, index, 'reconcile_required')
                raise
        proof = self._proof(expected_pre)
        state['verification_digest'] = proof['digest']; state['verified_count'] = proof['count']
        state['verified_container_counts'] = proof['container_counts']
        state = self._write(state)
        return state | {'complete': True, 'snapshot_digest': snapshot_digest}


def _private_json_write(path: Path, payload: dict[str, Any]) -> bytes:
    """Atomically write private deterministic JSON and return its exact bytes."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':')) + '\n').encode('utf-8')
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(name): os.unlink(name)
    return data


def create_logical_v3_snapshot(path: Path, current: dict[str, dict[str, Any]],
                               plan: dict[str, Any], *, expected_eligible: int,
                               expected_owner_private: int,
                               expected_family_shared: int) -> dict[str, Any]:
    """Persist and verify the exact backend-returned legacy projection."""
    if plan['already_v4'] or len(plan['replacements']) != expected_eligible:
        raise ReconciliationRequired('logical v3 snapshot requires the complete legacy set')
    records = []
    seen_ids: set[str] = set()
    for order, rel in enumerate(sorted(current)):
        doc = current[rel]; remote = plan['inventory'].get(doc['custom_id'])
        validate_legacy_backend_document(remote, doc)
        ident = str(_field(remote, 'id', '') or '')
        inventory_container = _field(remote, '_inventory_container')
        if inventory_container not in {CONTAINER, FAMILY_CONTAINER}:
            raise ReconciliationRequired('snapshot inventory container is unavailable or invalid')
        if ident in seen_ids: raise ReconciliationRequired('snapshot document ids are not unique')
        seen_ids.add(ident)
        records.append({'order': order, 'relative_path': rel, 'visibility': doc['visibility'],
                        'custom_id': doc['custom_id'], 'document_id': ident,
                        'content': _field(remote, 'content'),
                        'metadata': _field(remote, 'metadata'),
                        '_inventory_container': inventory_container})
    counts = (len(records), sum(r['visibility'] == 'owner_private' for r in records),
              sum(r['visibility'] == 'family_shared' for r in records))
    if counts != (expected_eligible, expected_owner_private, expected_family_shared):
        raise ReconciliationRequired('snapshot count drift')
    payload = {'snapshot_format': 'logical_backend_returned_v3', 'byte_original': False,
               'container': CONTAINER, 'count': len(records), 'records': records}
    data = _private_json_write(path, payload)
    digest = hashlib.sha256(data).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ReconciliationRequired('snapshot read-back hash mismatch')
    receipt = {'snapshot_format': payload['snapshot_format'], 'byte_original': False,
               'count': len(records), 'unique_custom_ids': len({r['custom_id'] for r in records}),
               'unique_document_ids': len(seen_ids), 'owner_private_count': counts[1],
               'family_shared_count': counts[2], 'snapshot_sha256': digest, 'verified': True}
    _private_json_write(path.with_suffix(path.suffix + '.receipt.json'), receipt)
    return receipt


def load_logical_v3_snapshot(path: Path, *, expected_eligible: int,
                             expected_owner_private: int,
                             expected_family_shared: int) -> dict[str, Any]:
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ReconciliationRequired('snapshot is absent or not mode 0600')
    data = path.read_bytes(); digest = hashlib.sha256(data).hexdigest()
    try:
        payload = json.loads(data); receipt = json.loads(path.with_suffix(path.suffix + '.receipt.json').read_text())
    except (OSError, json.JSONDecodeError):
        raise ReconciliationRequired('snapshot or receipt is unreadable') from None
    records = payload.get('records') if isinstance(payload, dict) else None
    if (payload.get('snapshot_format') != 'logical_backend_returned_v3'
            or payload.get('byte_original') is not False or payload.get('container') != CONTAINER
            or not isinstance(records, list) or payload.get('count') != len(records)
            or receipt.get('snapshot_sha256') != digest or receipt.get('verified') is not True):
        raise ReconciliationRequired('snapshot integrity receipt mismatch')
    ids = [str(r.get('custom_id') or '') for r in records]
    doc_ids = [str(r.get('document_id') or '') for r in records]
    counts = (len(records), sum(r.get('visibility') == 'owner_private' for r in records),
              sum(r.get('visibility') == 'family_shared' for r in records))
    if (counts != (expected_eligible, expected_owner_private, expected_family_shared)
            or len(set(ids)) != len(ids) or len(set(doc_ids)) != len(doc_ids)
            or any(not value for value in ids + doc_ids)
            or any(r.get('_inventory_container') not in {CONTAINER, FAMILY_CONTAINER}
                   for r in records)
            or [r.get('order') for r in records] != list(range(len(records)))):
        raise ReconciliationRequired('snapshot identity/count guard failed')
    return payload


def validate_snapshot_record(remote: Any, saved: dict[str, Any]) -> str:
    if remote is None or str(_field(remote, 'custom_id', '') or '') != saved['custom_id']:
        raise ReconciliationRequired('restored stable identity mismatch')
    if status_name(remote) != 'done':
        raise ReconciliationRequired('restored record is not terminal done')
    if _field(remote, 'content') != saved['content'] or _field(remote, 'metadata') != saved['metadata']:
        raise ReconciliationRequired('restored legacy content/metadata mismatch')
    ident = str(_field(remote, 'id', '') or '')
    if not ident: raise ReconciliationRequired('restored document id unavailable')
    return ident


def execute_schema_v3_rollback(client: Supermemory, current: dict[str, dict[str, Any]],
                               snapshot_path: Path, *, expected_eligible: int,
                               expected_owner_private: int,
                               expected_family_shared: int) -> dict[str, Any]:
    """Resumably replace only matching v4 identities with the logical v3 snapshot."""
    snapshot = load_logical_v3_snapshot(snapshot_path, expected_eligible=expected_eligible,
        expected_owner_private=expected_owner_private, expected_family_shared=expected_family_shared)
    records = snapshot['records']
    if set(current) != {r['relative_path'] for r in records}:
        raise ReconciliationRequired('snapshot/canonical path set drift')
    checkpoint_path = snapshot_path.with_suffix(snapshot_path.suffix + '.rollback.json')
    stages: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        try: stages = {r['custom_id']: r for r in json.loads(checkpoint_path.read_text()).get('records', [])}
        except (OSError, json.JSONDecodeError): raise ReconciliationRequired('rollback checkpoint unreadable') from None

    def save(saved: dict[str, Any], stage: str, replacement_id: str = '') -> None:
        stages[saved['custom_id']] = {'custom_id': saved['custom_id'], 'stage': stage,
                                      'v4_document_id': stages.get(saved['custom_id'], {}).get('v4_document_id', ''),
                                      'replacement_document_id': replacement_id}
        _private_json_write(checkpoint_path, {'mode': 'rollback_schema_v3', 'complete': False,
                            'expected_count': expected_eligible, 'records': list(stages.values())})

    # Preflight every not-yet-mutated record before the first deletion.
    pending = [r for r in records if stages.get(r['custom_id'], {}).get('stage') != 'done']
    inventory = schema_v4_backend_inventory(client, {r['custom_id'] for r in pending}) if pending else {}
    for saved in pending:
        stage = stages.get(saved['custom_id'], {}).get('stage', '')
        if stage in {'deleted', 'add_submitted'}: continue
        row = validate_backend_document(inventory.get(saved['custom_id']), current[saved['relative_path']])
        stages[saved['custom_id']] = {'custom_id': saved['custom_id'], 'stage': 'verified_v4',
                                      'v4_document_id': row['document_id'], 'replacement_document_id': ''}
    _private_json_write(checkpoint_path, {'mode': 'rollback_schema_v3', 'complete': False,
                        'expected_count': expected_eligible, 'records': list(stages.values())})

    for saved in records:
        state = stages.get(saved['custom_id'], {}); stage = state.get('stage')
        if stage == 'done': continue
        if stage == 'verified_v4':
            client.documents.delete(state['v4_document_id'], timeout=30.0)
            save(saved, 'deleted')
        if stages[saved['custom_id']]['stage'] == 'deleted':
            result = client.documents.add(
                content=saved['content'],
                container_tag=saved['_inventory_container'],
                custom_id=saved['custom_id'], task_type='superrag',
                metadata=saved['metadata'], timeout=30.0,
            )
            replacement_id = str(_field(result, 'id', '') or '')
            if not replacement_id: raise ReconciliationRequired('rollback add returned no document id')
            save(saved, 'add_submitted', replacement_id)
        replacement_id = stages[saved['custom_id']]['replacement_document_id']
        deadline = time.time() + 1800
        while time.time() < deadline:
            remote = client.documents.get(replacement_id, timeout=15.0)
            if status_name(remote) == 'done':
                validate_snapshot_record(remote, saved); save(saved, 'done', replacement_id); break
            if status_name(remote) in {'failed', 'error'}:
                raise ReconciliationRequired('rollback replacement failed')
            time.sleep(5)
        if stages[saved['custom_id']]['stage'] != 'done':
            raise ReconciliationRequired('rollback replacement did not reach terminal done')
    # Exact read-back also verifies legacy owner/missing ACL fields as captured.
    restored = schema_v4_backend_inventory(client, {r['custom_id'] for r in records})
    for saved in records: validate_snapshot_record(restored.get(saved['custom_id']), saved)
    receipt = {'mode': 'rollback_schema_v3', 'complete': True, 'restored_count': len(records),
               'owner_private_count': expected_owner_private, 'family_shared_count': expected_family_shared,
               'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
               'records': list(stages.values())}
    _private_json_write(checkpoint_path, receipt)
    return receipt


def execute_schema_v4_backfill(client: Supermemory, current: dict[str, dict[str, Any]],
                               plan: dict[str, Any]) -> dict[str, Any]:
    """Execute a pre-verified plan using durable delete/add checkpoints."""
    rows = dict(plan['completed']) | {rel: dict(row) for rel, row in plan['replacements'].items()}

    def save(rel: str, row: dict[str, Any]) -> None:
        rows[rel] = row
        atomic_receipt({
            'schema_version': INDEX_SCHEMA_VERSION,
            'mode': 'backfill_schema_v4',
            'reconciliation_complete': False,
            'expected_count': plan['eligible'],
            'documents': sorted(rows.values(), key=lambda value: value['relative_path']),
        })

    # The first durable receipt captures the entire verified replacement set.
    atomic_receipt({
        'schema_version': INDEX_SCHEMA_VERSION,
        'mode': 'backfill_schema_v4',
        'reconciliation_complete': False,
        'expected_count': plan['eligible'],
        'documents': sorted(rows.values(), key=lambda value: value['relative_path']),
    })
    for rel in sorted(plan['replacements']):
        old = rows[rel]
        if old.get('replacement_stage') == 'verified_legacy':
            old = update(client, current[rel], old)
            save(rel, old)
        row = recover_submission(client, current[rel], old,
                                 lambda value, rel=rel: save(rel, value))
        document_id = str(row.get('document_id') or '')
        deadline = time.time() + 1800
        while document_id and row.get('replacement_stage') != 'done' and time.time() < deadline:
            try:
                state = status_name(client.documents.get(document_id, timeout=15.0))
            except Exception:
                state = ''
            if state:
                row['final_status'] = state
                row['submission_status'] = state if state in TERMINAL else row.get('submission_status', '')
                row['replacement_stage'] = 'done' if state == 'done' else (
                    'add_failed' if state in TERMINAL else 'pending')
                save(rel, row)
                if state in TERMINAL:
                    break
            time.sleep(5)
        if row.get('replacement_stage') != 'done':
            raise ReconciliationRequired('schema-v4 replacement did not reach terminal done')

    inventory = schema_v4_backend_inventory(
        client, {doc['custom_id'] for doc in current.values()})
    reconciled = {
        rel: validate_backend_document(inventory.get(doc['custom_id']), doc)
        for rel, doc in current.items()
    }
    receipt = {
        'schema_version': INDEX_SCHEMA_VERSION,
        'mode': 'backfill_schema_v4',
        **canonical_container_fields(current),
        'expected_count': plan['eligible'],
        'replaced_count': len(plan['replacements']),
        'already_v4_count': plan['already_v4'],
        'backend_reconciled_count': len(reconciled),
        'reconciliation_complete': len(reconciled) == plan['eligible'],
        'documents': sorted(reconciled.values(), key=lambda value: value['relative_path']),
    }
    receipt.update(completion_readiness(client, current))
    atomic_receipt(receipt)
    if not receipt['reconciliation_complete']:
        raise ReconciliationRequired('schema-v4 search readiness incomplete')
    return receipt


def load_previous() -> dict[str, dict[str, Any]]:
    if not OUT.exists():
        return {}
    try:
        payload = json.loads(OUT.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError('Existing sync manifest is unreadable') from None
    return {
        str(row['relative_path']): row
        for row in payload.get('documents', [])
        if isinstance(row, dict) and row.get('relative_path')
    }


def atomic_receipt(payload: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=OUT.name + '.', dir=OUT.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, 0o600)
        os.replace(name, OUT)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def recover_submission(client: Supermemory, doc: dict[str, Any], old: dict[str, Any],
                       checkpoint: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Resume a non-terminal row without blindly duplicating an accepted add."""
    row = dict(old or {})
    stage = str(row.get('replacement_stage') or '')
    document_id = str(row.get('document_id') or '')
    if document_id and stage in {'add_submitted', 'pending', 'add_failed'}:
        try:
            state = status_name(client.documents.get(document_id, timeout=15.0))
        except Exception as exc:
            if not is_not_found(exc):
                row['replacement_stage'] = 'pending'
                checkpoint(row)
                return row
            state = 'not_found'
        if state == 'done':
            row.update({'final_status': 'done', 'submission_status': 'done',
                        'replacement_stage': 'done'})
            checkpoint(row)
            return row
        if state not in TERMINAL and state != 'not_found':
            row.update({'final_status': state, 'replacement_stage': 'pending'})
            checkpoint(row)
            return row
        try:
            client.documents.delete(document_id, timeout=30.0)
        except Exception as exc:
            if not is_not_found(exc):
                row.update({'final_status': state, 'replacement_stage': 'add_failed'})
                checkpoint(row)
                return row
        row.update({'document_id': '', 'final_status': '', 'submission_status': '',
                    'replacement_stage': 'deleted',
                    'replaced_document_id': row.get('replaced_document_id') or document_id})
        checkpoint(row)

    # An idless row may represent an add accepted by the backend whose response
    # was lost. Resolve stable provider identity before any possible resubmit.
    try:
        remote = lookup_backend_document(client, doc)
    except Exception:
        row.update({'relative_path': doc['relative_path'], 'sha256': doc['sha256'],
                    'custom_id': doc['custom_id'], 'index_schema_version': INDEX_SCHEMA_VERSION,
                    'replacement_stage': 'reconcile_required', 'final_status': 'unknown'})
        checkpoint(row)
        raise
    if remote is not None:
        try:
            reconciled = validate_backend_document(remote, doc)
        except Exception:
            row.update({'relative_path': doc['relative_path'], 'sha256': doc['sha256'],
                        'custom_id': doc['custom_id'], 'index_schema_version': INDEX_SCHEMA_VERSION,
                        'replacement_stage': 'reconcile_required', 'final_status': 'unknown'})
            checkpoint(row)
            raise
        reconciled['replaced_document_id'] = row.get('replaced_document_id', '')
        checkpoint(reconciled)
        return reconciled

    try:
        submitted = add(client, doc)
    except Exception:
        row.update({'relative_path': doc['relative_path'], 'sha256': doc['sha256'],
                    'custom_id': doc['custom_id'], 'index_schema_version': INDEX_SCHEMA_VERSION,
                    'replacement_stage': 'reconcile_required', 'final_status': 'unknown'})
        checkpoint(row)
        raise
    submitted.update({'replacement_stage': 'add_submitted',
                      'replaced_document_id': row.get('replaced_document_id', '')})
    checkpoint(submitted)
    return submitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Reconcile canonical Obsidian Markdown into Owner Supermemory.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Report changes without writing Supermemory or manifest.')
    mode.add_argument('--verify-only', action='store_true',
                      help='Exhaustively verify canonical documents in Supermemory without any writes.')
    mode.add_argument('--backfill-schema-v4', action='store_true',
                      help='Replace only exhaustively verified legacy canonical records with schema-v4 metadata.')
    mode.add_argument('--rollback-schema-v3', action='store_true',
                      help='Restore a verified logical backend-returned v3 snapshot.')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path,
                        help='Normal-mode output path; read-only modes are stdout-only.')
    parser.add_argument('--manifest', type=Path, help='Manifest path (primarily for isolated tests).')
    parser.add_argument('--storage-receipt', type=Path,
                        help='Separate storage reconciliation receipt path.')
    parser.add_argument('--vector-receipt', type=Path,
                        help='Separate per-document vector readiness receipt path.')
    parser.add_argument('--behavioral-receipt', type=Path,
                        help='Read-only external behavioral benchmark receipt path.')
    parser.add_argument('--expected-eligible', type=int)
    parser.add_argument('--expected-owner-private', type=int)
    parser.add_argument('--expected-family-shared', type=int)
    args = parser.parse_args()
    if (args.dry_run or args.verify_only) and args.output is not None:
        parser.error('read-only modes are stdout-only; --output is not allowed')
    expected = (args.expected_eligible, args.expected_owner_private, args.expected_family_shared)
    migration_mode = args.backfill_schema_v4 or args.rollback_schema_v3
    if migration_mode and any(value is None for value in expected):
        parser.error('migration modes require all three --expected-* counts')
    if migration_mode and args.snapshot is None:
        parser.error('migration modes require --snapshot PATH')
    if not migration_mode and (any(value is not None for value in expected) or args.snapshot is not None):
        parser.error('--expected-* counts and --snapshot require a migration mode')
    if any(value is not None and value < 0 for value in expected):
        parser.error('--expected-* counts must be non-negative')
    return args


def eligible_documents(scanned: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return project_canonical_documents(scanned)['documents']


def project_canonical_documents(scanned: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate and filter a scan into a deterministic path-only projection."""
    documents: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, str]] = []
    for doc in sorted(scanned, key=lambda value: value.get('relative_path', '')):
        rel = doc.get('relative_path')
        scope = canonical_scope_from_path(rel)
        if scope is None or not isinstance(rel, str) or doc.get('custom_id') != stable_custom_id_from_path(rel):
            raise ValueError('invalid canonical projection')
        if doc.get('visibility') != canonical_visibility_from_path(rel):
            raise ValueError('invalid canonical projection')
        if not isinstance(doc.get('sha256'), str) or not isinstance(doc.get('bytes'), int):
            raise ValueError('invalid canonical projection')
        source = doc.get('content', '').split('[/canonical-identity]', 1)[-1].lstrip('\n')
        reason = canonical_exclusion_reason(rel, source, str(doc.get('identity', {}).get('entity_type', '')))
        if reason:
            excluded.append({'relative_path': rel, 'reason': reason})
        else:
            if rel in documents:
                raise ValueError('duplicate canonical path')
            documents[rel] = doc
    return {'documents': documents, 'excluded': tuple(excluded)}


def backend_reconcile_all(client: Supermemory, current: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Read back and validate every expected stable identity from the backend."""
    reconciled = {}
    for rel, doc in current.items():
        reconciled[rel] = validate_backend_document(lookup_backend_document(client, doc), doc)
    return reconciled


def verify_backend(client: Supermemory, current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Verify every eligible source identity, collecting path-only failures."""
    verified = 0
    failed_paths: list[str] = []
    for rel in sorted(current):
        try:
            validate_backend_document(lookup_backend_document(client, current[rel]), current[rel])
        except Exception:
            failed_paths.append(rel)
        else:
            verified += 1
    return {
        'backend_verified_count': verified,
        'failure_count': len(failed_paths),
        'failed_paths': failed_paths,
        'verification_complete': verified == len(current) and not failed_paths,
    }


def canonical_container_fields(current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        'canonical_containers': [CONTAINER, FAMILY_CONTAINER],
        'canonical_container_counts': {
            container: sum(container_for_doc(doc) == container for doc in current.values())
            for container in (CONTAINER, FAMILY_CONTAINER)
        },
        'conversation_container_mutated': False,
    }


def indexed_content_probes(doc: dict[str, Any]) -> list[str]:
    """Build deterministic credential-safe probes from indexed body content."""
    content = doc.get('content')
    if not isinstance(content, str):
        raise ReconciliationRequired('indexed content is unavailable for search proof')
    marker = '[/canonical-identity]'
    if marker not in content:
        raise ReconciliationRequired('indexed content identity envelope is malformed')
    body = content.split(marker, 1)[1].lstrip('\r\n')
    if body.startswith('---\n'):
        end = body.find('\n---\n', 4)
        if end < 0:
            raise ReconciliationRequired('indexed content frontmatter is malformed')
        body = body[end + 5:]

    # Queries stay inside the same private provider, but credentials must never
    # be copied into a search request. Remove URL query/fragment material and
    # assignment values while retaining surrounding searchable prose.
    body = re.sub(r'(?i)(https?://[^\s?#]+)[?#][^\s]*', r'\1', body)
    body = re.sub(
        r'(?i)\b(?:api[_-]?key|password|secret|(?:access[_-]?)?token)\s*[:=]\s*\S+',
        '[redacted]', body,
    )
    entity_name = str(doc.get('identity', {}).get('entity_name') or '').strip().casefold()
    candidates = []
    title_only = []
    for raw_line in body.splitlines():
        line = re.sub(
            r'^\s*(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+|\[[ xX]\]\s*)?',
            '', raw_line,
        ).strip()
        if not line or line == '[redacted]':
            continue
        if raw_line.lstrip().startswith('#') or line.casefold() == entity_name:
            title_only.append(line)
        else:
            candidates.append(line)
    # Prefer body prose, but retain indexed headings as deterministic fallback
    # probes for sparse or unusually chunked documents.
    lines = candidates + title_only
    all_words = re.findall(r"[^\s]+", ' '.join(lines))
    probe_width = 8
    line_sample_count = min(16, len(lines))
    line_indexes = sorted({
        index * (len(lines) - 1) // max(1, line_sample_count - 1)
        for index in range(line_sample_count)
    })
    probes = [
        ' '.join(re.findall(r"[^\s]+", lines[index])[:probe_width])
        for index in line_indexes
    ]
    if all_words:
        last = max(0, len(all_words) - probe_width)
        word_sample_count = min(16, last + 1)
        offsets = sorted({
            index * last // max(1, word_sample_count - 1)
            for index in range(word_sample_count)
        })
        probes.extend(
            ' '.join(all_words[offset:offset + probe_width]) for offset in offsets
        )
        # Four-token windows are short enough to stay inside provider chunk
        # boundaries. Keep deterministic early and corpus-spanning samples so
        # prose that straddles an internal chunk boundary still has a usable
        # high-information query.
        short_last = max(0, len(all_words) - 4)
        short_offsets = list(range(min(16, short_last + 1)))
        short_offsets.extend(
            index * short_last // 15 for index in range(16)
        )
        probes.extend(
            ' '.join(all_words[offset:offset + 4])
            for offset in sorted(set(short_offsets))
        )
    unique = []
    for probe in probes:
        probe = probe[:320].strip()
        if (probe and probe not in unique and not SENSITIVE_CONTENT.search(probe)
                and _distinctive_proof_tokens(_proof_tokens(probe))):
            unique.append(probe)
    if not unique:
        raise ReconciliationRequired(
            'privacy-safe distinctive indexed content probe is unavailable'
        )
    return unique


def indexed_content_probe(doc: dict[str, Any]) -> str:
    """Return the canonical first probe (kept separate for deterministic testing)."""
    return indexed_content_probes(doc)[0]


def _proof_tokens(value: str) -> list[str]:
    """Normalize provider text without retaining or emitting source content."""
    normalized = unicodedata.normalize('NFKC', value).casefold()
    return re.findall(r'\w+', normalized, flags=re.UNICODE)


_PROOF_STOPWORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'for', 'from',
    'had', 'has', 'have', 'he', 'her', 'his', 'i', 'in', 'is', 'it', 'its', 'of',
    'on', 'or', 'our', 'she', 'that', 'the', 'their', 'them', 'there', 'they',
    'this', 'to', 'was', 'we', 'were', 'will', 'with', 'you', 'your',
})


def _distinctive_proof_tokens(tokens: list[str]) -> bool:
    """Require enough non-common material to make a content proof meaningful."""
    informative = [token for token in tokens if token not in _PROOF_STOPWORDS]
    return (len(tokens) >= 4 and len(informative) >= 3
            and sum(len(token) for token in informative) >= 18)


def _contains_token_run(haystack: list[str], needle: list[str]) -> bool:
    return bool(needle) and any(
        haystack[index:index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


def _shared_distinctive_run(left: list[str], right: list[str]) -> bool:
    """Find a normalized, ordered, high-information shared token run."""
    for width in range(min(len(left), len(right)), 2, -1):
        for index in range(len(left) - width + 1):
            candidate = left[index:index + width]
            informative = [token for token in candidate if token not in _PROOF_STOPWORDS]
            if (len(informative) >= 3
                    and sum(len(token) for token in informative) >= 18
                    and _contains_token_run(right, candidate)):
                return True
    return False


def indexed_chunk_proves_probe(chunk: str, probe: str, canonical: str) -> bool:
    """Prove a returned chunk materially matches its current canonical probe."""
    if not all(isinstance(value, str) and value.strip()
               for value in (chunk, probe, canonical)):
        return False
    chunk_tokens = _proof_tokens(chunk)
    probe_tokens = _proof_tokens(probe)
    canonical_tokens = _proof_tokens(canonical)
    if (not _distinctive_proof_tokens(probe_tokens)
            or not _contains_token_run(canonical_tokens, probe_tokens)):
        return False
    return _shared_distinctive_run(chunk_tokens, probe_tokens)


def verify_indexed_documents(client: Supermemory, current: dict[str, dict[str, Any]],
                             inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    """Probe every source and fail closed on ambiguity or stale chunks.

    Each filtered query accepts at most one provider chunk. Candidate probes
    continue until that chunk proves the same distinctive canonical token run.
    """
    verified = 0
    failed = []
    for rel, doc in sorted(current.items()):
        try:
            remote = (inventory.get(doc['custom_id']) if inventory is not None
                      else lookup_backend_document(client, doc))
            row = validate_backend_document(remote, doc)
            results = None
            selected_probe = None
            for probe in indexed_content_probes(doc):
                response = search_documents_v4(
                    client, probe, container_tag=container_for_doc(doc),
                    limit=1, timeout=30.0,
                    filters={'AND': [{'key': 'source', 'value': 'obsidian'},
                                     {'key': 'relative_path', 'value': rel}]},
                )
                candidate_results = _field(response, 'results')
                total = _field(response, 'total')
                if (not isinstance(candidate_results, list) or len(candidate_results) > 1
                        or isinstance(total, bool) or not isinstance(total, (int, float))
                        or not float(total).is_integer()
                        or int(total) != len(candidate_results)):
                    raise ReconciliationRequired('search completeness unavailable')
                if (candidate_results
                        and all(indexed_chunk_proves_probe(_field(result, 'chunk'), probe,
                                                          doc['content'])
                                for result in candidate_results)):
                    results = candidate_results
                    selected_probe = probe
                    break
            if not results:
                raise ReconciliationRequired('indexed content is not retrievable')
            seen = set()
            for result in results:
                validate_search_result_metadata(result, doc)
                chunk = _field(result, 'chunk')
                if (not isinstance(selected_probe, str)
                        or not indexed_chunk_proves_probe(chunk, selected_probe, doc['content'])):
                    raise ReconciliationRequired('search chunk content does not prove current source')
                proof = dict(remote) if isinstance(remote, dict) else remote.model_dump()
                proof['container_tags'] = [_field(remote, '_inventory_container',
                                                  container_for_doc(doc))]
                normalized = normalize_document_chunk(result, container_for_doc(doc), proof)
                if normalized is None or normalized['id'] in seen:
                    raise ReconciliationRequired('search chunk identity unavailable or duplicated')
                seen.add(normalized['id'])
                if normalized['_parent_document_id'] not in {row['document_id'], doc['custom_id']}:
                    raise ReconciliationRequired('search parent identity does not match backend')
                custom_id = normalized['_source_custom_id']
                if custom_id and custom_id != doc['custom_id']:
                    raise ReconciliationRequired('search stable identity does not match backend')
        except Exception:
            failed.append(rel)
        else:
            verified += 1
    return {'search_verified_count': verified, 'search_failure_count': len(failed),
            'search_readiness_complete': verified == len(current) and not failed}


def completion_readiness(client: Supermemory, current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine closed inventory, backend validation, and exhaustive indexed readback."""
    try:
        closed = provider_inventory(client)
        classification = classify_inventory(current, closed)
        blockers = {key: value for key, value in classification.items()
                    if key != 'expected' and value}
        all_rows = [(container, row) for container in sorted(closed) for row in closed[container]]
        inventory: dict[str, Any] | None = {}
        for container, raw in all_rows:
            row = dict(raw)
            row['_inventory_container'] = container
            custom_id = _alias(row, 'custom_id', 'customId')
            if custom_id in inventory:
                raise ReconciliationRequired('duplicate provider stable identity')
            inventory[custom_id] = row
        exact_set = (not blockers and len(all_rows) == len(current)
                     and classification['expected'] == tuple(sorted(current)))
    except Exception:
        inventory = None
        exact_set = False
    reconciled = 0
    if inventory is not None and exact_set:
        for doc in current.values():
            try:
                validate_backend_document(inventory.get(doc['custom_id']), doc)
            except Exception:
                continue
            reconciled += 1
    inventory_complete = exact_set and reconciled == len(current)
    search = verify_indexed_documents(client, current, inventory)
    return {**canonical_container_fields(current), **search,
            'backend_reconciled_count': reconciled,
            'inventory_complete': inventory_complete,
            'reconciliation_complete': inventory_complete and search['search_readiness_complete'],
            'submission_failure_count': 0, 'still_pending_count': 0}


def write_verified_readiness_receipts(
        private_root: Path, current: dict[str, dict[str, Any]], inventory: list[dict[str, Any]],
        vectors: dict[str, list[Any]], *, generation: str, generated_at: Any,
        initialize_key: bool = False, rotate_key: bool = False) -> dict[str, Any]:
    """Persist separate proofs only after their complete in-memory validation.

    This intentionally has no live CLI call site in this slice. Behavioral
    benchmark receipts are external artifacts and are never accepted here.
    """
    key_path = private_root / 'readiness/receipt-hmac.key'
    if initialize_key or rotate_key:
        generate_receipt_key(key_path, rotate=rotate_key)
    key = read_receipt_key(key_path)
    if key is None:
        raise ReconciliationRequired('readiness receipt key is unavailable or unsafe')
    storage = build_storage_reconciliation_receipt(
        current, inventory, generation=generation, generated_at=generated_at, key=key,
        owner_canonical_container=CONTAINER)
    vector = build_vector_readiness_receipt(
        current, vectors, generation=generation, generated_at=generated_at, key=key,
        owner_canonical_container=CONTAINER,
    )
    if not validate_readiness_receipts(
            storage, vector, key=key,
            current_fingerprint=source_fingerprint_from_documents(
                current, owner_canonical_container=CONTAINER), now=generated_at):
        raise ReconciliationRequired('readiness receipt proof is incomplete')
    storage_digest = private_json_write(
        private_root, 'readiness/storage-reconciliation.json', storage,
    )
    vector_digest = private_json_write(
        private_root, 'readiness/vector-readiness.json', vector,
    )
    return {
        'generation': generation,
        'storage_receipt_sha256': storage_digest,
        'vector_receipt_sha256': vector_digest,
    }


def legacy_main() -> None:
    global OUT
    args = parse_args()
    if args.manifest is not None:
        OUT = args.manifest
    started = time.time()
    if args.backfill_schema_v4 or args.rollback_schema_v3:
        if CONTAINER == CONVERSATION_CONTAINER:
            raise ReconciliationRequired('canonical and conversation containers are not isolated')
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        with LOCK.open('a+', encoding='utf-8') as lock_handle:
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('sync already running') from None
            scanned = canonical_documents()
            current = eligible_documents(scanned)
            client = Supermemory(api_key=api_key(), base_url=BASE_URL, timeout=30.0, max_retries=1)
            if args.rollback_schema_v3:
                receipt = execute_schema_v3_rollback(client, current, args.snapshot,
                    expected_eligible=args.expected_eligible,
                    expected_owner_private=args.expected_owner_private,
                    expected_family_shared=args.expected_family_shared)
            else:
                previous = load_previous()
                plan = build_schema_v4_backfill_plan(client, current, previous,
                    expected_eligible=args.expected_eligible,
                    expected_owner_private=args.expected_owner_private,
                    expected_family_shared=args.expected_family_shared)
                if args.snapshot.exists():
                    load_logical_v3_snapshot(args.snapshot,
                        expected_eligible=args.expected_eligible,
                        expected_owner_private=args.expected_owner_private,
                        expected_family_shared=args.expected_family_shared)
                else:
                    create_logical_v3_snapshot(args.snapshot, current, plan,
                        expected_eligible=args.expected_eligible,
                        expected_owner_private=args.expected_owner_private,
                        expected_family_shared=args.expected_family_shared)
                receipt = execute_schema_v4_backfill(client, current, plan)
        print(json.dumps({
            'schema_version': 3 if args.rollback_schema_v3 else INDEX_SCHEMA_VERSION,
            'mode': 'rollback_schema_v3' if args.rollback_schema_v3 else 'backfill_schema_v4',
            'eligible': len(current),
            'owner_private_count': sum(doc['visibility'] == 'owner_private' for doc in current.values()),
            'family_shared_count': sum(doc['visibility'] == 'family_shared' for doc in current.values()),
            'replaced_count': receipt.get('replaced_count', receipt.get('restored_count')),
            'already_v4_count': receipt.get('already_v4_count', 0),
            'backend_reconciled_count': receipt.get('backend_reconciled_count', receipt.get('restored_count')),
            **canonical_container_fields(current),
            'search_readiness_complete': receipt.get('search_readiness_complete', False),
            'search_verified_count': receipt.get('search_verified_count', 0),
            'search_failure_count': receipt.get('search_failure_count', 0),
            'reconciliation_complete': receipt.get('reconciliation_complete', receipt.get('complete')),
        }, indent=2))
        return
    if args.verify_only:
        scanned = canonical_documents()
        current = eligible_documents(scanned)
        previous = load_previous()
        client = Supermemory(api_key=api_key(), base_url=BASE_URL, timeout=30.0, max_retries=1)
        verification = verify_backend(client, current)
        readiness = completion_readiness(client, current)
        verification['verification_complete'] &= readiness['reconciliation_complete']
        verification.update(readiness)
        summary = {
            'schema_version': INDEX_SCHEMA_VERSION,
            'scanned': len(scanned),
            'eligible': len(current),
            'owner_private_count': sum(doc['visibility'] == 'owner_private' for doc in current.values()),
            'family_shared_count': sum(doc['visibility'] == 'family_shared' for doc in current.values()),
            'manifest_count': len(previous),
            **verification,
            'verify_only': True,
            'filesystem_mutated': False,
            'backend_mutated': False,
        }
        print(json.dumps(summary, indent=2))
        if not verification['verification_complete']:
            raise SystemExit(1)
        return
    if args.dry_run:
        scanned = canonical_documents(); previous = load_previous()
        current = eligible_documents(scanned)
        changed = {r for r in set(current) & set(previous) if current[r]['sha256'] != previous[r].get('sha256') or previous[r].get('index_schema_version') != INDEX_SCHEMA_VERSION or container_for_row(previous[r]) != container_for_doc(current[r])}
        plan = ([{'action': 'add', 'relative_path': path} for path in sorted(set(current)-set(previous))]
                + [{'action': 'replace', 'relative_path': path} for path in sorted(changed)]
                + [{'action': 'delete', 'relative_path': path} for path in sorted(set(previous)-set(current))])
        print(json.dumps({'scanned': len(scanned), 'eligible': len(current), 'new': len(set(current)-set(previous)), 'changed': len(changed), 'unchanged': len(set(current)&set(previous)-changed), 'removed': len(set(previous)-set(current)), 'plan': plan, 'dry_run': True, 'filesystem_mutated': False, 'backend_mutated': False}, indent=2)); return

    # A true unchanged run is read-only: do not create/touch the lock or rewrite
    # the manifest. Readiness still comes from exhaustive backend validation.
    previous_probe = load_previous()
    scanned_probe = canonical_documents()
    current_probe = eligible_documents(scanned_probe)
    changed_probe = {rel for rel in set(current_probe) & set(previous_probe)
                     if current_probe[rel]['sha256'] != previous_probe[rel].get('sha256')
                     or previous_probe[rel].get('index_schema_version') != INDEX_SCHEMA_VERSION
                     or container_for_row(previous_probe[rel]) != container_for_doc(current_probe[rel])
                     or previous_probe[rel].get('final_status') != 'done'}
    if (set(current_probe) == set(previous_probe) and not changed_probe):
        client = Supermemory(api_key=api_key(), base_url=BASE_URL, timeout=30.0, max_retries=1)
        readiness = completion_readiness(client, current_probe)
        print(json.dumps({'schema_version': INDEX_SCHEMA_VERSION, 'expected_count': len(current_probe),
                          **readiness,
                          'unchanged': len(current_probe), 'manifest_mutated': False}, indent=2))
        if not readiness['reconciliation_complete']:
            raise SystemExit(1)
        return
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+', encoding='utf-8') as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('sync already running') from None

        previous = load_previous()
        scanned = canonical_documents()
        current: dict[str, dict[str, Any]] = {}
        skipped_sensitive: list[dict[str, str]] = []
        for doc in scanned:
            if SENSITIVE_CONTENT.search(doc['content']):
                skipped_sensitive.append({'relative_path': doc['relative_path'], 'reason': 'sensitive_content'})
            elif doc['is_template']:
                skipped_sensitive.append({'relative_path': doc['relative_path'], 'reason': 'template'})
            else:
                current[doc['relative_path']] = doc

        new_paths = sorted(set(current) - set(previous))
        changed_paths = sorted(
            rel for rel in set(current) & set(previous)
            if (current[rel]['sha256'] != previous[rel].get('sha256')
                or previous[rel].get('index_schema_version') != INDEX_SCHEMA_VERSION
                or container_for_row(previous[rel]) != container_for_doc(current[rel])
                or previous[rel].get('final_status') != 'done')
        )
        unchanged_paths = sorted(set(current) & set(previous) - set(changed_paths))
        removed_paths = sorted(set(previous) - set(current))

        summary = {
            'scanned': len(scanned), 'eligible': len(current),
            'new': len(new_paths), 'changed': len(changed_paths),
            'unchanged': len(unchanged_paths), 'removed': len(removed_paths),
            'skipped_sensitive': len(skipped_sensitive), 'dry_run': args.dry_run,
        }
        client = Supermemory(api_key=api_key(), base_url=BASE_URL, timeout=30.0, max_retries=1)
        next_records = {rel: dict(previous[rel]) for rel in unchanged_paths}
        # Submission status is the provider's initial receipt (often "queued").
        # Once polling records a terminal final status, normalize unchanged rows
        # so the durable manifest reflects current backend state rather than the
        # historical acknowledgement forever.
        for row in next_records.values():
            final_status = str(row.get('final_status') or '').lower()
            if final_status in TERMINAL:
                row['submission_status'] = final_status
        failures: list[dict[str, str]] = []
        submitted: list[dict[str, Any]] = []

        def checkpoint(rel: str, row: dict[str, Any]) -> None:
            next_records[rel] = row
            atomic_receipt({'schema_version': INDEX_SCHEMA_VERSION, 'reconciliation_complete': False,
                            'documents': sorted(next_records.values(), key=lambda value: value['relative_path'])})

        for rel in new_paths:
            try:
                row = recover_submission(client, current[rel], previous.get(rel, {}),
                                         lambda value, rel=rel: checkpoint(rel, value))
                if row.get('final_status') != 'done':
                    submitted.append(row)
            except Exception as exc:
                failures.append({'relative_path': rel, 'action': 'add', 'error_type': type(exc).__name__})

        for rel in changed_paths:
            old = previous[rel]
            try:
                if (old.get('replacement_stage') in {
                        'deleted', 'add_submitted', 'pending', 'add_failed', 'reconcile_required'}
                        and old.get('sha256') == current[rel]['sha256']
                        and old.get('index_schema_version') == INDEX_SCHEMA_VERSION):
                    deleted_row = old
                else:
                    deleted_row = update(client, current[rel], old)
                    checkpoint(rel, deleted_row)
            except Exception as exc:
                next_records[rel] = old
                failures.append({'relative_path': rel, 'action': 'delete_for_replace', 'error_type': type(exc).__name__})
                continue
            try:
                row = recover_submission(client, current[rel], deleted_row,
                                         lambda value, rel=rel: checkpoint(rel, value))
                if row.get('final_status') != 'done':
                    submitted.append(row)
            except Exception as exc:
                failures.append({'relative_path': rel, 'action': 'add_after_delete', 'error_type': type(exc).__name__})

        deleted_count = 0
        for rel in removed_paths:
            old = previous[rel]
            document_id = str(old.get('document_id') or '')
            if not document_id:
                next_records[rel] = old
                failures.append({'relative_path': rel, 'action': 'delete', 'error_type': 'MissingDocumentId'})
                continue
            try:
                client.documents.delete(document_id, timeout=30.0)
                deleted_count += 1
            except Exception as exc:
                if is_not_found(exc):
                    deleted_count += 1
                else:
                    next_records[rel] = old
                    failures.append({'relative_path': rel, 'action': 'delete', 'error_type': type(exc).__name__})

        pending = {row['document_id']: row for row in submitted if row.get('document_id')}
        deadline = time.time() + 1800
        while pending and time.time() < deadline:
            for document_id in list(pending):
                try:
                    state = status_name(client.documents.get(document_id, timeout=15.0))
                except Exception:
                    continue
                pending[document_id]['final_status'] = state
                pending[document_id]['replacement_stage'] = 'done' if state == 'done' else (
                    'add_failed' if state in TERMINAL else 'pending')
                checkpoint(pending[document_id]['relative_path'], pending[document_id])
                if state in TERMINAL:
                    pending.pop(document_id, None)
            if pending:
                time.sleep(5)

        # Advance canonical projection state only after confirmed ingestion.
        # Failed/error/pending updates keep their previous manifest row so the
        # next run retries; failed new documents remain absent for the same
        # reason.
        for row in submitted:
            rel = row['relative_path']
            if row.get('final_status') == 'done':
                row['replacement_stage'] = 'done'
                next_records[rel] = row
            else:
                row['replacement_stage'] = 'pending' if not row.get('final_status') else 'add_failed'
                next_records[rel] = row

        final_counts: dict[str, int] = {}
        for row in submitted:
            state = row.get('final_status') or 'pending'
            final_counts[state] = final_counts.get(state, 0) + 1
        try:
            backend_records = backend_reconcile_all(client, current)
        except Exception as exc:
            failures.append({'relative_path': '*', 'action': 'backend_reconcile_all',
                             'error_type': type(exc).__name__})
            backend_records = {}
        # Completion state is a backend receipt, not a count of manifest rows.
        if backend_records:
            next_records = backend_records
        receipt = {
            'schema_version': INDEX_SCHEMA_VERSION,
            'root': str(ROOT), **canonical_container_fields(current),
            'authority': 'Obsidian -> Supermemory',
            'file_count': len(scanned), 'eligible_file_count': len(current),
            'skipped_sensitive_count': len(skipped_sensitive),
            'byte_count': sum(doc['bytes'] for doc in current.values()),
            'added_count': len(new_paths), 'changed_count': len(changed_paths),
            'unchanged_count': len(unchanged_paths), 'removed_count': deleted_count,
            'submission_failure_count': len(failures),
            'final_status_counts': final_counts, 'still_pending_count': len(pending),
            'expected_count': len(current),
            'backend_reconciled_count': len(backend_records),
            'total_seconds': round(time.time() - started, 3),
            'documents': sorted(next_records.values(), key=lambda row: row['relative_path']),
            'failures': failures, 'skipped_sensitive': skipped_sensitive,
        }
        receipt['reconciliation_complete'] = (
            not failures and not pending and len(receipt['documents']) == receipt['expected_count']
            and receipt['backend_reconciled_count'] == receipt['expected_count']
            and all(row.get('final_status') == 'done' and row.get('index_schema_version') == INDEX_SCHEMA_VERSION
                    for row in receipt['documents'])
        )
        readiness = completion_readiness(client, current)
        prior_complete = receipt['reconciliation_complete']
        receipt.update({key: value for key, value in readiness.items()
                        if key not in {'submission_failure_count', 'still_pending_count'}})
        receipt['reconciliation_complete'] &= prior_complete
        atomic_receipt(receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k not in {'documents', 'failures', 'skipped_sensitive'}}, indent=2))
        if (not receipt['reconciliation_complete'] or failures or pending or final_counts.get('pending', 0)
                or final_counts.get('failed', 0) or final_counts.get('error', 0)):
            raise SystemExit(1)


# Phase-2 command surface.  The original flags above remain accepted for the
# migration utility, but all scheduled/new invocations use these explicit
# subcommands and the durable executors.
PHASE2_COMMANDS = frozenset({
    'dry-run', 'plan', 'verify-only', 'reconcile', 'rollback', 'key-init', 'key-rotate',
    'install', 'reload-instructions', 'legacy-reconcile',
})


def _phase2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Secure canonical-memory reconciliation')
    sub = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--source-root', type=Path, required=True)
    common.add_argument('--private-root', type=Path, required=True)
    common.add_argument('--manifest', type=Path)
    common.add_argument('--base-url', default=BASE_URL)
    common.add_argument('--owner-canonical-container', default=OWNER_CONTAINER,
                        help='Exact validated destination for Owner canonical documents')
    common.add_argument('--owner-explicit-container', default=OWNER_CONTAINER,
                        help='Exact validated destination reserved for Owner explicit memories')
    sub.add_parser('dry-run', parents=[common])
    plan = sub.add_parser('plan', parents=[common])
    plan.add_argument('--transaction-id', required=True, type=_transaction_id_argument)
    sub.add_parser('verify-only', parents=[common])
    reconcile = sub.add_parser('reconcile', parents=[common])
    reconcile.add_argument('--execute', action='store_true', required=True)
    reconcile.add_argument('--transaction-id', required=True, type=_transaction_id_argument)
    reconcile.add_argument('--confirm-plan', required=True, type=_plan_digest_argument)
    reconcile.add_argument('--confirm', required=True,
                           help='Must exactly equal the transaction id')
    reconcile.add_argument('--plist', type=Path, required=True,
                           help='Installed LaunchAgent plist to preserve in the backup')
    reconcile.add_argument('--first-install', action='store_true',
                           help='Approve absent legacy manifest/plist only for a reviewed first install')
    rollback = sub.add_parser('rollback')
    rollback.add_argument('--private-root', type=Path, required=True)
    rollback.add_argument('--transaction-id', required=True, type=_transaction_id_argument)
    rollback.add_argument('--forward-transaction-id', required=True, type=_transaction_id_argument)
    rollback.add_argument('--confirm', required=True,
                          help='Must exactly equal ROLLBACK:<forward transaction id>')
    rollback.add_argument('--base-url', default=BASE_URL)
    rollback.add_argument('--owner-canonical-container', default=OWNER_CONTAINER)
    rollback.add_argument('--owner-explicit-container', default=OWNER_CONTAINER)
    for name in ('key-init', 'key-rotate'):
        key = sub.add_parser(name)
        key.add_argument('--private-root', type=Path, required=True)
    install = sub.add_parser('install')
    install.add_argument('--tracked-importer', type=Path, required=True)
    install.add_argument('--expected-sha256', required=True)
    install.add_argument('--stable-path', type=Path, required=True)
    install.add_argument('--backup-root', type=Path, required=True)
    install.add_argument('--plist', type=Path, required=True)
    install.add_argument('--interpreter', type=Path, required=True)
    reload_parser = sub.add_parser('reload-instructions')
    reload_parser.add_argument('--plist', type=Path, required=True)
    legacy = sub.add_parser('legacy-reconcile', add_help=False)
    legacy.add_argument('legacy_args', nargs=argparse.REMAINDER)
    return parser


def _transaction_id_argument(value: str) -> str:
    try:
        _validate_transaction_id(value)
    except TransactionJournalError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def _plan_digest_argument(value: str) -> str:
    if not _is_sha256(value):
        raise argparse.ArgumentTypeError('plan digest must be 64 lowercase hexadecimal characters')
    return value


def _phase2_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    root = Path(args.private_root)
    manifest = args.manifest or root / 'manifest/current.json'
    return (root, manifest, root / 'readiness/storage-reconciliation.json',
            root / 'readiness/vector-readiness.json')


def _preflight_phase2_paths(args: argparse.Namespace) -> None:
    """Reject unsafe input/output topology before clients, locks, or writes."""
    root = Path(args.private_root)
    source = Path(args.source_root)
    if (not root.is_absolute() or not source.is_absolute()
            or any(part in {'.', '..'} for part in root.parts[1:] + source.parts[1:])):
        raise SystemExit('source and private roots must be canonical absolute paths')
    if source.is_symlink() or not source.is_dir():
        raise SystemExit('source root must be a real directory')
    manifest = Path(args.manifest) if args.manifest is not None else root / 'manifest/current.json'
    if not manifest.is_absolute():
        raise SystemExit('manifest path must be absolute')
    try:
        child = manifest.relative_to(root).as_posix()
    except ValueError:
        raise SystemExit('manifest must be confined below private root') from None
    _private_child_parts(child)
    if root.exists():
        fd = _open_private_root(root, create=False)
        os.close(fd)
        if manifest.exists():
            private_json_read(root, child)
    elif args.command not in {'dry-run'}:
        raise SystemExit('private root must already exist')


def _client(base_url: str) -> Supermemory:
    # Credentials are obtained only from private configuration and are never
    # accepted on argv or included in summaries.
    return Supermemory(api_key=api_key(), base_url=base_url, timeout=30.0, max_retries=1)


def _scan_at(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    global ROOT
    old = ROOT
    try:
        ROOT = Path(root).resolve()
        scanned = canonical_documents()
        return scanned, eligible_documents(scanned)
    finally:
        ROOT = old


def _manifest_rows(root: Path, path: Path) -> dict[str, dict[str, Any]]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        raise PrivateArtifactError('manifest must be below the private root') from None
    try:
        payload = private_json_read(root, relative)
    except PrivateArtifactError as exc:
        if not path.exists():
            return {}
        raise exc
    rows = payload.get('documents')
    if not isinstance(rows, list):
        raise ReconciliationRequired('current manifest is malformed')
    return {row['relative_path']: row for row in rows
            if isinstance(row, dict) and isinstance(row.get('relative_path'), str)}


def _safe_summary(classification: dict[str, tuple[str, ...]], plan: Any) -> dict[str, Any]:
    # Paths and provider identifiers are intentionally absent from CLI output.
    actions: dict[str, int] = {}
    for row in plan:
        actions[row['action']] = actions.get(row['action'], 0) + 1
    return {'inventory': {key: len(value) for key, value in classification.items()},
            'actions': actions, 'change_count': sum(actions.values())}


def transaction_plan_from_inventory(
        current: dict[str, dict[str, Any]], inventory: dict[str, tuple[dict[str, Any], ...]],
        plan: Any) -> tuple[dict[str, Any], ...]:
    """Bind a privacy-safe approved plan to exact provider/canonical identities."""
    if any(row['action'] == 'operator_review' for row in plan):
        raise ReconciliationRequired('operator review is required before mutation')
    by_path: dict[str, list[dict[str, Any]]] = {}
    for container, rows in inventory.items():
        for raw in rows:
            row = dict(raw); row['_inventory_container'] = container
            meta = row.get('metadata')
            rel = meta.get('relative_path') if isinstance(meta, dict) else None
            if isinstance(rel, str):
                by_path.setdefault(rel, []).append(row)
    bound = []
    for approved in plan:
        action, rel = approved['action'], approved['relative_path']
        doc = current.get(rel)
        matches = by_path.get(rel, [])
        if action in {'replace', 'delete'} and len(matches) != 1:
            raise ReconciliationRequired('planned destructive identity is not unique')
        pre = provider_identity(matches[0], matches[0]['_inventory_container']) if matches else None
        post = None if action == 'delete' else {
            'custom_id': doc['custom_id'], 'container': container_for_doc(doc),
            'sha256': doc['sha256'], 'bytes': doc['bytes'], 'document_id': None,
        }
        expected = pre if action == 'delete' else post
        bound.append({'action': action, 'relative_path': rel,
                      'custom_id': expected['custom_id'],
                      'source_container': pre['container'] if pre else None,
                      'target_container': post['container'] if post else None,
                      'expected_sha256': expected['sha256'], 'expected_bytes': expected['bytes'],
                      'expected_pre_identity': pre, 'expected_post_identity': post})
    return _validated_transaction_plan(bound)


def _receipt_inventory(client: Supermemory, current: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    inventory = provider_inventory(client)
    rows = []
    for rel, doc in sorted(current.items()):
        matches = [(container, raw) for container, values in inventory.items() for raw in values
                   if _alias(raw, 'custom_id', 'customId') == doc['custom_id']]
        if len(matches) != 1:
            raise ReconciliationRequired('storage receipt identity count is not one')
        container, listed = matches[0]
        hydrated = client.documents.get(str(_field(listed, 'id')), timeout=15.0)
        validated = validate_backend_document(hydrated, doc)
        rows.append({'relative_path': rel, 'document_id': validated['document_id'],
                     'custom_id': doc['custom_id'], 'container': container,
                     'sha256': doc['sha256'], 'bytes': doc['bytes'], 'status': 'done',
                     'index_schema_version': INDEX_SCHEMA_VERSION, 'task_type': 'superrag',
                     'provenance': 'canonical_obsidian'})
    if len(rows) != sum(len(value) for value in inventory.values()):
        raise ReconciliationRequired('storage inventory contains collateral objects')
    return rows


def _vector_observations(client: Supermemory,
                         current: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    observations = {}
    for rel, doc in sorted(current.items()):
        accepted = None
        for probe in indexed_content_probes(doc):
            response = search_documents_v4(
                client, probe, container_tag=container_for_doc(doc), limit=1, timeout=30.0,
                filters={'AND': [{'key': 'source', 'value': 'obsidian'},
                                 {'key': 'relative_path', 'value': rel}]},
            )
            results = _field(response, 'results')
            total = _field(response, 'total')
            if (isinstance(results, list) and len(results) == 1 and total == 1
                    and indexed_chunk_proves_probe(str(_field(results[0], 'chunk', '')),
                                                   probe, doc['content'])):
                accepted = results
                break
        if accepted is None:
            raise ReconciliationRequired('vector proof failed for one or more documents')
        observations[rel] = accepted
    return observations


def _write_storage_receipt(root: Path, current: dict[str, dict[str, Any]], inventory: list[dict[str, Any]],
                           generation: str, generated_at: datetime) -> str:
    key = read_receipt_key(root / 'readiness/receipt-hmac.key')
    if key is None:
        raise ReconciliationRequired('readiness receipt key is unavailable or unsafe')
    value = build_storage_reconciliation_receipt(
        current, inventory, generation=generation, generated_at=generated_at, key=key,
        owner_canonical_container=CONTAINER)
    return private_json_write(root, 'readiness/storage-reconciliation.json', value)


def _write_vector_receipt(root: Path, current: dict[str, dict[str, Any]], observations: dict[str, list[Any]],
                          generation: str, generated_at: datetime) -> str:
    key = read_receipt_key(root / 'readiness/receipt-hmac.key')
    if key is None:
        raise ReconciliationRequired('readiness receipt key is unavailable or unsafe')
    value = build_vector_readiness_receipt(
        current, observations, generation=generation, generated_at=generated_at, key=key,
        owner_canonical_container=CONTAINER)
    return private_json_write(root, 'readiness/vector-readiness.json', value)


PUBLICATION_ARTIFACTS = (
    ('storage', 'readiness/storage-reconciliation.json'),
    ('vector', 'readiness/vector-readiness.json'),
    ('manifest', 'manifest/current.json'),
)
PUBLICATION_MARKER = 'publication/prepared.json'


def _private_unlink(root: Path, child: str) -> None:
    parts = _private_child_parts(child)
    root_fd = _open_private_root(root, create=False)
    parent_fd = _open_private_parent(root_fd, parts, create=False)
    try:
        try:
            os.unlink(parts[-1], dir_fd=parent_fd)
        except FileNotFoundError:
            return
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
        os.close(root_fd)


def _publication_marker_mac(marker: dict[str, Any], key: bytes) -> str:
    unsigned = {name: value for name, value in marker.items() if name != 'marker_hmac'}
    return hmac.new(key, canonical_bytes(unsigned), hashlib.sha256).hexdigest()


def _private_regular_exists(root: Path, child: str) -> bool:
    """Check for a confined regular file without creating or following anything."""
    parts = _private_child_parts(child)
    root_fd = parent_fd = -1
    try:
        root_fd = _open_private_root(root, create=False)
        try:
            parent_fd = _open_private_parent(root_fd, parts, create=False)
        except FileNotFoundError:
            return False
        try:
            info = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        _verify_private_stat(info, directory=False, label='private artifact')
        return True
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)
        if root_fd >= 0:
            os.close(root_fd)


def publication_recovery_required_read_only(root: Path) -> bool:
    """Inspect an interrupted publication without repairing or changing it."""
    try:
        marker_present = _private_regular_exists(root, PUBLICATION_MARKER)
    except Exception:
        # An unsafe marker entry is still an interrupted publication requiring
        # an operator-controlled mutating recovery path.
        return True
    if not marker_present:
        return False
    try:
        key = read_receipt_key(root / 'readiness/receipt-hmac.key')
        marker = private_json_read(root, PUBLICATION_MARKER)
        mac = marker.get('marker_hmac') if isinstance(marker, dict) else None
        if (key is None or not isinstance(mac, str)
                or not hmac.compare_digest(mac, _publication_marker_mac(marker, key))
                or marker.get('schema_version') != 1
                or marker.get('state') not in {'prepared', 'committed'}):
            raise ReconciliationRequired('invalid publication marker')
        artifacts = marker.get('artifacts')
        if not isinstance(artifacts, dict) or set(artifacts) != {
                name for name, _ in PUBLICATION_ARTIFACTS}:
            raise ReconciliationRequired('invalid publication marker')
        for name, canonical_live in PUBLICATION_ARTIFACTS:
            row = artifacts.get(name)
            if not isinstance(row, dict) or set(row) != {'live', 'old', 'new'}:
                raise ReconciliationRequired('invalid publication marker')
            live_child = row.get('live')
            if not isinstance(live_child, str):
                raise ReconciliationRequired('invalid publication marker')
            if name != 'manifest' and live_child != canonical_live:
                raise ReconciliationRequired('invalid publication marker')
            _private_child_parts(live_child)
            for version in ('old', 'new'):
                binding = row.get(version)
                if (not isinstance(binding, dict) or set(binding) != {'stage', 'sha256'}
                        or not isinstance(binding.get('stage'), str)):
                    raise ReconciliationRequired('invalid publication marker')
                _private_child_parts(binding['stage'])
                digest = binding.get('sha256')
                if digest is not None:
                    if not _is_sha256(digest):
                        raise ReconciliationRequired('invalid publication marker')
                    private_json_read(root, binding['stage'], expected_sha256=digest)
    except Exception:
        # Presence alone requires mutating recovery. Authentication failures and
        # partial triples deliberately collapse to the same privacy-safe result.
        pass
    return True


def _read_artifact(root: Path, child: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = private_json_read(root, child)
    except PrivateArtifactError:
        if not (root / child).exists():
            return None, None
        raise
    return value, hashlib.sha256(_json_bytes(value)).hexdigest()


def recover_publication(root: Path, *, fault_injector: Callable[[str], None] | None = None) -> bool:
    """Finish or restore an interrupted receipt+manifest generation."""
    marker_path = root / PUBLICATION_MARKER
    if not marker_path.exists():
        return False
    key = read_receipt_key(root / 'readiness/receipt-hmac.key')
    if key is None:
        raise ReconciliationRequired('publication marker cannot be authenticated')
    marker = private_json_read(root, PUBLICATION_MARKER)
    mac = marker.get('marker_hmac') if isinstance(marker, dict) else None
    if (not isinstance(mac, str) or not hmac.compare_digest(mac, _publication_marker_mac(marker, key))
            or marker.get('schema_version') != 1 or marker.get('state') not in {'prepared', 'committed'}):
        raise ReconciliationRequired('publication marker is malformed or unauthenticated')
    artifacts = marker.get('artifacts')
    if not isinstance(artifacts, dict) or set(artifacts) != {name for name, _ in PUBLICATION_ARTIFACTS}:
        raise ReconciliationRequired('publication marker artifact set is invalid')

    # Prefer completing the new generation whenever its complete staged triple
    # remains hash-exact. Otherwise restore the complete old triple.
    target = 'new'
    for name, _ in PUBLICATION_ARTIFACTS:
        row = artifacts[name]
        if not isinstance(row, dict) or set(row) != {'live', 'old', 'new'}:
            raise ReconciliationRequired('publication marker artifact binding is invalid')
        staged, digest = _read_artifact(root, row['new']['stage'])
        if staged is None or digest != row['new']['sha256']:
            target = 'old'
    if target == 'old':
        for name, _ in PUBLICATION_ARTIFACTS:
            old = artifacts[name]['old']
            if old['sha256'] is not None:
                staged, digest = _read_artifact(root, old['stage'])
                if staged is None or digest != old['sha256']:
                    raise ReconciliationRequired('neither publication generation is recoverable')

    for name, _ in PUBLICATION_ARTIFACTS:
        row = artifacts[name]
        selected = row[target]
        if selected['sha256'] is None:
            _private_unlink(root, row['live'])
        else:
            value = private_json_read(root, selected['stage'], expected_sha256=selected['sha256'])
            observed = private_json_write(root, row['live'], value)
            if observed != selected['sha256']:
                raise ReconciliationRequired('publication recovery write mismatch')
        if fault_injector:
            fault_injector(f'after_recover_{name}')
    marker['state'] = 'committed'
    marker['outcome'] = target
    marker['marker_hmac'] = _publication_marker_mac(marker, key)
    private_json_write(root, PUBLICATION_MARKER, marker)
    if fault_injector:
        fault_injector('after_committed_marker')
    _private_unlink(root, PUBLICATION_MARKER)
    if fault_injector:
        fault_injector('after_marker_cleanup')
    return True


def publish_readiness_pair(
        root: Path, current: dict[str, dict[str, Any]], inventory: list[dict[str, Any]],
        observations: dict[str, list[Any]], generation: str, generated_at: datetime,
        *, manifest: dict[str, Any] | None = None,
        manifest_child: str = 'manifest/current.json',
        fault_injector: Callable[[str], None] | None = None) -> tuple[str, str]:
    """Durably publish storage receipt, vector receipt, and manifest together."""
    recover_publication(root, fault_injector=fault_injector)
    key = read_receipt_key(root / 'readiness/receipt-hmac.key')
    if key is None:
        raise ReconciliationRequired('readiness receipt key is unavailable or unsafe')
    storage = build_storage_reconciliation_receipt(
        current, inventory, generation=generation, generated_at=generated_at, key=key,
        owner_canonical_container=CONTAINER)
    vector = build_vector_readiness_receipt(
        current, observations, generation=generation, generated_at=generated_at, key=key,
        owner_canonical_container=CONTAINER)
    fingerprint = source_fingerprint_from_documents(
        current, owner_canonical_container=CONTAINER)
    if not validate_readiness_receipts(storage, vector, key=key,
                                       current_fingerprint=fingerprint, now=generated_at):
        raise ReconciliationRequired('staged readiness receipt pair failed validation')
    tx = generation.split(':', 1)[0]
    if manifest is None:
        manifest = {'schema_version': INDEX_SCHEMA_VERSION, 'transaction_id': tx,
                    'generation': generation,
                    'documents': sorted(inventory, key=lambda row: row['relative_path'])}
    if manifest.get('generation') != generation or manifest.get('transaction_id') != tx:
        raise ReconciliationRequired('manifest generation is not bound to publication')
    values = {'storage': storage, 'vector': vector, 'manifest': manifest}
    live = dict(PUBLICATION_ARTIFACTS)
    live['manifest'] = manifest_child
    artifacts: dict[str, Any] = {}
    for name in ('storage', 'vector', 'manifest'):
        old, old_digest = _read_artifact(root, live[name])
        old_stage = f'transactions/{tx}/publication/old-{name}.json'
        if old is not None:
            assert private_json_write(root, old_stage, old) == old_digest
        new_stage = f'transactions/{tx}/publication/new-{name}.json'
        new_digest = private_json_write(root, new_stage, values[name])
        artifacts[name] = {'live': live[name],
                           'old': {'stage': old_stage, 'sha256': old_digest},
                           'new': {'stage': new_stage, 'sha256': new_digest}}
        if fault_injector:
            fault_injector(f'after_stage_{name}')
    marker = {'schema_version': 1, 'transaction_id': tx, 'generation': generation,
              'source_fingerprint': fingerprint, 'state': 'prepared', 'artifacts': artifacts}
    marker['marker_hmac'] = _publication_marker_mac(marker, key)
    private_json_write(root, PUBLICATION_MARKER, marker)
    if fault_injector:
        fault_injector('after_prepared_marker')
    recover_publication(root, fault_injector=fault_injector)
    return artifacts['storage']['new']['sha256'], artifacts['vector']['new']['sha256']


def _copy_private(source: Path, destination: Path, *, private_root: Path | None = None) -> str:
    data = _read_confined_regular(source, label='backup source')
    root = private_root or destination.parent
    try:
        child = destination.relative_to(root).as_posix()
    except ValueError:
        raise PrivateArtifactError('backup destination escapes private root') from None
    parts = _private_child_parts(child)
    root_fd = parent_fd = fd = -1
    try:
        root_fd = _open_private_root(root, create=True)
        parent_fd = _open_private_parent(root_fd, parts, create=True)
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent_fd)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError('short private backup write')
            view = view[written:]
        os.fsync(fd); os.close(fd); fd = -1; os.fsync(parent_fd)
    except OSError as exc:
        raise PrivateArtifactError(f'private backup write failed: {type(exc).__name__}') from None
    finally:
        if fd >= 0: os.close(fd)
        if parent_fd >= 0: os.close(parent_fd)
        if root_fd >= 0: os.close(root_fd)
    root_fd = parent_fd = fd = -1
    try:
        root_fd = _open_private_root(root, create=False)
        parent_fd = _open_private_parent(root_fd, parts, create=False)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        info = os.fstat(fd)
        _verify_private_stat(info, directory=False, label='private backup')
        observed = bytearray()
        while len(observed) <= len(data):
            block = os.read(fd, min(65536, len(data) + 1 - len(observed)))
            if not block: break
            observed.extend(block)
        if bytes(observed) != data:
            raise PrivateArtifactError('backup read-back mismatch')
    finally:
        if fd >= 0: os.close(fd)
        if parent_fd >= 0: os.close(parent_fd)
        if root_fd >= 0: os.close(root_fd)
    return hashlib.sha256(data).hexdigest()


def create_release_backup(root: Path, transaction_id: str, *, importer_path: Path | None = None,
                          manifest_path: Path | None = None, plist_path: Path | None = None,
                          first_install: bool = False) -> dict[str, Any]:
    """Create an immutable dated private backup without exposing key contents."""
    private_fd = _open_private_root(root, create=True)
    os.close(private_fd)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    child = f'backups/{stamp}-{transaction_id}'
    hashes = {}
    artifacts = {}
    sources = (('importer', importer_path), ('legacy-manifest', manifest_path),
               ('launch-agent.plist', plist_path))
    classified = []
    # Classify the complete required set before creating any backup artifact.
    for label, source in sources:
        if source is None:
            if not first_install:
                raise PrivateArtifactError(f'required backup source is unspecified: {label}')
            artifacts[label] = {'status': 'absent', 'reason': 'approved-first-install'}
            continue
        try:
            _read_confined_regular(source, label=f'backup source {label}')
        except PrivateArtifactError:
            if source.exists():
                raise
            if not first_install:
                raise PrivateArtifactError(f'required backup source is unexpectedly absent: {label}') from None
            artifacts[label] = {'status': 'absent', 'reason': 'approved-first-install'}
            continue
        classified.append((label, source))
    for label, source in classified:
        hashes[label] = _copy_private(source, root / child / label, private_root=root)
        artifacts[label] = {'status': 'present-backed-up-hash-verified', 'sha256': hashes[label]}
    key = read_receipt_key(root / 'readiness/receipt-hmac.key')
    if key is None:
        raise ReconciliationRequired('readiness receipt key is unavailable or unsafe')
    metadata_value = {'transaction_id': transaction_id, 'created_at': stamp, 'hashes': hashes,
                      'artifacts': artifacts,
                      'receipt_key': {'present': True,
                                      'key_id': hashlib.sha256(key).hexdigest()[:16]},
                      'key_contents_backed_up': False}
    digest = private_json_write(root, f'{child}/metadata.json', metadata_value)
    if private_json_read(root, f'{child}/metadata.json', expected_sha256=digest) != metadata_value:
        raise PrivateArtifactError('backup metadata read-back mismatch')
    return {'path': child, 'metadata_sha256': digest, 'hashes': hashes}


def secure_install(tracked: Path, stable: Path, backup_root: Path, expected_sha256: str,
                   plist_path: Path, interpreter: Path) -> dict[str, Any]:
    _require_private_io_capabilities()
    if not _is_sha256(expected_sha256):
        raise PrivateArtifactError('tracked importer or expected hash is invalid')
    source = _read_confined_regular(tracked, label='tracked importer')
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise PrivateArtifactError('tracked importer SHA-256 mismatch')
    _read_confined_regular(interpreter, label='interpreter')
    try:
        plist = plistlib.loads(_read_confined_regular(plist_path, label='LaunchAgent plist'))
    except PrivateArtifactError:
        raise
    except Exception:
        raise PrivateArtifactError('LaunchAgent plist is malformed') from None
    arguments = plist.get('ProgramArguments')
    if (not isinstance(arguments, list) or len(arguments) < 2
            or arguments[0] != str(interpreter) or arguments[1] != str(stable)
            or 'verify-only' not in arguments or 'reconcile' in arguments
            or '--execute' in arguments):
        raise PrivateArtifactError('LaunchAgent plist does not point at safe noninteractive command')
    backup = None
    backup_fd = _open_private_root(backup_root, create=True); os.close(backup_fd)
    if stable.exists():
        backup = backup_root / f'{stable.name}.{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
        _copy_private(stable, backup)
    stable_root_fd = _open_private_root(stable.parent, create=True)
    fd = -1; name = f'.{stable.name}.{secrets.token_hex(16)}.tmp'
    try:
        existing = None
        try: existing = os.stat(stable.name, dir_fd=stable_root_fd, follow_symlinks=False)
        except FileNotFoundError: pass
        if existing is not None:
            if not stat.S_ISREG(existing.st_mode) or existing.st_uid != os.getuid() or stat.S_IMODE(existing.st_mode) != 0o700:
                raise PrivateArtifactError('installed importer target is unsafe')
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o700, dir_fd=stable_root_fd)
        view = memoryview(source)
        while view:
            written = os.write(fd, view)
            if written <= 0: raise OSError('short install write')
            view = view[written:]
        os.fsync(fd); os.close(fd); fd = -1
        os.replace(name, stable.name, src_dir_fd=stable_root_fd, dst_dir_fd=stable_root_fd)
        name = ''; os.fsync(stable_root_fd)
    finally:
        if fd >= 0: os.close(fd)
        if name:
            try: os.unlink(name, dir_fd=stable_root_fd)
            except FileNotFoundError: pass
        os.close(stable_root_fd)
    installed_fd = os.open(stable, os.O_RDONLY | os.O_NOFOLLOW)
    try: installed = os.read(installed_fd, len(source) + 1)
    finally: os.close(installed_fd)
    if installed != source or hashlib.sha256(installed).hexdigest() != expected_sha256:
        raise PrivateArtifactError('installed importer read-back hash mismatch')
    return {'installed_sha256': expected_sha256, 'backup_created': backup is not None,
            'launch_agent_reloaded': False}


def _phase2_main(argv: list[str]) -> None:
    if argv and argv[0] == 'legacy-reconcile':
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], *argv[1:]]
            legacy_main()
        finally:
            sys.argv = old_argv
        return
    args = _phase2_parser().parse_args(argv)
    if hasattr(args, 'owner_canonical_container'):
        configure_destinations(args.owner_canonical_container, args.owner_explicit_container)
    if args.command in {'key-init', 'key-rotate'}:
        key_id = generate_receipt_key(
            args.private_root / 'readiness/receipt-hmac.key', rotate=args.command == 'key-rotate')
        print(json.dumps({'command': args.command, 'key_id': key_id,
                          'receipts_valid': False, 'reverification_required': True}))
        return
    if args.command == 'install':
        result = secure_install(args.tracked_importer, args.stable_path, args.backup_root,
                                args.expected_sha256, args.plist, args.interpreter)
        print(json.dumps(result)); return
    if args.command == 'reload-instructions':
        print(json.dumps({'launch_agent_reloaded': False, 'manual_command':
                          f'launchctl bootout gui/$UID {args.plist} && launchctl bootstrap gui/$UID {args.plist}'}))
        return
    if args.command == 'rollback':
        expected = f'ROLLBACK:{args.forward_transaction_id}'
        if args.confirm != expected:
            raise SystemExit('rollback confirmation token mismatch')
        root = Path(args.private_root)
        plan_payload = private_json_read(
            root, f'transactions/{args.forward_transaction_id}/plan.json')
        if plan_payload.get('transaction_id') != args.forward_transaction_id:
            raise ReconciliationRequired('forward plan transaction mismatch')
        state = DurableRollbackExecutor(_client(args.base_url), root).execute(
            plan_payload['records'], args.forward_transaction_id, args.transaction_id)
        print(json.dumps({'mode': 'rollback', 'complete': state['complete'],
                          'transaction_id': args.transaction_id, 'readiness': False}))
        return

    _preflight_phase2_paths(args)
    root, manifest_path, _, _ = _phase2_paths(args)
    if args.command == 'verify-only' and publication_recovery_required_read_only(root):
        print(json.dumps({'mode': 'verify-only', 'publication_recovery_required': True,
                          'filesystem_mutated': False, 'backend_mutated': False}))
        raise SystemExit(1)
    if args.command == 'reconcile':
        recover_publication(root)
    scanned, current = _scan_at(args.source_root)
    if args.command == 'dry-run':
        previous = _manifest_rows(root, manifest_path) if manifest_path.exists() else {}
        changed = {rel for rel in set(current) & set(previous)
                   if current[rel]['sha256'] != previous[rel].get('sha256')
                   or container_for_doc(current[rel]) != container_for_row(previous[rel])}
        counts = {'add': len(set(current) - set(previous)), 'replace': len(changed),
                  'delete': len(set(previous) - set(current))}
        print(json.dumps({'mode': 'dry-run', 'scanned': len(scanned), 'eligible': len(current),
                          'actions': counts, 'filesystem_mutated': False,
                          'provider_client_created': False, 'backend_mutated': False}))
        return
    client = _client(args.base_url)
    inventory = provider_inventory(client)
    classification = classify_inventory(current, inventory)
    approved = reconciliation_plan(classification)
    summary = _safe_summary(classification, approved)
    if args.command == 'verify-only':
        storage_rows = _receipt_inventory(client, current)
        observations = _vector_observations(client, current)
        key = read_receipt_key(root / 'readiness/receipt-hmac.key')
        try:
            storage = private_json_read(root, 'readiness/storage-reconciliation.json')
            vector = private_json_read(root, 'readiness/vector-readiness.json')
            receipts_valid = key is not None and validate_readiness_receipts(
                storage, vector, key=key,
                current_fingerprint=source_fingerprint_from_documents(
                    current, owner_canonical_container=CONTAINER),
                now=datetime.now(timezone.utc))
            # Re-run the complete vector receipt validator against the live
            # search/get observations without persisting its transient proof.
            live_vector_valid = key is not None and bool(build_vector_readiness_receipt(
                current, observations, generation=str(vector.get('generation', '')),
                generated_at=datetime.now(timezone.utc), key=key,
                owner_canonical_container=CONTAINER))
        except Exception:
            receipts_valid = False
            live_vector_valid = False
        complete = (not approved and classification['expected'] == tuple(sorted(current))
                    and len(storage_rows) == len(current) and len(observations) == len(current)
                    and receipts_valid and live_vector_valid)
        print(json.dumps({'mode': 'verify-only', **summary, 'verification_complete': complete,
                          'storage_verified_count': len(storage_rows),
                          'vector_verified_count': len(observations),
                          'receipts_valid': receipts_valid,
                          'filesystem_mutated': False, 'backend_mutated': False}))
        if not complete: raise SystemExit(1)
        return
    proposed_plan = transaction_plan_from_inventory(current, inventory, approved)
    proposed_digest = transaction_plan_digest(proposed_plan)
    if args.command == 'plan':
        print(json.dumps({'mode': 'plan', **summary, 'transaction_id': args.transaction_id,
                          'plan_digest': proposed_digest,
                          'containers': sorted(CANONICAL_CONTAINERS),
                          'reason_codes': sorted({row['action'] for row in proposed_plan}),
                          'filesystem_mutated': False, 'backend_mutated': False}))
        return
    if args.confirm != args.transaction_id:
        raise SystemExit('reconcile confirmation token mismatch')

    plan_child = f'transactions/{args.transaction_id}/plan.json'
    plan_path = root / plan_child
    if plan_path.exists():
        plan_payload = private_json_read(root, plan_child)
        plan = _validated_transaction_plan(plan_payload.get('records'))
        digest = transaction_plan_digest(plan)
        if plan_payload.get('transaction_id') != args.transaction_id or plan_payload.get('plan_digest') != digest:
            raise TransactionJournalError('persisted transaction plan binding is invalid')
        summary = _safe_summary(classification, plan)
    else:
        plan = proposed_plan
        digest = proposed_digest
    if args.confirm_plan != digest:
        raise SystemExit('reconcile plan digest mismatch')

    if not plan_path.exists():
        backup = create_release_backup(
            root, args.transaction_id, importer_path=Path(__file__).resolve(),
            manifest_path=manifest_path, plist_path=args.plist, first_install=args.first_install,
        )
        plan_payload = {'transaction_id': args.transaction_id, 'records': _plain_json(plan),
                        'plan_digest': digest, 'backup': backup}
        plan_digest = private_json_write(root, plan_child, plan_payload)
        if private_json_read(root, plan_child, expected_sha256=plan_digest) != plan_payload:
            raise PrivateArtifactError('forward plan read-back mismatch')
    try:
        state = DurableReconciliationExecutor(client, root).execute(current, plan, args.transaction_id)
        generated_at = datetime.now(timezone.utc)
        generation = f'{args.transaction_id}:{state["verification_digest"]}'
        storage_rows = _receipt_inventory(client, current)
        # Both deep proofs complete before either public receipt is replaced.
        observations = _vector_observations(client, current)
        manifest = {'schema_version': INDEX_SCHEMA_VERSION, 'transaction_id': args.transaction_id,
                    'generation': generation,
                    'documents': sorted(storage_rows, key=lambda row: row['relative_path'])}
        storage_digest, vector_digest = publish_readiness_pair(
            root, current, storage_rows, observations, generation, generated_at,
            manifest=manifest, manifest_child=manifest_path.relative_to(root).as_posix())
    except Exception:
        print(json.dumps({'mode': 'reconcile', 'transaction_id': args.transaction_id,
                          'readiness': False, 'resume': f'reconcile transaction {args.transaction_id}',
                          'rollback': f'rollback forward transaction {args.transaction_id}'}))
        raise SystemExit(1) from None
    print(json.dumps({'mode': 'reconcile', **summary, 'transaction_id': args.transaction_id,
                      'storage_receipt_sha256': storage_digest,
                      'vector_receipt_sha256': vector_digest, 'behavioral_receipt_written': False,
                      'readiness': True}))


def main() -> None:
    _phase2_main(sys.argv[1:])


if __name__ == '__main__':
    main()
