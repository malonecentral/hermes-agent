#!/Users/dennis/.hermes/hermes-agent/venv/bin/python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable

from supermemory import Supermemory
from plugins.memory.supermemory.canonical_policy import (
    CANONICAL_CONTAINERS,
    FAMILY_CONTAINER,
    OWNER_CONTAINER,
    canonical_scope_from_path,
    canonical_visibility_from_path,
    stable_custom_id_from_path,
)
from plugins.memory.supermemory.search_v4 import normalize_document_chunk, search_documents_v4

ROOT = Path('/Users/dennis/Documents/ObsidianVault/Personal/Hermes').resolve()
ENV = Path('/Users/dennis/.hermes/.env')
OUT = Path('/Users/dennis/.hermes/obsidian-supermemory-import.json')
LOCK = Path('/Users/dennis/.hermes/obsidian-supermemory-sync.lock')
BASE_URL = 'http://127.0.0.1:6767'
CONTAINER = OWNER_CONTAINER
CONVERSATION_CONTAINER = 'owner_conversations'
INDEX_SCHEMA_VERSION = 4
SENSITIVE_CONTENT = re.compile(
    r'(?i)(?:[#?&](?:access_)?token=|(?:api[_-]?key|password|secret)\s*[:=])'
)
TERMINAL = {'done', 'failed', 'error'}


class ReconciliationRequired(RuntimeError):
    """Backend identity could not be proven; mutation must stop."""

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
    return scope


def container_for_row(row: dict[str, Any]) -> str:
    # Pre-isolation manifests implicitly placed every canonical record in Owner.
    return str(row.get('container') or CONTAINER)

def _frontmatter_values(source: str) -> dict[str, list[str]]:
    values = {}
    if not source.startswith("---\n"): return values
    end = source.find("\n---\n", 4)
    if end < 0: raise ValueError("malformed frontmatter")
    for line in source[4:end].splitlines():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*):\s*(.*?)\s*", line)
        if not match: continue
        key, raw = match.groups()
        if key in values: raise ValueError("duplicate frontmatter field")
        if raw.startswith("["):
            if not raw.endswith("]"): raise ValueError("malformed frontmatter list")
            values[key] = [v.strip().strip("\"'") for v in raw[1:-1].split(",") if v.strip()]
        elif raw in {"", "null", "[]"}: values[key] = []
        elif raw[:1] in {"|", ">"}: raise ValueError("unsupported multiline frontmatter")
        else: values[key] = [raw.strip("\"'")]
    return values

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
    entity_type = fields.get("type") or ("restaurant" if "/Food/Restaurants/" in f"/{rel}" else "dish" if "/Food/Dishes/" in f"/{rel}" else "person" if "/People/" in f"/{rel}" else "document")
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
        derived_container = canonical_scope_from_path(rel)
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
    raw_rows = [(container, raw) for container in sorted(inventory) for raw in inventory[container]]
    tainted_rows, tainted_tokens = _globally_tainted_rows(raw_rows)
    if tainted_rows:
        categories['identity_collision'].add('*')
    for row_index, (container, raw) in enumerate(raw_rows):
        if not isinstance(raw, dict):
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
    parser.add_argument('--expected-eligible', type=int)
    parser.add_argument('--expected-owner-private', type=int)
    parser.add_argument('--expected-family-shared', type=int)
    args = parser.parse_args()
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
        reason = ('sensitive_content' if SENSITIVE_CONTENT.search(doc.get('content', ''))
                  else 'template' if doc.get('is_template') else '')
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
    """Combine dual inventory, backend validation, and exhaustive indexed readback."""
    try:
        inventory = schema_v4_backend_inventory(client, {doc['custom_id'] for doc in current.values()})
    except Exception:
        inventory = None
    reconciled = 0
    if inventory is not None:
        for doc in current.values():
            try:
                validate_backend_document(inventory.get(doc['custom_id']), doc)
            except Exception:
                continue
            reconciled += 1
    inventory_complete = inventory is not None and reconciled == len(current)
    search = verify_indexed_documents(client, current, inventory)
    return {**canonical_container_fields(current), **search,
            'backend_reconciled_count': reconciled,
            'inventory_complete': inventory_complete,
            'reconciliation_complete': inventory_complete and search['search_readiness_complete'],
            'submission_failure_count': 0, 'still_pending_count': 0}


def main() -> None:
    args = parse_args()
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
        previous = load_previous(); scanned = canonical_documents()
        current = eligible_documents(scanned)
        changed = {r for r in set(current) & set(previous) if current[r]['sha256'] != previous[r].get('sha256') or previous[r].get('index_schema_version') != INDEX_SCHEMA_VERSION or container_for_row(previous[r]) != container_for_doc(current[r])}
        print(json.dumps({'scanned': len(scanned), 'eligible': len(current), 'new': len(set(current)-set(previous)), 'changed': len(changed), 'unchanged': len(set(current)&set(previous)-changed), 'removed': len(set(previous)-set(current)), 'dry_run': True}, indent=2)); return

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


if __name__ == '__main__':
    main()
