"""Authenticated, source-bound readiness receipts for canonical indexing.

The receipt key detects offline receipt tampering by processes that cannot read the
private key. Compromise by the same OS user (and therefore of the key) is outside
this boundary; receipts are not a substitute for an isolated signing service.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .canonical_policy import (
    CANONICAL_CONTAINERS,
    canonical_entity_type,
    canonical_exclusion_reason,
    canonical_frontmatter_values,
    canonical_scope_from_path,
    stable_custom_id_from_path,
)

SHA256 = re.compile(r"[0-9a-f]{64}")
STORAGE_SCHEMA = "hermes.storage-reconciliation/v2"
VECTOR_SCHEMA = "hermes.vector-readiness/v2"
VECTOR_METHOD = "normalized-contiguous-current-token-run"
VECTOR_METHOD_VERSION = 1
UNIQUE_RUN_TOKENS = 6
UNIQUE_RUN_INFORMATIVE_TOKENS = 4
UNIQUE_RUN_INFORMATIVE_CHARS = 28
DEFAULT_MAX_AGE = timedelta(days=7)
MAX_SOURCE_FILES = 100_000
MAX_SOURCE_BYTES = 512 * 1024 * 1024
MAX_RECEIPT_BYTES = 1024 * 1024
KEY_BYTES = 32


def _current_uid() -> int:
    getter = getattr(os, "getuid", None)
    if getter is None:
        raise OSError("owned private receipts are unsupported on this platform")
    return int(getter())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("receipt timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_rows(current: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path, doc in current.items():
        container = canonical_scope_from_path(path)
        if (container is None or doc.get("relative_path") != path
                or doc.get("custom_id") != stable_custom_id_from_path(path)
                or not isinstance(doc.get("sha256"), str) or not SHA256.fullmatch(doc["sha256"])
                or type(doc.get("bytes")) is not int or doc["bytes"] < 0):
            raise ValueError("invalid canonical source document")
        rows.append({"relative_path": path, "sha256": doc["sha256"], "bytes": doc["bytes"],
                     "container": container})
    if not rows:
        raise ValueError("canonical source set is empty")
    return sorted(rows, key=lambda row: row["relative_path"])


def source_fingerprint_from_documents(current: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = _source_rows(current)
    counts = {name: sum(row["container"] == name for row in rows)
              for name in sorted(CANONICAL_CONTAINERS)}
    return {"method": "canonical-relative-path-byte-sha256", "method_version": 1,
            "digest": canonical_digest(rows), "document_count": len(rows),
            "byte_count": sum(row["bytes"] for row in rows), "container_counts": counts}


def scan_canonical_source(root: Path) -> dict[str, Any]:
    """Read a bounded canonical tree without following symlinks; fail on any scan error."""
    root = Path(root)
    rows: list[dict[str, Any]] = []
    total = 0
    root_fd = -1
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_id = (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino)
        expected = {".": root_id}
        visited: set[str] = set()
        for directory, names, files, directory_fd in os.fwalk(
                ".", topdown=True, follow_symlinks=False, dir_fd=root_fd,
                onerror=lambda exc: (_ for _ in ()).throw(exc)):
            key = Path(directory).as_posix()
            info = os.fstat(directory_fd)
            if expected.get(key) != (info.st_dev, info.st_ino):
                raise OSError("canonical directory identity changed")
            visited.add(key)
            kept = []
            for name in sorted(names):
                if name == ".obsidian" or name.startswith("."):
                    continue
                if key == "." and name not in {"Jarvis", "Skills"}:
                    continue
                child = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if not stat.S_ISDIR(child.st_mode):
                    raise OSError("canonical directory entry is unsafe")
                child_key = (Path(directory) / name).as_posix()
                expected[child_key] = (child.st_dev, child.st_ino)
                kept.append(name)
            names[:] = kept
            base = Path() if key == "." else Path(directory)
            for name in sorted(files):
                if name.startswith(".") or not name.endswith(".md"):
                    continue
                rel = (base / name).as_posix()
                if canonical_scope_from_path(rel) is None:
                    raise OSError("non-canonical markdown path")
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                try:
                    file_info = os.fstat(fd)
                    if not stat.S_ISREG(file_info.st_mode):
                        raise OSError("canonical source is not regular")
                    size = file_info.st_size
                    total += size
                    if len(rows) >= MAX_SOURCE_FILES or total > MAX_SOURCE_BYTES:
                        raise OSError("canonical source bounds exceeded")
                    raw = bytearray()
                    while len(raw) < size:
                        block = os.read(fd, min(1024 * 1024, size - len(raw)))
                        if not block:
                            break
                        raw.extend(block)
                    if len(raw) != size or os.fstat(fd).st_size != size:
                        raise OSError("canonical source changed during read")
                finally:
                    os.close(fd)
                try:
                    source = bytes(raw).decode("utf-8")
                    parsed = canonical_frontmatter_values(source)
                    fields = {k: v[0] for k, v in parsed.items() if len(v) == 1}
                    entity_type = canonical_entity_type(rel, fields)
                    excluded = canonical_exclusion_reason(rel, source, entity_type)
                except (UnicodeDecodeError, ValueError) as exc:
                    raise OSError("canonical source decode or policy failed") from exc
                if not excluded:
                    rows.append({"relative_path": rel, "sha256": hashlib.sha256(raw).hexdigest(),
                                 "bytes": size, "container": canonical_scope_from_path(rel)})
        if visited != set(expected) or (os.fstat(root_fd).st_dev, os.fstat(root_fd).st_ino) != root_id:
            raise OSError("canonical traversal changed or was incomplete")
    except OSError as exc:
        raise RuntimeError(f"canonical source scan failed: {type(exc).__name__}") from None
    finally:
        if root_fd >= 0:
            os.close(root_fd)
    if not rows:
        raise RuntimeError("canonical source scan found no documents")
    rows.sort(key=lambda row: row["relative_path"])
    counts = {name: sum(row["container"] == name for row in rows)
              for name in sorted(CANONICAL_CONTAINERS)}
    return {"method": "canonical-relative-path-byte-sha256", "method_version": 1,
            "digest": canonical_digest(rows), "document_count": len(rows),
            "byte_count": sum(row["bytes"] for row in rows),
            "container_counts": counts}


def _safe_private_dir(path: Path) -> None:
    path = Path(path)
    pending = []
    cursor = path
    while not cursor.exists():
        pending.append(cursor)
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise OSError("private root ancestor is unsafe")
    for directory in reversed(pending):
        os.mkdir(directory, 0o700)
    absolute = path.absolute()
    chain = [absolute]
    while chain[-1] != chain[-1].parent:
        chain.append(chain[-1].parent)
    for directory in reversed(chain):
        info = os.lstat(directory)
        if not stat.S_ISDIR(info.st_mode):
            raise OSError("private directory path contains a symlink")
    info = os.lstat(path)
    if info.st_uid != _current_uid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise OSError("private directory is not owned mode 0700")


def generate_receipt_key(key_path: Path, *, rotate: bool = False) -> str:
    """Explicitly create or rotate a private receipt key; never called by validation."""
    key_path = Path(key_path)
    _safe_private_dir(key_path.parent)
    if key_path.exists() and not rotate:
        raise FileExistsError("receipt key already exists")
    if rotate and read_receipt_key(key_path) is None:
        raise OSError("existing receipt key is unavailable or unsafe")
    temporary = key_path.parent / f".{key_path.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        key = secrets.token_bytes(KEY_BYTES)
        os.write(fd, key)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, key_path)
    os.chmod(key_path, 0o600, follow_symlinks=False)
    return hashlib.sha256(key).hexdigest()[:16]


def read_receipt_key(path: Path) -> bytes | None:
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != _current_uid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size < KEY_BYTES or info.st_size > 4096):
            return None
        key = os.read(fd, info.st_size + 1)
        return key if len(key) == info.st_size else None
    except OSError:
        return None
    finally:
        if fd >= 0:
            os.close(fd)


def _sign(payload: dict[str, Any], key: bytes) -> dict[str, Any]:
    if not isinstance(key, bytes) or len(key) < KEY_BYTES:
        raise ValueError("receipt authentication key is too short")
    unsigned = dict(payload)
    unsigned["authentication"] = {"algorithm": "HMAC-SHA256",
                                  "key_id": hashlib.sha256(key).hexdigest()[:16]}
    unsigned["authentication"]["tag"] = hmac.new(key, canonical_bytes(unsigned), hashlib.sha256).hexdigest()
    return unsigned


def _authenticate(value: Any, schema: str, key: bytes) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("receipt_schema") != schema:
        return None
    auth = value.get("authentication")
    if not isinstance(auth, dict) or auth.get("algorithm") != "HMAC-SHA256" or auth.get("key_id") != hashlib.sha256(key).hexdigest()[:16]:
        return None
    tag = auth.get("tag")
    unsigned = dict(value); unsigned_auth = dict(auth); unsigned_auth.pop("tag", None)
    unsigned["authentication"] = unsigned_auth
    expected = hmac.new(key, canonical_bytes(unsigned), hashlib.sha256).hexdigest()
    return value if isinstance(tag, str) and hmac.compare_digest(tag, expected) else None


def _base(schema: str, fingerprint: dict[str, Any], generation: str, generated_at: datetime) -> dict[str, Any]:
    if not isinstance(generation, str) or not generation:
        raise ValueError("generation is required")
    return {"receipt_schema": schema, "generation": generation, "generated_at": _timestamp(generated_at),
            "index_schema_version": 4, "source_fingerprint": fingerprint}


def build_storage_reconciliation_receipt(current: Mapping[str, Mapping[str, Any]], inventory: list[dict[str, Any]], *, generation: str, generated_at: datetime, key: bytes) -> dict[str, Any]:
    fingerprint = source_fingerprint_from_documents(current)
    expected = {row["relative_path"]: row for row in _source_rows(current)}
    seen: set[str] = set()
    document_ids: set[str] = set()
    for row in inventory:
        path = row.get("relative_path") if isinstance(row, dict) else None
        source = expected.get(path)
        identity = row.get("document_id") if isinstance(row, dict) else None
        if (source is None or path in seen or not isinstance(identity, str) or not identity
                or identity in document_ids or row.get("custom_id") != stable_custom_id_from_path(path)
                or row.get("container") != source["container"] or row.get("sha256") != source["sha256"]
                or row.get("bytes") != source["bytes"] or row.get("status") != "done"
                or row.get("index_schema_version") != 4 or row.get("task_type") != "superrag"
                or row.get("provenance") != "canonical_obsidian"):
            raise ValueError("storage inventory does not exactly prove canonical source")
        seen.add(str(path))
        document_ids.add(identity)
    if seen != set(expected):
        raise ValueError("storage inventory is incomplete")
    payload = _base(STORAGE_SCHEMA, fingerprint, generation, generated_at) | {
        "task_type": "superrag", "provenance": "canonical_obsidian",
        "expected_count": len(expected), "verified_count": len(seen),
        "orphan_count": 0, "duplicate_count": 0, "pending_count": 0,
        "failed_count": 0, "malformed_count": 0,
        "inventory_proof_digest": canonical_digest(sorted(
            ({"path_hash": hashlib.sha256(path.encode()).hexdigest(),
              "document_id_hash": hashlib.sha256(str(row["document_id"]).encode()).hexdigest()}
             for path, row in ((r["relative_path"], r) for r in inventory)),
            key=lambda item: item["path_hash"])),
    }
    return _sign(payload, key)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


_MISSING = object()
_PROOF_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "there", "they",
    "this", "to", "was", "we", "were", "will", "with", "you", "your",
})


def _alias(value: Any, snake: str, camel: str, *, required: bool = False) -> Any:
    """Accept real SDK/wire variants, but never ambiguous duplicate aliases."""
    left = _field(value, snake, _MISSING)
    right = _field(value, camel, _MISSING)
    if left is not _MISSING and right is not _MISSING and left != right:
        raise ValueError("vector observation aliases conflict")
    found = left if left is not _MISSING else right
    if found is _MISSING and required:
        raise ValueError("vector observation required field is missing")
    return None if found is _MISSING else found


def _container_claim(value: Any, expected: str, *, required: bool) -> None:
    claims = []
    for name, plural in (("container_tags", True), ("containerTags", True),
                         ("container_tag", False), ("containerTag", False)):
        claim = _field(value, name, _MISSING)
        if claim is not _MISSING:
            claims.append((claim, plural))
    if (required and len(claims) != 1) or (not required and len(claims) > 1):
        raise ValueError("vector container claim is missing or ambiguous")
    if claims:
        claim, plural = claims[0]
        if (plural and (type(claim) is not list or claim != [expected])) or (
                not plural and (type(claim) is not str or claim != expected)):
            raise ValueError("vector container claim is invalid")


def _qualifying_run(run: tuple[str, ...]) -> bool:
    informative = [token for token in run if token not in _PROOF_STOPWORDS]
    return (len(informative) >= UNIQUE_RUN_INFORMATIVE_TOKENS
            and sum(map(len, informative)) >= UNIQUE_RUN_INFORMATIVE_CHARS)


def _valid_provider_id(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", value))


def _unique_run_index(current: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, ...], frozenset[str]]:
    """Build the corpus-wide fixed-width proof index once per receipt build."""
    owners: dict[tuple[str, ...], set[str]] = {}
    for path, doc in current.items():
        body = str(doc.get("content", ""))
        # The importer prepends a generated identity envelope. It is useful for
        # retrieval but must not turn a tiny/boilerplate source into content
        # proof merely because its path is unique.
        marker = "[/canonical-identity]"
        source_body = body.split(marker, 1)[1] if marker in body else body
        tokens = _tokens(source_body)
        for offset in range(len(tokens) - UNIQUE_RUN_TOKENS + 1):
            run = tuple(tokens[offset:offset + UNIQUE_RUN_TOKENS])
            if _qualifying_run(run):
                owners.setdefault(run, set()).add(path)
    return {run: frozenset(paths) for run, paths in owners.items()}


def build_vector_readiness_receipt(current: Mapping[str, Mapping[str, Any]], observations: Mapping[str, list[Any]], *, generation: str, generated_at: datetime, key: bytes) -> dict[str, Any]:
    """Verify production-shaped indexed observations and derive all proof digests."""
    fingerprint = source_fingerprint_from_documents(current)
    if set(observations) != set(current):
        raise ValueError("vector observations do not exactly cover canonical source")
    proof_rows = []
    unique_runs = _unique_run_index(current)
    parent_ids: set[str] = set()
    chunk_ids: set[str] = set()
    used_runs: set[tuple[str, ...]] = set()
    for path in sorted(current):
        doc = current[path]; results = observations[path]
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError("each canonical document requires exactly one observation")
        result = results[0]; metadata = _field(result, "metadata")
        parents = _field(result, "documents", [])
        chunk = _field(result, "chunk")
        if not isinstance(metadata, dict) or not isinstance(parents, list) or len(parents) != 1 or not isinstance(chunk, str):
            raise ValueError("vector observation shape is invalid")
        parent = parents[0]; parent_meta = _field(parent, "metadata")
        expected_meta = {"source": "obsidian", "authority": "canonical", "relative_path": path,
                         "content_sha256": doc["sha256"], "content_bytes": doc["bytes"],
                         "index_schema_version": 4,
                         "visibility": "family_shared" if canonical_scope_from_path(path) == "family_shared" else "owner_private",
                         "identity_scope": "owner", "canonical_root": "owner", **dict(doc.get("identity", {})),
                         **dict(doc.get("governed_identity", {})),
                         **{key: doc[key] for key in ("event_date", "event_date_ordinal", "event_year", "event_month")
                            if doc.get(key) not in (None, "")}}
        expected_meta = {key: value for key, value in expected_meta.items() if value not in (None, "")}
        parent_id = _field(parent, "id", "")
        chunk_id = _field(result, "id", "")
        parent_custom = str(_alias(parent, "custom_id", "customId", required=True) or "")
        document_id = str(_alias(result, "document_id", "documentId", required=True) or "")
        aggregated = _alias(result, "is_aggregated", "isAggregated")
        task_type = _alias(parent, "task_type", "taskType", required=True)
        status = str(_field(parent, "status", "") or "").lower()
        container = canonical_scope_from_path(path)
        _container_claim(parent, str(container), required=True)
        _container_claim(result, str(container), required=False)
        if (metadata != parent_meta or metadata != expected_meta
                or not _valid_provider_id(chunk_id) or chunk_id in chunk_ids
                or not _valid_provider_id(parent_id) or parent_id in parent_ids
                or document_id != parent_id or parent_custom != doc["custom_id"]
                or aggregated not in (None, False) or task_type != "superrag" or status != "done"):
            raise ValueError("vector identity, schema, provenance, or container is invalid")
        body_tokens = _tokens(str(doc.get("content", ""))); chunk_tokens = _tokens(chunk)
        if not any(body_tokens[i:i + len(chunk_tokens)] == chunk_tokens
                   for i in range(len(body_tokens) - len(chunk_tokens) + 1)):
            raise ValueError("vector chunk is not a current contiguous token run")
        proof_run = next((tuple(chunk_tokens[i:i + UNIQUE_RUN_TOKENS])
                          for i in range(len(chunk_tokens) - UNIQUE_RUN_TOKENS + 1)
                          if unique_runs.get(tuple(chunk_tokens[i:i + UNIQUE_RUN_TOKENS])) == frozenset({path})
                          and tuple(chunk_tokens[i:i + UNIQUE_RUN_TOKENS]) not in used_runs), None)
        if proof_run is None:
            raise ValueError("vector chunk has no unused corpus-unique content proof")
        proof_rows.append({"current_body_sha256": hashlib.sha256(str(doc["content"]).encode()).hexdigest(),
                           "observed_chunk_digest": hashlib.sha256("\u001f".join(proof_run).encode()).hexdigest(),
                           "parent_id_digest": hashlib.sha256(parent_id.encode()).hexdigest()})
        parent_ids.add(parent_id)
        chunk_ids.add(chunk_id)
        used_runs.add(proof_run)
    payload = _base(VECTOR_SCHEMA, fingerprint, generation, generated_at) | {
        "method": VECTOR_METHOD, "method_version": VECTOR_METHOD_VERSION,
        "expected_document_count": len(current), "verified_document_count": len(proof_rows),
        "verified_chunk_count": len(proof_rows), "failure_count": 0,
        "proof_set_digest": canonical_digest(proof_rows),
    }
    return _sign(payload, key)


def _fresh(value: dict[str, Any], now: datetime, max_age: timedelta) -> bool:
    try:
        generated = datetime.fromisoformat(value["generated_at"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return False
    return generated.tzinfo is not None and timedelta(0) <= now.astimezone(timezone.utc) - generated <= max_age


def validate_readiness_receipts(storage: Any, vector: Any, *, key: bytes, current_fingerprint: Mapping[str, Any], now: datetime | None = None, max_age: timedelta = DEFAULT_MAX_AGE) -> bool:
    storage = _authenticate(storage, STORAGE_SCHEMA, key); vector = _authenticate(vector, VECTOR_SCHEMA, key)
    now = now or datetime.now(timezone.utc)
    if not storage or not vector or not _fresh(storage, now, max_age) or not _fresh(vector, now, max_age):
        return False
    fingerprint = dict(current_fingerprint)
    expected = fingerprint.get("document_count")
    return bool(type(expected) is int and expected > 0
                and storage.get("generation") == vector.get("generation")
                and storage.get("source_fingerprint") == vector.get("source_fingerprint") == fingerprint
                and storage.get("expected_count") == storage.get("verified_count") == expected
                and vector.get("expected_document_count") == vector.get("verified_document_count") == expected
                and vector.get("verified_chunk_count") == expected and vector.get("failure_count") == 0
                and all(storage.get(k) == 0 for k in ("orphan_count", "duplicate_count", "pending_count", "failed_count", "malformed_count"))
                and storage.get("index_schema_version") == vector.get("index_schema_version") == 4
                and storage.get("task_type") == "superrag" and storage.get("provenance") == "canonical_obsidian"
                and vector.get("method") == VECTOR_METHOD and vector.get("method_version") == VECTOR_METHOD_VERSION)


def read_receipt(path: Path) -> dict[str, Any] | None:
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != _current_uid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > MAX_RECEIPT_BYTES):
            return None
        data = os.read(fd, info.st_size + 1)
        if len(data) != info.st_size:
            return None
        value = json.loads(data)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None
    finally:
        if fd >= 0:
            os.close(fd)


def readiness_from_paths(storage_path: Path, vector_path: Path, *, key_path: Path, source_root: Path, now: datetime | None = None) -> bool:
    """Runtime gate: read existing key/receipts and recompute live source once."""
    key = read_receipt_key(key_path)
    if key is None:
        return False
    try:
        fingerprint = scan_canonical_source(source_root)
    except RuntimeError:
        return False
    return validate_readiness_receipts(read_receipt(storage_path), read_receipt(vector_path),
                                       key=key, current_fingerprint=fingerprint, now=now)
