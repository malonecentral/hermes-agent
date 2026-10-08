"""Phase 1: local observations only. No remediation or execution authority.

Only explicit scalar metadata enters this module; never pass exception objects,
arguments, results, prompts, tracebacks, locals or arbitrary payloads.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat

from hermes_constants import get_hermes_home

VERSION = 1
SOURCE = 'tools.registry.dispatch'
CAPTURE_TIMEOUT_SECONDS = 0.05


def classify(exception_type: str, message: str = '', *, evidence: str = '') -> str:
    """Ordered descriptive labels. Message text is NEVER trusted provenance.

    Evidence is reserved for positively established facts at trusted callsites.
    Dispatch supplies none: exception names alone cannot prove a local defect.
    """
    groups = (
        ('external', {'HTTPError', 'APIError'}),
        ('network', {'ConnectionError', 'TimeoutError', 'ConnectError', 'ReadTimeout'}),
        ('upstream', {'InternalServerError', 'ServiceUnavailableError'}),
        ('rate_limit', {'RateLimitError'}),
        ('auth', {'AuthenticationError', 'PermissionDeniedError'}),
        ('cancellation', {'CancelledError', 'KeyboardInterrupt'}),
    )
    for category, types in groups:
        if exception_type in types:
            return 'deferred.' + category
    if evidence == 'deliberate_test':
        return 'deferred.deliberate_test'
    if evidence in {'local', 'internal', 'schema', 'config', 'import', 'protocol'}:
        return 'investigation.' + evidence
    return 'review.unknown'


def normalize(text: str) -> str:
    """Normalize already-redacted diagnostic text, preserving stable codes."""
    text = re.sub(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b', '<uuid>', text)
    text = re.sub(r'\b(pid\s*[=:]\s*)\d+\b', r'\1<pid>', text, flags=re.I)
    text = re.sub(r'(?:/private)?/(?:tmp|var/folders)/[^\s\"\']+', '<temp>', text)
    text = re.sub(r'\b0x[0-9a-fA-F]+\b', '<address>', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b', '<address>', text)
    return ' '.join(text.split())


@dataclass(frozen=True)
class Observation:
    tool: str
    component: str
    operation: str
    exception_type: str
    code: str
    message: str
    session_id: str
    source: str
    category: str
    fingerprint: str


def prepare(*, tool: str, exception_type: str, message: str,
            component: str = 'tools.registry', operation: str = 'dispatch',
            code: str = '', session_id: str = '') -> Observation | None:
    """Force redaction of every scalar BEFORE normalization, bounds or hashing.

    Redactor failure omits the entire observation, silently. No raw fallback.
    """
    values = (tool, component, operation, exception_type, code, message, session_id)
    if any(type(value) is not str for value in values):
        raise TypeError('incident metadata must be plain strings')
    try:
        from agent.redact import redact_sensitive_text
        safe = [redact_sensitive_text(v, force=True, redact_url_credentials=True) for v in values]
        if any(type(v) is not str for v in safe):
            return None
    except Exception:
        return None
    tool, component, operation, exception_type, code, message, session_id = safe
    tool, component, operation, exception_type, code = [normalize(v)[:128] for v in (tool, component, operation, exception_type, code)]
    message = normalize(message)[:2048]
    session_id = normalize(session_id)[:128]
    canonical = json.dumps([VERSION, tool, component, operation, exception_type, code, message], ensure_ascii=True, separators=(',', ':'))
    fingerprint = 'v1:' + hashlib.sha256(canonical.encode()).hexdigest()
    return Observation(tool, component, operation, exception_type, code, message,
                       session_id, SOURCE, classify(exception_type), fingerprint)


def db_path() -> Path:
    return get_hermes_home() / 'self-healing' / 'incidents.db'


def _safe_target(path: Path, directory: bool = False) -> None:
    st = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(st.st_mode) or (hasattr(os, 'getuid') and st.st_uid != os.getuid()):
        raise OSError('unsafe incident ledger target')
    if not directory and st.st_nlink != 1:
        raise OSError('linked incident ledger target')
    if st.st_mode & 0o022:
        raise OSError('writable incident ledger target')


def _writer_path() -> Path:
    path = db_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _safe_target(path.parent, directory=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        os.close(fd)
    _safe_target(path)
    return path


def _connect(path: Path, *, readonly: bool = False, timeout: float = 2):
    # SQLite can open these implicitly; validate them too without chmod/unlink.
    for suffix in ('-journal', '-wal', '-shm'):
        sidecar = Path(str(path) + suffix)
        try:
            _safe_target(sidecar)
        except FileNotFoundError:
            pass
    conn = sqlite3.connect(path.absolute().as_uri() + ('?mode=ro&immutable=1' if readonly else '?mode=rw'),
                           uri=True, timeout=timeout)
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        if readonly:
            conn.execute('PRAGMA query_only=ON')
        return conn
    except Exception:
        conn.close()
        raise


def _schema(conn, *, initialize: bool = False):
    version = conn.execute('PRAGMA user_version').fetchone()[0]
    if version == VERSION:
        return
    if version != 0 or not initialize:
        raise ValueError('unsupported incident schema version')
    # Individual statements, not executescript (which implicitly commits).
    for statement in (
        'CREATE TABLE signatures (fingerprint TEXT PRIMARY KEY, version INTEGER NOT NULL, tool TEXT NOT NULL, component TEXT NOT NULL, operation TEXT NOT NULL, exception_type TEXT NOT NULL, code TEXT NOT NULL, message TEXT NOT NULL, category TEXT NOT NULL)',
        'CREATE TABLE incidents (fingerprint TEXT PRIMARY KEY REFERENCES signatures(fingerprint), count INTEGER NOT NULL CHECK(count > 0), first_seen TEXT NOT NULL, last_seen TEXT NOT NULL)',
        'CREATE TABLE occurrences (id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL REFERENCES incidents(fingerprint), seen TEXT NOT NULL, session_id TEXT NOT NULL, source TEXT NOT NULL)',
        'CREATE INDEX occurrence_signature ON occurrences(fingerprint, id)',
        'PRAGMA user_version=1',
    ):
        conn.execute(statement)


def record(observation: Observation, *, timeout: float = 2) -> None:
    """Atomic aggregate + occurrence write; independent connection per operation.

    Re-prepare even a constructed Observation so raw scalar metadata can never
    bypass the redaction boundary via this storage API.
    Direct callers retain the normal write budget; dispatch supplies a short
    best-effort budget so contention does not stall tool error handling.
    """
    if type(observation) is not Observation:
        raise TypeError('expected prepared observation')
    e = prepare(tool=observation.tool, component=observation.component,
                operation=observation.operation, exception_type=observation.exception_type,
                code=observation.code, message=observation.message, session_id=observation.session_id)
    if e is None:
        return
    conn = _connect(_writer_path(), timeout=timeout)
    try:
        with conn:
            conn.execute('BEGIN IMMEDIATE')
            _schema(conn, initialize=True)
            now = datetime.now(timezone.utc).isoformat(timespec='microseconds')
            conn.execute('INSERT INTO signatures VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(fingerprint) DO NOTHING',
                         (e.fingerprint, VERSION, e.tool, e.component, e.operation, e.exception_type, e.code, e.message, e.category))
            conn.execute('INSERT INTO incidents VALUES (?,1,?,?) ON CONFLICT(fingerprint) DO UPDATE SET count=count+1, first_seen=min(first_seen,excluded.first_seen), last_seen=max(last_seen,excluded.last_seen)', (e.fingerprint, now, now))
            conn.execute('INSERT INTO occurrences(fingerprint,seen,session_id,source) VALUES (?,?,?,?)', (e.fingerprint, now, e.session_id, SOURCE))
    finally:
        conn.close()


def enabled() -> bool:
    from hermes_cli.config import load_config_readonly
    section = load_config_readonly(side_effects=False).get('self_healing', {})
    return isinstance(section, dict) and section.get('enabled') is True


def capture_dispatch(*, tool: str, exception_type: str, message: str, session_id: str = '') -> bool:
    """Best effort, no logs, no exceptions, no filesystem creation when disabled."""
    try:
        if not enabled():
            return False
        observation = prepare(tool=tool, exception_type=exception_type, message=message, session_id=session_id)
        if observation is None:
            return False
        record(observation, timeout=CAPTURE_TIMEOUT_SECONDS)
        return True
    except Exception:
        return False


def _report_state(path: Path):
    """Immutable reads require a quiescent, rollback-mode ledger.

    Reject WAL mode even without sidecars: immutable SQLite ignores WAL and
    must not silently return a checkpoint older than the committed data.
    Never recover journals, checkpoint, or change the database's journal mode.
    """
    for suffix in ('-journal', '-wal', '-shm'):
        if os.path.lexists(str(path) + suffix):
            raise OSError('incident report requires a quiescent rollback-mode ledger')
    st = path.stat()
    with path.open('rb') as stream:
        header = stream.read(20)
    if header[:16] == b'SQLite format 3\x00' and 2 in header[18:20]:
        raise OSError('incident report does not support WAL mode')
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def report(*, limit: int = 100, examples: int = 3) -> list[dict]:
    """Non-mutating, offline report; callers must quiesce writers first.

    Immutable SQLite creates no sidecars and takes no locks. Refuse WAL/journals
    and discard results if write metadata changes during the read. This is not
    an online snapshot protocol; filesystem access-time accounting may occur.
    """
    if type(limit) is not int or not 1 <= limit <= 1000 or type(examples) is not int or not 0 <= examples <= 10:
        raise ValueError('report bounds: limit 1..1000, examples 0..10')
    path = db_path()
    if not path.exists():
        return []
    _safe_target(path.parent, directory=True)
    _safe_target(path)
    before = _report_state(path)
    conn = _connect(path, readonly=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('BEGIN')
        _schema(conn)
        rows = [dict(row) for row in conn.execute('SELECT s.*, i.count, i.first_seen, i.last_seen FROM signatures s JOIN incidents i USING(fingerprint) ORDER BY i.last_seen DESC, s.fingerprint ASC LIMIT ?', (limit,))]
        for row in rows:
            row['examples'] = [dict(r) for r in conn.execute('SELECT seen, session_id, source FROM occurrences WHERE fingerprint=? ORDER BY id DESC LIMIT ?', (row['fingerprint'], examples))]
        if _report_state(path) != before:
            raise OSError('incident ledger changed during report')
        return rows
    finally:
        conn.close()
