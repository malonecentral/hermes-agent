"""Observation-only ledger contracts; conftest isolates home before imports."""
import ast
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
from unittest.mock import Mock

import pytest

def test_phase1_module_exists():
    assert importlib.util.find_spec('agent.incidents') is not None, 'Phase 1 ledger missing'

@pytest.fixture
def ledger():
    assert importlib.util.find_spec('agent.incidents') is not None, 'Phase 1 ledger missing'
    return importlib.import_module('agent.incidents')

def event(m, **kw):
    return m.prepare(tool='demo', exception_type='RuntimeError', message=kw.pop('message', 'broken'), **kw)

def test_redaction_before_persistence(ledger, monkeypatch):
    from agent import redact
    monkeypatch.setattr(redact, '_REDACT_ENABLED', False)
    secret = 'https://alice:supersecret@example.org/?token=privatevalue'
    e = event(ledger, message=secret)
    ledger.record(e)
    assert 'supersecret' not in ledger.db_path().read_bytes().decode('latin1')
    assert 'privatevalue' not in json.dumps(ledger.report())
    seen = []
    def scrub(text, **kw):
        assert kw == dict(force=True, redact_url_credentials=True)
        seen.append(text)
        return 'SAFE'
    monkeypatch.setattr(redact, 'redact_sensitive_text', scrub)
    e = event(ledger, message='x' * 10000)
    assert 'x' * 10000 in seen
    assert e.message == 'SAFE'

def test_redaction_failure_omits(ledger, monkeypatch):
    from agent import redact
    monkeypatch.setattr(redact, 'redact_sensitive_text', Mock(side_effect=ValueError('secret')))
    assert event(ledger, message='secret') is None
    assert ledger.capture_dispatch(tool='demo', exception_type='RuntimeError', message='secret') is False
    assert not ledger.db_path().exists()

@pytest.mark.parametrize('kind,expected', [('ConnectionError','deferred.network'), ('TimeoutError','deferred.network'), ('RateLimitError','deferred.rate_limit'), ('AuthenticationError','deferred.auth'), ('CancelledError','deferred.cancellation'), ('HTTPError','deferred.external'), ('InternalServerError','deferred.upstream'), ('RuntimeError','review.unknown'), ('ImportError','review.unknown'), ('TypeError','review.unknown')])
def test_classification(ledger, kind, expected):
    assert ledger.classify(kind, 'local internal schema config import protocol deliberate test') == expected

@pytest.mark.parametrize('evidence', ['local','internal','schema','config','import','protocol'])
def test_positive_evidence(ledger, evidence):
    assert ledger.classify('RuntimeError', 'noise', evidence=evidence) == 'investigation.' + evidence
    assert ledger.classify('ConnectionError', 'noise', evidence=evidence) == 'deferred.network'
    assert ledger.classify('RuntimeError', 'noise', evidence='deliberate_test') == 'deferred.deliberate_test'

def test_fingerprints(ledger):
    a = event(ledger, message='pid=12 /tmp/abc/file 123e4567-e89b-12d3-a456-426614174000 at 0xabc  hi', session_id='one')
    b = event(ledger, message='pid=99 /tmp/xyz/file 123e4567-e89b-12d3-a456-426614174001 at 0xdef\nhi', session_id='two')
    assert a.fingerprint == b.fingerprint
    for kw in [dict(tool='other'), dict(component='other'), dict(operation='other'), dict(exception_type='ValueError'), dict(code='E42')]:
        values = dict(tool='demo', exception_type='RuntimeError', message='broken')
        values.update(kw)
        assert ledger.prepare(**values).fingerprint != event(ledger).fingerprint
    with pytest.raises(TypeError):
        ledger.prepare(tool='x', exception_type='X', message={'payload':'no'})

def test_concurrent_initialization_counts_permissions(ledger):
    e = event(ledger)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: ledger.record(e), range(40)))
    rows = ledger.report(examples=2)
    assert len(rows) == 1 and rows[0]['count'] == 40
    assert len(rows[0]['examples']) == 2
    assert rows[0]['first_seen'] <= rows[0]['last_seen']
    with sqlite3.connect(ledger.db_path()) as conn:
        assert conn.execute('select count(*) from occurrences').fetchone()[0] == 40
        assert conn.execute('pragma journal_mode').fetchone()[0] != 'wal'
    assert ledger.db_path().stat().st_mode & 0o777 == 0o600
    assert ledger.db_path().parent.stat().st_mode & 0o777 == 0o700

def test_atomic_rollback(ledger):
    ledger.record(event(ledger))
    with sqlite3.connect(ledger.db_path()) as conn:
        conn.execute("CREATE TRIGGER deny BEFORE INSERT ON occurrences BEGIN SELECT RAISE(ABORT, 'deny'); END")
    with pytest.raises(sqlite3.DatabaseError):
        ledger.record(event(ledger))
    assert ledger.report()[0]['count'] == 1

def test_newer_schema(ledger):
    ledger.record(event(ledger))
    with sqlite3.connect(ledger.db_path()) as conn:
        conn.execute('pragma user_version=999')
    before = ledger.db_path().read_bytes()
    for fn in [lambda: ledger.record(event(ledger)), ledger.report]:
        with pytest.raises(ValueError, match='version'):
            fn()
    assert ledger.db_path().read_bytes() == before

def test_disabled_profiles_and_fail_open(ledger, monkeypatch, tmp_path):
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG['self_healing']['enabled'] is False
    assert not ledger.capture_dispatch(tool='x', exception_type='X', message='y')
    assert not ledger.db_path().exists()
    ledger.record(event(ledger))
    first = ledger.db_path()
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'other'))
    assert ledger.report() == [] and not ledger.db_path().exists()
    monkeypatch.setattr(ledger, 'enabled', lambda: True)
    monkeypatch.setattr(ledger, 'record', Mock(side_effect=OSError('broken')))
    assert not ledger.capture_dispatch(tool='x', exception_type='X', message='y')
    assert first.exists()

def test_unsafe_targets_not_mutated(ledger, tmp_path):
    target = tmp_path / 'outside'
    target.mkdir(mode=0o755)
    ledger.db_path().parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        ledger.record(event(ledger))
    assert list(target.iterdir()) == []
    assert target.stat().st_mode & 0o777 == 0o755

def test_report_readonly_bounds_order(ledger):
    assert ledger.report() == []
    assert not ledger.db_path().parent.exists()
    for msg in ['z', 'a', 'b']:
        ledger.record(event(ledger, message=msg))
    before = ledger.db_path().read_bytes()
    rows = ledger.report(limit=2, examples=0)
    assert len(rows) == 2 and all(r['examples'] == [] for r in rows)
    assert rows == ledger.report(limit=2, examples=0)
    assert ledger.db_path().read_bytes() == before
    with pytest.raises(ValueError):
        ledger.report(limit=1001)

def test_registry_capture_once_unchanged(ledger, monkeypatch):
    from tools.registry import ToolRegistry, tool_error
    from model_tools import _sanitize_tool_error
    reg = ToolRegistry()
    def fail(args, **kwargs):
        raise ValueError('oops')
    reg.register(name='demo', toolset='test', schema={}, handler=fail)
    capture = Mock(side_effect=RuntimeError('ledger failed'))
    monkeypatch.setattr(ledger, 'capture_dispatch', capture)
    result = reg.dispatch('demo', {'secret':'not captured'}, task_id='task-1')
    assert result == tool_error(_sanitize_tool_error('Tool execution failed: ValueError: oops'))
    assert capture.call_count == 1
    assert capture.call_args.kwargs == dict(tool='demo', exception_type='ValueError', message='oops', session_id='task-1')

def test_unsafe_sidecar_not_mutated(ledger, tmp_path):
    ledger.record(event(ledger))
    outside = tmp_path / 'outside-journal'
    outside.write_text('untouched')
    Path(str(ledger.db_path()) + '-journal').symlink_to(outside)
    with pytest.raises(OSError):
        ledger.record(event(ledger))
    assert outside.read_text() == 'untouched'


def test_enabled_dispatch_real_io_no_execution(ledger, monkeypatch):
    import socket
    import subprocess
    from tools.registry import ToolRegistry
    from hermes_constants import get_hermes_home
    get_hermes_home().mkdir(parents=True, exist_ok=True)
    (get_hermes_home() / 'config.yaml').write_text('self_healing:\n  enabled: true\n')
    deny = Mock(side_effect=AssertionError('execution forbidden'))
    monkeypatch.setattr(subprocess, 'Popen', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    reg = ToolRegistry()
    def fail(args, **kwargs):
        raise RuntimeError('https://alice:passwordvalue@example.com/?token=secretvalue')
    reg.register(name='demo', toolset='test', schema={}, handler=fail)
    reg.dispatch('demo', {'private':'not recorded'}, task_id='task-1')
    rows = ledger.report()
    assert len(rows) == 1 and rows[0]['count'] == 1
    assert rows[0]['category'] == 'review.unknown'
    assert rows[0]['examples'][0]['session_id'] == 'task-1'
    assert 'passwordvalue' not in json.dumps(rows)
    deny.assert_not_called()


def test_standalone_missing_report(ledger, monkeypatch, capsys):
    from agent.incident_report import main
    monkeypatch.setattr('sys.argv', ['incident_report', '--examples', '0'])
    main()
    assert json.loads(capsys.readouterr().out) == []
    assert not ledger.db_path().parent.exists()


def test_observation_only_dependencies(ledger):
    tree = ast.parse(Path(ledger.__file__).read_text())
    forbidden = {'subprocess', 'requests', 'httpx', 'socket', 'run_agent', 'scheduler', 'shutil'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not forbidden.intersection(a.name.split('.')[0] for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in forbidden
    assert not any(hasattr(ledger, n) for n in ['repair','deploy','enqueue','launch'])
