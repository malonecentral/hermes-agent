"""Phase 1 review regressions: non-mutating reports and best-effort capture."""

from pathlib import Path
import sqlite3
import time

import pytest

from agent import incidents as ledger


def seed():
    observation = ledger.prepare(tool='demo', exception_type='RuntimeError', message='broken')
    assert observation is not None
    ledger.record(observation)


def snapshot(directory):
    # atime is deliberately excluded: filesystem read-access accounting is not
    # controlled by an application. Include directory entries and write metadata.
    return {
        p.name: (p.stat().st_ino, p.stat().st_mode, p.stat().st_size,
                 p.stat().st_mtime_ns, p.stat().st_ctime_ns,
                 p.read_bytes() if p.is_file() else None)
        for p in [directory, *directory.iterdir()]
    }


@pytest.mark.parametrize('active', [False, True])
def test_wal_report_fails_closed_without_sidecar_mutation(active):
    seed()
    path = ledger.db_path()
    writer = sqlite3.connect(path)
    try:
        assert writer.execute('PRAGMA journal_mode=WAL').fetchone() == ('wal',)
        if active:
            writer.execute('UPDATE incidents SET count=count+1')
            writer.commit()
            assert Path(str(path) + '-wal').stat().st_size > 0
        else:
            writer.close()
            assert not Path(str(path) + '-wal').exists()
            assert not Path(str(path) + '-shm').exists()
        before = snapshot(path.parent)
        with pytest.raises((OSError, sqlite3.Error)):
            ledger.report()
        assert snapshot(path.parent) == before
    finally:
        writer.close()


def test_report_uses_immutable_uri_and_changes_nothing(monkeypatch):
    seed()
    before = snapshot(ledger.db_path().parent)
    connect = sqlite3.connect
    calls = []

    def spy(database, **kwargs):
        calls.append(database)
        return connect(database, **kwargs)

    monkeypatch.setattr(ledger.sqlite3, 'connect', spy)
    assert ledger.report()[0]['count'] == 1
    assert calls and all('immutable=1' in uri and 'mode=ro' in uri for uri in calls)
    assert snapshot(ledger.db_path().parent) == before


def test_managed_disable_overrides_user_enable_without_creation(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_cli import config, managed_scope
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text('self_healing:\n  enabled: true\n')
    managed = tmp_path / 'managed'
    managed.mkdir()
    (managed / 'config.yaml').write_text('self_healing:\n  enabled: false\n')
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(managed))
    config._LOAD_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    before = snapshot(home)
    assert not ledger.capture_dispatch(tool='demo', exception_type='RuntimeError', message='broken')
    assert not ledger.db_path().parent.exists()
    assert snapshot(home) == before


def test_writer_contention_is_short_best_effort(monkeypatch):
    seed()
    monkeypatch.setattr(ledger, 'enabled', lambda: True)
    # Warm redaction/imports before timing; hold the lock for the entire call.
    ledger.prepare(tool='demo', exception_type='RuntimeError', message='broken')
    connect = ledger._connect
    budgets = []

    def timed_connect(path, **kwargs):
        conn = connect(path, **kwargs)
        budgets.append(conn.execute('PRAGMA busy_timeout').fetchone()[0])
        return conn

    monkeypatch.setattr(ledger, '_connect', timed_connect)
    writer = sqlite3.connect(ledger.db_path())
    try:
        writer.execute('BEGIN IMMEDIATE')
        started = time.monotonic()
        assert not ledger.capture_dispatch(tool='demo', exception_type='RuntimeError', message='broken')
        elapsed = time.monotonic() - started
        # A generous 1s ceiling avoids scheduler jitter while detecting the old
        # 2s wait. The configured budget is also asserted deterministically.
        assert elapsed < 1.0, elapsed
        assert len(budgets) == 1 and 0 <= budgets[0] <= 100
    finally:
        writer.rollback()
        writer.close()
    assert ledger.report()[0]['count'] == 1


@pytest.mark.parametrize('config_text', [None, 'self_healing: [invalid'])
def test_config_read_does_not_initialize_or_back_up(config_text, tmp_path, monkeypatch, capsys):
    home = tmp_path / 'uninitialized'
    monkeypatch.setenv('HERMES_HOME', str(home))
    if config_text is not None:
        home.mkdir()
        (home / 'config.yaml').write_text(config_text)
    before = snapshot(home) if home.exists() else None
    assert not ledger.capture_dispatch(tool='demo', exception_type='RuntimeError', message='broken')
    assert (snapshot(home) if home.exists() else None) == before
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''


def test_report_refuses_active_rollback_journal_without_mutation():
    seed()
    writer = sqlite3.connect(ledger.db_path())
    try:
        writer.execute('UPDATE incidents SET count=count+1')
        before = snapshot(ledger.db_path().parent)
        with pytest.raises(OSError):
            ledger.report()
        assert snapshot(ledger.db_path().parent) == before
    finally:
        writer.rollback()
        writer.close()


def test_report_discards_results_if_ledger_changes(monkeypatch):
    seed()
    connect = ledger._connect

    def changed_connect(path, **kwargs):
        conn = connect(path, **kwargs)
        # Deterministically model another writer between the initial metadata
        # check and the immutable read. No online snapshot is promised.
        with sqlite3.connect(path) as writer:
            writer.execute('UPDATE incidents SET count=count+1')
        return conn

    monkeypatch.setattr(ledger, '_connect', changed_connect)
    with pytest.raises(OSError, match='changed during report'):
        ledger.report()


@pytest.mark.parametrize('failure', ['corrupt', 'locked'])
def test_cli_sqlite_errors_are_bounded_and_sanitized(failure, monkeypatch, capsys):
    from agent import incident_report
    seed()
    if failure == 'corrupt':
        ledger.db_path().write_bytes(b'private-corrupt-data' * 100)
        expected = 'DatabaseError'
    else:
        # Exercise sqlite3's actual lock error, independently of the immutable
        # report's refusal of active journals/WAL.
        def locked_report(**kwargs):
            conn = sqlite3.connect(ledger.db_path(), timeout=0)
            try:
                conn.execute('SELECT * FROM incidents').fetchall()
            finally:
                conn.close()
        monkeypatch.setattr(incident_report, 'report', locked_report)
        expected = 'OperationalError'
    lock = sqlite3.connect(ledger.db_path()) if failure == 'locked' else None
    try:
        if lock:
            lock.execute('BEGIN EXCLUSIVE')
        monkeypatch.setattr('sys.argv', ['incident_report'])
        with pytest.raises(SystemExit) as exc:
            incident_report.main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert captured.out == ''
        assert captured.err == f'Incident report unavailable: {expected}\n'
    finally:
        if lock:
            lock.rollback()
            lock.close()
