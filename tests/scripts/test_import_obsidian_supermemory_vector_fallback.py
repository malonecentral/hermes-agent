"""Bounded readiness-only candidates; synthetic data, no provider calls."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/import-obsidian-supermemory.py'

@pytest.fixture
def harness(monkeypatch):
    spec = importlib.util.spec_from_file_location('vector_fallback', SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    doc = m.item('Skills/A.md', b'alpha cedar telescope marmalade compass current distinctive body content for readiness proof')
    probes = m.indexed_content_probes(doc)
    assert probes
    monkeypatch.setattr(m, 'indexed_content_probes', lambda _: [probes[0]])
    parent = dict(id='parent', custom_id=doc['custom_id'], metadata=m.metadata(doc),
                  container_tags=[m.container_for_doc(doc)], task_type='superrag', status='done')
    valid = {'total': 1, 'results': [{'id': 'chunk', 'chunk': doc['content'],
                                    'metadata': m.metadata(doc), 'documents': [{'id': 'parent'}]}]}
    calls = []
    gets = []
    def get(ident, **kwargs):
        gets.append(ident)
        return copy.deepcopy(parent)
    client = SimpleNamespace(documents=SimpleNamespace(get=get))
    def run(responses):
        def search(actual_client, query, **kwargs):
            assert actual_client is client
            assert query == probes[0]
            assert kwargs['container_tag'] == m.container_for_doc(doc)
            assert kwargs['threshold'] == 0.0
            assert kwargs['timeout'] == 30.0
            assert kwargs['filters'] == {'AND': [{'key': 'source', 'value': 'obsidian'},
                                                {'key': 'relative_path', 'value': doc['relative_path']}]}
            calls.append(kwargs['limit'])
            return copy.deepcopy(responses[min(len(calls)-1, len(responses)-1)])
        monkeypatch.setattr(m, 'search_documents_v4', search)
        return m._vector_observations(client, {doc['relative_path']: doc})
    return m, parent, valid, calls, gets, run

EMPTY = {'total': 0, 'results': []}

@pytest.mark.parametrize('empty_count,limits', [(0,[1]), (1,[1,5]), (2,[1,5,20])])
def test_valid_proof_preserves_fast_path_and_bounded_fallback(harness, empty_count, limits):
    m, parent, valid, calls, gets, run = harness
    result = run([EMPTY] * empty_count + [valid])
    assert calls == limits
    assert gets == ['parent']
    assert len(result) == 1


def test_empty_forever_fails_at_bound(harness):
    m, parent, valid, calls, gets, run = harness
    with pytest.raises(m.ReconciliationRequired):
        run([EMPTY])
    assert calls == [1,5,20]
    assert gets == []

@pytest.mark.parametrize('bad', [
    {}, {'total': 0, 'results': None}, {'total': False, 'results': []},
    {'total': '0', 'results': []}, {'total': 1, 'results': []},
])
def test_malformed_response_never_expands(harness, bad):
    m, parent, valid, calls, gets, run = harness
    with pytest.raises(m.ReconciliationRequired):
        run([bad, valid])
    assert calls == [1]
    assert gets == []

@pytest.mark.parametrize('kind', ['multiple', 'wrong_total', 'stale_chunk', 'wrong_parent', 'stale_hash', 'not_done'])
@pytest.mark.parametrize('after_empty', [False, True])
def test_invalid_proof_fails_without_further_expansion(harness, kind, after_empty):
    m, parent, valid, calls, gets, run = harness
    bad = copy.deepcopy(valid)
    if kind == 'multiple':
        bad['results'] *= 2
        bad['total'] = 2
    elif kind == 'wrong_total':
        bad['total'] = 2
    elif kind == 'stale_chunk':
        bad['results'][0]['chunk'] = 'unrelated obsolete orchard lantern'
    elif kind == 'wrong_parent':
        parent['custom_id'] = 'wrong-parent'
    elif kind == 'stale_hash':
        parent['metadata']['content_sha256'] = '0' * 64
    elif kind == 'not_done':
        parent['status'] = 'processing'
    with pytest.raises(m.ReconciliationRequired):
        run(([EMPTY] if after_empty else []) + [bad, valid])
    assert calls == ([1,5] if after_empty else [1])
