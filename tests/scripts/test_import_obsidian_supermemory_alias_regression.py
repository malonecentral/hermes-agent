"""Production-shaped duplicate SDK aliases must not strand canonical sync."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from supermemory.types.document_list_response import Memory

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/import-obsidian-supermemory.py'
@pytest.fixture
def importer():
    spec=importlib.util.spec_from_file_location('alias_regression_importer', SCRIPT)
    assert spec is not None and spec.loader is not None
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.configure_destinations('owner_canonical','owner_primary')
    return m

def orphan(m):
    doc=m.item('Jarvis/Family Shared/Fixtures/Retired.md',b'old fixture\n')
    return {'id':'fixture-provider-id','custom_id':doc['custom_id'],
            'status':'done','metadata':m.metadata(doc),
            'container_tags':['family_shared'],'containerTags':['family_shared']}

def test_exact_duplicate_sdk_aliases_preserve_trusted_orphan(importer):
    row=orphan(importer)
    assert importer._trusted_orphan_path(row,'family_shared')==row['metadata']['relative_path']
    classified=importer.classify_inventory({}, {'owner_canonical':(), 'family_shared':(row,)})
    assert classified['untrusted_orphans']==()
    assert [r['action'] for r in importer.reconciliation_plan(classified)]==['delete']

@pytest.mark.parametrize('left,right',[
    (['family_shared'],['owner_primary']), (['family_shared'],None),
    (['family_shared'],('family_shared',)), (['family_shared'],['family_shared','owner_primary']),
    (None,None), ([],[]), (True,1),
])
def test_conflicting_or_malformed_duplicate_aliases_stay_review_only(importer,left,right):
    row=orphan(importer); row.update(container_tags=left,containerTags=right)
    assert importer._trusted_orphan_path(row,'family_shared') is None
    plan=importer.reconciliation_plan(importer.classify_inventory({}, {'family_shared':(row,), 'owner_canonical':()}))
    assert [r['action'] for r in plan]==['operator_review']

def test_singular_plural_competing_claims_still_fail(importer):
    row=orphan(importer); row['container_tag']='family_shared'
    assert importer._trusted_orphan_path(row,'family_shared') is None

def test_duplicate_alias_does_not_override_global_collision(importer):
    row=orphan(importer); peer=dict(row,id='second-provider-id')
    plan=importer.reconciliation_plan(importer.classify_inventory({}, {'family_shared':(row,peer),'owner_canonical':()}))
    assert [r['action'] for r in plan]==['operator_review']

def test_actual_sdk_list_model_through_plan_cli_is_write_free(importer,tmp_path,monkeypatch,capsys):
    row=orphan(importer)
    sdk_row=Memory.model_construct(**dict(row, customId=row['custom_id']))
    # The live SDK produces both aliases; reproduce that exact parsed shape.
    assert sdk_row.__pydantic_extra__ is not None
    sdk_row.__pydantic_extra__['containerTags']=['family_shared']
    assert sdk_row.model_dump()['container_tags']==sdk_row.model_dump()['containerTags']
    calls=[]
    class Documents:
        def list(self, **kwargs):
            container=kwargs['container_tags'][0]; calls.append(container)
            assert container in {'owner_canonical','family_shared'}
            rows=[sdk_row] if container=='family_shared' else []
            return {'memories':rows,'pagination':{'current_page':1,'total_pages':int(bool(rows)),'total_items':len(rows)}}
    monkeypatch.setattr(importer,'_client',lambda _: SimpleNamespace(documents=Documents()))
    source=tmp_path/'source'; source.mkdir()
    private=tmp_path/'private'; private.mkdir(mode=0o700)
    import sys
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT),'plan','--transaction-id','alias_fixture','--source-root',str(source),
                   '--private-root',str(private),'--owner-canonical-container','owner_canonical',
                   '--owner-explicit-container','owner_primary'])
    importer.main()
    result=json.loads(capsys.readouterr().out)
    assert result['actions']=={'delete':1}
    assert result['backend_mutated'] is result['filesystem_mutated'] is False
    assert set(calls)=={'owner_canonical','family_shared'}
    assert list(private.iterdir())==[]
