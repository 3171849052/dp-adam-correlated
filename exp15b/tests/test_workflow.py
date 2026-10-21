from dataclasses import replace
import json
import pytest
from exp15b.config import Trial, METHODS, SEARCH_SEED, save_json, file_hash
from exp15b import search


def test_three_seed_review_can_reverse_rank_and_deduplicates_schemes(monkeypatch,tmp_path):
    jobs=search.extension_trials()
    groups={f'{task}:{method}':[] for task in ('cv','nlp') for method in METHODS}
    for c in jobs:groups[f'{c.task}:{c.method}'].append(c)
    preferred={}
    schemes={s:{} for s in ('A','B')}
    for key,group in groups.items():
        fixed=sorted([c for c in group if c.clip==1.],key=lambda c:c.lr)
        preferred[key]=[c.id for c in fixed[:2]]
        schemes['A'][key]=[c.id for c in fixed]
        schemes['B'][key]=[c.id for c in group]
    frozen=dict(candidates={c.id:c.values() for c in jobs},schemes=schemes)
    path=tmp_path/'extended_grid.json';path.write_text('{}\n')
    monkeypatch.setattr(search,'RESULTS',tmp_path)
    monkeypatch.setattr(search,'FROZEN',path)
    monkeypatch.setattr(search,'REVIEW',tmp_path/'review_plan.json')
    monkeypatch.setattr(search,'SELECTED',tmp_path/'selected_configs.json')
    monkeypatch.setattr(search,'load_frozen',lambda:frozen)
    monkeypatch.setattr(search,'reuse',lambda cfg:False)
    calls=[]
    def fake_batch(name,trials,allow_nonfinite=False):
        calls.append((name,list(trials)))
        rows=[]
        for c in trials:
            original=replace(c,stage='search',seed=SEARCH_SEED)
            ids=preferred[f'{c.task}:{c.method}']
            index=ids.index(original.id) if original.id in ids else 2
            accuracy=([.8,.7,.01] if name=='search' else [.1,.9,.01])[index]
            rows.append(dict(**c.values(),trial_id=c.id,status='completed',completed_epochs=5,
                             accuracy=accuracy,validation_loss=1-accuracy,official_evaluation_used=False))
        return rows
    monkeypatch.setattr(search,'run_batch',fake_batch)
    finals=search.search_and_review()
    assert [(name,len(jobs)) for name,jobs in calls]==[('search',150),('review',48)]
    selected=json.loads(search.SELECTED.read_text())
    assert len(finals)==120
    for winner in selected['winners']:
        assert winner['trial_id']==preferred[f"{winner['task']}:{winner['method']}"][1]
        assert winner['accuracy_mean']==pytest.approx((.7+.9+.9)/3)
    assert all(c.stage=='final' and c.seed in range(20261111,20261121) for c in finals)
    assert not selected['official_evaluation_used_for_selection']


def test_frozen_grid_tamper_and_unauthorized_review_fail(monkeypatch,tmp_path):
    path=tmp_path/'extended_grid.json';path.write_text('{}\n')
    (tmp_path/'extended_grid.sha256').write_text('wrong\n')
    monkeypatch.setattr(search,'RESULTS',tmp_path);monkeypatch.setattr(search,'FROZEN',path)
    with pytest.raises(ValueError,match='Frozen extension changed'):search.load_frozen()
    monkeypatch.setattr(search,'load_frozen',lambda:{'candidates':{}})
    with pytest.raises(ValueError,match='outside authorized'):
        search.authorize(Trial('cv','momentum+iid',2.,stage='search',seed=SEARCH_SEED))


def test_statistics_and_noise_control_comparison():
    from exp15b.report import stats,controlled_comparisons
    values=stats([1.,2.,3.])
    assert values['mean']==2. and values['sample_std']==1.
    common=dict(task='cv',stage='search',lr=2.,clip=1.,seed=20261101,status='completed',
        evaluation_scope='internal validation',initialization_sha256='i',classifier_initialization_sha256='h',
        train_order_sha256='o',rng_trace=['r'],noise_rng_start='n0',noise_rng_end='n1')
    left=dict(common,method='momentum',accuracy=.7,validation_loss=1.)
    right=dict(common,method='momentum+iid',accuracy=.6,validation_loss=1.2)
    rows=controlled_comparisons([left,right])
    assert rows[0]['difference']==pytest.approx(-.1)
    assert rows[1]['difference']==pytest.approx(.2)
    right['rng_trace']=['changed']
    with pytest.raises(ValueError,match='unpaired'):controlled_comparisons([left,right])
