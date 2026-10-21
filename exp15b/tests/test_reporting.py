import json
from dataclasses import replace
import pytest
from exp15b.config import Trial, METHODS, FINAL_SEEDS, SEARCH_SEED, file_hash, save_json
from exp15b import report


def test_complete_report_and_missing_review_fail_closed(monkeypatch,tmp_path):
    results=tmp_path/'results';results.mkdir()
    candidates={};search=[];formal=[];review=[];winners=[]
    def row(c):
        return dict(**c.values(),trial_id=c.id,status='completed',completed_epochs=5,actual_epsilon=8.,
            accuracy=.7+(.01 if c.method=='stpex2' else 0),validation_loss=.8,
            initialization_sha256='init',classifier_initialization_sha256='head',
            train_order_sha256='order',rng_trace=['paired'],noise_rng_start='start',noise_rng_end='end',
            evaluation_scope='official evaluation' if c.formal else 'internal validation')
    groups={f'{task}:{method}':[] for task in ('cv','nlp') for method in METHODS}
    for i in range(224):
        task,method=list(groups)[i%12].split(':')
        c=Trial(task,method,.01+i//12,stage='search',seed=SEARCH_SEED)
        candidates[c.id]=c.values();search.append(row(c));groups[f'{task}:{method}'].append(c.id)
    for task in ('cv','nlp'):
        for method in METHODS:
            c=Trial(task,method,.01,stage='search',seed=SEARCH_SEED)
            for scheme in ('A','B'):
                winners.append(dict(scheme=scheme,task=task,method=method,lr=c.lr,clip=c.clip,
                                    trial_id=c.id,accuracy_mean=.7,accuracy_std=.01))
            formal += [row(replace(c,stage='final',seed=seed)) for seed in FINAL_SEEDS]
            review += [row(replace(c,stage='review',seed=seed)) for seed in (20261102,20261103)]
    frozen=dict(candidates=candidates,schemes={s:groups for s in ('A','B')})
    frozen_path=results/'extended_grid.json';save_json(frozen_path,frozen)
    save_json(results/'search_results.json',search)
    save_json(results/'review_results.json',dict(rows=review))
    save_json(results/'formal_results.json',formal)
    save_json(results/'review_plan.json',dict(grid_sha256=file_hash(frozen_path),
        search_results_sha256=file_hash(results/'search_results.json'),trials=[report.cfg_of(r).values() for r in review]))
    selected_path=results/'selected_configs.json'
    selected=dict(grid_sha256=file_hash(frozen_path),review_results_sha256=file_hash(results/'review_results.json'),
        winners=winners,final_trials=[report.cfg_of(r).values() for r in formal])
    save_json(selected_path,selected)
    save_json(results/'protected_before.json',{})
    raw=tmp_path/'raw';raw.mkdir()
    def directory(r):return raw/r['trial_id']
    for r in search+review+formal:save_json(directory(r)/'summary.json',r)
    monkeypatch.setattr(report,'RESULTS',results)
    monkeypatch.setattr(report,'FROZEN',frozen_path)
    monkeypatch.setattr(report,'SELECTED',selected_path)
    monkeypatch.setattr(report,'load_frozen',lambda:frozen)
    monkeypatch.setattr(report,'completed',lambda c:True)
    monkeypatch.setattr(report,'result_dir',directory)
    monkeypatch.setattr(report,'protected_snapshot',lambda:{})
    monkeypatch.setattr(report,'diagnostic_noise',lambda r:dict(task=r['task'],method=r['method'],seed=r['seed']))
    monkeypatch.setattr(report,'historical_tables',lambda:([
        dict(experiment='Exp15',task=task,method='momentum',mean=.7) for task in ('cv','nlp')],{}))
    original_read=report.read
    def read(path):
        if str(path).endswith('exp15/results/statistics.json'):
            return dict(summary=[dict(scheme=w['scheme'],task=w['task'],method=w['method'],mean=.7)
                                 for w in winners if w['method']!='momentum+iid'])
        return original_read(path)
    monkeypatch.setattr(report,'read',read)
    report.main()
    text=(results/'final_report.md').read_text()
    assert '相同超参数' in text and 'STP-EX2' in text and '搜索边界' in text
    assert original_read(results/'final_audit.json')['status']=='passed'
    assert (results/'controlled_differences.csv').exists()
    (results/'final_report.md').unlink()
    # Even if the selected-result hash is updated, a missing scheduled review is rejected.
    save_json(results/'review_results.json',dict(rows=review[:-1]))
    selected['review_results_sha256']=file_hash(results/'review_results.json')
    save_json(selected_path,selected)
    with pytest.raises(ValueError,match='Missing planned training'):report.main()
    assert not (results/'final_report.md').exists()
