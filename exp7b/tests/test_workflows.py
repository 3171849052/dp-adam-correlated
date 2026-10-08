"""Exercise adaptive control flow and full report assembly without full GPU runs."""
import json
import math
import numpy as np
import pytest
from exp7b import config
from exp7b.config import *

@pytest.mark.parametrize('variant',['interior','scale_alternative','boundary'])
def test_bounded_search_uses_accuracy_only(tmp_path,monkeypatch,variant):
    from exp7b import search,report
    records = []
    monkeypatch.setattr(config,'BASE',tmp_path)
    monkeypatch.setattr(search,'BASE',tmp_path)
    monkeypatch.setattr(report,'BASE',tmp_path)
    monkeypatch.setattr(search,'rows',lambda category:list(records))
    monkeypatch.setattr(report,'rows',lambda category:list(records))
    monkeypatch.setattr(search,'reuse',lambda job:False)
    monkeypatch.setattr(search,'freeze',lambda selected:selected)
    def score(j):
        m = j['method']
        target_lr = .002 if m == SGD else .005
        target_C = 10 if m in (SGD,BIAS) else 100
        if m == BIAS:
            target_lr = .003 if variant != 'boundary' else .03
        if m in (MOMENTUM_SCALE,BIAS_SCALE):
            eps = (.3 if variant == 'scale_alternative' else 2 if variant == 'boundary' else .1)
            K = 40 if variant == 'boundary' else 10
            target_C = K/j['eps_scale']
            return .8 - .03*math.log(j['eps_scale']/eps)**2 - .02*math.log(j['C']/target_C)**2 - .02*math.log(j['lr']/target_lr)**2
        return .8-.02*math.log(j['C']/target_C)**2-.02*math.log(j['lr']/target_lr)**2
    def queue(jobs,gpus,category):
        assert gpus == [1,2,3] and category == 'search'
        existing = {trial_id({k:r[k] for k in ('method','seed','lr','C','eps_scale')}) for r in records}
        for j in jobs:
            if trial_id(j) in existing:
                continue
            assert j['seed'] == SEED
            records.append(dict(j,status='completed',smoke=False,source='new',final_test_top1=score(j),
                                workload_sha256='W',strategy_sha256='S',
                                train_loss=100*score(j),result_dir=f'exp7b/results/search/trials/{trial_id(j)}'))
        return records
    monkeypatch.setattr(search,'run_queue',queue)
    save_json(tmp_path/'results/platform_validation.json',dict(status='passed',smoke_trials=7))
    selected = search.search([1,2,3])
    assert set(selected) == set(SEARCH_METHODS)
    assert len(records) < 80, 'Refinement must be finite and bounded'
    for m in SEARCH_METHODS:
        assert selected[m]['final_test_top1'] == max(r['final_test_top1'] for r in records if r['method']==m)
    if variant == 'interior':
        assert selected[SGD]['lr'] == .002
        assert selected[MOMENTUM_SCALE]['eps_scale'] == .1
        assert not (tmp_path/f'results/search/{MOMENTUM_SCALE}_K_candidates.json').exists()
    if variant == 'boundary':
        assert selected[BIAS]['lr'] == .02
        assert selected[BIAS_SCALE]['eps_scale'] == 2
        assert selected[BIAS_SCALE]['C']*selected[BIAS_SCALE]['eps_scale'] == 40
    # A completed selection is loaded, with no new candidates or trials.
    before = len(records)
    assert search.search([1,2,3]) == selected and len(records) == before

def test_compatible_history_actual_artifacts(tmp_path,monkeypatch):
    from exp7b import history,audit
    monkeypatch.setattr(config,'BASE',tmp_path)
    monkeypatch.setattr(history,'BASE',tmp_path)
    # Explicit canonical paths stay under the real exp7b/runtime/tests directory.
    values = trial(MOMENTUM_SCALE,.005,100,.1)
    assert trial_id(values) in history.index()
    assert history.reuse(values)
    row = audit.audit_trial(trial_dir(values,'search'))
    assert row['source'] == 'historical' and row['final_test_top1'] == pytest.approx(.7450)
    assert not history.reuse(values)

def test_final_report_70_rows_and_paired_effects(tmp_path,monkeypatch):
    from exp7b import report,frozen
    monkeypatch.setattr(config,'BASE',tmp_path)
    monkeypatch.setattr(report,'BASE',tmp_path)
    settings = {m:dict(trial(m,.005,100,.1 if m.endswith('-scale') else None),source='search_selected') for m in METHODS}
    monkeypatch.setattr(frozen,'load_frozen',lambda:settings)
    monkeypatch.setattr(report,'audit_trial',lambda path:None)
    monkeypatch.setattr(report,'audit_pairing',lambda rows:None)
    records = []
    for i,m in enumerate(METHODS):
        for j,s in enumerate(FINAL_SEEDS):
            records.append(dict(trial(m,.005,100,.1 if m.endswith('-scale') else None,s),
                                final_test_top1=.6+i*.01+j*.001*(1+i/10),
                                result_dir=f'exp7b/runtime/test/{m}/{s}'))
    monkeypatch.setattr(report,'rows',lambda category:records)
    report.final_report()
    effects = json.loads((tmp_path/'results/paired_effects.json').read_text())
    assert len(effects) == 8 and all(e['wins']==10 for e in effects)
    expected = np.array([r['final_test_top1'] for r in records if r['method']==SGD])-np.array([r['final_test_top1'] for r in records if r['method']==IID])
    assert effects[0]['mean_difference'] == pytest.approx(expected.mean())
    assert effects[0]['sample_std'] == pytest.approx(expected.std(ddof=1))
    assert (tmp_path/'results/final_report.md').exists()
    assert len((tmp_path/'results/final_multiseed.csv').read_text().splitlines()) == 71
