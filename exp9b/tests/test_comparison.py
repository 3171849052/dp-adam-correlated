import ast
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from exp9b import ROOT
from exp9b.config import METHODS, FINAL_SEEDS, Trial
from exp9b.grid import GRIDS
from exp9b import comparison


def test_exp9_grid_and_training_kernel_are_preserved():
    tree=ast.parse((ROOT/'exp9/grid.py').read_text())
    original=next(ast.literal_eval(s.value) for s in tree.body if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GRIDS' for t in s.targets))
    assert GRIDS==original
    for name in ('clipping.py','cv_model.py','nlp_model.py','data.py','privacy.py'):
        expected=(ROOT/'exp9'/name).read_text().replace('exp9','exp9b')
        assert (ROOT/'exp9b'/name).read_text()==expected


def test_same_seed_historical_pairing_and_ci(tmp_path,monkeypatch):
    monkeypatch.setattr(comparison,'ROOT',tmp_path)
    monkeypatch.setattr(comparison,'RESULTS',tmp_path/'exp9b/results')
    current=[];old=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            for i,seed in enumerate(FINAL_SEEDS):
                r=dict(task=task,method=method,seed=seed,trial_id=f'{task}-{method}-{seed}',result_dir='matrix',
                       accuracy=.70+.001*i,lr=.001,C=10,eps_scale=.1 if method.endswith('-scale') else None,
                       initialization_sha256='init',classifier_initialization_sha256='head',train_order_sha256='order',rng_trace=['rng'])
                current.append(r)
                old.append(dict(r,method=method.replace('bandmf','bandinvmf'),accuracy=r['accuracy']-.01))
    for experiment in ('exp9','exp9b'):
        folder=tmp_path/experiment/'results/matrix';folder.mkdir(parents=True)
        np.savez(folder/'matrices.npz',workload=np.eye(2))
    raw,summary=comparison.paired_history(current,old)
    assert len(raw)==140 and len(summary)==14
    for r in summary:
        assert r['n']==10 and r['mean']==pytest.approx(.01)
        assert r['ci95_low']==pytest.approx(.01) and r['ci95_high']==pytest.approx(.01)
    old[0]['rng_trace']=['different']
    with pytest.raises(AssertionError): comparison.paired_history(current,old)
