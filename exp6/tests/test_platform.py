import json
from pathlib import Path
import pytest
from exp6.runtime import EXP, output_path
from exp6.config import METHODS, GPUS, Trial, FIXED, FINAL_SEEDS
from exp6.final_runner import final_jobs
from exp6.report import aggregate, comparisons
from exp6.search import rank


def test_output_paths_and_identity():
    assert output_path(EXP / 'results/new') == EXP / 'results/new'
    with pytest.raises(AssertionError): output_path(EXP.parent / 'exp5/results/new')
    assert GPUS == (1,2,3)
    trial = Trial(METHODS[0])
    assert trial.identity == Trial(**trial.asdict()).identity
    assert trial.identity != Trial(METHODS[0],geom_eps=.3).identity
    with pytest.raises(AssertionError): Trial(METHODS[0],geom_eps=0)
    with pytest.raises(AssertionError): Trial(METHODS[0],physical_batch_size=3)


def selected():
    return dict(status='frozen',fixed=FIXED,shared_scale_geom_eps=.01,
        methods={m:dict(status='frozen',method=m,fixed=FIXED,physical_batch_size=50,
          hyperparameters=dict(lr=.001*(i+1),C=.3*(i+1),geom_eps=.01 if m.endswith('-scale') else .1))
          for i,m in enumerate(METHODS)})


def test_independent_final_jobs_and_shared_scale_geometry():
    frozen = selected()
    jobs = final_jobs(frozen,list(FINAL_SEEDS))
    assert len(jobs) == 12 and len({j['result_dir'] for j in jobs}) == 12
    assert all(Path(j['result_dir']).is_relative_to(EXP / 'results/final') for j in jobs)
    assert len({j['trial']['lr'] for j in jobs}) == 4
    with pytest.raises(AssertionError): final_jobs(dict(frozen,status='draft'),list(FINAL_SEEDS))
    with pytest.raises(AssertionError): final_jobs(frozen,[11,12,13])
    frozen['methods'][METHODS[3]]['hyperparameters']['geom_eps'] = .1
    with pytest.raises(AssertionError): final_jobs(frozen,list(FINAL_SEEDS))


def test_top1_tie_break_and_factorial_effects():
    def score(acc,C,lr,**extra):
        return dict(final_test_top1=acc,trial=dict(C=C,lr=lr,geom_eps=.1),trial_id='x',**extra)
    assert rank(score(.6,3,.001)) < rank(score(.5,.1,.0001))
    assert rank(score(.6,1,.03)) < rank(score(.6,3,.001))
    assert rank(score(.6,1,.001,clipping_fraction=1)) < rank(score(.6,1,.003,clipping_fraction=0))
    assert aggregate([1.,3.])['std'] == pytest.approx(2**.5)
    effects = comparisons([[.4,.45,.5,.6],[.41,.44,.51,.59],[.39,.46,.49,.61]])
    assert effects['MF_gain_raw']['mean'] == pytest.approx(.05)
    assert effects['MF_gain_scale']['mean'] == pytest.approx(.1)
    assert effects['interaction']['mean'] == pytest.approx(.05)
