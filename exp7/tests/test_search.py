import pytest
from exp7.config import IID, SCALE, trial
from exp7.report import best
from exp7.search import nearby, EPS_GRID, lr_bracket

def row(lr, utility, **extra):
    return dict(trial(IID, lr, 30), status='completed', smoke=False, final_test_top1=utility, **extra)

def test_accuracy_only_and_smoke_failure_exclusion():
    a, b = row(.0005, .6, loss=2), row(.001, .7, loss=4)
    smoke = dict(row(.002, .99), smoke=True)
    failure = dict(row(.004, None), status='numerical_failure')
    assert best([a, b, smoke, failure]) == b
    assert best([a, row(.001, .6)]) == a

def test_neighbors_and_lr_stopping():
    assert nearby(EPS_GRID, .1) == (.01, .1, .3)
    assert nearby(EPS_GRID, 1e-8) == (1e-8, 1e-4)
    points = [row(.00025, .5), row(.0005, .6), row(.001, .55)]
    assert lr_bracket(points, best(points))
    points[-1]['final_test_top1'] = .7
    assert not lr_bracket(points, best(points))

def test_joint_coordinates_do_not_fix_C_across_epsilon():
    for eps in EPS_GRID:
        point = trial(SCALE, .002, 20/eps, eps)
        assert point['seed'] == 20261001 and point['C'] * eps == pytest.approx(20)
