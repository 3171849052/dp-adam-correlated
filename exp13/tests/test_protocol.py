import numpy as np
import pytest
from exp13.assets import coordinate_map
from exp13.io import local_path
from exp13.math import HORIZONS, SEEDS
from exp13.report import statistics
from exp13.run import protocol


def test_formal_protocol_and_disjoint_evaluation_seeds():
    p = protocol(False, 4096, 3)
    assert p['seeds'] == list(range(20261111, 20261121)) == list(SEEDS)
    assert p['horizons'] == [225, 250, 310] == list(HORIZONS)
    held = [seed+100_000_000+i*1_000_000 for seed in SEEDS for i in range(3)]
    assert len(set(held)) == 30 and not set(held) & set(SEEDS)
    assert p['coordinates'] == 4096


def test_coordinate_mapping_has_valid_unique_parameter_offsets():
    params = {'encoder.weight': [4, 5], 'head.bias': [2], 'head.weight': [2, 4]}
    mapping = coordinate_map(params, 11)
    assert mapping['parameter_count'] == 30
    coordinates = mapping['coordinates']
    assert len({(c['parameter'], c['offset']) for c in coordinates}) == 11
    for c in coordinates:
        assert 0 <= c['offset'] < np.prod(params[c['parameter']])


def test_student_t_cross_seed_statistics():
    result = statistics([1., 2., 3., 4.])
    assert result['n'] == 4 and result['mean'] == 2.5
    assert result['sample_std'] == pytest.approx(np.sqrt(5/3))
    assert result['ci95_low'] == pytest.approx(.445739743, abs=1e-8)
    assert result['ci95_high'] == pytest.approx(4.554260257, abs=1e-8)


def test_output_cannot_escape_exp13():
    with pytest.raises(AssertionError):
        local_path('exp9/results/exp13-test.json')
