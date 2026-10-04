import json
import pytest
from exp3 import search_records as records


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(records,'BASE',tmp_path)
    monkeypatch.setattr(records,'SEARCH',tmp_path/'search')
    return tmp_path


def values(path,method='nonprivate_hybrid',seed=20261001):
    return dict(method=method,seed=seed,result_dir=str(path),muon_lr=.01,adam_lr=.0005,max_grad_norm=10)


def test_search_only_tuning_seed_and_stage_order(isolated):
    with pytest.raises(AssertionError):
        records.plan('A','invalid',[values(isolated/'search/trial',seed=20261011)],'invalid validation seed')
    with pytest.raises(KeyError):
        records.plan('B','out_of_order',[values(isolated/'search/trial','iid_dp_hybrid')],'no completed Stage A')
    assert not (isolated/'search_history.json').exists()


def test_one_batch_observe_barrier_and_budget(isolated):
    records.plan('A','round_1',[values(isolated/'search/trial')],'first explicit batch')
    with pytest.raises(AssertionError,match='Observe'):
        records.plan('A','round_2',[values(isolated/'search/trial2')],'cannot run ahead')
    history=records.read_history()
    history['rounds'][0]['status']='observed'
    history['trials']=[dict(stage='A',reused_identity=False,spec=values(isolated/f'search/past{i}'),
                             **{p:1. for p in records.PARAMETERS}) for i in range(8)]
    records.save(history)
    with pytest.raises(AssertionError):
        records.plan('A','over_budget',[values(isolated/'search/new')],'cannot exceed eight')
