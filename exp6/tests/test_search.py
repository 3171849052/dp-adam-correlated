import json
import math
from pathlib import Path
import pytest
from exp6.config import METHODS, Trial
from exp6.runtime import EXP
import exp6.search as search


def test_staged_search_resume_dedup_freeze_and_no_final_launch(tmp_path,monkeypatch):
    plan = json.loads((EXP / 'search_plan.json').read_text())
    folder = tmp_path / 'search'
    trained = []
    def fake_read(job):
        return json.loads((Path(job['result_dir']) / 'summary.json').read_text())
    def fake_launch(jobs):
        new, reused = [], []
        for job in jobs:
            t=Trial(**job['trial']); path=Path(job['result_dir'])
            if (path / 'summary.json').exists():
                reused.append(t.identity); continue
            path.mkdir(parents=True)
            targets=[(3.,.003),(10.,.01),(.3,.001),(1.,.003)]
            C,lr=targets[METHODS.index(t.method)]
            acc=.6-.015*abs(math.log(t.C/C))-.015*abs(math.log(t.lr/lr))
            if t.scaled: acc-=.04*abs(math.log(t.geom_eps/.01))
            s=dict(status='completed',trial=t.asdict(),trial_id=t.identity,fingerprint='fingerprint-'+t.identity,
                final_test_top1=acc,final_privacy=dict(epsilon=8.,delta=1e-5),calibration=dict(target_mu=1.666))
            (path / 'summary.json').write_text(json.dumps(s))
            trained.append(t.identity);new.append(t.identity)
        return dict(trained=new,reused=reused)
    monkeypatch.setattr(search,'environment',lambda: {'test':'fixed'})
    monkeypatch.setattr(search,'launch',fake_launch)
    monkeypatch.setattr(search,'read_completed',fake_read)
    monkeypatch.setattr(search,'diagnostics',lambda folder: {'artifact':str(folder)})
    result=search.Search(plan,folder).run()
    assert len(trained) == len(set(trained)) == result['total_unique_tuning_trials']
    assert len(trained) < 90
    frozen=json.loads((folder / 'selected_configs.json').read_text())
    assert frozen['shared_scale_geom_eps'] == .01
    assert frozen['methods'][METHODS[0]]['hyperparameters']['C'] == 3.
    assert frozen['methods'][METHODS[1]]['hyperparameters']['C'] == 10.
    assert all(frozen['methods'][m]['hyperparameters']['geom_eps'] == .01 for m in METHODS[2:])
    manifest=json.loads((folder / 'prepared_final_manifest.json').read_text())
    assert manifest['status'] == 'prepared_not_launched' and len(manifest['jobs']) == 12
    count=len(trained)
    again=search.Search(plan,folder).run()
    assert len(trained) == count and again['actual_trained_trials'] == 0
    assert again['reused_preexisting_trials'] == count
    assert json.loads((folder / 'selected_configs.json').read_text()) == frozen
    with pytest.raises(AssertionError,match='conflict'):
        search.Search(dict(plan,initial_lr=.003),folder)
    bad=folder / 'trials' / 'incomplete';bad.mkdir()
    with pytest.raises(AssertionError,match='Incomplete'):
        search.Search(plan,folder)
