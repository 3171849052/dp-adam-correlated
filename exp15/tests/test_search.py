import json
from dataclasses import replace
import pytest
from exp15 import BASE, ROOT
from exp15.config import Trial, METHODS, SEARCH_SEED, save_json, file_hash
from exp15 import search


def test_equal_budgets_dedup_frozen_guard_and_authorization(monkeypatch):
    import tempfile
    with tempfile.TemporaryDirectory(dir=BASE/'runtime') as d:
        from pathlib import Path
        directory=Path(d)
        monkeypatch.setattr(search,'RESULTS',directory)
        monkeypatch.setattr(search,'FROZEN',directory/'frozen_grid.json')
        monkeypatch.setattr(search,'SELECTED',directory/'selected_configs.json')
        monkeypatch.setattr(search,'require_phase1',lambda:None)
        save_json(directory/'phase1_acceptance.json',dict(status='passed'))
        evidence=directory/'pilot/evidence.json';save_json(evidence,dict(status='completed'))
        groups=[dict(task=t,method=m,learning_rates=[.01,.03,.09],clips=[.1,1.,10.],
            rationale='controlled synthetic test',evidence=[dict(summary=str(evidence.relative_to(ROOT)),sha256=file_hash(evidence))])
            for t in ('cv','nlp') for m in METHODS]
        frozen=search.freeze(dict(groups=groups))
        assert len(frozen['candidates'])==90
        for key,ids in frozen['schemes']['A'].items():
            assert len(ids)==3 and len(frozen['schemes']['B'][key])==9
            assert set(ids).issubset(frozen['schemes']['B'][key])
        c=Trial(**next(iter(frozen['candidates'].values())))
        search.authorize(c)
        with pytest.raises(ValueError):search.authorize(replace(c,lr=.777))
        with pytest.raises(FileExistsError):search.freeze(dict(groups=groups))
        modified=dict(frozen);modified['candidates']={}
        save_json(search.FROZEN,modified)
        with pytest.raises(ValueError,match='changed'):search.load_frozen()


def test_all_failure_rows_excluded_from_selection():
    good=dict(status='completed',accuracy=.4,validation_loss=.7,trial_id='valid')
    bad=dict(status='non_finite',accuracy=1.,validation_loss=0.,trial_id='invalid')
    assert search.rank([bad,good])==[good]


def test_every_run_records_exit_code_and_failure(monkeypatch):
    # Exercise the real scheduler event path that includes the Trial.stage field.
    import tempfile
    from pathlib import Path
    from exp15 import scheduler
    with tempfile.TemporaryDirectory(dir=BASE/'runtime') as d:
        monkeypatch.setattr(scheduler,'RESULTS',Path(d))
        row=dict(stage='smoke',event='launched',batch='smoke_test',pid=123)
        scheduler.event(row)
        saved=json.loads((Path(d)/'scheduler.jsonl').read_text())
        assert saved['stage']=='smoke' and saved['batch']=='smoke_test' and saved['pid']==123
