import json
import tempfile
from pathlib import Path
from exp15 import BASE
from exp15 import config, report
from exp15.config import Trial, METHODS, FINAL_SEEDS, save_json


def test_ten_seed_report_pairs_and_reuses_identical_schemes(monkeypatch):
    with tempfile.TemporaryDirectory(dir=BASE/'runtime') as d:
        directory=Path(d)
        monkeypatch.setattr(config,'RESULTS',directory)
        monkeypatch.setattr(report,'RESULTS',directory)
        monkeypatch.setattr(report,'SELECTED',directory/'selected_configs.json')
        monkeypatch.setattr(report,'FROZEN',directory/'frozen_grid.json')
        monkeypatch.setattr(report,'protected_snapshot',lambda:{})
        winners=[];trials=[]
        for task in ('cv','nlp'):
            for mi,method in enumerate(METHODS):
                for scheme in ('A','B'):
                    winners.append(dict(scheme=scheme,task=task,method=method,lr=.03,clip=1.,
                                        validation_accuracy=.5,validation_loss=.7,trial_id='synthetic'))
                for si,seed in enumerate(FINAL_SEEDS):
                    c=Trial(task,method,.03,stage='final',seed=seed);trials.append(c.values())
                    save_json(c.output/'summary.json',dict(**c.values(),trial_id=c.id,status='completed',
                        completed_epochs=5,accuracy=.5+.001*si+.01*mi,validation_loss=.7-.001*si-.01*mi,
                        seconds=1.,peak_allocated_bytes=100))
                    (c.output/'diagnostics.jsonl').write_text('{"step":1,"update_norm":0.1}\n')
        save_json(report.SELECTED,dict(winners=winners,final_trials=trials))
        save_json(report.FROZEN,{})
        save_json(directory/'protected_before.json',{})
        report.main()
        statistics=json.loads((directory/'statistics.json').read_text())
        assert len(statistics['summary'])==20 and len(statistics['paired'])==100
        assert all(r['n']==10 for r in statistics['summary'])
        for r in statistics['paired']:
            assert r['sample_std']<1e-14
            if r['scheme']=='B_vs_A':assert r['mean']==0
            else:assert r['mean']>0 if r['metric']=='accuracy' else r['mean']<0
        assert json.loads((directory/'final_audit.json').read_text())['formal_runs']==100
        assert (directory/'final_report.md').exists()
