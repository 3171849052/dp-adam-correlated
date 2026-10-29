"""Exercise all orchestration phases without running synthetic trials as results."""
from pathlib import Path
from exp16d.config import METHODS,FINAL_SEEDS

def test_pipeline_reaches_all_40_formal_and_report(monkeypatch):
    import exp16d.pipeline as p
    import exp16d.stage1 as stage
    import exp16d.report as report
    saved={};calls=[];reported=[]
    monkeypatch.setattr(stage,'gpu_acceptance',lambda:None)
    monkeypatch.setattr(p,'phase',lambda *a,**k:None)
    monkeypatch.setattr(p,'save_json',lambda path,value:saved.__setitem__(Path(path).name,value))
    monkeypatch.setattr(p,'freeze_json',lambda name,value:saved.setdefault(name,value))
    monkeypatch.setattr(p,'read',lambda path: {'stage':'completed'} if 'exp16c' in str(path) else saved[Path(path).name])
    monkeypatch.setattr(p,'file_hash',lambda path:'test-hash')
    monkeypatch.setattr(report,'report',lambda:reported.append(True))
    completed={}
    def batch(name,jobs):
        jobs=list(jobs);calls.append((name,jobs));out=[]
        for c in jobs:
            if c.id not in completed:
                completed[c.id]=dict(**c.values(),trial_id=c.id,status='completed',accuracy=.5+c.lr,validation_loss=1.,official_evaluation_used=c.stage=='final')
            out.append(completed[c.id])
        return out
    monkeypatch.setattr(p,'run_batch',batch)
    p.run()
    assert [name for name,_ in calls]==['search','review','controlled','formal']
    assert len(saved['formal_results.json'])==40 and reported==[True]
    assert {(r['method'],r['seed']) for r in saved['formal_results.json']}=={(m,s) for m in METHODS for s in FINAL_SEEDS}
    assert 76<=len(completed)<=82
    assert len(saved['controlled_results.json'])==9
