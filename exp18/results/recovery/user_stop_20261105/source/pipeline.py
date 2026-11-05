import fcntl,traceback
from exp18 import RESULTS
from exp18.common import save,read,verify_seal,GROUPS
from exp18.search import coarse,trials,rank,fine,boundary
from exp18.scheduler import execute
from exp18.report import report

def plan(name,configs):
    p=RESULTS/f'{name}_plan.json'
    if p.exists():assert read(p)==configs
    else:save(p,configs)
    return configs

def main():
    with (RESULTS/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_seal()
        rows=execute(plan('coarse',trials(coarse(),'coarse',[20261101])),'coarse')
        f=trials(fine(rows),'fine',[20261102]);assert len(f)<=68;rows+=execute(plan('fine',f),'fine')
        b=trials(boundary(rows),'boundary',[20261102]);assert len(b)<=12;rows+=execute(plan('boundary',b),'boundary')
        top={g:rank(rows,g)[:2] for g in GROUPS};assert all(len(v)==2 for v in top.values())
        review=execute(plan('review',trials(top,'review',[20261103,20261104,20261105])),'review')
        selected={g:rank(review,g)[:1] for g in GROUPS};p=RESULTS/'hyperparameters.json'
        if p.exists():assert read(p)==selected
        else:save(p,selected)
        formal=execute(plan('formal',trials(selected,'formal',list(range(20261111,20261116)))),'formal')
        report(formal);save(RESULTS/'status.json',dict(status='complete',training_count=len(rows)+len(review)+len(formal)))
if __name__=='__main__':
    try:main()
    except Exception:
        save(RESULTS/'pipeline_failure.json',dict(traceback=traceback.format_exc()));raise
