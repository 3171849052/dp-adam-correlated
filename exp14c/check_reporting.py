"""Synthetic stage-transition and report checks, isolated from real results."""
import json
from dataclasses import replace
from exp14c import RESULTS
from exp14c.config import save_json
from exp14c.support import grid, historical

def main():
    from exp14c import pipeline, report
    sandbox=RESULTS/'verification';sandbox.mkdir(exist_ok=True)
    pipeline.RESULTS=sandbox;report.RESULTS=sandbox
    rows=[]
    for task in ('cv','nlp'):
        for c in grid(task):
            r=dict(c.values(),trial_id=c.id,status='completed',accuracy=.5,epochs=[dict(validation_loss=.7)]*5,official_evaluation_used=False,result_dir=f'search/{task}/{c.id}')
            d=sandbox/r['result_dir'];d.mkdir(parents=True,exist_ok=True)
            for name in ('summary.json','config.json','metrics.csv'): (d/name).write_text(json.dumps(r))
            rows.append(r)
    save_json(sandbox/'search_grid.json',[r['trial_id'] for r in rows])
    finals=pipeline.freeze(rows)
    assert len(finals)==80 and len({c.id for c in finals})==80
    old=json.loads((historical/'final_results.json').read_text());formal=[]
    for c in finals:
        ref=next(o for o in old if o['task']==c.task and o['method']==c.method and o['seed']==c.seed and o['epsilon']==8)
        r=dict(c.values(),trial_id=c.id,accuracy=ref['accuracy'],epochs=[dict(epoch=i,accuracy=ref['accuracy'],train_loss=.7) for i in range(1,6)],result_dir=f'final/{c.task}/{c.id}')
        d=sandbox/r['result_dir'];d.mkdir(parents=True,exist_ok=True)
        (d/'mechanism_metrics.csv').write_text('step,clip_fraction\n1,0.5\n2,0.4\n')
        (d/'optimizer_metrics.jsonl').write_text('\n'.join(json.dumps(dict(step=i,update_norm=.01,v_mean=.001,vhat_mean=.01)) for i in (1,2)))
        formal.append(r)
    save_json(sandbox/'search_results.json',rows);save_json(sandbox/'formal_results.json',formal)
    report.main()
    assert len(list((sandbox/'plots').glob('*.png')))==16
    stats=json.loads((sandbox/'statistics.json').read_text())
    assert all(s['paired_delta']['mean']==0 and s['paired_delta']['ci95_low']==0 for s in stats)
    (sandbox/'SYNTHETIC_ONLY.txt').write_text('Synthetic test artifacts only. Not experimental results.\n')
    save_json(RESULTS/'reporting_tests.json',dict(status='passed',synthetic_only=True,freeze_120_to_80=True,eight_groups=True,plots=16,paired_zero_difference=True))
if __name__=='__main__': main()
