"""FIFO scheduling, four physical GPUs, no retries; selection before test access."""
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from exp14b import ROOT, RESULTS
from exp14b.config import save_json
from exp14b.support import jobs, METHODS, write_csv

def rank(rows):
    return sorted(rows,key=lambda r:(-r['accuracy'],r['epochs'][-1]['validation_loss'],r['number']))

def run_queue(queue,prefix):
    queue=list(queue);active={}
    states={c.id:dict(**c.values(),status='queued') for c in queue}
    def event(kind,**details):
        with (RESULTS/f'{prefix}.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),event=kind,**details))+'\n')
        save_json(RESULTS/f'{prefix}_status.json',states)
    event('started',pid=os.getpid())
    while queue or active:
        for gpu,(p,cfg,log) in list(active.items()):
            code=p.poll()
            if code is None: continue
            log.close();states[cfg.id].update(status='completed' if code==0 else 'failed',exit_code=code,ended=time.time())
            event('finished',gpu=gpu,trial_id=cfg.id);del active[gpu]
        for gpu in range(4):
            if gpu in active or not queue: continue
            cfg=queue.pop(0);cfg.output.mkdir(parents=True,exist_ok=True)
            assert not (cfg.output/'summary.json').exists()
            command=[sys.executable,'-B','-m','exp14b.train','--trial-json',json.dumps(cfg.values(),sort_keys=True)]
            log=(cfg.output/'train.log').open('w')
            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(p,cfg,log)
            states[cfg.id].update(status='running',gpu=gpu,pid=p.pid,command='CUDA_VISIBLE_DEVICES='+str(gpu)+' '+shlex.join(command),log=str(cfg.output/'train.log'),started=time.time())
            save_json(cfg.output/'process.json',states[cfg.id]);event('launched',trial_id=cfg.id,**states[cfg.id])
        if active: time.sleep(3)
    assert all(s['status']=='completed' for s in states.values()), 'Failed tasks: see status/logs; no retries'
    return [json.loads((RESULTS/s['stage']/'cv'/i/'summary.json').read_text()) for i,s in states.items()]

def main():
    if '--smoke' in sys.argv:
        run_queue(jobs('smoke'),'smoke');return
    grid=jobs();save_json(RESULTS/'search_grid.json',[c.protocol()|dict(trial_id=c.id) for c in grid])
    rows=run_queue(sorted(grid,key=lambda c:((c.number-1)%9,(c.number-1)//9)),'search');save_json(RESULTS/'search_results.json',rows)
    ranking=[];winners=[]
    for method in METHODS:
        ordered=rank([r for r in rows if r['method']==method])
        ranking.extend(dict(rank=i+1,number=r['number'],method=method,lr=r['lr'],eps=r['eps'],accuracy=r['accuracy'],validation_loss=r['epochs'][-1]['validation_loss'],trial_id=r['trial_id']) for i,r in enumerate(ordered))
        winners.append(replace(next(c for c in grid if c.id==ordered[0]['trial_id']),stage='final'))
    write_csv(RESULTS/'search_ranking.csv',ranking)
    save_json(RESULTS/'best_configs.json',[c.values() for c in winners])
    final=run_queue(winners,'confirmation');save_json(RESULTS/'confirmation_results.json',final)
    from exp14b.report import main as report
    report()
    save_json(RESULTS/'pipeline_status.json',dict(status='completed',search=36,confirmation=4))
if __name__=='__main__': main()
