"""Four GPU FIFO pipeline. Failed jobs are recorded, never automatically retried."""
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from exp14c import ROOT, RESULTS
from exp14c.config import Trial, METHODS, FINAL_SEEDS, save_json, file_hash
from exp14c.support import grid, smoke_jobs, write_csv, code_hashes

def rank(rows):
    return sorted(rows,key=lambda r:(-r['accuracy'],r['epochs'][-1]['validation_loss'],r['trial_id']))

def occupied_gpus():
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    mapping={u.strip():int(i) for i,u in (line.split(',') for line in uuids.splitlines())}
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True)
    return {mapping[u.strip()] for u in procs.splitlines() if u.strip() in mapping}

def run_queue(queue,prefix):
    configs=list(queue);queue=list(configs);active={}
    states={c.id:dict(**c.values(),status='queued') for c in queue}
    def event(kind,**details):
        with (RESULTS/'scheduler.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),queue=prefix,event=kind,**details))+'\n')
        save_json(RESULTS/f'{prefix}_status.json',states)
    event('started',pid=os.getpid())
    while queue or active:
        for gpu,(p,cfg,log) in list(active.items()):
            code=p.poll()
            if code is None: continue
            log.close()
            valid=code==0 and (cfg.output/'summary.json').exists() and json.loads((cfg.output/'summary.json').read_text())['status']=='completed'
            states[cfg.id].update(status='completed' if valid else 'failed',exit_code=code,ended=time.time())
            event('finished',gpu=gpu,trial_id=cfg.id,status=states[cfg.id]['status']);del active[gpu]
            if not valid:
                with (RESULTS/'failures.jsonl').open('a') as f: f.write(json.dumps(states[cfg.id])+'\n')
        busy=occupied_gpus() if queue else set()
        for gpu in range(4):
            if gpu in active or gpu in busy or not queue: continue
            cfg=queue.pop(0);cfg.output.mkdir(parents=True,exist_ok=True)
            assert not (cfg.output/'summary.json').exists()
            command=[sys.executable,'-B','-m','exp14c.train','--trial-json',json.dumps(cfg.values(),sort_keys=True)]
            log=(cfg.output/'train.log').open('w')
            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(p,cfg,log)
            states[cfg.id].update(status='running',gpu=gpu,pid=p.pid,command='CUDA_VISIBLE_DEVICES='+str(gpu)+' '+shlex.join(command),log=str(cfg.output/'train.log'),started=time.time())
            save_json(cfg.output/'process.json',states[cfg.id]);event('launched',trial_id=cfg.id,**states[cfg.id])
        if active or queue: time.sleep(3)
    assert all(s['status']=='completed' for s in states.values()), 'Failed tasks: see failures.jsonl; no retries'
    return [json.loads((c.output/'summary.json').read_text()) for c in configs]

def freeze(rows):
    assert len(rows)==120 and {r['trial_id'] for r in rows}=={c.id for t in ('cv','nlp') for c in grid(t)}
    winners=[];ranking=[];evidence={}
    for r in rows:
        assert r['status']=='completed' and len(r['epochs'])==5 and not r['official_evaluation_used']
        directory=RESULTS/r['result_dir']
        for name in ('summary.json','config.json','metrics.csv'):
            evidence[str((directory/name).relative_to(RESULTS))]=file_hash(directory/name)
    for task in ('cv','nlp'):
        for method in METHODS:
            ordered=rank([r for r in rows if r['task']==task and r['method']==method])
            winners.append({k:ordered[0][k] for k in ('task','method','lr','C','trial_id','accuracy')})
            ranking.extend(dict(rank=i+1,**{k:r[k] for k in ('task','method','lr','C','trial_id','accuracy')},validation_loss=r['epochs'][-1]['validation_loss']) for i,r in enumerate(ordered))
    write_csv(RESULTS/'search_ranking.csv',ranking)
    save_json(RESULTS/'best_configs.json',winners)
    save_json(RESULTS/'frozen_manifest.json',dict(winners=winners,search_evidence_sha256=evidence,code_sha256=code_hashes(),
        search_grid_sha256=file_hash(RESULTS/'search_grid.json'),selection='last internal validation accuracy desc, loss asc, trial_id asc',frozen_at=time.time()))
    save_json(RESULTS/'frozen_manifest.sha256.json',dict(sha256=file_hash(RESULTS/'frozen_manifest.json')))
    return [Trial(w['task'],w['method'],w['lr'],w['C'],seed=seed,stage='final') for w in winners for seed in FINAL_SEEDS]

def main():
    if '--smoke' in sys.argv:
        run_queue(smoke_jobs(),'smoke');return
    try:
        assert json.loads((RESULTS/'tests.json').read_text())['status']=='passed'
        assert json.loads((RESULTS/'source_provenance.json').read_text())==code_hashes()
        rows=[]
        for task in ('cv','nlp'):
            save_json(RESULTS/'pipeline_status.json',dict(status='running',phase=task+'_search',pid=os.getpid()))
            current=run_queue(grid(task),task+'_search');rows.extend(current)
            save_json(RESULTS/f'{task}_search_results.json',current)
        save_json(RESULTS/'search_results.json',rows)
        finals=freeze(rows)
        save_json(RESULTS/'formal_grid.json',[c.protocol()|dict(trial_id=c.id) for c in finals])
        from exp14c.prepare import prepare
        prepare(official=True)
        save_json(RESULTS/'pipeline_status.json',dict(status='running',phase='formal',pid=os.getpid()))
        final=run_queue(finals,'formal');save_json(RESULTS/'formal_results.json',final)
        from exp14c.audit import main as audit
        audit()
        from exp14c.report import main as report
        report()
        save_json(RESULTS/'pipeline_status.json',dict(status='completed',search=120,formal=80,total=200))
    except Exception:
        save_json(RESULTS/'pipeline_status.json',dict(status='failed',traceback=traceback.format_exc(),pid=os.getpid()))
        raise
if __name__=='__main__': main()
