"""Four GPU FIFO pipeline. Failed jobs are recorded, never automatically retried."""
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from exp14d import ROOT, RESULTS
from exp14d.config import Trial, METHODS, FINAL_SEEDS, save_json, file_hash
from exp14d.support import grid, boundary_grid, smoke_jobs, write_csv, code_hashes, historical_rows, protected_snapshot

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
            command=[sys.executable,'-B','-m','exp14d.train','--trial-json',json.dumps(cfg.values(),sort_keys=True)]
            log=(cfg.output/'train.log').open('w')
            p=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(p,cfg,log)
            states[cfg.id].update(status='running',gpu=gpu,pid=p.pid,command='CUDA_VISIBLE_DEVICES='+str(gpu)+' '+shlex.join(command),log=str(cfg.output/'train.log'),started=time.time())
            save_json(cfg.output/'process.json',states[cfg.id]);event('launched',trial_id=cfg.id,**states[cfg.id])
        if active or queue: time.sleep(3)
    assert all(s['status']=='completed' for s in states.values()), 'Failed tasks: see failures.jsonl; no retries'
    return [json.loads((c.output/'summary.json').read_text()) for c in configs]

def freeze(rows,module):
    expected=128 if module=='A' else 36
    assert len(rows)==expected
    for r in rows:
        assert r['status']=='completed' and len(r['epochs'])==5 and not r['official_evaluation_used']
    winners=[];ranking=[];evidence={}
    for task,method in sorted({(r['task'],r['method']) for r in rows}):
        ordered=rank([r for r in rows if (r['task'],r['method'])==(task,method)])
        w=ordered[0]
        winners.append({k:w.get(k) for k in ('task','method','lr','C','eps_scale','trial_id','accuracy','source_experiment','source_summary')})
        ranking.extend(dict(rank=i+1,**{k:r.get(k) for k in ('task','method','lr','C','eps_scale','trial_id','accuracy','source_experiment','source_summary')},validation_loss=r['epochs'][-1]['validation_loss']) for i,r in enumerate(ordered))
    for r in rows:
        p=__import__('pathlib').Path(r.get('source_summary') or RESULTS/r['result_dir']/'summary.json')
        for name in ('summary.json','config.json','metrics.csv'):
            evidence[str(p.parent/name)]=file_hash(p.parent/name)
    write_csv(RESULTS/f'{module}_search_ranking.csv',ranking)
    manifest=dict(winners=winners,search_evidence_sha256=evidence,code_sha256=code_hashes(),selection='last internal validation accuracy descending, loss ascending, fixed trial_id ascending',frozen_at=time.time())
    save_json(RESULTS/f'{module}_frozen_manifest.json',manifest)
    save_json(RESULTS/f'{module}_frozen_manifest.sha256.json',dict(sha256=file_hash(RESULTS/f'{module}_frozen_manifest.json')))
    return winners

def finals(winners,module):
    return [Trial(w['task'],w['method'],w['lr'],w['C'],eps_scale=w.get('eps_scale'),seed=seed,stage='final',module=module) for w in winners for seed in FINAL_SEEDS]

def phase(name):
    save_json(RESULTS/'pipeline_status.json',dict(status='running',phase=name,pid=os.getpid(),time=time.time()))

def main():
    if '--smoke' in sys.argv:
        run_queue(smoke_jobs(),'smoke');return
    try:
        assert json.loads((RESULTS/'tests.json').read_text())['status']=='passed'
        assert json.loads((RESULTS/'prelaunch_audit.json').read_text())['status']=='passed'
        assert json.loads((RESULTS/'source_provenance.json').read_text())==code_hashes()
        phase('A_boundary_search')
        added=run_queue(boundary_grid(),'A_search');save_json(RESULTS/'A_new_search_results.json',added)
        old=historical_rows('exp14c','search')
        merged=old+added;save_json(RESULTS/'A_merged_search_results.json',merged)
        winners=freeze(merged,'A')
        original=json.loads((ROOT/'exp14c/results/best_configs.json').read_text())
        changes=[];changed=[]
        for w in winners:
            o=next(r for r in original if r['task']==w['task'] and r['method']+'-adam3'==w['method'])
            different=(w['lr'],w['C'])!=(o['lr'],o['C'])
            changes.append(dict(task=w['task'],method=w['method'],original=o,updated=w,changed=different,original_manifest_sha256=file_hash(ROOT/'exp14c/results/frozen_manifest.json')))
            if different:changed.append(w)
        assert len(changed)<=4
        save_json(RESULTS/'A_config_changes.json',changes)
        from exp14d.prepare import prepare
        # Official assets are accessed only after internal selection is frozen.
        prepare(official=True)
        jobs=finals(changed,'A');save_json(RESULTS/'A_formal_grid.json',[c.protocol()|dict(trial_id=c.id) for c in jobs])
        phase('A_formal');new=run_queue(jobs,'A_formal');save_json(RESULTS/'A_new_formal_results.json',new)
        changed_keys={(w['task'],w['method']) for w in changed}
        reused=[r for r in historical_rows('exp14c','final') if (r['task'],r['method']) not in changed_keys]
        save_json(RESULTS/'A_formal_results.json',new+reused)
        phase('B_search');rows=[]
        for task in ('cv','nlp'):
            current=run_queue(grid(task),'B_'+task+'_search');rows.extend(current)
            save_json(RESULTS/f'B_{task}_search_results.json',current)
        save_json(RESULTS/'B_search_results.json',rows)
        winners=freeze(rows,'B');jobs=finals(winners,'B')
        assert len(jobs)==120
        save_json(RESULTS/'B_formal_grid.json',[c.protocol()|dict(trial_id=c.id) for c in jobs])
        phase('B_formal');rows=run_queue(jobs,'B_formal');save_json(RESULTS/'B_formal_results.json',rows)
        phase('report')
        from exp14d.report import main as report
        report()
        assert protected_snapshot()==json.loads((RESULTS/'protected_before.json').read_text())
        save_json(RESULTS/'final_audit.json',dict(status='passed',history_unchanged=True,new_search=44,new_formal=120+len(new)))
        save_json(RESULTS/'pipeline_status.json',dict(status='completed',search=44,formal=120+len(new),total=164+len(new)))
    except Exception:
        error=dict(status='failed',traceback=traceback.format_exc(),pid=os.getpid(),time=time.time())
        save_json(RESULTS/'pipeline_status.json',error)
        with (RESULTS/'failures.jsonl').open('a') as f:f.write(json.dumps(error)+'\n')
        raise
if __name__=='__main__': main()
