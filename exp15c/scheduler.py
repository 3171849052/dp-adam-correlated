"""One subprocess per idle GPU, durable status, fail closed, resumable successes."""
import json
import math
import os
import shlex
import subprocess
import sys
import time
from exp15c import ROOT, RESULTS
from exp15c.config import save_json, file_hash, json_hash


def event(row):
    row=dict(time=time.time(),**row)
    with (RESULTS/'scheduler.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


def idle(gpu):
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(gpu)],text=True)
    return int(memory.strip())<256


def valid_metrics(row):
    return (row['status']=='completed'
        and all(isinstance(row.get(k),(int,float)) and math.isfinite(row[k]) for k in ('accuracy','validation_loss'))
        and 0 <= row['accuracy'] <= 1 and row['validation_loss'] >= 0)


def verify_config(cfg,row):
    if row['trial_id']!=cfg.id or any(row[k]!=v for k,v in cfg.values().items()):
        raise ValueError('Wrong result identity/configuration')
    resolved=json.loads((cfg.output/'config.json').read_text())
    digest=resolved.pop('config_sha256')
    if json_hash(resolved)!=digest: raise ValueError('Resolved configuration hash mismatch')
    if any(resolved[k]!=v for k,v in cfg.protocol().items()): raise ValueError('Trial protocol mismatch')
    from exp15c.support import code_hashes
    current=code_hashes()
    if any(current[k]!=resolved['code_sha256'][k] for k in CORE):
        raise ValueError('Training core changed since trial')
    if row['status']=='completed':
        if row['config_sha256']!=digest or any(current[k]!=row['code_sha256'][k] for k in CORE):
            raise ValueError('Summary configuration/core hash mismatch')


def non_finite(cfg):
    """Only explicit numerical divergence in search/review is an exclusion."""
    path=cfg.output/'summary.json'
    if not path.exists(): return False
    row=json.loads(path.read_text())
    if row['status']!='non_finite': return False
    verify_config(cfg,row)
    if cfg.stage not in ('search','review'): return False
    if not row.get('error','').startswith('FloatingPointError:'):
        raise ValueError(f'Nonfinite result lacks explicit numerical failure: {path}')
    return True


def completed(cfg):
    p=cfg.output/'summary.json'
    if not p.exists():return False
    r=json.loads(p.read_text())
    if r['status']!='completed':raise RuntimeError(f'Failed run requires inspection: {p}')
    verify_config(cfg,r)
    if not valid_metrics(r): raise ValueError(f'Invalid completed metrics: {p}')
    for name,h in r['file_sha256'].items():
        if file_hash(cfg.output/name)!=h:raise ValueError(f'Result artifact changed: {cfg.output/name}')
    return True


CORE={f'exp15c/{s}.py' for s in ('__init__','config','train','data','cv_model','nlp_model','optimizer','clipping','privacy','noise','bandinvmf','support')}


def run_batch(name,jobs):
    jobs=list({(c.id,c.replicate):c for c in jobs}.values())
    status={};pending=[];active={};failed=False
    for c in jobs:
        key=f'{c.id}:{c.replicate}'
        if non_finite(c):
            row=json.loads((c.output/'summary.json').read_text())
            status[key]=dict(**c.values(),status='non_finite',reused=True,error=row['error'],
                traceback=row['traceback'],log=str(c.output/'train.log'))
        elif completed(c):status[key]=dict(**c.values(),status='completed',reused=True)
        else:
            pending.append(c);status[key]=dict(**c.values(),status='pending')
    path=RESULTS/f'{name}_status.json';save_json(path,status)
    while pending or active:
        for gpu in (1,2,3):
            if failed or not pending or gpu in active or not idle(gpu):continue
            c=pending.pop(0);c.output.mkdir(parents=True,exist_ok=True)
            if (c.output/'config.json').exists():raise RuntimeError(f'Partial run requires inspection: {c.output}')
            log=(c.output/'train.log').open('a')
            argv=[sys.executable,'-B','-m','exp15c.train','--trial-json',json.dumps(c.values(),sort_keys=True)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
            process=subprocess.Popen(argv,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            key=f'{c.id}:{c.replicate}'
            status[key]=dict(**c.values(),status='running',gpu=gpu,pid=process.pid,
                command=f'CUDA_VISIBLE_DEVICES={gpu} '+shlex.join(argv),log=str(c.output/'train.log'),started=time.time())
            active[gpu]=(c,process,log);event(dict(batch=name,event='launched',**status[key]));save_json(path,status)
        for gpu,(c,process,log) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            log.close();del active[gpu];key=f'{c.id}:{c.replicate}'
            excluded=non_finite(c)
            okay=not excluded and rc==0 and completed(c)
            status[key].update(status='non_finite' if excluded else 'completed' if okay else 'failed',exit_code=rc,ended=time.time())
            if excluded:
                row=json.loads((c.output/'summary.json').read_text())
                status[key].update(error=row['error'],traceback=row['traceback'])
            event(dict(batch=name,event='finished',**status[key]));save_json(path,status)
            failed|=not (okay or excluded)
        if failed and not active:raise RuntimeError(f'{name} failed; see {path}; no automatic retry')
        if pending or active:time.sleep(3)
    return [json.loads((c.output/'summary.json').read_text()) for c in jobs]
