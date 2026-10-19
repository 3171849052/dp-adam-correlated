"""One subprocess per idle GPU, durable status, fail closed, resumable successes."""
import json
import os
import shlex
import subprocess
import sys
import time
from exp15 import ROOT, RESULTS
from exp15.config import save_json, file_hash


def event(row):
    row=dict(time=time.time(),**row)
    with (RESULTS/'scheduler.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')


def idle(gpu):
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(gpu)],text=True)
    return int(memory.strip())<256


def completed(cfg):
    p=cfg.output/'summary.json'
    if not p.exists():return False
    r=json.loads(p.read_text())
    if r['status']!='completed':raise RuntimeError(f'Failed run requires inspection: {p}')
    if r['trial_id']!=cfg.id:raise ValueError('Wrong result identity')
    from exp15.support import code_hashes
    current=code_hashes()
    for key,value in r['code_sha256'].items():
        if key in CORE and current.get(key)!=value:raise ValueError(f'Code changed since trial: {key}')
    for name,h in r['file_sha256'].items():
        if file_hash(cfg.output/name)!=h:raise ValueError(f'Result artifact changed: {cfg.output/name}')
    return True


CORE={f'exp15/{s}.py' for s in ('config','train','data','cv_model','nlp_model','optimizer','clipping','privacy','noise','bandinvmf','support')}


def run_batch(name,jobs,allow_nonfinite=False):
    jobs=list({(c.id,c.replicate):c for c in jobs}.values())
    status={};pending=[];active={};failed=False
    for c in jobs:
        key=f'{c.id}:{c.replicate}'
        existing=c.output/'summary.json'
        if allow_nonfinite and existing.exists() and json.loads(existing.read_text())['status']=='non_finite':
            status[key]=dict(**c.values(),status='non_finite',reused=True)
        elif completed(c):status[key]=dict(**c.values(),status='completed',reused=True)
        else:pending.append(c)
    path=RESULTS/f'{name}_status.json';save_json(path,status)
    while pending or active:
        for gpu in (0,1,2,3):
            if failed or not pending or gpu in active or not idle(gpu):continue
            c=pending.pop(0);c.output.mkdir(parents=True,exist_ok=True)
            log=(c.output/'train.log').open('w')
            argv=[sys.executable,'-B','-m','exp15.train','--trial-json',json.dumps(c.values(),sort_keys=True)]
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
            summary=c.output/'summary.json'
            unstable=allow_nonfinite and summary.exists() and json.loads(summary.read_text())['status']=='non_finite'
            okay=rc==0 and completed(c)
            status[key].update(status='completed' if okay else 'non_finite' if unstable else 'failed',exit_code=rc,ended=time.time())
            event(dict(batch=name,event='finished',**status[key]));save_json(path,status)
            failed|=not (okay or unstable)
        if failed and not active:raise RuntimeError(f'{name} failed; see {path}; no automatic retry')
        if pending or active:time.sleep(3)
    return [json.loads((c.output/'summary.json').read_text()) for c in jobs]
