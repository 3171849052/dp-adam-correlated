"""Deterministic global FIFO, round-robin GPU selection, task-exclusive slots."""
from collections import deque
import json
import os
import subprocess
import sys
import time
from exp9 import ROOT, RESULTS
from exp9.config import save_json

def run_queue(jobs,gpus=(0,1,2,3),cv_per_gpu=1,nlp_per_gpu=2,command=None,verify=None,poll_seconds=1.):
    from exp9.audit import audit_trial
    verify=verify or audit_trial
    assert len(gpus)==len(set(gpus)) and set(gpus)<=set(range(4)) and gpus
    assert cv_per_gpu==1 and nlp_per_gpu in (1,2)
    jobs=list(jobs);assert len({j.id for j in jobs})==len(jobs)
    RESULTS.mkdir(parents=True,exist_ok=True)
    queue_id=f'{time.time_ns()}'
    pending=deque(enumerate(jobs));active={};results={};pointer=0;fatal=[]
    def event(kind,**fields):
        with (RESULTS/'scheduler.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=kind,time=time.time(),queue=queue_id,**fields))+'\n')
    def reuse(index,job):
        row=verify(job,allow_failure=job.stage=='search')
        results[job.id]=row;event('reuse',trial_id=job.id,task=job.task,queue_index=index,status=row['status'])
    for index,job in list(pending):
        if (job.output/'summary.json').exists(): reuse(index,job)
        elif job.output.exists(): raise RuntimeError(f'Inspect incomplete trial directory: {job.output}')
    pending=deque((i,j) for i,j in pending if j.id not in results)
    last_telemetry=0.
    try:
        while pending or active:
            for trial_id,item in list(active.items()):
                process,job,gpu,log,started=item
                code=process.poll()
                if code is None: continue
                log.close();del active[trial_id]
                row=json.loads((job.output/'summary.json').read_text()) if (job.output/'summary.json').exists() else None
                event('finish',trial_id=trial_id,task=job.task,gpu=gpu,exit_code=code,
                      status=row['status'] if row else 'failed_before_summary',seconds=time.monotonic()-started,
                      peak_allocated_bytes=row.get('peak_allocated_bytes') if row else None)
                if code==0 or row and row['status']=='non_finite' and job.stage=='search':
                    results[trial_id]=verify(job,allow_failure=job.stage=='search')
                else:
                    fatal.append(f'{job.output}: exit {code}, {row.get("error") if row else "see train.log"}')
            if fatal:
                # Already-running trials finish and keep their evidence; no new launches.
                pending.clear()
            while pending:
                index,job=pending[0];capacity=cv_per_gpu if job.task=='cv' else nlp_per_gpu
                selected=None
                for offset in range(len(gpus)):
                    position=(pointer+offset)%len(gpus);gpu=gpus[position]
                    existing=[item for item in active.values() if item[2]==gpu]
                    if len(existing)<capacity and all(item[1].task==job.task for item in existing):
                        selected=(position,gpu);break
                if selected is None: break
                position,gpu=selected;pointer=(position+1)%len(gpus);pending.popleft()
                job.output.mkdir(parents=True)
                log=(job.output/'train.log').open('w',buffering=1)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
                args=command(job) if command else [sys.executable,'-B','-m','exp9.train','--trial-json',json.dumps(job.values())]
                process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[job.id]=(process,job,gpu,log,time.monotonic())
                event('start',trial_id=job.id,task=job.task,gpu=gpu,queue_index=index,
                      pid=process.pid,capacity=capacity,physical_batch_size=job.physical_batch)
                print(f'[{job.stage}] GPU {gpu} {job.task} {job.method} {job.id}',flush=True)
            if time.monotonic()-last_telemetry>=5 and active:
                sample=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],
                                      capture_output=True,text=True,check=True)
                with (RESULTS/'gpu_telemetry.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),queue=queue_id,raw=sample.stdout.strip(),active=list(active)))+'\n')
                last_telemetry=time.monotonic()
            if active: time.sleep(poll_seconds)
        if fatal: raise RuntimeError('Worker failure; parameters unchanged:\n'+'\n'.join(fatal))
        return [results[j.id] for j in jobs]
    finally:
        for trial_id,(process,job,gpu,log,started) in active.items():
            process.terminate();process.wait();log.close()
            event('interrupted',trial_id=trial_id,task=job.task,gpu=gpu)
