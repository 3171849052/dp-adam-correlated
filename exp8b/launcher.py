"""A sequential single-GPU FIFO. Each trial gets an isolated process and a log."""
import json
import os
import subprocess
import sys
import time
from exp8b import ROOT,RESULTS
from exp8b.config import *


def run_queue(jobs,gpu=0,category='search'):
    from exp8b.audit import audit_trial
    records=[]
    for job in jobs:
        directory=trial_dir(job,category);directory.mkdir(parents=True,exist_ok=True)
        summary=directory/'summary.json'
        if summary.exists():
            r=json.loads(summary.read_text())
            if r['status']=='completed': r=audit_trial(directory)
            elif r['status'] not in ('non_finite','oom'): raise RuntimeError(f'Inspect failed trial: {directory}')
            records.append(r);continue
        def event(kind,**extra):
            with (RESULTS/'scheduler.jsonl').open('a') as f:
                f.write(json.dumps(dict(event=kind,trial_id=trial_id(job),category=category,gpu=gpu,time=time.time(),**extra))+'\n')
        cmd=[sys.executable,'-B','-m','exp8b.train','--method',job['method'],'--lr',str(job['lr']),
             '--C',str(job['C']),'--seed',str(job['seed']),'--gpu','0','--category',category]
        if job['eps_scale'] is not None: cmd+=['--eps-scale',str(job['eps_scale'])]
        event('start')
        print(f"{category}: {job}",flush=True)
        with (directory/'train.log').open('w') as log:
            code=subprocess.call(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
        event('finish',returncode=code)
        if not summary.exists():
            save_json(summary,dict(job,status='failed',category=category,error=f'worker exited {code}',result_dir=str(directory.relative_to(ROOT))))
        r=json.loads(summary.read_text())
        if r['status']=='completed':
            assert code==0;r=audit_trial(directory)
        elif r['status']=='failed' or category=='smoke' or r['status']=='oom':
            # OOM is a platform failure, never resolved by changing physical batch.
            raise RuntimeError(f"{r['status']}: inspect {directory/'train.log'}")
        records.append(r)
    return records


class MultiGPUQueue:
    """Six simple worker threads launch isolated trials on two slots per GPU."""
    def __init__(self,gpus=(0,1,2),per_gpu=2):
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue
        import threading
        assert len(set(gpus))==len(gpus) and all(g in (0,1,2) for g in gpus)
        assert per_gpu in (1,2)
        self.gpus=list(gpus);self.per_gpu=per_gpu
        self.failed=threading.Event();self.local=threading.local()
        slots=Queue()
        for g in gpus:
            for _ in range(per_gpu): slots.put(g)
        def initialize(): self.local.gpu=slots.get()
        self.executor=ThreadPoolExecutor(max_workers=len(gpus)*per_gpu,initializer=initialize)

    def _one(self,job,category):
        if self.failed.is_set(): raise RuntimeError('GPU queue stopped after a worker failure')
        try:
            return run_queue([job],gpu=self.local.gpu,category=category)[0]
        except Exception:
            self.failed.set()
            raise

    def run(self,jobs,gpu=0,category='search'):
        futures=[self.executor.submit(self._one,j,category) for j in jobs]
        return [f.result() for f in futures]

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
