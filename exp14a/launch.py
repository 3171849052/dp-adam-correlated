"""One worker per physical GPU; FIFO, no retries. Invoke through detached Popen."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from exp14a import ROOT, RESULTS
from exp14a.config import save_json
from exp14a.support import jobs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    stage='smoke' if args.smoke else 'final'
    queue=jobs(stage);active={};states={c.id:dict(task=c.task,method=c.method,status='queued') for c in queue}
    prefix='smoke' if args.smoke else 'scheduler'
    save_json(RESULTS/f'{prefix}_process.json',dict(pid=os.getpid(),command=shlex.join([sys.executable,'-B','-m','exp14a.launch']+(['--smoke'] if args.smoke else [])),cwd=str(ROOT)))
    def event(kind,**details):
        with (RESULTS/f'{prefix}.jsonl').open('a') as f: f.write(json.dumps(dict(time=time.time(),event=kind,**details))+'\n')
        save_json(RESULTS/f'{prefix}_status.json',states)
    event('started',pid=os.getpid())
    while queue or active:
        for gpu,(p,cfg,log) in list(active.items()):
            code=p.poll()
            if code is None: continue
            log.close();states[cfg.id].update(status='completed' if code==0 else 'failed',exit_code=code,ended=time.time())
            event('finished',gpu=gpu,trial_id=cfg.id,exit_code=code);del active[gpu]
        for gpu in range(4):
            if gpu in active or not queue: continue
            cfg=queue.pop(0);cfg.output.mkdir(parents=True,exist_ok=True)
            assert not (cfg.output/'summary.json').exists(), 'Do not overwrite previous runs'
            command=[sys.executable,'-B','-m','exp14a.train','--trial-json',json.dumps(cfg.values(),sort_keys=True)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
            log=(cfg.output/'train.log').open('w')
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(p,cfg,log)
            states[cfg.id].update(status='running',gpu=gpu,pid=p.pid,command='CUDA_VISIBLE_DEVICES='+str(gpu)+' '+shlex.join(command),log=str(cfg.output/'train.log'),started=time.time())
            save_json(cfg.output/'process.json',states[cfg.id]);event('launched',**states[cfg.id],trial_id=cfg.id)
        if active: time.sleep(5)
    if not args.smoke:
        from exp14a.report import main as report
        report()
    event('all_finished')
    if any(s['status']!='completed' for s in states.values()): sys.exit(1)

if __name__=='__main__': main()
