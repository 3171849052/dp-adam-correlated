"""One isolated worker per physical GPU, durable phase plans, strict reuse."""
import os
import subprocess
import time
from exp17b.common import ROOT, RESULTS, PYTHON, save, read, sha, success, output, trial_id


def occupied():
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={line.split(',')[1].strip():int(line.split(',')[0]) for line in query.splitlines()}
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return {mapping[line.split(',')[0].strip()] for line in query.splitlines() if line.split(',')[0].strip() in mapping}


def run_phase(name,trials):
    plan=RESULTS/(name+'_plan.json')
    if plan.exists() and read(plan)!=trials: raise RuntimeError('Phase plan changed')
    save(plan,trials)
    if name!='smoke':
        auth=read(RESULTS/'authorized.json') if (RESULTS/'authorized.json').exists() else []
        for c in trials:
            if c not in auth: auth.append(c)
        save(RESULTS/'authorized.json',auth)
    identity=sha(RESULTS/'protocol_freeze.json');pending=[];done=[];active={};failure=None
    for c in trials:
        result=success(c,identity)
        if result: done.append(result)
        elif (output(c)/'summary.json').exists(): raise RuntimeError('Retained failure: '+str(output(c)))
        else: pending.append(c)
    try:
        while pending or active:
            for gpu,(p,c,log) in list(active.items()):
                code=p.poll()
                if code is None: continue
                log.close();del active[gpu]
                if code!=0: failure='Worker failed: '+trial_id(c)
                else:
                    result=success(c,identity)
                    if result is None: failure='Worker lacks verified completion: '+trial_id(c)
                    else: done.append(result)
            if failure and not active: raise RuntimeError(failure)
            busy=occupied()
            if not failure:
                for gpu in (2,3):
                    if gpu in busy or gpu in active or not pending: continue
                    c=pending.pop(0);folder=output(c);folder.mkdir(parents=True,exist_ok=True)
                    log=(folder/'worker.log').open('a')
                    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                    command=[PYTHON,'-B','-m','exp17b.train','--config',__import__('json').dumps(c)]
                    p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    active[gpu]=(p,c,log)
                    with (RESULTS/'scheduler.jsonl').open('a') as f:
                        f.write(__import__('json').dumps(dict(event='start',time=time.time(),pid=p.pid,gpu=gpu,config=c,command=command))+'\n')
            save(RESULTS/'pipeline_status.json',dict(phase=name,pid=os.getpid(),time=time.time(),
                completed=len(done),planned=len(trials),pending=len(pending),failure=failure,
                active=[dict(gpu=g,pid=p.pid,trial=trial_id(c),log=str(output(c)/'worker.log')) for g,(p,c,l) in active.items()]))
            if pending or active: time.sleep(5)
    finally:
        # Unexpected scheduler failure must not leave untracked training processes.
        for p,c,log in active.values():
            p.terminate()
        for p,c,log in active.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()
    save(RESULTS/(name+'_results.json'),done)
    return done
