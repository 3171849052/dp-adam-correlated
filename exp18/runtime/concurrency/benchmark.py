import sys,subprocess,time,json
from pathlib import Path
from exp18 import BASE,ROOT,RESULTS
from exp18.common import env,PYTHON,save,read
OUT=RESULTS/'concurrency';OUT.mkdir(exist_ok=True)

def test(n,group='stpex2',repeat=0):
    batch=OUT/f'{group}_{n}_{repeat}';batch.mkdir();procs=[];start=time.time();peak=0;utils=[];samples=[]
    for i in range(n):
        p=batch/str(i);cfg=dict(stage='benchmark',group=group,method={'adam':'adam-b2lt-sgd','momentum':'momentum-b2lt-momentum','stpex2':'stpex2-b2lt-momentum-bias'}[group],lr=.001,clip=10 if group=='stpex2' else 1,eps=.001,seed=20261102)
        save(p/'config.json',cfg);f=(p/'worker.log').open('w');e=env(1);e['PYTHONPATH']=str(ROOT);e['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        proc=subprocess.Popen([PYTHON,'-B',str(BASE/'runtime/concurrency/worker.py'),str(p/'config.json'),str(n)],cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT);procs.append((proc,f,p))
    while any(p.poll() is None for p,f,d in procs):
        s=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();mem,util=map(int,s.split(','));peak=max(peak,mem);utils.append(util);samples.append(dict(time=time.time(),memory_mib=mem,utilization=util))
        failed=[p.returncode for p,f,d in procs if p.poll() not in (None,0)]
        if failed:
            (batch/'abort').touch()
            for p,f,d in procs:
                if p.poll() is None:p.terminate()
        if time.time()-start>300:
            for p,f,d in procs:
                if p.poll() is None:p.terminate()
        time.sleep(.5)
    for p,f,d in procs:p.wait();f.close()
    ok=all(p.returncode==0 for p,f,d in procs);wall=time.time()-start
    row=dict(n=n,group=group,repeat=repeat,success=ok,wall_seconds=wall,trials_per_minute=60*n/wall if ok else 0,peak_gpu_mib=peak,mean_utilization=sum(utils)/len(utils),returncodes=[p.returncode for p,f,d in procs],all_cuda_ready=len(list(batch.glob('*/ready.json')))==n)
    if ok:
        times=[read(d/'summary.json')['seconds'] for p,f,d in procs];row['max_training_seconds']=max(times);row['training_trials_per_minute']=60*n/max(times)
    save(batch/'telemetry.json',samples);save(batch/'result.json',row)
    with (OUT/'results.jsonl').open('a',buffering=1) as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True);return ok

if __name__=='__main__':
    last=0;failed=None
    for n in [1,2,4,8,12,16,20,24,28,32]:
        if test(n):last=n
        else:failed=n;break
    if failed:
        for n in range(last+1,failed):
            if test(n):last=n
            else:break
    save(OUT/'capacity.json',dict(max_success=last,first_coarse_failure=failed))
