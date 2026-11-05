import subprocess,time,os
from exp18 import RESULTS,ROOT
from exp18.common import save,read,sha,env,PYTHON
from exp18.search import key

def completed(cfg):
    p=RESULTS/'trials'/key(cfg)
    if not (p/'complete.json').exists():return None
    assert read(p/'config.json')==cfg
    for name,h in read(p/'complete.json').items():assert sha(p/name)==h,f'Artifact changed: {p/name}'
    return read(p/'summary.json')

def telemetry():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(time=time.time(),gpu=gpu,processes=procs)

def execute(configs,stage):
    pending=[];rows=[];active=[];capacity={g:2 for g in (1,2,3)};peaks={};failure=None;last=0;overlap=False
    for cfg in configs:
        old=completed(cfg)
        if old:rows.append(old);continue
        p=RESULTS/'trials'/key(cfg)
        if p.exists():raise RuntimeError(f'Incomplete trial preserved at {p}; inspect and archive before retry')
        pending.append(cfg)
    with (RESULTS/'gpu_telemetry.jsonl').open('a',buffering=1) as log:
        while pending or active:
            now=time.time()
            if now-last>=5 or last==0:
                tel=telemetry();log.write(__import__('json').dumps(tel)+'\n');last=now
                metrics={int(v[0]):[int(x) for x in v[1:]] for line in tel['gpu'].splitlines() for v in [line.split(',')]}
                for g,(total,used,util) in metrics.items():
                    if g not in capacity:continue
                    if g in peaks and util<75 and total-used>max(2000,peaks[g]*1.5):capacity[g]=min(4,capacity[g]+1)
                    elif used>total*.9:capacity[g]=max(1,capacity[g]-1)
            for item in list(active):
                proc,cfg,g,handle=item
                if proc.poll() is None:continue
                handle.close();active.remove(item)
                if proc.returncode:
                    failure=dict(trial=key(cfg),returncode=proc.returncode);pending.clear()
                else:
                    row=completed(cfg);assert row;rows.append(row);peaks[g]=max(peaks.get(g,0),row['peak_memory']/1024**2)
            if not failure:
                for g in capacity:
                    while pending and sum(a[2]==g for a in active)<capacity[g]:
                        # Leave headroom for external processes; never change trial protocol.
                        total,used,util=metrics[g]
                        if used>total*.85:break
                        cfg=pending.pop(0);p=RESULTS/'trials'/key(cfg);save(p/'config.json',cfg)
                        handle=(p/'worker.log').open('w');e=env(g);e['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                        proc=subprocess.Popen([PYTHON,'-B','-m','exp18.train',str(p/'config.json')],cwd=ROOT,env=e,stdout=handle,stderr=subprocess.STDOUT)
                        active.append((proc,cfg,g,handle))
            if all(sum(a[2]==g for a in active)>=2 for g in (1,2,3)):overlap=True
            state=dict(stage=stage,pending=len(pending),completed=len(rows),capacity=capacity,active=[dict(pid=p.pid,trial=key(c),physical_gpu=g) for p,c,g,h in active],failure=failure,six_process_overlap=overlap,time=time.time())
            save(RESULTS/f'{stage}_status.json',state);save(RESULTS/'status.json',state)
            if active or pending:time.sleep(2)
    if failure:raise RuntimeError(f'Trial failed, pipeline stopped: {failure}')
    return rows
