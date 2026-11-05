import subprocess,time,os,json
from exp18 import RESULTS,ROOT,BASE
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

def admission_limit(total,used,own_resident_mib,policy):
    # Count unallocated/new workers as reservations too. A stale NVML sample
    # must never authorize a whole wave against the same free-memory value.
    external=max(0,used-own_resident_mib)
    available=max(0,total-policy['reserve_mib']-external)
    return min(policy['target_per_gpu'],int(available//policy['worker_budget_mib']))

def execute(configs,stage):
    policy=read(BASE/'scheduling.json');pending=[];rows=[];active=[];failure=None;last=0;overlap=False
    capacity={g:policy['target_per_gpu'] for g in (1,2,3)}
    for cfg in configs:
        old=completed(cfg)
        if old:rows.append(old);continue
        p=RESULTS/'trials'/key(cfg)
        if p.exists():raise RuntimeError(f'Incomplete trial preserved at {p}; inspect and archive before retry')
        pending.append(cfg)
    with (RESULTS/'gpu_telemetry.jsonl').open('a',buffering=1) as log:
        while pending or active:
            now=time.time()
            if now-last>=2 or last==0:
                tel=telemetry();log.write(json.dumps(tel)+'\n');last=now
                metrics={int(v[0]):[int(x) for x in v[1:]] for line in tel['gpu'].splitlines() for v in [line.split(',')]}
            for item in list(active):
                proc,cfg,g,handle=item
                if proc.poll() is None:continue
                handle.close();active.remove(item)
                if proc.returncode:failure=dict(trial=key(cfg),returncode=proc.returncode);pending.clear()
                else:
                    row=completed(cfg);assert row;rows.append(row)
            resident={}
            for line in tel['processes'].splitlines():
                fields=line.split(',')
                if len(fields)==3 and fields[1].strip().isdigit() and fields[2].strip().isdigit():resident[int(fields[1])]=int(fields[2])
            memory_limits={g:admission_limit(metrics[g][0],metrics[g][1],sum(resident.get(p.pid,0) for p,c,gpu,h in active if gpu==g),policy) for g in (1,2,3)}
            for g,limit in memory_limits.items():
                if limit<capacity[g]:capacity[g]=limit
                elif limit>capacity[g] and metrics[g][2]<90:capacity[g]=min(limit,capacity[g]+1)
            if not failure:
                # Round robin prevents concentrating an entire startup wave on one GPU.
                while pending:
                    choices=[g for g in capacity if sum(a[2]==g for a in active)<capacity[g]]
                    if not choices:break
                    g=min(choices,key=lambda g:sum(a[2]==g for a in active))
                    cfg=pending.pop(0);p=RESULTS/'trials'/key(cfg);save(p/'config.json',cfg)
                    handle=(p/'worker.log').open('w');e=env(g);e['CUDA_DEVICE_ORDER']='PCI_BUS_ID';e['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                    proc=subprocess.Popen([PYTHON,'-B','-m','exp18.train',str(p/'config.json')],cwd=ROOT,env=e,stdout=handle,stderr=subprocess.STDOUT)
                    active.append((proc,cfg,g,handle))
            if all(sum(a[2]==g for a in active)>=2 for g in (1,2,3)):overlap=True
            state=dict(stage=stage,pending=len(pending),completed=len(rows),capacity=capacity,scheduling=policy,active=[dict(pid=p.pid,trial=key(c),physical_gpu=g) for p,c,g,h in active],failure=failure,six_process_overlap=overlap,time=time.time())
            save(RESULTS/f'{stage}_status.json',state);save(RESULTS/'status.json',state)
            if active or pending:time.sleep(1)
    if failure:raise RuntimeError(f'Trial failed, pipeline stopped: {failure}')
    return rows
