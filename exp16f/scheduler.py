"""Task-aware 4-GPU scheduler, hash-verified reuse, no implicit retries."""
import json,math,os,shlex,subprocess,sys,time
from exp16f import ROOT,RESULTS
from exp16f.config import save_json,file_hash,json_hash
from exp16f.support import read,code_hashes

def event(**row):
    with (RESULTS/'scheduler.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**row))+'\n')
GPUS=(1,2,3)
def capacity(task,existing):return not existing or task=='nlp' and existing==['nlp']
def runnable(task,active):return task!='nlp' or sum(c.task=='nlp' for c,p,g,log in active.values())<6
def gpu_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    mapping={r.split(',')[1].strip():int(r.split(',')[0]) for r in uuids.splitlines()}
    out={g:set() for g in range(4)}
    for row in text.splitlines():
        uuid,pid=row.split(',');out[mapping[uuid.strip()]].add(int(pid.strip()))
    return out
def verify_config(cfg,row):
    if row['trial_id']!=cfg.id or any(row[k]!=v for k,v in cfg.values().items()):raise ValueError('Result configuration mismatch')
    resolved=read(cfg.output/'config.json');h=resolved.pop('config_sha256')
    if json_hash(resolved)!=h or resolved['code_sha256']!=code_hashes():raise ValueError('Code/config hash mismatch')
    if any(resolved[k]!=v for k,v in cfg.protocol().items()):raise ValueError('Protocol mismatch')
    if row['status']=='completed' and (row['config_sha256']!=h or row['code_sha256']!=code_hashes()):raise ValueError('Summary hash mismatch')
    from exp16f.support import verify_assets
    if resolved['input_assets_sha256']!=verify_assets():raise ValueError('Input asset mismatch')
    if cfg.stage=='final':
        from exp16f.support import official_assets
        if resolved['official_assets_sha256']!=official_assets() or resolved['final_freeze_sha256']!=file_hash(RESULTS/'final_freeze.json'):raise ValueError('Official/final freeze mismatch')
    if cfg.private:
        from exp16f.matrices import load
        a=load(cfg.workload,cfg.mechanism,cfg.total_steps)
        if row['status']=='completed' and row['mechanism']['matrix_sha256']!=file_hash(a['path']):raise ValueError('Result matrix mismatch')
def seal(cfg):
    """After worker exit, bind summary, logs and every output artifact together."""
    path=cfg.output/'completion.json'
    if path.exists():raise ValueError('Completion manifest already exists')
    files={p.name:file_hash(p) for p in sorted(cfg.output.iterdir()) if p.is_file()}
    save_json(path,dict(trial_id=cfg.id,core_sha256=code_hashes(),file_sha256=files))

def verify(cfg):
    row=read(cfg.output/'summary.json');verify_config(cfg,row)
    manifest=read(cfg.output/'completion.json')
    if manifest['trial_id']!=cfg.id or manifest['core_sha256']!=code_hashes():raise ValueError('Completion identity/core mismatch')
    current={p.name for p in cfg.output.iterdir() if p.is_file() and p.name!='completion.json'}
    if current!=set(manifest['file_sha256']):raise ValueError('Trial file set changed')
    for name,h in manifest['file_sha256'].items():
        if file_hash(cfg.output/name)!=h:raise ValueError(f'Artifact mismatch: {cfg.output/name}')
    if row['status']=='non_finite' and cfg.stage in ('search','review'):
        if not row.get('error','').startswith('FloatingPointError:'):raise ValueError('Invalid numerical exclusion')
        return row
    if row['status']!='completed':raise RuntimeError(f'Failed trial requires inspection: {cfg.output}')
    if not (0<=row['accuracy']<=1 and math.isfinite(row['validation_loss']) and row['validation_loss']>=0):raise ValueError('Invalid metrics')
    if row['optimizer_steps']!=(2 if cfg.stage=='smoke' else cfg.total_steps):raise ValueError('Incomplete successful run')
    if cfg.private and cfg.stage!='smoke' and abs(row['actual_epsilon']-8)>1e-7:raise ValueError('Privacy mismatch')
    for name,h in row['file_sha256'].items():
        if file_hash(cfg.output/name)!=h:raise ValueError(f'Artifact mismatch: {cfg.output/name}')
    return row
def run_batch(name,jobs):
    jobs=list({c.id:c for c in jobs}.values());pending=[];status={};active={};results={};fatal=[];pointer=0
    status_path=RESULTS/f'{name}_status.json'
    for c in jobs:
        if (c.output/'summary.json').exists():
            r=verify(c);results[c.id]=r;status[c.id]=dict(**c.values(),status=r['status'],reused=True)
        elif c.output.exists():raise RuntimeError(f'Partial run requires inspection: {c.output}')
        else:pending.append(c);status[c.id]=dict(**c.values(),status='pending')
    save_json(status_path,status);last_sample=0
    while pending or active:
        for key,(c,p,gpu,log) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            log.close();del active[key]
            try:
                raw=read(c.output/'summary.json')
                reusable=(rc==0 and raw['status']=='completed') or (raw['status']=='non_finite' and c.stage in ('search','review'))
                if not reusable:raise RuntimeError(f"Worker {raw['status']}: {raw.get('error')}")
                seal(c)
                row=verify(c)
                if rc!=0 and row['status']!='non_finite':raise RuntimeError(f'Worker exit {rc}')
                results[key]=row;status[key].update(status=row['status'],exit_code=rc,ended=time.time())
            except Exception as error:
                fatal.append(str(error));failure=read(c.output/'summary.json') if (c.output/'summary.json').exists() else {}
                status[key].update(status='oom' if failure.get('status')=='oom' else 'failed',worker_status=failure.get('status'),exit_code=rc,error=str(error),ended=time.time())
            event(batch=name,event='finished',**status[key]);save_json(status_path,status)
        if not fatal and pending:
            occupied=gpu_processes()
            free={int(r.split(',')[0]):int(r.split(',')[1]) for r in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
            # Greedy oldest runnable task, rotating physical GPU priority.
            for gpu in [GPUS[(pointer+offset)%3] for offset in range(3)]:
                own={p.pid for c,p,g,log in active.values() if g==gpu}
                if occupied[gpu]-own:continue
                existing=[c.task for c,p,g,log in active.values() if g==gpu]
                while True:
                    index=next((i for i,c in enumerate(pending) if capacity(c.task,existing) and runnable(c.task,active) and free[gpu]>=(6000 if c.task=='cv' else 4800)),None)
                    if index is None:break
                    c=pending.pop(index);c.output.mkdir(parents=True,exist_ok=True)
                    log=(c.output/'train.log').open('a')
                    argv=[sys.executable,'-B','-m','exp16f.train','--trial-json',json.dumps(c.values(),sort_keys=True)]
                    p=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    active[c.id]=(c,p,gpu,log);existing.append(c.task);free[gpu]-=6000 if c.task=='cv' else 4800
                    status[c.id]=dict(**c.values(),status='running',gpu=gpu,pid=p.pid,started=time.time(),log=str(c.output/'train.log'),command=f'CUDA_VISIBLE_DEVICES={gpu} '+shlex.join(argv))
                    save_json(c.output/'process.json',status[c.id]);event(batch=name,event='launched',**status[c.id]);save_json(status_path,status)
                    print(f'{name}: GPU {gpu} {c.task} {c.method} pid={p.pid}',flush=True)
                pointer=(GPUS.index(gpu)+1)%3
        if active and time.time()-last_sample>15:
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
            with (RESULTS/'gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),batch=name,raw=sample,active=list(active)))+'\n')
            last_sample=time.time()
        if fatal and not active:raise RuntimeError(f'{name} stopped without retry: '+str(fatal))
        if pending or active:time.sleep(2)
    return [results[c.id] for c in jobs]
