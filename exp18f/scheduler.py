"""Exactly three UUIDs, <=2 independent workers per GPU, dynamic memory admission."""
import collections
import json
import os
import subprocess
import time
from exp18f.common import BASE,ROOT,PYTHON,configs,outdir,terminal,read,save,now,env

def gpu_snapshot():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    return {int(v[0]):dict(uuid=v[1].strip(),name=v[2].strip(),total_mib=int(v[3]),used_mib=int(v[4]),
        free_mib=int(v[5]),utilization=int(v[6])) for line in output.splitlines() for v in [line.split(',')]}

def process_memory():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return {int(v[0]):int(v[1]) for line in output.splitlines() if line.strip() for v in [line.split(',')]}

def run(stage):
    if stage!='smoke':
        save(BASE/'pipeline_status.json',dict(status='running',stage=stage,pid=os.getpid(),time=now()))
    jobs=configs(stage);queue=collections.deque(c for c in jobs if terminal(c) is None)
    active={};finished=[c for c in jobs if terminal(c) is not None];settings=read(BASE/'scheduling.json')
    assert len(settings['physical_gpus'])==len(set(settings['gpu_uuids'].values()))==3
    assert settings['max_per_gpu']==2 and settings['max_processes']==6
    statefile=BASE/'logs'/f'{stage}_status.json';eventfile=BASE/'logs'/f'{stage}_events.jsonl'
    peaks=[] if stage=='smoke' else [terminal(c)['peak_reserved_bytes']/1048576 for c in configs('smoke')]
    reservation=max([settings['worker_reservation_mib']]+[p+settings['peak_margin_mib'] for p in peaks])
    last_admission=time.monotonic()
    def event(value):
        with eventfile.open('a') as f:f.write(json.dumps(dict(time=now(),**value))+'\n')
    def snapshot(status):
        counts=collections.Counter(g for g,p,c,f in active.values())
        assert len(active)<=6 and max(counts.values(),default=0)<=2
        save(statefile,dict(status=status,stage=stage,pid=os.getpid(),time=now(),pending=len(queue),
            completed=sum(terminal(c)['status']=='complete' for c in finished),
            numerical_failures=sum(terminal(c)['status']=='numerical_failure' for c in finished),
            terminal=len(finished),total=len(jobs),reservation_mib=reservation,
            active=[dict(gpu=g,uuid=settings['gpu_uuids'][str(g)],worker_pid=p.pid,config=c,
                directory=str(outdir(c).relative_to(BASE))) for g,p,c,f in active.values()]))
    try:
        while queue or active:
            changed=False
            for pid,(gpu,process,cfg,log) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();result=terminal(cfg,verify=True)
                if code!=0 or result is None:
                    save(outdir(cfg)/'infrastructure_failure.json',dict(exit_code=code,pid=pid,time=now()))
                    raise RuntimeError(f'Infrastructure failure: {outdir(cfg)}, exit={code}')
                finished.append(cfg);del active[pid];changed=True
                event(dict(event=result['status'],pid=pid,gpu=gpu,config=cfg))
                print(f"{result['status'].upper()} {stage} {len(finished)}/{len(jobs)} {cfg}",flush=True)
            if changed and stage!='smoke':
                from exp18f.report import report
                report(plots=False)
            inventory=gpu_snapshot();used=process_memory()
            with (BASE/'logs/gpu_telemetry.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=now(),stage=stage,gpus=inventory,worker_pids=list(active)))+'\n')
            counts=collections.Counter(g for g,p,c,f in active.values())
            candidates=sorted(settings['physical_gpus'],key=lambda g:(counts[g],-inventory[g]['free_mib'],g))
            for gpu in candidates:
                if not queue or len(active)>=6:break
                if counts[gpu]>=2:continue
                info=inventory[gpu];assert info['uuid']==settings['gpu_uuids'][str(gpu)]
                # Account for each worker's unallocated peak, including CUDA startup.
                deficit=sum(max(0,reservation-used.get(p.pid,0)) for g,p,c,f in active.values() if g==gpu)
                if info['free_mib']-deficit<reservation+settings['headroom_mib']:continue
                cfg=queue.popleft();out=outdir(cfg)
                assert not out.exists(),f'Incomplete output requires inspection: {out}'
                out.mkdir(parents=True);save(out/'config.json',cfg)
                log=(out/'worker.log').open('a',buffering=1)
                command=[PYTHON,'-B','-m','exp18f.train',str(out/'config.json')]
                process=subprocess.Popen(command,cwd=ROOT,env=env(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                active[process.pid]=(gpu,process,cfg,log);counts[gpu]+=1;last_admission=time.monotonic()
                event(dict(event='start',pid=process.pid,gpu=gpu,uuid=info['uuid'],command=command,config=cfg))
                print(f'START {stage} gpu={gpu} uuid={info["uuid"]} pid={process.pid} {cfg}',flush=True)
            if queue and not active and time.monotonic()-last_admission>300:
                raise RuntimeError('No three-GPU memory admission possible for 300 seconds')
            snapshot('running');time.sleep(2)
        snapshot('complete')
    except BaseException:
        # Infrastructure stops the stage and preserves every worker directory. No retries.
        for gpu,process,cfg,log in active.values():
            if process.poll() is None:process.terminate()
        for gpu,process,cfg,log in active.values():
            code=process.wait();log.close()
            if terminal(cfg) is None:save(outdir(cfg)/'infrastructure_failure.json',dict(exit_code=code,reason='stage stopped',time=now()))
        active.clear();snapshot('infrastructure_failed');raise
    return finished

if __name__=='__main__':
    import sys
    run(sys.argv[1])
