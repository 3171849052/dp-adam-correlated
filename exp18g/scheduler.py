"""Independent UUID-isolated workers with measured GPU-memory admission."""
import collections
import json
import os
import subprocess
import time
from exp18g.common import BASE,ROOT,PYTHON,configs,outdir,completed,read,save,now,env


def gpu_snapshot():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    return {int(v[0]):dict(uuid=v[1].strip(),total_mib=int(v[2]),used_mib=int(v[3]),
                free_mib=int(v[4]),utilization=int(v[5])) for line in output.splitlines() for v in [line.split(',')]}

def run(stage):
    queue=collections.deque(configs(stage));active={};finished=[];settings=read(BASE/'scheduling.json')
    statefile=BASE/'logs'/f'{stage}_status.json';eventfile=BASE/'logs'/f'{stage}_events.jsonl'
    peaks=[]
    if stage!='smoke':
        peaks=[completed(c)['peak_reserved_bytes']/1048576 for c in configs('smoke')]
    reservation=max([settings['worker_reservation_mib']]+[p+800 for p in peaks])
    def event(value):
        with eventfile.open('a') as f:f.write(json.dumps(dict(time=now(),**value))+'\n')
    def snapshot(status):
        save(statefile,dict(status=status,pid=os.getpid(),time=now(),pending=len(queue),completed=len(finished),
            total=len(configs(stage)),reservation_mib=reservation,
            active=[dict(gpu=g,uuid=settings['gpu_uuids'][str(g)],worker_pid=p.pid,config=c,
                         directory=str(outdir(c).relative_to(BASE))) for g,p,c,f in active.values()]))
    try:
        while queue or active:
            for pid,(gpu,process,cfg,log) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close()
                if code!=0 or not completed(cfg):
                    save(outdir(cfg)/'failure.json',dict(exit_code=code,pid=pid,time=now()))
                    raise RuntimeError(f'Worker failed {outdir(cfg)}, exit={code}')
                finished.append(cfg);del active[pid]
                event(dict(event='complete',pid=pid,gpu=gpu,config=cfg))
                print(f'COMPLETE {stage} {cfg["method"]} seed={cfg["seed"]} ({len(finished)}/{len(configs(stage))})',flush=True)
            inventory=gpu_snapshot()
            with (BASE/'logs/gpu_telemetry.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=now(),stage=stage,gpus=inventory,worker_pids=list(active)))+'\n')
            # Reserve expected peaks for newly launched processes before CUDA allocations become visible.
            counts=collections.Counter(g for g,p,c,f in active.values())
            candidates=sorted(settings['physical_gpus'],key=lambda g:(counts[g],-inventory[g]['free_mib'],g))
            for gpu in candidates:
                if not queue or len(active)>=settings['max_processes']:break
                if stage=='smoke' and queue[0]['gpu']!=gpu:continue
                if counts[gpu]>=settings['max_per_gpu']:continue
                info=inventory[gpu]
                assert info['uuid']==settings['gpu_uuids'][str(gpu)]
                pending_reserve=sum(reservation for g,p,c,f in active.values()
                    if g==gpu and time.monotonic()-p.launch_time<15)
                if info['free_mib']-pending_reserve<reservation+settings['headroom_mib']:continue
                cfg=queue.popleft();out=outdir(cfg)
                if completed(cfg):finished.append(cfg);continue
                assert not out.exists(),f'Incomplete output requires inspection: {out}'
                out.mkdir(parents=True);save(out/'config.json',cfg)
                log=(out/'worker.log').open('a',buffering=1)
                command=[PYTHON,'-B','-m','exp18g.train',str(out/'config.json')]
                process=subprocess.Popen(command,cwd=ROOT,env=env(gpu),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                process.launch_time=time.monotonic();active[process.pid]=(gpu,process,cfg,log);counts[gpu]+=1
                event(dict(event='start',pid=process.pid,gpu=gpu,uuid=info['uuid'],command=command,config=cfg))
                print(f'START {stage} gpu={gpu} uuid={info["uuid"]} pid={process.pid} {cfg}',flush=True)
            snapshot('running');time.sleep(1)
        snapshot('complete')
    except BaseException:
        # A failed worker stops peers. Preserve their directories and all diagnostics; no silent retries.
        for gpu,process,cfg,log in active.values():
            if process.poll() is None:process.terminate()
        for gpu,process,cfg,log in active.values():
            code=process.wait();log.close()
            if not completed(cfg):save(outdir(cfg)/'failure.json',dict(exit_code=code,reason='stage stopped',time=now()))
        snapshot('failed');raise
    return finished

if __name__=='__main__':
    import sys
    run(sys.argv[1])
