"""Six fixed process slots, two per physical GPU; fail visibly without retries."""
import collections
import os
import subprocess
import time
from exp18d.common import (BASE, ROOT, PYTHON, configs, outdir, completed, read, save, now, env)


def run(stage):
    queue = collections.deque(configs(stage)); active = {}; finished = []; events = []
    slots = read(BASE / 'scheduling.json')['slots']
    statefile = BASE / f'{stage}_status.json'
    def snapshot(status):
        save(statefile,dict(status=status,pid=os.getpid(),time=now(),pending=len(queue),completed=len(finished),
            active=[dict(slot=s,gpu=g,worker_pid=p.pid,method=c['method'],seed=c['seed'],
                         directory=str(outdir(c).relative_to(BASE))) for s,(g,p,c,f) in active.items()]))
    try:
        while queue or active:
            for slot,gpu in enumerate(slots):
                if slot in active or not queue: continue
                cfg = queue.popleft(); out = outdir(cfg)
                if completed(cfg):
                    finished.append(cfg); continue
                assert not out.exists(), f'Incomplete output requires inspection: {out}'
                out.mkdir(parents=True); save(out / 'config.json',cfg)
                log = (BASE / 'logs' / f"{stage}_{out.name}.log").open('a',buffering=1)
                cmd = [PYTHON,'-B','-m','exp18d.train',str(out / 'config.json')]
                process = subprocess.Popen(cmd,cwd=ROOT,env=env(gpu),stdout=log,stderr=subprocess.STDOUT)
                active[slot] = (gpu,process,cfg,log)
                print(f"START slot={slot} gpu={gpu} pid={process.pid} {cfg['method']} seed={cfg['seed']}",flush=True)
                events.append(dict(event='start',slot=slot,gpu=gpu,pid=process.pid,config=cfg,time=now()))
                save(BASE / f'{stage}_events.json',events)
            snapshot('running')
            for slot,(gpu,process,cfg,log) in list(active.items()):
                code = process.poll()
                if code is None: continue
                log.close()
                if code != 0 or not completed(cfg):
                    save(outdir(cfg)/'failure.json',dict(exit_code=code,pid=process.pid,time=now(),
                        log=str(BASE / 'logs' / f"{stage}_{outdir(cfg).name}.log")))
                    raise RuntimeError(f"Failed {cfg['method']} seed={cfg['seed']}, exit={code}")
                finished.append(cfg); del active[slot]
                print(f"COMPLETE {cfg['method']} seed={cfg['seed']} ({len(finished)}/{len(configs(stage))})",flush=True)
                events.append(dict(event='complete',slot=slot,gpu=gpu,pid=process.pid,time=now()))
                save(BASE / f'{stage}_events.json',events)
                if stage=='formal':
                    from exp18d.report import report
                    report()
            if stage=='smoke':
                telemetry = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                    '--format=csv,noheader,nounits'],text=True)
                with (BASE / 'logs/smoke_gpu_telemetry.jsonl').open('a') as f:
                    import json
                    f.write(json.dumps(dict(time=now(),processes=telemetry))+'\n')
            time.sleep(1)
        snapshot('complete')
    except BaseException:
        # Stop all peers and reap them; preserve every output for inspection.
        for gpu,process,cfg,log in active.values():
            if process.poll() is None: process.terminate()
        for gpu,process,cfg,log in active.values():
            code=process.wait(); log.close()
            if not completed(cfg):
                save(outdir(cfg)/'failure.json',dict(exit_code=code,reason='stage stopped after failure or interruption',time=now()))
        snapshot('failed')
        raise
    return finished

if __name__ == '__main__':
    import sys
    assert sys.argv[1] in ('smoke','formal')
    run(sys.argv[1])
