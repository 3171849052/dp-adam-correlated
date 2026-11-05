"""Freeze only after tests and all six full-size smoke workers succeed."""
import json
import sys
from exp18d.common import BASE, configs, completed, read, save, now, seal


def accept():
    assert read(BASE / 'unit_tests.json')['passed']
    assert read(BASE / 'smoke_status.json')['status']=='complete'
    results=[completed(c) for c in configs('smoke')]
    assert len(results)==6 and all(results)
    records=[json.loads(line) for line in (BASE / 'logs/smoke_gpu_telemetry.jsonl').read_text().splitlines()]
    uuids=read(BASE / 'scheduling.json')['gpu_uuids']
    found=[]
    for row in records:
        gpu_pids={g:[] for g in uuids}
        for line in row['processes'].splitlines():
            pid,uuid,memory=[x.strip() for x in line.split(',')]
            for g,u in uuids.items():
                if uuid==u: gpu_pids[g].append(int(pid))
        assert all(len(pids)<=2 for pids in gpu_pids.values())
        if all(len(pids)==2 for pids in gpu_pids.values()): found.append(row['time'])
    assert found,'No observed six-process CUDA overlap'
    assert all(r['steps']==3 and not r['formal'] for r in results)
    verification=dict(passed=True,time=now(),six_process_overlap=found,
        workers=[read((BASE / c['stage'] / f"{c['method']}_seed{c['seed']}")/'process.json') for c in configs('smoke')],
        peak_reserved_bytes=[r['peak_reserved_bytes'] for r in results])
    save(BASE / 'smoke_verification.json',verification)
    save(BASE / 'acceptance.json',dict(passed=True,time=now(),code=seal(),smoke=verification,python=sys.executable))
    print('ACCEPTED: tests, full-size six-process smoke, 2 workers/GPU and immutable seal')

if __name__=='__main__': accept()
