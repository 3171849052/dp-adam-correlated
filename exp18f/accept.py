"""Seal only after tests, privacy/matrix audits and six simultaneous three-GPU workers."""
import json
from collections import Counter
from exp18f.common import BASE,ROOT,read,save,now,configs,outdir,terminal,seal

def accept():
    assert read(BASE/'unit_tests.json')['passed']
    for name in ('data_audit','matrix_audit','privacy_audit'):assert read(BASE/f'{name}.json')['passed']
    results=[terminal(c,verify=True) for c in configs('smoke')]
    assert len(results)==6 and all(r and r['status']=='complete' and r['steps']==3 for r in results)
    assert all(r['pairing']==results[0]['pairing'] for r in results)
    processes=[read(outdir(c)/'process.json') for c in configs('smoke')]
    settings=read(BASE/'scheduling.json');counts=Counter(p['physical_gpu'] for p in processes)
    assert counts==Counter({g:2 for g in settings['physical_gpus']})
    assert all(p['visible_device_count']==1 and '/envs/curve/' in p['python'] and
        p['CUDA_VISIBLE_DEVICES']==settings['gpu_uuids'][str(p['physical_gpu'])] for p in processes)
    events=[json.loads(l) for l in (BASE/'logs/smoke_events.jsonl').read_text().splitlines()]
    starts=[e for e in events if e['event']=='start'];ends=[e for e in events if e['event']=='complete']
    assert len(starts)==len(ends)==6 and max(e['time'] for e in starts)<min(e['time'] for e in ends)
    assert all((outdir(c)/'ready.json').exists() for c in configs('smoke'))
    assert len(configs('axis'))==108 and all(c['steps']==310 for c in configs('axis'))
    assert {c['method'] for c in configs('axis')}=={'stpex2-iid','stpex2-b2lt-momentum-bias'}
    snapshot={str(p.relative_to(ROOT/'exp18e')):[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'exp18e').rglob('*') if p.is_file()}
    assert snapshot==read(BASE/'exp18e_snapshot.json')
    save(BASE/'smoke_verification.json',dict(passed=True,six_simultaneous=True,processes=processes,
        results=[dict(config=r['config'],peak_reserved_bytes=r['peak_reserved_bytes'],seconds=r['seconds']) for r in results],time=now()))
    save(BASE/'acceptance.json',dict(passed=True,time=now(),code=seal(),three_gpu_smoke=True,
        full_trainings_max=156,axis_trainings=108,gpu_uuids=settings['gpu_uuids'],exp18e_unchanged=True))
    print('ACCEPTED: tests, privacy, assets, six concurrent workers, exactly three UUIDs, Exp18e unchanged',flush=True)

if __name__=='__main__':accept()
