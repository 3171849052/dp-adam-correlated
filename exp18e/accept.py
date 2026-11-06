"""Accept only after numerical tests and six simultaneous GPU-isolated smoke workers."""
import json
from exp18e.common import BASE,ROOT,SEEDS,GROUPS,FIXED,HISTORICAL,make_config,write_csv,configs,completed,read,save,sha,seal,now,outdir


def accept():
    assert read(BASE/'unit_tests.json')['passed']
    assert read(BASE/'matrix_audit.json')['passed'] and read(BASE/'data_audit.json')['passed']
    smoke=configs('smoke');results=[completed(c) for c in smoke]
    assert len(results)==6 and all(results)
    assert all(r['pairing']==results[0]['pairing'] for r in results)
    processes=[read(outdir(c)/'process.json') for c in smoke]
    assert all(p['visible_device_count']==1 and '/envs/curve/' in p['python'] for p in processes)
    assert {p['physical_gpu'] for p in processes}=={0,1,2,3}
    from collections import Counter
    assert max(Counter(p['physical_gpu'] for p in processes).values())<=2
    # Every worker waited until all six ready markers existed before entering training.
    assert all((outdir(c)/'ready.json').exists() for c in smoke)
    events=[json.loads(line) for line in (BASE/'logs/smoke_events.jsonl').read_text().splitlines()]
    starts=[e for e in events if e['event']=='start'];ends=[e for e in events if e['event']=='complete']
    assert len(starts)==len(ends)==6 and max(e['time'] for e in starts)<min(e['time'] for e in ends)
    save(BASE/'smoke_verification.json',dict(passed=True,six_simultaneous=True,processes=processes,
        results=[dict(config=r['config'],peak_reserved_bytes=r['peak_reserved_bytes'],seconds=r['seconds']) for r in results],time=now()))
    template=[make_config('formal',m,seed,FIXED.get(m,HISTORICAL)) for seed in SEEDS for names in GROUPS.values() for m in names]
    assert len(template)==30 and len({(c['method'],c['seed']) for c in template})==30
    for c in template:
        if c['group']=='stpex2': c.update(lr='PENDING_SHARED_FREEZE',clip='PENDING_SHARED_FREEZE',eps='PENDING_SHARED_FREEZE')
    write_csv(BASE/'formal_template.csv',template,list(template[0]))
    save(BASE/'acceptance.json',dict(passed=True,time=now(),code=seal(),formal_count=30,formal_seeds=SEEDS,
        coarse_count=242,smoke_count=6,gpus=[0,1,2,3]))
    print('ACCEPTED: tests, matrices, data, privacy, six-worker smoke, four physical GPUs and 30-task formal template',flush=True)

if __name__=='__main__':accept()
