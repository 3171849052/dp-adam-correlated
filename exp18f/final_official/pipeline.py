"""Background smoke, formal training, then verified official reporting."""
import os
from exp18f.final_official.common import BASE, METHODS, configs, completed, read, save, now, verify_seal
from exp18f.final_official.scheduler import run


def pipeline():
    verify_seal()
    save(BASE/'pipeline_status.json', dict(status='running', stage='smoke', pid=os.getpid(), time=now()))
    run('smoke')
    results = [completed(c) for c in configs('smoke')]
    assert len(results)==6 and all(r['steps']==3 and r['evaluation']=='internal_smoke64' for r in results)
    assert len({r['gpu_uuid'] for r in results})==3
    assert len({r['pairing']['initial_model_sha256'] for r in results})==1
    assert len({r['pairing']['order_sha256'] for r in results})==1
    for m in METHODS:
        assert len({r['gpu_uuid'] for r in results if r['config']['method']==m})==3
    peaks = [r['peak_reserved_bytes']/1048576 for r in results]
    save(BASE/'smoke_verification.json', dict(passed=True, six_completed=True, three_gpus=True,
        pairing_verified=True, max_peak_reserved_mib=max(peaks), results=results, time=now()))
    save(BASE/'pipeline_status.json', dict(status='running', stage='formal', pid=os.getpid(), time=now()))
    run('formal')
    from exp18f.final_official.report import report
    report()
    save(BASE/'pipeline_status.json', dict(status='complete', stage='finished', pid=os.getpid(),
        completed_formal=10, time=now()))


if __name__ == '__main__': pipeline()
