"""Check four real GPU smoke runs before any formal launch."""
from exp18g.common import *

def accept():
    verify_seal()
    for n in ('unit_tests','matrix_audit','privacy_audit','data_audit'):
        assert read(BASE/f'acceptance/{n}.json')['passed']
    results=[completed(c) for c in configs('smoke')]
    assert len(results)==4 and all(r['steps']==3 and r['evaluation']=='internal_smoke64' for r in results)
    assert {r['physical_gpu'] for r in results}=={0,1,2,3}
    assert len({r['gpu_uuid'] for r in results})==4
    reference=read(ROOT/'exp18f/final_official/formal/stpex2-iid_seed20261111/pairing.json')
    assert all(r['pairing']==reference for r in results)
    save(BASE/'acceptance/smoke_results.json',dict(passed=True,four_gpus=True,pairing_verified=True,
        max_peak_reserved_mib=max(r['peak_reserved_bytes']/1048576 for r in results),results=results,time=now()))
    save(BASE/'acceptance/preflight.json',dict(passed=True,formal_runs=5,search=False,time=now()))
    print('ACCEPTANCE PASS: four GPU smoke runs; five fixed formal runs ready',flush=True)
if __name__=='__main__':accept()
