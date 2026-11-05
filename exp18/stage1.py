import subprocess,time
from exp18 import RESULTS,ROOT,BASE
from exp18.common import save,read,seal,PYTHON,GROUPS
from exp18.scheduler import execute
from exp18.search import trials
from exp18.matrices import load,audit

def main():
    with (RESULTS/'unit_tests.log').open('w') as f:
        subprocess.run([PYTHON,'-B','-m','unittest','discover','-s','exp18/tests','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    audits={}
    for name in ('sgd','momentum','momentum-bias'):
        m=load(name);audits[name]=audit(m['A'],m['C'],m['D'])
    candidates={g:[dict(lr=.001,clip=10,eps=.001) if g=='stpex2' else dict(lr=.001,clip=1)] for g in GROUPS}
    rows=execute(trials(candidates,'smoke',[20261102]),'smoke')
    assert len(rows)==6 and read(RESULTS/'smoke_status.json')['six_process_overlap']
    ready=[read(p) for p in (RESULTS/'smoke_ready/20261102').glob('*.json')]
    assert len(ready)==6 and all(r['cuda_memory']>0 for r in ready)
    assert all(sum(r['physical_gpu']==str(g) for r in ready)==2 for g in (1,2,3))
    save(RESULTS/'acceptance.json',dict(passed=True,time=time.time(),code=seal(),matrix_audits=audits,smoke_trials=6,unit_tests='unit_tests.log',physical_batch=1000,accumulation=1,precision='FP32',physical_gpus=[1,2,3]))
    print('Acceptance passed',flush=True)
if __name__=='__main__':main()
