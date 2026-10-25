"""Audit actual artifacts and print current progress, including incomplete stages."""
from collections import Counter
from exp16a import RESULTS
from exp16a.support import read,verify_protected
from exp16a.config import Trial,save_json,file_hash
from exp16a.scheduler import verify

def main():
    verify_protected();stages={}
    for name in ('smoke','coarse','fine','review','formal'):
        p=RESULTS/f'{name}_status.json'
        if not p.exists():continue
        states=read(p);stages[name]=dict(Counter(r['status'] for r in states.values()))
        for r in states.values():
            if r['status'] in ('completed','non_finite'):verify(Trial(**{k:r[k] for k in Trial.__dataclass_fields__}))
    from exp16a.stage1 import verify_acceptance
    if (RESULTS/'phase1_acceptance.json').exists():verify_acceptance()
    out=dict(read_only_verified=True,stages=stages,pipeline=read(RESULTS/'pipeline_status.json') if (RESULTS/'pipeline_status.json').exists() else None)
    save_json(RESULTS/'current_audit.json',out);print(out)
if __name__=='__main__':main()
