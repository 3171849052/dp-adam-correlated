"""Two real-model steps, each horizon, repeated all three methods per task."""
from dataclasses import replace
import torch
from exp15c import RESULTS
from exp15c.config import Trial,METHODS,save_json,file_hash
from exp15c.scheduler import run_batch
from exp15c.support import code_hashes,verify_assets,protected_snapshot,read


def jobs():
    primary=[Trial(t,m,.001,1. if m=='momentum-mf-native' else None,smoke_phase=p)
        for t in ('cv','nlp') for m in METHODS for p in (('search','final') if t=='cv' else ('search',))]
    return primary+[replace(c,replicate='repeat') for c in primary if c.smoke_phase=='search']


def main():
    tests=read(RESULTS/'test_results.json')
    if tests['status']!='passed' or tests['code_sha256']!=code_hashes():raise ValueError('Passing tests for current code required')
    assets=verify_assets();rows=run_batch('smoke',jobs())
    for r in rows:
        if not r['finite'] or r['optimizer_steps']!=2 or r['official_evaluation_used']:raise ValueError('Smoke failed')
        if r['noise_draws']!=(2 if r['method']=='momentum-mf-native' else 0):raise ValueError('Wrong noise count')
    for task in ('cv','nlp'):
        group=[r for r in rows if r['task']==task and r['smoke_phase']=='search' and r['replicate']=='primary']
        for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
            if any(r[key]!=group[0][key] for r in group):raise ValueError(f'Unpaired {key}')
        for first in group:
            second=next(r for r in rows if r['task']==task and r['method']==first['method'] and r['replicate']=='repeat')
            def checkpoint(r):
                cfg=Trial(**{k:r[k] for k in Trial.__dataclass_fields__})
                return torch.load(cfg.output/'checkpoint.pt',weights_only=True,map_location='cpu')['model']
            a,b=checkpoint(first),checkpoint(second)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            for key in ('epochs','rng_trace','noise_rng_start','noise_rng_end','actual_epsilon'):
                if first[key]!=second[key]:raise ValueError(f'Nonreproducible {key}')
    if protected_snapshot()!=read(RESULTS/'protected_before.json'):raise ValueError('History modified')
    save_json(RESULTS/'phase1_acceptance.json',dict(status='passed',runs=len(rows),code_sha256=code_hashes(),
        tests_log_sha256=file_hash(RESULTS/'tests.log'),tests=read(RESULTS/'test_results.json'),assets=assets,
        reproducibility='bit-exact checkpoints and metrics, all 3 methods, CV and NLP',protected_files_unchanged=True))
    print(f'Smoke passed: {len(rows)} runs',flush=True)


if __name__=='__main__':main()
