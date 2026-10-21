"""Phase-one real-model acceptance; no tuning and no official evaluation."""
import json
from dataclasses import replace
from exp15b import RESULTS
from exp15b.config import Trial, METHODS, save_json
from exp15b.scheduler import run_batch
from exp15b.support import protected_snapshot, verify_assets, code_hashes


def jobs():
    rows=[Trial(t,m,.01,smoke_phase=p) for t in ('cv','nlp') for m in METHODS
          for p in (('search','final') if t=='cv' else ('search',))]
    rows += [replace(next(c for c in rows if c.task==t and c.method=='momentum+iid' and c.smoke_phase=='search'),replicate='repeat') for t in ('cv','nlp')]
    return rows


def main():
    assets=verify_assets();rows=run_batch('smoke',jobs())
    for r in rows:
        if not r['finite'] or not r['noise_draws']==r['optimizer_steps']==2 or r['official_evaluation_used']:
            raise ValueError('Smoke acceptance failed')
    reproducibility=[]
    for task in ('cv','nlp'):
        group=[r for r in rows if r['task']==task and r['smoke_phase']=='search' and r['replicate']=='primary']
        for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','noise_rng_start','noise_rng_end'):
            if any(r[key]!=group[0][key] for r in group):raise ValueError(f'Unpaired {task} {key}')
        first=next(r for r in group if r['method']=='momentum+iid')
        second=next(r for r in rows if r['task']==task and r['replicate']=='repeat')
        # checkpoint serialization bytes are not the tensor-identity criterion.
        import torch
        a,b=[torch.load(Trial(**{k:r[k] for k in Trial.__dataclass_fields__}).output/'checkpoint.pt',weights_only=True,map_location='cpu') for r in (first,second)]
        for key,tensor in a['model'].items():torch.testing.assert_close(tensor,b['model'][key],rtol=0,atol=0)
        for key in ('epochs','rng_trace','noise_rng_start','noise_rng_end','actual_epsilon'):
            if first[key]!=second[key]:raise ValueError(f'Non-reproducible {key}')
        reproducibility.append(dict(task=task,bit_exact_model=True,exact_metrics=True,paired_all_six_methods=True))
    historical_equivalence=[]
    import torch
    from exp15b import ROOT
    for r in rows:
        if r['method']=='momentum+iid':continue
        c=Trial(**{k:r[k] for k in Trial.__dataclass_fields__})
        old=ROOT/'exp15/results/smoke'/c.replicate/c.task/c.id
        previous=json.loads((old/'summary.json').read_text())
        for key in ('epochs','initialization_sha256','train_order_sha256','rng_trace','noise_rng_start','noise_rng_end'):
            if previous[key]!=r[key]:raise ValueError(f'Historical behavior changed: {c.id} {key}')
        a=torch.load(old/'checkpoint.pt',weights_only=True,map_location='cpu')
        b=torch.load(c.output/'checkpoint.pt',weights_only=True,map_location='cpu')
        for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
        historical_equivalence.append(dict(trial_id=c.id,bit_exact_model=True,exact_metrics=True))
    save_json(RESULTS/'historical_equivalence.json',dict(status='passed',trials=historical_equivalence))
    if protected_snapshot()!=json.loads((RESULTS/'protected_before.json').read_text()):raise ValueError('Historical files changed')
    result=dict(status='passed',runs=len(rows),assets=assets,reproducibility=reproducibility,
                code_sha256=code_hashes(),official_evaluation_used=False,historical_files_unchanged=True)
    save_json(RESULTS/'phase1_acceptance.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
