"""Bounded independent coordinate search. Diagnostics only generate candidates."""
import argparse
import csv
import json
import math
import numpy as np
from exp8b import RESULTS,ROOT
from exp8b.config import *
from exp8b.launcher import run_queue


def best(candidates):
    valid=[r for r in candidates if r['status']=='completed' and r['category']=='search' and r['completed_epochs']==5 and math.isfinite(r['accuracy'])]
    assert valid,'No finite completed epoch-5 search candidate'
    return max(valid,key=lambda r:(r['accuracy'],trial_id({k:r[k] for k in ('method','seed','lr','C','eps_scale')})))


def neighbors(value,grid):
    """Interior geometric midpoints; one outward candidate at a boundary."""
    grid=sorted(set(grid))
    if value<=grid[0]: return [value/3,math.sqrt(value*grid[1])]
    if value>=grid[-1]: return [math.sqrt(value*grid[-2]),value*3]
    lower=max(x for x in grid if x<value);upper=min(x for x in grid if x>value)
    return [math.sqrt(lower*value),math.sqrt(upper*value)]


def search_method(method,gpu=0,runner=run_queue):
    scale=METHODS[method]['geometry']=='scale';budget=18 if scale else 12
    records=[]; seen=set(); stages=[]
    def evaluate_jobs(jobs,label):
        new=[j for j in jobs if trial_id(j) not in seen]
        # Bound includes failures and reused evidence, not only successful trials.
        remaining=budget-len(seen)
        if len(new)>remaining: new=new[:remaining]
        if not new: return
        seen.update(trial_id(j) for j in new)
        stages.append(dict(stage=label,candidates=new,budget_used=len(seen)))
        save_json(RESULTS/f'search/{method}_candidates.json',stages)
        records.extend(runner(new,gpu=gpu,category='search'))
        save_json(RESULTS/f'search/{method}_curve.json',dict(records=records,stages=stages,budget=budget,trials=len(seen)))
    eps=.1 if scale else None
    smoke=json.loads((trial_dir(trial(method,1e-4,1.,eps),'smoke')/'summary.json').read_text())
    center=max(.01,smoke['first_step_diagnostics']['gradient_norm_p50']*.3) if scale else 10.
    evaluate_jobs([trial(method,lr,center,eps) for lr in LRS],'coarse_lr')
    winner=best(records)
    if scale:
        # Use observed scaled norms AND clipping rate of the converging anchor.
        path=ROOT/winner['result_dir']/'mechanism_metrics.csv'
        with path.open() as f: diagnostics=list(csv.DictReader(f))
        median=float(np.median([float(r['gradient_norm_p50']) for r in diagnostics]))
        clip=float(np.mean([float(r['clip_fraction']) for r in diagnostics]))
        center=max(.01,median*(.3 if clip>.5 else 1.))
        cs=[center*x for x in (.1,.3,1.,3.,10.)]
        provenance=dict(smoke_norm=smoke['first_step_diagnostics']['gradient_norm_p50'],
                        anchor=winner['result_dir'],observed_norm_p50=median,observed_clip_fraction=clip,C_candidates=cs)
    else:
        cs=list(STANDARD_CS);provenance=dict(C_candidates=cs)
    save_json(RESULTS/f'search/{method}_range_provenance.json',provenance)
    evaluate_jobs([trial(method,winner['lr'],c,eps) for c in cs],'coarse_C')
    winner=best(records)
    if scale:
        K=winner['C']*winner['eps_scale']
        evaluate_jobs([trial(method,winner['lr'],K/e,e) for e in SCALE_EPS],'coarse_eps_scale_at_K')
        winner=best(records)
    # Refine each coordinate once, including at most one outward boundary expansion.
    lr_grid=list(LRS)
    evaluate_jobs([trial(method,x,winner['C'],winner['eps_scale']) for x in neighbors(winner['lr'],lr_grid)],'refine_lr_once')
    winner=best(records)
    if scale:
        # Translate the C grid to the selected eps only to ORGANIZE candidates.
        c_grid=[c*.1/winner['eps_scale'] for c in cs]
        c_grid.append(winner['C'])
    else: c_grid=cs
    evaluate_jobs([trial(method,winner['lr'],c,winner['eps_scale']) for c in neighbors(winner['C'],c_grid)],'refine_C_once')
    winner=best(records)
    if scale:
        K=winner['C']*winner['eps_scale']
        evaluate_jobs([trial(method,winner['lr'],K/e,e) for e in neighbors(winner['eps_scale'],SCALE_EPS)],'refine_eps_scale_once')
        winner=best(records)
    reason='trial_budget_reached' if len(seen)==budget else 'bounded_coordinate_schedule_completed'
    save_json(RESULTS/f'search/{method}_search.json',dict(method=method,budget=budget,trials=len(seen),
              stop_reason=reason,selection_objective='epoch-5 search-validation Accuracy',winner=winner,
              records=records,stages=stages,range_provenance=provenance,seed=SEED))
    return winner


def search(gpu=0,queue=None):
    from exp8b.audit import check_platform
    check_platform()
    assert not (RESULTS/'frozen_configs.json').exists(),'Frozen configuration exists: use final runner'
    if queue is None:
        selected={m:search_method(m,gpu) for m in METHODS}
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(METHODS)) as methods:
            futures={m:methods.submit(search_method,m,gpu,queue.run) for m in METHODS}
            selected={m:f.result() for m,f in futures.items()}
    save_json(RESULTS/'search/selected_configs.json',selected)
    proofs={m:json.loads((RESULTS/f'search/{m}_search.json').read_text()) for m in METHODS}
    save_json(RESULTS/'search/search_summary.json',proofs)
    from exp8b.report import write_csv
    fields=('method','seed','lr','C','eps_scale','accuracy','status','result_dir')
    write_csv(RESULTS/'search/search_summary.csv',[{k:r.get(k) for k in fields} for proof in proofs.values() for r in proof['records']])
    lines=['# Exp8b search','','Selection: seed 20261001 epoch-5 search-validation Accuracy only.',
           '', '| Method | LR | C | eps_scale | Accuracy | Trials / budget | Stop reason |',
           '|---|---:|---:|---:|---:|---:|---|']
    for m,r in selected.items():
        proof=proofs[m]
        lines.append(f"| {m} | {r['lr']} | {r['C']} | {r['eps_scale']} | {r['accuracy']:.5f} | {proof['trials']}/{proof['budget']} | {proof['stop_reason']} |")
    (RESULTS/'search/search_report.md').write_text('\n'.join(lines)+'\n')
    from exp8b.frozen import freeze
    freeze(selected)
    return selected

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);search(p.parse_args().gpu)
