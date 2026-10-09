"""Sequential single-GPU search; never invoked by Stage 1."""
import argparse
from dataclasses import replace
import json
import csv
from exp8a import RESULTS
from exp8a.config import Config, save_json, file_hash, MODEL_PATH
from exp8a.benchmark_batch import search, select_batch
from exp8a.benchmark_length import benchmark_lengths, train_lengths, summarize_lengths
from exp8a.train import run, mechanism
from exp8a.audit import audit_runs, report, verify_stage1


def pilots(cfg,gpu,count,require_convergence=True):
    candidates=((5e-4,1.,.1),(5e-5,1.,.1),(1e-4,1.,.1),(5e-5,3.,.1))
    results=[]
    for i,(lr,C,eps_scale) in enumerate(candidates[:count]):
        trial=replace(cfg,lr=lr,C=C,eps_scale=eps_scale,max_length=64)
        output=RESULTS/f'pilots/pilot{i+1}'
        summary=output/'summary.json'
        if summary.exists():
            result=json.loads(summary.read_text())
            assert result['status']=='completed' and result['optimizer_steps']==result['noise_draws']==62
            for key in ('lr','C','eps_scale','seed','physical_batch_size','max_length'):
                assert result[key]==getattr(trial,key), f'Completed pilot differs in {key}'
        else:
            result=run(trial,gpu,steps=62,output=output)
        results.append(result)
    if not results:
        save_json(RESULTS/'pilots.json',dict(count=0,selection='configured defaults; no pilots'))
        return cfg
    # Only search validation, same one-epoch duration and paired seed.
    best=max(results,key=lambda r:(r['accuracy'],-r['loss']))
    counts=json.loads((RESULTS/'token_lengths.json').read_text())['search_validation_label_counts']
    baseline=max(counts)/sum(counts)
    converged=best['loss']<.6931471805599453 and best['accuracy']>baseline+.003
    save_json(RESULTS/'pilots.json',dict(count=count,results=results,converged=converged,majority_baseline_accuracy=baseline,
              selected=dict(lr=best['lr'],C=best['C'],eps_scale=best['eps_scale'])))
    if require_convergence:
        assert converged, 'Four-pilot limit exhausted without usable convergence; do not start new full runs'
    return replace(cfg,lr=best['lr'],C=best['C'],eps_scale=best['eps_scale'])


def completed_batch_search(cfg):
    # Explicit continuation after the batch phase; read the original worker JSONs.
    with open(RESULTS/'physical_batch_benchmark.csv') as f:
        entries=list(csv.DictReader(f))
    rows=[json.loads((RESULTS/'benchmarks'/f"{r['geometry']}_length64_batch{r['physical_batch_size']}.json").read_text())
          for r in entries]
    tested={r['physical_batch_size'] for r in rows}
    assert {8,20,40,50,100}<=tested
    assert len(rows)==2*len(tested)
    for row in rows:
        assert row['warmup_steps']==1 and row['measured_steps']==3
        for key in ('seed','lr','C','eps_scale','max_length'):
            assert row[key]==getattr(cfg,key)
    if any(r['oom'] for r in rows):
        assert 4 in tested
    p100=[r for r in rows if r['physical_batch_size']==100]
    p50=[r for r in rows if r['physical_batch_size']==50]
    if all(r['status']=='completed' for r in p100+p50):
        if sum(1/r['logical_steps_per_second'] for r in p50)>1.05*sum(1/r['logical_steps_per_second'] for r in p100):
            assert {125,200,250}<=tested
    chosen,candidates=select_batch(rows)
    return chosen,rows,candidates


def execute(gpu=0,pilot_count=4,after_batch=False,after_length=False):
    verify_stage1()
    assert not after_length or after_batch
    assert after_length or not (RESULTS/'selected_config.json').exists() or json.loads((RESULTS/'selected_config.json').read_text())['status']=='stage1_only'
    cfg=Config()
    chosen,batch_rows,candidates=completed_batch_search(cfg) if after_batch else search(gpu,cfg)
    cfg=replace(cfg,physical_batch_size=chosen['physical_batch_size'])
    cfg=pilots(cfg,gpu,pilot_count,require_convergence=not after_length)
    save_json(RESULTS/'frozen_hyperparameters.json',dict(lr=cfg.lr,C=cfg.C,eps_scale=cfg.eps_scale,seed=cfg.seed,
                                                     selection_length=64,frozen_for_all_lengths=True))
    if after_length:
        length_benchmarks=[json.loads((RESULTS/'length_benchmarks'/f'{g}_length{n}_batch{cfg.physical_batch_size}.json').read_text())
                           for n in (32,64,128) for g in ('standard','scale')]
        runs=[json.loads(p.read_text()) for p in sorted((RESULTS/'runs').glob('*/summary.json'))]
        audit_runs(runs)
        for row in runs+length_benchmarks:
            for key in ('lr','C','eps_scale','physical_batch_size'):
                assert row[key]==getattr(cfg,key)
        length,runs,length_reason=summarize_lengths(runs,length_benchmarks)
    else:
        length_benchmarks=benchmark_lengths(cfg,gpu)
        # OOM at 128 must not kill a 5-epoch run. Rechoose one common safe batch before the paired design.
        if not all(r['status']=='completed' and r['headroom_fraction']>=.2 for r in length_benchmarks):
            safe,_,_=search(gpu,cfg,length=128,path=RESULTS/'batch_at_128.csv')
            cfg=replace(cfg,physical_batch_size=safe['physical_batch_size'])
            length_benchmarks=benchmark_lengths(cfg,gpu)
            assert all(r['status']=='completed' and r['headroom_fraction']>=.2 for r in length_benchmarks)
        length,runs,length_reason=train_lengths(cfg,gpu,length_benchmarks)
    cfg=replace(cfg,max_length=length)
    final_batch,final_rows,final_candidates=search(gpu,cfg,length=length,path=RESULTS/'physical_batch_final_benchmark.csv')
    cfg=replace(cfg,physical_batch_size=final_batch['physical_batch_size'])
    audit_runs(runs)
    d,S,W,meta,privacy=mechanism(cfg)
    selected=dict(status='final',physical_batch_size=cfg.physical_batch_size,max_length=cfg.max_length,
                  logical_batch_size=1000,training_protocol=cfg.protocol(),
                  split_sha256=file_hash(RESULTS/'split.npz'),checkpoint_sha256=file_hash(MODEL_PATH/'pytorch_model.bin'),
                  privacy_calibration=privacy,bandinvmf=dict(coefficients=d.tolist(),optimization=meta),
                  hyperparameters=dict(lr=cfg.lr,C=cfg.C,eps_scale=cfg.eps_scale),
                  selection_basis=dict(batch=final_batch,batch_candidates=final_candidates,length=length_reason,
                                       initial_batch=chosen,initial_batch_candidates=candidates,
                                       truncation=json.loads((RESULTS/'token_lengths.json').read_text())['truncation_ratio']),
                  full_runs=len(runs),gpu=gpu,
                  hyperparameters_converged=length_reason.get('convergence_evidence',True),
                  length_evidence='fallback_insufficient_learning' if length_reason.get('convergence_evidence') is False else 'paired_comparison')
    save_json(RESULTS/'selected_config.json',selected)
    report(selected,batch_rows,length_benchmarks,runs,final_rows)
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--after-length',action='store_true',help='With --after-batch, audit completed length runs and repeat only final selection/benchmark')
    p.add_argument('--after-batch',action='store_true',help='Continue from the completed physical-batch benchmark')
    p.add_argument('--pilots',type=int,choices=range(5),default=4,help='0-4 one-epoch pilots at length 64')
    a=p.parse_args()
    execute(a.gpu,a.pilots,a.after_batch,a.after_length)

if __name__=='__main__': main()
