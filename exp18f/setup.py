"""Reuse verified existing tokens/split/matrix; inspect local assets without regenerating."""
import sys
import numpy as np
import torch
from exp18f.common import (BASE,ROOT,TOKENS,SPLIT,MODEL_PATH,BASELINE,METHODS,SEARCH_SEEDS,
    read,save,sha,now,make_config,load_matrix,calibration)
from exp18f.matrices import audit,workload
from exp18f.scheduler import gpu_snapshot
from exp18f.search import axis_candidates,stage_plan

def prepare():
    assert '/envs/curve/' in sys.executable
    assert not (BASE/'acceptance.json').exists()
    protocol=read(ROOT/'exp18e/protocol.json')
    for key in ('selection','formal_trainings','historical_results','seeds','methods','stpex2_parameters'):protocol.pop(key,None)
    protocol.update(experiment='Exp18f',methods=list(METHODS),baseline=BASELINE,
        evaluation='Internal validation only, every epoch; official validation never loaded',
        gpu='Exactly three selected UUIDs; dynamic memory admission; <=2 workers each; <=6 total; no DDP',
        total_training_limit=156,stage_limits=[108,18,30],
        eps_definition='optimizer stability coefficient, NOT privacy epsilon',
        stability=dict(numerical='NaN/Inf loss, clipping norm, noisy gradient, alpha, optimizer state or model parameter; terminal numerical_failure',
            predictive='finite but final mean accuracy < baseline - .05 OR mean loss > max(1.0,1.5*baseline loss)',
            baseline_unavailable='absolute accuracy .55 and loss 1.0 thresholds only if no baseline trial succeeds',
            alpha='decay alone never failure',nonmonotonic='any unstable-to-stable return triggers local densification',
            censored='never invent an unobserved stable/unstable endpoint'),
        statistics='seed is observational unit; ddof=1 sample SD and Student-t 95% CI; paired method differences',
        stpex2_parameters='Shared triplet frozen ONLY after five fresh review seeds',
        privacy_scope='Per-training epsilon=8,delta=1e-5; combined search/review is NOT epsilon=8; no sampling amplification')
    protocol['B2LT']['source']='Byte-identical Exp18e momentum-bias matrix, independently re-audited'
    save(BASE/'protocol.json',protocol)
    space=dict(axes=dict(lr=[.01,.015,.02,.024,.0282842712474619,.03,.032,.036,.04,.048,.056,.064,.08],
        clip=[50,75,100,125,150,200],eps=[.0003,.0005,.0007,.001,.0015,.002,.003,.005]),
        baseline=BASELINE,axis_seeds=SEARCH_SEEDS,axis_candidates=27,axis_trainings=108,
        baseline_duplicates='Keep one baseline entry on EACH axis, as requested; no pooled duplicate-seed statistics',
        turning_seeds=[20261123,20261124,20261125],turning_rounds_max=3,turning_trainings_max=18,
        midpoint='sqrt(low*high); nonmonotone -> local densification; no bracket -> largest performance-drop gap',
        rounded_lrs=[.025,.03,.035],rounded_seeds=[20261131,20261132,20261133,20261134,20261135],
        rounded_trainings_max=30,selection='Complete ten-run candidates only; five-seed paired method mean final accuracy, then loss',
        official_validation_used=False)
    save(BASE/'search_space.json',space)
    inventory=gpu_snapshot();selected=[0,1,2]
    assert all(inventory[g]['free_mib']>11000 for g in selected)
    save(BASE/'scheduling.json',dict(physical_gpus=selected,gpu_uuids={str(g):inventory[g]['uuid'] for g in selected},
        initial_inventory=inventory,max_per_gpu=2,max_processes=6,worker_reservation_mib=5200,
        peak_margin_mib=500,headroom_mib=700))
    # Existing audited tensors are read directly; never tokenize or copy the dataset.
    old=read(ROOT/'exp18e/assets.json')
    for p in (TOKENS,SPLIT):assert sha(p)==old[str(p.relative_to(ROOT))]
    tensors=torch.load(TOKENS,weights_only=True)
    assert set(tensors)=={'input_ids','attention_mask','token_type_ids','labels'}
    assert tensors['input_ids'].shape==(67349,128)
    assert set(tensors['labels'].unique().tolist())=={0,1}
    with np.load(SPLIT) as split:
        train,valid=split['train'],split['search_validation']
        assert len(train)==62000 and len(valid)==5349
        assert len(np.unique(np.r_[train,valid]))==67349
    inputs=[TOKENS,SPLIT,ROOT/'data/sst2/train.parquet',ROOT/'data/sst2/validation.parquet']
    inputs+=list(MODEL_PATH.glob('*'))+list((ROOT/'exp18e').glob('*.py'))
    inputs+=[ROOT/'exp18e/assets.json',ROOT/'exp18e/protocol.json',ROOT/'exp18e/matrices/momentum-bias.npz']
    save(BASE/'inputs.json',{str(p.relative_to(ROOT)):sha(p) for p in inputs if p.is_file()})
    save(BASE/'exp18e_snapshot.json',{str(p.relative_to(ROOT/'exp18e')):[p.stat().st_size,p.stat().st_mtime_ns]
        for p in (ROOT/'exp18e').rglob('*') if p.is_file()})
    m=load_matrix();assert m['metadata']['first_order_pass'] and m['metadata']['selected']=='b2lt'
    np.testing.assert_allclose(m['A'],workload('momentum-bias'),atol=1e-12)
    save(BASE/'matrix_audit.json',dict(passed=True,audit=audit(m['A'],m['C'],m['D']),sha256=m['metadata']['sha256']))
    candidates=axis_candidates();assert len(candidates)==27
    jobs=stage_plan('axis',candidates,SEARCH_SEEDS);assert len(jobs)==108
    smoke=[make_config('smoke',f'gpu_pair_{i}',m,20261121,BASELINE) for i in range(3) for m in METHODS]
    save(BASE/'plans/smoke.json',smoke)
    privacy={f"{cid}/{method}":calibration(make_config('axis',cid,method,SEARCH_SEEDS[0],p)) for cid,p in candidates for method in METHODS}
    save(BASE/'privacy_audit.json',dict(passed=True,calibrations=privacy,per_trial_recalibration=True,
        composition_epsilon8_claim=False,sampling_amplification=False))
    save(BASE/'data_audit.json',dict(passed=True,downloads=False,tokenization=False,train=62000,internal_validation=5349,
        tokens=str(TOKENS),split=str(SPLIT),model=str(MODEL_PATH),official_validation_loaded=False,python=sys.executable,time=now()))
    save(BASE/'rounded_freeze.json',dict(status='pending_five_seed_review',frozen=False,parameters=None))
    save(BASE/'turning_point.json',dict(status='pending_axis_scan'))
    print('PREPARED 27 axis candidates / 108 trainings; offline assets; three UUIDs; privacy audit',flush=True)

if __name__=='__main__':prepare()
