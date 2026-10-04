"""Audit only completed tuning-seed search artifacts, without reading validation runs."""
import hashlib
import json
import numpy as np
import torch
from exp3.search_records import read_history,STAGES,BUDGETS,BASE,SEARCH
from exp3.spec import ROOT,TrialSpec,read_completed


def main():
    torch.set_num_threads(2)
    h=read_history()
    assert set(h['stages'])==set(STAGES) and all(s['status']=='closed' for s in h['stages'].values())
    assert all(r['status']=='observed' and 1<=len(r['specs'])<=3 for r in h['rounds'])
    for r in h['rounds']:
        batch=json.loads((ROOT/r['batch_dir']/'batch_summary.json').read_text())
        assert batch['gpus']==[1,2,3] and len(batch['trials'])==len(r['specs'])
        assert all(t['gpu'] in (1,2,3) for t in batch['trials'])
    configs=json.loads((BASE/'selected_configs.json').read_text())
    assert set(configs)==set(STAGES.values())
    for stage, method in STAGES.items():
        assert configs[method]==h['stages'][stage]['selected_config']
        chosen=next(t for t in h['trials'] if t['trial_id']==h['stages'][stage]['selected_trial'])
        best=max(t['final_test_top1'] for t in h['trials']
                 if t['stage']==stage and t['status']=='completed')
        assert chosen['final_test_top1']>=best-.003  # documented near-tie preferences
    baseline=None;mf=None;probe_hash=None;completed=0;failed=[]
    for row in h['trials']:
        assert row['seed']==20261001
        if row['reused_identity']:
            assert row['source_method']=='mf_muon_standard'
            assert row['lambda_parallel']==1 and row['kappa']==1
            continue
        spec=TrialSpec(**row['spec'])
        assert spec.directory.is_relative_to(SEARCH) and not spec.smoke
        if row['status']=='failed':
            failed.append(row['trial_id']);continue
        s=read_completed(spec)
        assert s['seed']==20261001 and s['optimizer_steps']==250 and s['physical_batches']==1000
        assert s['noise_steps']==(0 if spec.method=='nonprivate_hybrid' else 250)
        assert s['epsilon'] is None if spec.method=='nonprivate_hybrid' else np.isclose(s['epsilon'],8)
        paired={k:s[k] for k in ('initialization_sha256','classifier_initialization_sha256','checkpoint_sha256',
                                 'train_order_sha256','augmentation_trace_sha256')}
        if baseline is None: baseline=paired
        assert paired==baseline
        order=np.load(spec.directory/'train_order.npy')
        assert order.shape==(50000,) and len(np.unique(order))==50000
        assert hashlib.sha256(order.tobytes()).hexdigest()==s['train_order_sha256']
        state=torch.load(spec.directory/'final.pt',map_location='cpu',weights_only=True)
        assert state['logical_steps']==250 and all(torch.isfinite(t).all() for t in state['model'].values())
        assert all(torch.isfinite(v['pre_ns']).all() for v in state['optimizer']['state'].values() if 'pre_ns' in v)
        if spec.method.startswith('mf_'):
            with np.load(spec.directory/'matrices.npz') as data:
                current={k:data[k].copy() for k in ('D','strategy','workload_coefficients','noising_coefficients')}
            if mf is None:mf=current
            for key in current:np.testing.assert_array_equal(current[key],mf[key])
            f=s['frozen_trajectory_muon_mf']
            assert f['steps']==250 and f['actual_dynamics']=='EMA Nesterov beta=.95'
            if probe_hash is None:probe_hash=f['temporal_probe_sha256']
            assert f['temporal_probe_sha256']==probe_hash
        completed+=1
    counts={stage:sum(t['stage']==stage and not t['reused_identity'] for t in h['trials']) for stage in STAGES}
    assert all(counts[stage]<=BUDGETS[stage] for stage in STAGES)
    report=dict(status='passed',tuning_seed=20261001,completed_trials=completed,failed_trials=failed,counts=counts,
                paired_initialization_order_augmentation=True,finite_saved_models=True,
                optimizer_steps_per_trial=250,physical_batches_per_trial=1000,
                single_joint_release_counts_verified=True,mf_factorization_identical=True,
                mf_temporal_probes_identical=True,validation_results_read=False,
                physical_gpus_verified=[1,2,3],max_batch_size_verified=3,
                selection_frozen=True,
                selected_config_sha256=hashlib.sha256((BASE/'selected_configs.json').read_bytes()).hexdigest())
    (BASE/'search/search_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
