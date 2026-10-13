"""Read-only historical evidence and same-method/same-seed comparisons."""
import csv
import json
import numpy as np
from exp9b import ROOT, RESULTS
from exp9b.config import METHODS, IID, FINAL_SEEDS, file_hash, save_json
from exp9b.noise import materialize, inverse_coefficients
from exp9b.privacy import fixed_epoch_sensitivity
from exp9b.bandmf import matrix_bundle


def baseline_evidence():
    history=ROOT/'exp9/results'
    rows=json.loads((history/'final_results.json').read_text())
    assert len(rows)==140
    manifest=json.loads((history/'frozen_manifest.json').read_text())
    assert file_hash(history/'frozen_manifest.json')==(history/'frozen_manifest.sha256').read_text().strip()
    for name,expected in {**manifest['files_sha256'],**manifest['trial_summary_sha256']}.items():
        assert file_hash(history/name)==expected, name
    provenance=json.loads((ROOT/'exp9b/kernel_provenance.json').read_text())['source_sha256']
    for name,expected in provenance.items(): assert file_hash(ROOT/name)==expected
    assets=json.loads((RESULTS/'assets_manifest.json').read_text())
    old_assets=json.loads((history/'assets_manifest.json').read_text())
    for key in ('cv_pretrained','nlp_pretrained','cifar_train','sst2_train','split_seed'):
        assert assets[key]==old_assets[key]
    for task in ('cv','nlp'):
        assert assets['splits'][task]['indices_sha256']==old_assets['splits'][task]['indices_sha256']
    hashes={name:file_hash(history/name) for name in ('final_results.json','frozen_manifest.json','assets_manifest.json','smoke_summary.json')}
    for r in rows:
        assert r['status']=='completed' and r['stage']=='final' and r['epsilon']==8
        assert r['seed'] in FINAL_SEEDS and r['optimizer_steps']==(250 if r['task']=='cv' else 310)
        assert r['completed_epochs']==5 and abs(r['actual_epsilon']-8)<1e-7
        directory=history/r['result_dir']
        assert json.loads((directory/'summary.json').read_text())==r
        for name in ('config.json','metrics.csv','matrices.npz','train_order.npy','mechanism_metrics.csv'):
            assert file_hash(directory/name)==r['file_sha256'][name]
        metrics=list(csv.DictReader((directory/'metrics.csv').open()))
        assert len(metrics)==5 and float(metrics[-1]['accuracy'])==r['accuracy']
        resolved=json.loads((directory/'config.json').read_text())
        assert resolved['optimizer']==dict(beta1=.9,beta2=.999,eps=1e-8,weight_decay=0.,lr=r['lr'])
        hashes[str((directory/'summary.json').relative_to(history))]=file_hash(directory/'summary.json')
    save_json(RESULTS/'exp9_comparison_sources.json',dict(source='exp9/results',reused_training_results=0,
              comparison_only=True,validated_final_records=len(rows),files_sha256=hashes))
    return rows


def paired_history(formal, baseline):
    from exp9b.report import statistics
    indexed={(r['task'],r['method'],r['seed']):r for r in baseline}
    raw=[];summary=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            old_method=method.replace('bandmf','bandinvmf')
            runs=sorted([r for r in formal if r['task']==task and r['method']==method],key=lambda r:r['seed'])
            assert [r['seed'] for r in runs]==list(FINAL_SEEDS)
            differences=[]
            for r in runs:
                old=indexed[task,old_method,r['seed']]
                for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','workload_sha256'):
                    if key=='workload_sha256':
                        # Independent materialization may differ at roundoff only.
                        with np.load(RESULTS/r['result_dir']/'matrices.npz') as a, np.load(ROOT/'exp9/results'/old['result_dir']/'matrices.npz') as b:
                            np.testing.assert_allclose(a['workload'],b['workload'],rtol=1e-13,atol=1e-13)
                    else: assert r[key]==old[key], (task,method,r['seed'],key)
                diff=r['accuracy']-old['accuracy'];differences.append(diff)
                raw.append(dict(task=task,method=method,seed=r['seed'],bandmf_accuracy=r['accuracy'],
                                bandinvmf_accuracy=old['accuracy'],difference=diff,
                                bandmf_trial=r['trial_id'],exp9_trial=old['trial_id'],
                                bandmf_lr=r['lr'],exp9_lr=old['lr'],bandmf_C=r['C'],exp9_C=old['C'],
                                bandmf_eps_scale=r['eps_scale'],exp9_eps_scale=old['eps_scale']))
            summary.append(dict(task=task,method=method,comparison='Exp9b minus Exp9',**statistics(differences),
                                wins=int(np.sum(np.asarray(differences)>0)),
                                interpretation='same method and seed; independently selected hyperparameters'))
    return raw,summary


def matrix_geometry():
    history=ROOT/'exp9/results'
    smokes=json.loads((history/'smoke_summary.json').read_text())
    rows=[]
    for n in (225,250,310):
        for method in METHODS[:4]:
            noise='iid' if method==IID else {'sgd':'prefix_bandmf','momentum':'momentum_bandmf','momentum-bias':'momentum_bias_bandmf'}[method.removeprefix('dp-adam-').removesuffix('-bandmf')]
            c,S,W,meta=matrix_bundle(noise,n)
            old=next(r for r in smokes if r['planned_total_steps']==n and r['method']==method.replace('bandmf','bandinvmf'))
            with np.load(history/old['result_dir']/'matrices.npz') as v:
                old_c,old_S,old_W=(v[key] for key in ('coefficients','strategy','workload'))
            np.testing.assert_allclose(W,old_W,rtol=1e-13,atol=1e-13)
            for label,coef,strategy,inverse in (('BandMF',c,S,materialize(inverse_coefficients(c,n),n)),
                                                ('Exp9 BandInvMF',old_c,old_S,materialize(old_c,n))):
                sensitivity=fixed_epoch_sensitivity(strategy,5,n//5)
                variance=np.cumsum(inverse[:,0]**2);error=float(np.sum((W@inverse)**2)/n)
                rows.append(dict(experiment=label,method=method,T=n,coefficients=coef.tolist(),
                                 sensitivity=sensitivity,mean_full_workload_error=error,
                                 objective=sensitivity**2*error,unit_innovation_variance_final=float(variance[-1]),
                                 unit_innovation_variance_last_epoch_mean=float(variance[-n//5:].mean()),
                                 unit_C_calibrated_variance_final=float((sensitivity/old['calibration']['target_mu'])**2*variance[-1]),
                                 tail_definition='finite-horizon final variance and final-epoch average',
                                 inverse_residual=float(np.max(np.abs(strategy@inverse-np.eye(n))))))
    return rows


def generate_comparison(formal):
    from exp9b.report import write_csv
    baseline=baseline_evidence();raw,paired=paired_history(formal,baseline)
    write_csv(RESULTS/'exp9_paired_raw.csv',raw);write_csv(RESULTS/'exp9_paired_summary.csv',paired)
    save_json(RESULTS/'exp9_paired_summary.json',paired)
    geometry=matrix_geometry();write_csv(RESULTS/'matrix_geometry.csv',geometry);save_json(RESULTS/'matrix_geometry.json',geometry)
    variances=[]
    for r in formal:
        n=r['planned_total_steps'];c=r['coefficients'];h=inverse_coefficients(c,n)
        v=r['calibration']['innovation_std_sum']**2*np.cumsum(h**2)
        variances.append(dict(task=r['task'],method=r['method'],seed=r['seed'],C=r['C'],
                              variance_sum_final=float(v[-1]),variance_sum_last_epoch_mean=float(v[-n//5:].mean()),
                              variance_mean_gradient_final=float(v[-1]/1000**2),
                              space='scaled clipped sum before inverse-scale' if r['method'].endswith('-scale') else 'clipped gradient sum'))
    write_csv(RESULTS/'noise_tail_variance.csv',variances)
    lines=['','Exp9 same-method/same-seed paired accuracy differences (Exp9b minus Exp9; percentage points).',
           'Both experiments tune independently using the identical grid and selection protocol. These differences include selected-hyperparameter changes.',
           '', '| Task | Method | Paired mean (pp) | Paired t 95% CI (pp) |','|---|---|---:|---|']
    lines += [f"| {r['task']} | {r['method']} | {100*r['mean']:.3f} | [{100*r['ci95_low']:.3f}, {100*r['ci95_high']:.3f}] |" for r in paired]
    lines += ['', '[Raw historical pairs](exp9_paired_raw.csv), [matrix errors, sensitivities and finite-horizon noise tails](matrix_geometry.csv), '
              '[calibrated per-run tail variances](noise_tail_variance.csv), [historical source hashes](exp9_comparison_sources.json).',
              'Scale variance is measured in scaled sum space; inverse-scaling is coordinate-dependent and follows unchanged Exp9 semantics.']
    return lines
