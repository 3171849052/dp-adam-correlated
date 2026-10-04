"""Export the frozen search choices and diagnostics; does not launch experiments."""
import json
from exp3.search_records import BASE, STAGES, read_history


def main():
    history=read_history()
    assert all(history['stages'][s]['status']=='closed' for s in STAGES)
    selected={method:next(t for t in history['trials']
                         if t['trial_id']==history['stages'][stage]['selected_trial'])
              for stage,method in STAGES.items()}
    standard=selected['mf_muon_standard']['final_test_top1']
    version_a=selected['mf_muon_normscale']['final_test_top1']
    version_b=selected['mf_muon_spectralscale']['final_test_top1']
    controls=[t for t in history['trials'] if t['status']=='completed' and
              (t['method']=='mf_muon_standard' or
               t['method']=='mf_muon_normscale' and t['lambda_parallel']==1 or
               t['method']=='mf_muon_spectralscale' and t['kappa']==1)]
    comparisons=[]
    for t in history['trials']:
        if t['stage'] not in ('D','E') or t['reused_identity'] or t['status']!='completed':
            continue
        matches=[c for c in controls if all(c[k]==t[k] for k in ('muon_lr','adam_lr','max_grad_norm'))]
        c=matches[0] if matches else None
        comparisons.append(dict(trial=t['trial_id'],standard_control=c['trial_id'] if c else None,
                                utility_difference_percentage_points=100*(t['final_test_top1']-c['final_test_top1']) if c else None,
                                final_clip_fraction=t['epoch_clip_fraction'][-1],
                                control_final_clip_fraction=c['epoch_clip_fraction'][-1] if c else None,
                                temporal_gain_cv=t['mean_temporal_update_gain_cv'],
                                control_temporal_gain_cv=c['mean_temporal_update_gain_cv'] if c else None,
                                noise_weighted_gain=t['mean_noise_weighted_update_gain'],
                                control_noise_weighted_gain=c['mean_noise_weighted_update_gain'] if c else None))
    report=dict(tuning_seed=20261001,selection_frozen=True,
                validation_results_used=False,final_validation_started=False,
                stages=history['stages'],selected_trials=selected,
                matched_standard_comparisons=comparisons,
                differences_percentage_points=dict(version_a_minus_standard=100*(version_a-standard),
                                                   version_b_minus_standard=100*(version_b-standard),
                                                   version_b_minus_version_a=100*(version_b-version_a)),
                diagnostics_interpretation='Frozen-state first-order estimates, not exact nonlinear training dynamics; different LRs scale cumulative RMSE, so these diagnostics do not rank utility.',
                final_validation_command='conda activate curve\nPYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/exp3/tmp" python -m exp3.final_runner --frozen-config exp3/results/selected_configs.json --result-dir exp3/results/final')
    (BASE/'search/search_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Frozen sequential search', '', 'Only tuning seed 20261001 was used. Final validation has not been run.', '',
           '| Stage | Method | New trials | Top-1 (%) | Muon LR | Adam LR | C | lambda | kappa | rho |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for stage,method in STAGES.items():
        t=selected[method]
        lines.append(f"| {stage} | {method} | {history['stages'][stage]['trials_run']} | {100*t['final_test_top1']:.2f} | {t['muon_lr']} | {t['adam_lr']} | {t['max_grad_norm']} | {t['lambda_parallel']} | {t['kappa']} | {t['rho']} |")
    for stage in STAGES:
        s=history['stages'][stage]
        lines.extend(['',f"## Stage {stage}",'',s['observed_trend'],'',f"Stopped: {s['why_search_stopped']}"])
    lines.extend(['','## Diagnostics','',report['diagnostics_interpretation'],'',
                  'Detailed diagnostics, epoch curves, and identity reuse references are in search_report.json and search_history.json.',
                  '', '## Final validation command', '', '```bash', report['final_validation_command'], '```'])
    (BASE/'search/search_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report['differences_percentage_points']))


if __name__=='__main__':
    main()
