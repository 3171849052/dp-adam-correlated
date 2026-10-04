"""Bookkeeping for manually reasoned sequential batches; contains no search policy/grid."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from exp3.spec import EXP3, ROOT, METHODS, TrialSpec, read_completed

BASE=EXP3/'results'
SEARCH=BASE/'search'
STAGES=dict(zip('ABCDE',METHODS))
BUDGETS=dict(A=8,B=12,C=12,D=16,E=16)
PARAMETERS=('muon_lr','adam_lr','max_grad_norm','lambda_parallel','kappa','rho')


def read_history():
    path=BASE/'search_history.json'
    if path.exists():
        return json.loads(path.read_text())
    return dict(protocol=dict(tuning_seed=20261001,objective='maximize final_test_top1',gpus=[1,2,3],
                              max_batch_size=3,budgets=BUDGETS,selection='utility; no composite score'),
                rounds=[],stages={},trials=[])


def save(history):
    BASE.mkdir(exist_ok=True)
    (BASE/'search_history.json').write_text(json.dumps(history,indent=2,allow_nan=False))
    fields=['stage','trial_id','method','source_method','seed','reused_identity','status',*PARAMETERS,
            'final_test_top1','best_test_top1','train_loss','mean_clip_fraction','innovation_std_sum',
            'noise_std_gradient','mean_update_gain_cv','mean_layer_update_gain_cv','mean_temporal_update_gain_cv',
            'mean_noise_weighted_update_gain','frozen_final_cumulative_rmse','result_dir']
    with (BASE/'search_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(history['trials'])
    rationale={stage:dict(tested_region={p:sorted({t[p] for t in history['trials'] if t['stage']==stage}) for p in PARAMETERS},
                         rounds=[dict(round=r['name'],why_next_trials_were_selected=r['reasoning'],observed=r.get('observed'))
                                 for r in history['rounds'] if r['stage']==stage],
                         **history['stages'].get(stage,{})) for stage in STAGES}
    (BASE/'search_rationale.json').write_text(json.dumps(rationale,indent=2,allow_nan=False))


def plan(stage,name,values,reasoning):
    assert stage in STAGES and 1<=len(values)<=3 and reasoning
    history=read_history()
    for previous in list(STAGES)[:list(STAGES).index(stage)]:
        assert history['stages'][previous]['status']=='closed'
    assert history['stages'].get(stage,{}).get('status')!='closed'
    assert not any(r['status']=='running' for r in history['rounds']), 'Observe the current batch first'
    assert name not in {r['name'] for r in history['rounds']}
    specs=[TrialSpec(**v) for v in values]
    assert all(s.seed==20261001 and not s.smoke and s.method==STAGES[stage] for s in specs)
    assert all(s.directory.is_relative_to(SEARCH) for s in specs)
    count=sum(t['stage']==stage and not t['reused_identity'] for t in history['trials'])
    assert count+len(specs)<=BUDGETS[stage]
    assert not any(s.directory==TrialSpec(**r['spec']).directory for s in specs for r in history['trials'])
    directory=SEARCH/'batches'/name
    directory.mkdir(parents=True)
    specs_path=directory/'specs.json'
    specs_path.write_text(json.dumps([s.mapping() for s in specs],indent=2))
    history['rounds'].append(dict(stage=stage,name=name,status='running',reasoning=reasoning,
                                  planned_at=datetime.now(timezone.utc).isoformat(),specs=[s.mapping() for s in specs],
                                  batch_dir=str(directory.relative_to(ROOT))))
    save(history)
    print(str(specs_path.relative_to(ROOT)))
    return specs_path


def row_from_summary(stage,spec,summary,reused=False,source_method=None):
    assert summary['seed']==20261001 and not summary['smoke'] and summary['optimizer_steps']==250
    assert len(summary['epoch_test_top1'])==5
    diag=summary['diagnostics']
    clips=[v for v in summary['epoch_clip_fraction'] if v is not None]
    avg=lambda v:sum(v)/len(v)
    frozen=summary['frozen_trajectory_muon_mf']
    row=dict(stage=stage,trial_id=spec.directory.name,method=spec.method,source_method=source_method or spec.method,
             seed=20261001,reused_identity=reused,status=summary['status'],spec=spec.mapping(),
             **{p:getattr(spec,p) for p in PARAMETERS},result_dir=spec.mapping()['result_dir'],
             final_test_top1=summary['final_test_top1'],best_test_top1=summary['best_test_top1'],
             train_loss=summary['train_loss'],epoch_test_top1=summary['epoch_test_top1'],
             epoch_clip_fraction=summary['epoch_clip_fraction'],mean_clip_fraction=avg(clips) if clips else None,
             innovation_std_sum=summary['innovation_std_sum'],noise_std_gradient=summary['innovation_std_sum']/1000,
             mean_update_gain_cv=avg([r['update_gain_cv'] for r in diag['records']]),
             mean_layer_update_gain_cv=avg(list(diag['layer_update_gain_cv'].values())),
             mean_temporal_update_gain_cv=avg(list(diag['temporal_update_gain_cv'].values())),
             mean_noise_weighted_update_gain=avg([r['noise_weighted_update_gain_mean'] for r in diag['records']]) if spec.method!='nonprivate_hybrid' else None,
             frozen_final_cumulative_rmse=frozen['aggregate']['final_cumulative_rmse']['mean'] if frozen else None,
             update_statistics=summary['update_statistics'])
    return row


def observe(name,observed):
    history=read_history()
    batch=next(r for r in history['rounds'] if r['name']==name)
    assert batch['status']=='running'
    result=json.loads((ROOT/batch['batch_dir']/'batch_summary.json').read_text())
    for child in result['trials']:
        spec=TrialSpec(**child['spec'])
        if child['returncode']==0:
            row=row_from_summary(batch['stage'],spec,read_completed(spec))
        else:
            row=dict(stage=batch['stage'],trial_id=spec.directory.name,method=spec.method,source_method=spec.method,
                     seed=spec.seed,reused_identity=False,status='failed',spec=spec.mapping(),
                     **{p:getattr(spec,p) for p in PARAMETERS},result_dir=spec.mapping()['result_dir'],
                     error=child.get('error',f"Child exit {child['returncode']}"))
        history['trials'].append(row)
    batch.update(status='observed',observed=observed,completed_at=datetime.now(timezone.utc).isoformat())
    save(history)
    print(json.dumps([dict(trial=t['trial_id'],top1=t.get('final_test_top1'),curve=t.get('epoch_test_top1'),
                          clip=t.get('epoch_clip_fraction'),noise=t.get('noise_std_gradient'),loss=t.get('train_loss'),
                          gain_cv=t.get('mean_update_gain_cv'),temporal_cv=t.get('mean_temporal_update_gain_cv'),
                          frozen_rmse=t.get('frozen_final_cumulative_rmse')) for t in history['trials'] if t['stage']==batch['stage']],indent=2))


def identity_control(stage,source_trial):
    assert stage in ('D','E')
    history=read_history()
    source=next(t for t in history['trials'] if t['trial_id']==source_trial)
    assert source['stage']=='C' and source['status']=='completed'
    values=dict(source['spec'],method=STAGES[stage],lambda_parallel=1.,kappa=1.)
    spec=TrialSpec(**values)
    summary=read_completed(TrialSpec(**source['spec']))
    row=row_from_summary(stage,spec,summary,True,'mf_muon_standard')
    row['trial_id']=f'{stage.lower()}_identity_reuse_{source_trial}'
    row['source_trial']=source_trial
    row['reason']='Exact identity geometry, same optimizer/clipping/noise/order; reuse Standard without rerunning an equivalent trial'
    assert row['trial_id'] not in {t['trial_id'] for t in history['trials']}
    history['trials'].append(row);save(history)


def close_stage(stage,trial_id,trend,stop):
    history=read_history()
    candidates=[t for t in history['trials'] if t['stage']==stage and t['status']=='completed']
    selected=next(t for t in candidates if t['trial_id']==trial_id)
    assert selected['final_test_top1']>=max(t['final_test_top1'] for t in candidates)-.003
    config={p:selected[p] for p in PARAMETERS}
    history['stages'][stage]=dict(status='closed',observed_trend=trend,why_search_stopped=stop,
                                  trials_run=sum(t['stage']==stage and not t['reused_identity'] for t in history['trials']),
                                  identity_controls_reused=sum(t['stage']==stage and t['reused_identity'] for t in history['trials']),
                                  selected_trial=trial_id,selected_config=config,selected_final_test_top1=selected['final_test_top1'])
    history['stages'][stage]['best_observed_final_test_top1']=max(t['final_test_top1'] for t in candidates)
    history['stages'][stage]['selection_gap_percentage_points']=100*(history['stages'][stage]['best_observed_final_test_top1']-selected['final_test_top1'])
    save(history)
    if set(history['stages'])==set(STAGES) and all(v['status']=='closed' for v in history['stages'].values()):
        configs={STAGES[s]:history['stages'][s]['selected_config'] for s in STAGES}
        (BASE/'selected_configs.json').write_text(json.dumps(configs,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe')
    parser.add_argument('--observed',default='')
    args=parser.parse_args()
    assert args.observe
    observe(args.observe,args.observed)
