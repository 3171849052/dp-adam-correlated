"""Public validation results and postprocessing of DP optimizer state only."""
import csv
import json
import numpy as np
from scipy.stats import t
from exp10b import ROOT,RESULTS
from exp10b.config import LAMBDAS,FINAL_SEEDS,Trial,save_json,file_hash

def write_csv(path,rows):
    rows=list(rows)
    if not rows: return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def statistics(values):
    values=np.asarray(list(values),dtype=float);n=len(values);assert n>=2
    mean=float(values.mean());sd=float(values.std(ddof=1));se=sd/np.sqrt(n)
    width=float(t.ppf(.975,n-1)*se)
    return dict(n=n,mean=mean,std=sd,se=float(se),ci95_low=mean-width,ci95_high=mean+width)

def summarize(rows):
    metrics=('accuracy','validation_loss','seconds','peak_allocated_bytes','peak_reserved_bytes')
    return [dict(task=task,module=lam,metric=metric,**statistics(r[metric] for r in rows
            if r['task']==task and r['module']==lam))
            for task in ('cv','nlp') for lam in sorted({r['module'] for r in rows if r['task']==task})
            for metric in metrics]

def paired_differences(rows,winners):
    raw=[];summary=[]
    for task in ('cv','nlp'):
        chosen=winners[task]['module']
        base={r['seed']:r for r in rows if r['task']==task and r['module']==0}
        positive={r['seed']:r for r in rows if r['task']==task and r['module']==chosen}
        assert set(base)==set(positive)==set(FINAL_SEEDS)
        for metric in ('accuracy','validation_loss','seconds','peak_allocated_bytes','peak_reserved_bytes'):
            differences=[]
            for seed in FINAL_SEEDS:
                difference=positive[seed][metric]-base[seed][metric];differences.append(difference)
                raw.append(dict(task=task,module=chosen,seed=seed,metric=metric,
                                baseline=base[seed][metric],positive=positive[seed][metric],difference=difference))
            summary.append(dict(task=task,module=chosen,metric=metric,**statistics(differences)))
    return raw,summary

def matrix_diagnostics():
    from exp10b.matrices import get_matrix
    rows=[]
    for n in (225,250,310):
        C=1. if n==310 else 30.
        for lam in LAMBDAS:
            d,S,W,meta=get_matrix(n,lam)
            rows.append(dict(horizon=n,task='nlp' if n==310 else 'cv',module=lam,C=C,
                J1_ratio=meta['J1_ratio'],J2_ratio=meta['J2_ratio'],
                J1=meta['J1'],J2=meta['J2']*C**4,J2_diag=meta['J2_diag']*C**4,
                J2_cross=meta['J2_cross']*C**4,R_corr=meta['R_corr'],
                sensitivity=meta['sensitivity'],sigma_avg=meta['sigma_avg_C1']*C,
                marginal_variance_mean=meta['covariance_diagonal_mean']*C**2,
                lag1_covariance_mean=meta['covariance_lag1_mean']*C**2,
                d0=float(d[0]),d1=float(d[1]),d2=float(d[2]),d3=float(d[3]),
                strategy_sha256=meta['strategy_sha256']))
    write_csv(RESULTS/'matrix_diagnostics.csv',rows);save_json(RESULTS/'matrix_diagnostics.json',rows)
    return rows

def preconditioner_summary(rows,name):
    values=[]
    for task in ('cv','nlp'):
        for lam in sorted({r['module'] for r in rows if r['task']==task}):
            group=[r for r in rows if r['task']==task and r['module']==lam]
            paths=[RESULTS/r['result_dir']/'mechanism_metrics.csv' for r in group]
            for field in ('preconditioner_mean','preconditioner_std','preconditioner_change_relative'):
                run_means=[]
                for path in paths:
                    with path.open() as f: run_means.append(np.mean([float(step[field]) for step in csv.DictReader(f)]))
                values.append(dict(task=task,module=lam,metric=field,**statistics(run_means)))
    write_csv(RESULTS/name,values)

def search_report(rows,evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stats=summarize(rows);write_csv(RESULTS/'internal_statistics.csv',stats)
    save_json(RESULTS/'internal_statistics.json',stats)
    diagnostics=matrix_diagnostics()
    figures=RESULTS/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axes,('cv','nlp')):
        group=[r for r in stats if r['task']==task and r['metric']=='accuracy']
        ax.errorbar(range(6),[r['mean'] for r in group],
            yerr=[r['ci95_high']-r['mean'] for r in group],marker='o',capsize=4)
        ax.set(xticks=range(6),xticklabels=[f'{x:g}' for x in LAMBDAS],xlabel='lambda',
               ylabel='Internal validation accuracy',title=f'{task.upper()}: 10 seeds, t 95% CI')
    fig.tight_layout();fig.savefig(figures/'lambda_accuracy.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,n in zip(axes,(225,250,310)):
        group=[r for r in diagnostics if r['horizon']==n]
        ax.plot([r['J1_ratio'] for r in group],[r['J2_ratio'] for r in group],'-o')
        for r in group: ax.annotate(f"{r['module']:g}",(r['J1_ratio'],r['J2_ratio']),fontsize=8)
        ax.set(xlabel='J1 / J1(D0)',ylabel='J2 / J2(D0)',title=f'T={n}')
    fig.tight_layout();fig.savefig(figures/'j1_j2_pareto.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,n in zip(axes,(225,250,310)):
        group=[r for r in diagnostics if r['horizon']==n];base=group[0]['J2']
        diag=[r['J2_diag']/base for r in group];cross=[r['J2_cross']/base for r in group]
        ax.bar(range(6),diag,label='J2 diag');ax.bar(range(6),cross,bottom=diag,label='J2 cross')
        ax.set(xticks=range(6),xticklabels=[f'{x:g}' for x in LAMBDAS],xlabel='lambda',
               ylabel='Contribution / J2(D0)',title=f'T={n}');ax.legend()
    fig.tight_layout();fig.savefig(figures/'j2_decomposition.png',dpi=180);plt.close(fig)
    (RESULTS/'selection_report.md').write_text('# Exp10b internal lambda ablation\n\n'
        'All six lambdas completed ten seeds per task. Selection uses final-epoch internal accuracy only; ties use the smaller lambda.\n\n'+
        '\n'.join(f"- {task.upper()}: best positive lambda={evidence[task]['best_positive']['module']:g}, "
                  f"mean={evidence[task]['best_positive']['mean_internal_accuracy']:.6f}; "
                  f"global best including zero: lambda={evidence[task]['global_best']['module']:g}, "
                  f"mean={evidence[task]['global_best']['mean_internal_accuracy']:.6f}." for task in ('cv','nlp'))+'\n')

def historical_cv_reproduction(rows):
    """Compare actual Exp9 artifacts, never retrain or select on official scores."""
    import torch
    old={}
    for stage in ('search','recheck','final'):
        for path in sorted((ROOT/'exp9/results'/stage/'cv').glob('*/summary.json')):
            value=json.loads(path.read_text())
            if (value.get('status')=='completed' and value.get('method')=='dp-adam-momentum-bias-bandinvmf'
                and value['lr']==.003 and value['C']==30):
                old[value['seed'],value['planned_total_steps']]=(path,value)
    checks=[]
    for row in rows:
        if row['task']!='cv' or row['module']!=0: continue
        reference=old.get((row['seed'],row['planned_total_steps']))
        if reference is None:
            checks.append(dict(seed=row['seed'],horizon=row['planned_total_steps'],status='no_same_seed_Exp9_artifact'))
            continue
        path,historical=reference
        fields=('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','strategy_sha256','coefficients')
        equal={name:row[name]==historical[name] for name in fields}
        calibration_equal=all(row['calibration'][key]==historical['calibration'][key]
            for key in ('target_mu','sensitivity','innovation_std_sum'))
        # Same generator seed, shapes, ordering and FP32 FIR implementation are
        # checked against the historical source by CPU and GPU unit tests.
        noise_equal=calibration_equal and equal['coefficients'] and row['innovation_seed']==historical['seed']+1
        new_ckpt=torch.load(RESULTS/row['result_dir']/'checkpoint.pt',map_location='cpu',weights_only=True)
        old_ckpt=torch.load(path.parent/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert set(new_ckpt['model'])==set(old_ckpt['model'])
        max_error=max(float((new_ckpt['model'][k]-old_ckpt['model'][k]).abs().max()) for k in new_ckpt['model'])
        tensors_equal=all(torch.equal(new_ckpt['model'][k],old_ckpt['model'][k]) for k in new_ckpt['model'])
        old_loss=historical['epochs'][-1]['validation_loss']
        checks.append(dict(seed=row['seed'],horizon=row['planned_total_steps'],
            reference=str(path.relative_to(ROOT)),reference_summary_sha256=file_hash(path),
            paired_checks=equal,calibration_equal=calibration_equal,
            noise_protocol_equal=noise_equal,noise_evidence='seed+1, parameter shapes/order, exact coefficients/sigma; independent CPU/GPU source-equivalence tests',
            final_model_tensors_exact=tensors_equal,final_model_max_abs_difference=max_error,
            accuracy_difference=row['accuracy']-historical['accuracy'],
            validation_loss_difference=row['validation_loss']-old_loss,
            status='exact_reproduction' if all(equal.values()) and noise_equal and tensors_equal
                and row['accuracy']==historical['accuracy'] and row['validation_loss']==old_loss else 'difference_recorded'))
    save_json(RESULTS/'exp9_cv_reproduction.json',checks)
    return checks

def final_report(internal,final,evidence,winners,audit):
    import matplotlib.pyplot as plt
    stats=summarize(final);raw,paired=paired_differences(final,winners)
    write_csv(RESULTS/'final_statistics.csv',stats);write_csv(RESULTS/'paired_seed_differences.csv',raw)
    save_json(RESULTS/'paired_differences.json',paired)
    preconditioner_summary(internal,'internal_preconditioner_statistics.csv')
    preconditioner_summary(final,'final_preconditioner_statistics.csv')
    reproduction=historical_cv_reproduction(internal+final)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axes,('cv','nlp')):
        group=[r for r in raw if r['task']==task and r['metric']=='accuracy']
        mean=next(r for r in paired if r['task']==task and r['metric']=='accuracy')
        ax.plot(range(10),[r['difference'] for r in group],'o');ax.axhline(0,color='gray')
        ax.axhline(mean['mean'],color='red');ax.axhspan(mean['ci95_low'],mean['ci95_high'],alpha=.2,color='red')
        ax.set(xticks=range(10),xticklabels=[str(s)[-2:] for s in FINAL_SEEDS],xlabel='Paired seed suffix',
               ylabel='Accuracy difference (positive - zero)',title=f'{task.upper()}, lambda={winners[task]["module"]:g}')
    fig.tight_layout();fig.savefig(RESULTS/'figures/final_paired_differences.png',dpi=180);plt.close(fig)
    lines=['# Exp10b formal paired comparison','',
        'Completed 36 + 84 + 40 = 160 full five-epoch trainings. All positive winners were frozen after all 120 internal runs and before official evaluation.', '',
        '| Task | Lambda | Accuracy mean | Sample SD | 95% t CI |','|---|---:|---:|---:|---|']
    for row in stats:
        if row['metric']=='accuracy': lines.append(f"| {row['task']} | {row['module']:g} | {row['mean']:.6f} | {row['std']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    lines+=['','Paired differences compare the frozen best positive lambda against lambda zero on ten common seeds.']
    for row in paired:
        if row['metric']=='accuracy': lines.append(f"- {row['task'].upper()}: difference={row['mean']:.6f}, 95% CI [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}].")
    final_reproduction=[r for r in reproduction if r['horizon']==250]
    lines+=['',f"CV same-seed Exp9 formal reproduction: {sum(r['status']=='exact_reproduction' for r in final_reproduction)}/10 exact; details and any differences are in exp9_cv_reproduction.json.",'',
        'Accuracy, validation loss, elapsed time, allocated/reserved GPU memory, matrix objectives and preconditioner changes are saved in the accompanying JSON/CSV files. Covariance and marginal variances are saved in matrix NPZ files.', '',
        'Privacy applies separately to each epsilon=8, delta=1e-5 training, using Exp9 fixed-participation GDP calibration without sampling amplification. Internal validation is public/non-protected. No epsilon=8 guarantee is claimed for the complete selection/repetition pipeline. Only public evaluation and DP optimizer-state postprocessing diagnostics are released.', '',
        'Historical CV official test and NLP official validation have already been viewed. These are not new blind tests. Exp9 comparison scores do not affect lambda selection.']
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    from exp10b.frozen import load_frozen
    from exp10b.audit import full_audit
    internal=json.loads((RESULTS/'stage1_results.json').read_text())+json.loads((RESULTS/'stage2_results.json').read_text())
    final=json.loads((RESULTS/'final_results.json').read_text());evidence=json.loads((RESULTS/'selection_evidence.json').read_text())
    search_report(internal,evidence);final_report(internal,final,evidence,load_frozen(),full_audit())
