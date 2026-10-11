"""Public-validation results and diagnostics from DP state/public Gaussian geometry."""
import csv
import json
import numpy as np
from scipy.stats import t
from exp10 import RESULTS
from exp10.config import METHODS,FINAL_SEEDS,save_json

def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def trial_table(rows):
    return [{k:r[k] for k in ('trial_id','task','method','module','lr','C','stage','seed','status','accuracy',
        'validation_loss','actual_epsilon','seconds','train_seconds','peak_allocated_bytes','physical_gpu')} for r in rows]
def statistics(values):
    a=np.asarray(values,dtype=float);assert len(a)>=2 and np.isfinite(a).all()
    mean=float(a.mean());std=float(a.std(ddof=1));se=std/np.sqrt(len(a));half=float(t.ppf(.975,len(a)-1)*se)
    return dict(n=len(a),mean=mean,std=std,se=float(se),ci95_low=mean-half,ci95_high=mean+half)
def summarize(rows):
    assert len(rows)==40
    summary=[];paired=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            runs=sorted((r for r in rows if r['task']==task and r['method']==method),key=lambda r:r['seed'])
            assert [r['seed'] for r in runs]==list(FINAL_SEEDS)
            for metric in ('accuracy','validation_loss','actual_epsilon','seconds','peak_allocated_bytes'):
                summary.append(dict(task=task,method=method,metric=metric,**statistics([r[metric] for r in runs])))
        by={(r['method'],r['seed']):r for r in rows if r['task']==task}
        for metric in ('accuracy','validation_loss'):
            differences=[by[METHODS[0],seed][metric]-by[METHODS[1],seed][metric] for seed in FINAL_SEEDS]
            paired.append(dict(task=task,metric=metric,contrast='Adam-aware MF minus BlockAdam-MF',
                differences=differences,positive=sum(x>0 for x in differences),**statistics(differences)))
    return summary,paired

def mechanism_figures(rows,winners):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from exp10.matrices import get_matrix,weights
    from exp10.noise import materialize
    figures=RESULTS/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    details=[]
    for ti,task in enumerate(('cv','nlp')):
        n=250 if task=='cv' else 310
        for method in METHODS:
            cfg=winners[task][method];d,S,W,meta=get_matrix(n,cfg['module'] if method==METHODS[0] else None)
            D=materialize(d,n); covariance=(meta['sigma_avg_C1']*cfg['C'])**2*(D@D.T)
            w=weights(n);second_var=2*np.sum((w@covariance**2)*w,axis=1)
            cumulative_var=np.sum((.1*W@D)**2,axis=1)*(meta['sigma_avg_C1']*cfg['C'])**2
            axes[ti,0].plot(np.arange(1,n+1),second_var,label=method)
            axes[ti,1].plot(np.arange(1,n+1),cumulative_var,label=method)
            # Deterministic pure Gaussian simulation; independent of all private data.
            innovation=np.random.default_rng(20261100).normal(size=(n,4096))
            correlated=meta['sigma_avg_C1']*cfg['C']*(D@innovation)
            mean_square=w@(correlated**2)
            empirical=mean_square.var(axis=1,ddof=1)
            details.append(dict(task=task,method=method,coefficients=d.tolist(),**meta,
                actual_C=cfg['C'],J2_actual_C=meta['J2']*cfg['C']**4,
                theoretical_J2=float(second_var.mean()),monte_carlo_J2=float(empirical.mean()),
                cumulative_variance_final=float(cumulative_var[-1]),innovation_std_avg=meta['sigma_avg_C1']*cfg['C']))
            traces=[]
            for r in rows:
                if r['task']==task and r['method']==method:
                    with (RESULTS/r['result_dir']/'mechanism_metrics.csv').open() as f: traces.append(list(csv.DictReader(f)))
            for field,style in (('preconditioner_change_relative','-'),('denominator_lag_relative','--')):
                means=np.mean([[float(r[field]) for r in trace] for trace in traces],axis=0)
                axes[ti,2].plot(np.arange(1,n+1),means,style,label=f'{method}: {field}')
        axes[ti,0].set_title(f'{task}: pure Gaussian second-moment variance')
        axes[ti,1].set_title(f'{task}: pure MF cumulative first-moment noise')
        axes[ti,2].set_title(f'{task}: DP preconditioner changes and lag')
        for ax in axes[ti]: ax.set_xlabel('Logical step');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'mechanisms.png',dpi=160);fig.savefig(figures/'mechanisms.pdf');plt.close(fig)
    save_json(RESULTS/'mechanism_summary.json',details)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    search=json.loads((RESULTS/'search_results.json').read_text())
    initial={r['trial_id'] for r in json.loads((RESULTS/'initial_search_results.json').read_text())}
    for ax,task in zip(axes,('cv','nlp')):
        for method in METHODS:
            candidates=[r for r in search if r['task']==task and r['method']==method]
            ax.plot(range(1,31),[r['accuracy'] for r in candidates],'.-',label=method)
            assert all((r['trial_id'] in initial)==(i<24) for i,r in enumerate(candidates))
        ax.axvline(24.5,color='grey',linestyle='--');ax.set_title(f'{task}: final internal-validation accuracy')
        ax.set_xlabel('Trial within method');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'search_trajectory.png',dpi=160);plt.close(fig)
    return details

def generate_report(rows,audit):
    winners=json.loads((RESULTS/'frozen_configs.json').read_text());summary,paired=summarize(rows)
    save_json(RESULTS/'summary_statistics.json',summary);write_csv(RESULTS/'summary_statistics.csv',summary)
    save_json(RESULTS/'paired_differences.json',paired);write_csv(RESULTS/'paired_differences.csv',paired)
    details=mechanism_figures(rows,winners)
    common=['All reported Exp10 full trainings are newly executed: 120 search + 24 review + 40 formal = 184.',
        'Search: 20261101, five epochs, exactly 30 configurations per task/method (24 initial + six internal-validation refinements).',
        'Top-2 reviewed on 20261102/03/04. Winner selected by four-seed mean internal accuracy; loss and trial ID break ties.',
        'Formal paired seeds: 20261111–20261120. CV T=250, NLP T=310; search CV T=225.',
        'Physical GPUs 1/2/3 only. FIFO; one CV or two NLP workers per GPU; no task mixing on a GPU.',
        'Per-run epsilon=8, delta=1e-5, no sampling amplification. No epsilon=8 claim for composition of model selection and repeated training.',
        'Internal validation is assumed public/non-protected. Diagnostics use DP optimizer state and public matrix/Gaussian geometry; raw training losses and clipping statistics are not logged.',
        'Exp9 results are historical references, not new Exp10 trainings. Historical CV test-based tuning and prior viewing of SST-2 validation preclude any new blind-test claim.',
        'Uncertainty: sample standard deviation (ddof=1), two-sided Student t 95% confidence intervals over ten seeds. Paired differences use each common seed.']
    table=['| Task | Method | Mean accuracy | SD | 95% CI | LR | C | Module |','|---|---|---:|---:|---|---:|---:|---:|']
    for stat in summary:
        if stat['metric']!='accuracy': continue
        task,method=stat['task'],stat['method'];cfg=winners[task][method]
        table.append(f"| {task} | {method} | {stat['mean']:.4f} | {stat['std']:.4f} | [{stat['ci95_low']:.4f}, {stat['ci95_high']:.4f}] | {cfg['lr']} | {cfg['C']} | {cfg['module']} |")
    for method,filename,description in ((METHODS[0],'module_a_report.md',
        'Standard PyTorch Adam is unchanged. A 4-band inverse MF minimizes normalized Exp9 full Momentum-Bias J1 plus lambda times normalized Gaussian J2. C_clip=1 during optimization; each training recalibrates true strategy sensitivity at its chosen C.'),
        (METHODS[1],'module_b_report.md',
        'The matrix equals Exp9 Momentum-Bias at the same horizon. Every step updates both moments and bias correction. Each block freezes the denominator from its first step after updating v; incomplete blocks follow the same rule.')):
        lines=[f'# {method}','',description,'',*common,'','\n'.join(table),'','Formal mechanism geometry:']
        for item in details:
            if item['method']==method:
                lines.append(f"- {item['task']}: d={item['coefficients']}, sensitivity={item['sensitivity']:.8g}, J1={item['J1']:.8g}, J2(C=1)={item['J2']:.8g}, J2(actual C)={item['J2_actual_C']:.8g}.")
        lines+=['','![Mechanisms](figures/mechanisms.png)','',
            'Preconditioner diagnostics use the first 64 flattened coordinates per parameter tensor. Denominator lag is the mean absolute relative difference between the frozen and live DP denominator.',
            'Pure MF cumulative first-moment noise includes (1-beta1)=0.1; it excludes gradients, adaptive preconditioning, and parameter feedback. It is a mechanism diagnostic, not total optimization-error variance.']
        (RESULTS/filename).write_text('\n\n'.join(lines)+'\n')
    lines=['# Exp10 final report','',*common,'','\n'.join(table),'','Paired accuracy differences (A minus B):']
    for p in paired:
        if p['metric']=='accuracy': lines.append(f"- {p['task']}: mean={p['mean']:.5f}, SD={p['std']:.5f}, 95% CI=[{p['ci95_low']:.5f}, {p['ci95_high']:.5f}].")
    lines+=['','![Search trajectory](figures/search_trajectory.png)','![Mechanisms](figures/mechanisms.png)','',
        '[Module A](module_a_report.md) · [Module B](module_b_report.md) · [Freeze](frozen_manifest.json) · [Audit](audit_summary.json)',
        f"Audit: {audit['status']}; protected historical/data/cache files unchanged: {audit['historical_files_unchanged']}.",
        'Full numeric accuracy/loss/epsilon/runtime/memory statistics are in summary_statistics.csv. All candidate and five-epoch results are retained in search_results.json, recheck_results.json and final_results.json.']
    (RESULTS/'final_report.md').write_text('\n\n'.join(lines)+'\n')
