"""Machine-readable tables, paired statistics, figures and the paper report."""
import csv
import json
import itertools
import numpy as np
from scipy.stats import t
from exp9 import RESULTS
from exp9.config import METHODS, IID, MF, FINAL_SEEDS, save_json

def write_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def trial_table(rows):
    keys=('task','method','stage','seed','epsilon','lr','C','eps_scale','trial_id','status','accuracy',
          'optimizer_steps','noise_draws','actual_epsilon','seconds','train_seconds','logical_step_seconds',
          'peak_allocated_bytes','peak_reserved_bytes','physical_gpu','result_dir','error')
    return [dict({k:row.get(k) for k in keys},train_loss=row.get('epochs',[{}])[-1].get('train_loss'),
                 clip_fraction=row.get('epochs',[{}])[-1].get('clip_fraction')) for row in rows]

def statistics(values):
    values=np.asarray(values,dtype=float);assert len(values)>=2 and np.isfinite(values).all()
    n=len(values);mean=float(values.mean());std=float(values.std(ddof=1));se=std/np.sqrt(n)
    margin=float(t.ppf(.975,n-1)*se)
    return dict(n=n,mean=mean,std=std,se=float(se),ci95_low=mean-margin,ci95_high=mean+margin)

def summarize(rows):
    assert len(rows)==140 and all(r['status']=='completed' and r['epsilon']==8 for r in rows)
    summaries=[];paired=[]
    for task in ('cv','nlp'):
        arrays={}
        for method in METHODS:
            runs=sorted([r for r in rows if r['task']==task and r['method']==method],key=lambda r:r['seed'])
            assert [r['seed'] for r in runs]==list(FINAL_SEEDS)
            arrays[method]=np.array([r['accuracy'] for r in runs])
            summaries.append(dict(task=task,method=method,**statistics(arrays[method])))
        for a,b in itertools.combinations(METHODS,2):
            diff=arrays[b]-arrays[a]
            paired.append(dict(task=task,comparison=f'{b} minus {a}',method_a=a,method_b=b,
                               **statistics(diff),wins=int((diff>0).sum()),ties=int((diff==0).sum()),
                               inference='descriptive, no multiple-comparison correction'))
    return summaries,paired

def plot_grid(rows):
    import matplotlib.pyplot as plt
    for task in ('cv','nlp'):
        for method in METHODS:
            records=[r for r in rows if r['task']==task and r['method']==method]
            epsilons=sorted({r['eps_scale'] for r in records},key=lambda v:v or 0.)
            lrs=sorted({r['lr'] for r in records});cs=sorted({r['C'] for r in records})
            fig,axes=plt.subplots(1,len(epsilons),figsize=(4*len(epsilons),3.5),squeeze=False)
            for ax,eps in zip(axes[0],epsilons):
                table=np.full((len(cs),len(lrs)),np.nan)
                for r in records:
                    if r['eps_scale']==eps and r['status']=='completed': table[cs.index(r['C']),lrs.index(r['lr'])]=100*r['accuracy']
                im=ax.imshow(table,aspect='auto',origin='lower')
                ax.set_xticks(range(len(lrs)),[f'{v:g}' for v in lrs]);ax.set_yticks(range(len(cs)),[f'{v:g}' for v in cs])
                ax.set_xlabel('LR');ax.set_ylabel('C');ax.set_title(f'eps_scale={eps}' if eps is not None else 'Standard')
                fig.colorbar(im,ax=ax,label='Internal validation accuracy (%)')
            fig.suptitle(f'{task.upper()} {method} — seed 20261101, epoch 5');fig.tight_layout()
            path=RESULTS/'figures'/f'grid_{task}_{method}.png';path.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(path,dpi=180);plt.close(fig)

def compute_report(rows):
    write_csv(RESULTS/'compute_trials.csv',trial_table(rows))
    summaries=[]
    for task in ('cv','nlp'):
        for stage in ('smoke','search','recheck','final','sweep'):
            group=[r for r in rows if r['task']==task and r['stage']==stage]
            if group: summaries.append(dict(task=task,stage=stage,trials=len(group),
                          worker_hours=sum(r['seconds'] for r in group)/3600,
                          train_hours=sum(r.get('train_seconds',0) for r in group)/3600,
                          max_peak_allocated_GiB=max(r.get('peak_allocated_bytes',0) for r in group)/2**30))
    write_csv(RESULTS/'compute_summary.csv',summaries)
    telemetry=[]
    path=RESULTS/'gpu_telemetry.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            record=json.loads(line)
            for raw in record['raw'].splitlines():
                gpu,util,mem=(int(v.strip()) for v in raw.split(','))
                telemetry.append(dict(time=record['time'],queue=record['queue'],gpu=gpu,utilization_percent=util,memory_MiB=mem))
    write_csv(RESULTS/'gpu_utilization.csv',telemetry)
    if telemetry:
        import matplotlib.pyplot as plt
        origin=min(r['time'] for r in telemetry)
        fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
        utilization=[]
        for gpu in range(4):
            samples=[r for r in telemetry if r['gpu']==gpu]
            if not samples: continue
            hours=[(r['time']-origin)/3600 for r in samples]
            axes[0].plot(hours,[r['utilization_percent'] for r in samples],label=f'GPU {gpu}')
            axes[1].plot(hours,[r['memory_MiB']/1024 for r in samples],label=f'GPU {gpu}')
            utilization.append(dict(gpu=gpu,samples=len(samples),
                               mean_sampled_utilization_percent=float(np.mean([r['utilization_percent'] for r in samples])),
                               peak_sampled_memory_GiB=max(r['memory_MiB'] for r in samples)/1024,
                               sampling='every ~5 seconds while workers active; includes initialization and idle transitions'))
        axes[0].set_ylabel('GPU utilization (%)');axes[0].legend(ncol=4)
        axes[1].set_ylabel('Device memory (GiB)');axes[1].set_xlabel('Hours since first telemetry sample')
        (RESULTS/'figures').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(RESULTS/'figures/gpu_utilization.png',dpi=180);plt.close(fig)
        write_csv(RESULTS/'gpu_utilization_summary.csv',utilization)
    events=RESULTS/'scheduler.jsonl'
    if events.exists():
        import matplotlib.pyplot as plt
        parsed=[json.loads(s) for s in events.read_text().splitlines()];starts={e['trial_id']:e for e in parsed if e['event']=='start'}
        ends=[e for e in parsed if e['event']=='finish'];fig,ax=plt.subplots(figsize=(12,4))
        origin=min(e['time'] for e in starts.values())
        for end in ends:
            start=starts[end['trial_id']]
            ax.broken_barh([((start['time']-origin)/3600,(end['time']-start['time'])/3600)],(start['gpu']-.35,.7),
                          facecolors='tab:blue' if start['task']=='cv' else 'tab:orange',alpha=.55)
        ax.set_xlabel('Hours since first worker launch');ax.set_ylabel('Physical GPU');ax.set_yticks(range(4))
        ax.set_title('Worker occupancy (CV blue / NLP orange; concurrent NLP bars overlap)')
        (RESULTS/'figures').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(RESULTS/'figures/gpu_timeline.png',dpi=180);plt.close(fig)
    return summaries

def generate_report():
    from exp9.audit import full_audit
    from exp9.frozen import load_frozen
    winners=load_frozen()
    grid=json.loads((RESULTS/'grid_results.json').read_text());recheck=json.loads((RESULTS/'recheck_results.json').read_text())
    formal=json.loads((RESULTS/'final_results.json').read_text());sweep=json.loads((RESULTS/'sweep_results.json').read_text())
    assert len(grid)==321 and len(recheck)==56 and len(sweep)==126 and all(r['status']=='completed' for r in sweep)
    summary,paired=summarize(formal)
    save_json(RESULTS/'method_summary.json',summary);write_csv(RESULTS/'main_table.csv',summary)
    save_json(RESULTS/'paired_effects.json',paired);write_csv(RESULTS/'paired_effects.csv',paired)
    lookup={(r['task'],r['method']):r for r in summary}
    workload_table=[dict(task=task,workload=method.removeprefix('dp-adam-').removesuffix('-bandinvmf'),
                         standard_mean=lookup[task,method]['mean'],standard_std=lookup[task,method]['std'],
                         scale_mean=lookup[task,method+'-scale']['mean'],scale_std=lookup[task,method+'-scale']['std'],
                         paired_mean_difference=lookup[task,method+'-scale']['mean']-lookup[task,method]['mean'])
                    for task in ('cv','nlp') for method in MF]
    write_csv(RESULTS/'workload_geometry.csv',workload_table)
    frozen_rows=[dict(task=task,method=m,**winners[task][m]) for task in ('cv','nlp') for m in METHODS]
    write_csv(RESULTS/'frozen_configs.csv',frozen_rows)
    top2=json.loads((RESULTS/'selection_evidence.json').read_text())
    write_csv(RESULTS/'top2.csv',[dict(task=task,method=m,rank=i+1,**r) for task in ('cv','nlp') for m in METHODS for i,r in enumerate(top2[task][m])])
    utility=[]
    for task in ('cv','nlp'):
        for method in METHODS:
            for epsilon in (2,4,8,16):
                runs=[r for r in formal+sweep if r['task']==task and r['method']==method and r['epsilon']==epsilon and r['seed'] in FINAL_SEEDS[:3]]
                assert sorted(r['seed'] for r in runs)==list(FINAL_SEEDS[:3])
                utility.append(dict(task=task,method=method,epsilon=epsilon,protocol='fixed-hyperparameter (selected at epsilon=8)',
                                    **statistics([r['accuracy'] for r in runs])))
    write_csv(RESULTS/'privacy_utility.csv',utility);save_json(RESULTS/'privacy_utility.json',utility)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,task in zip(axes,('cv','nlp')):
        for method in METHODS:
            points=[r for r in utility if r['task']==task and r['method']==method]
            ax.errorbar([r['epsilon'] for r in points],[100*r['mean'] for r in points],
                        yerr=[100*r['std'] for r in points],marker='o',label=method.removeprefix('dp-adam-'))
        ax.set_title(task.upper());ax.set_xticks([2,4,8,16]);ax.set_xlabel('Per-training epsilon');ax.set_ylabel('Official accuracy (%)');ax.legend(fontsize=7)
    fig.suptitle('Fixed hyperparameters selected at epsilon=8; common 3 seeds; error bars: sample std')
    (RESULTS/'figures').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(RESULTS/'figures/privacy_utility.png',dpi=180);plt.close(fig)
    plot_grid(grid);compute_report(grid+recheck+formal+sweep);audit=full_audit()
    save_json(RESULTS/'completion.json',dict(status='completed',search=321,recheck=56,final=140,sweep=126,total=643,audit=audit))
    lines=['# Exp9 paper experiments','','321 fixed-grid + 56 Top-2 review + 140 formal + 126 fixed-hyperparameter sweep trials.',
           'Selection uses only final-epoch internal validation. The 14 frozen configurations and SHA256 manifest precede official evaluations.',
           '', '| Task | Method | Mean accuracy (%) | Sample std (%) | t 95% CI (%) |', '|---|---|---:|---:|---|']
    lines += [f"| {r['task']} | {r['method']} | {100*r['mean']:.3f} | {100*r['std']:.3f} | [{100*r['ci95_low']:.3f}, {100*r['ci95_high']:.3f}] |" for r in summary]
    lines += ['', 'Ten common seeds at epsilon=8; paired differences and all 21 method-pair comparisons per task are in paired_effects.csv. '
                  'Intervals and comparisons are descriptive and have no multiple-comparison correction.',
              '', 'Privacy–Utility uses epsilon={2,4,8,16}, three common seeds, and epsilon=8 frozen LR/C/eps_scale. '
                  'This is a fixed-hyperparameter protocol; other epsilon values were not independently tuned.',
              '', 'Each successful training has its own audited add/remove zero-out GDP budget, delta=1e-5, without sampling amplification. '
                  'Internal validation is treated as public/non-protected; repeated training, selection, and released validation scores '
                  'are not automatically a single epsilon=8 DP procedure. Private validation would require separate protection/accounting.',
              '', 'Evaluation limitations: previous CV experiments selected parameters using official test; NLP official validation '
                  'was previously viewed. These evaluations are not newly blind held-out tests.',
              '', 'Artifacts: [raw final](final_raw.csv), [main table](main_table.csv), [workload/geometry](workload_geometry.csv), '
                  '[paired differences](paired_effects.csv), [privacy–utility](privacy_utility.csv), [freeze](frozen_manifest.json), '
                  '[audit](audit_summary.json), [compute](compute_summary.csv), [GPU samples](gpu_utilization.csv).',
              '', '![Privacy–Utility](figures/privacy_utility.png)', '![GPU timeline](figures/gpu_timeline.png)',
              '![GPU utilization](figures/gpu_utilization.png)']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': generate_report()
