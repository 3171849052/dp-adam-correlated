"""Reports only verified completed results; paired seed-level Student-t intervals."""
import csv
import json
import numpy as np
from scipy.stats import t
from exp17.common import RESULTS, METHODS, PROTOCOL, save, read, sha, success, output


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(values):
    x=np.asarray(values);n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1))
    half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    return dict(n=n,mean=mean,std=sd,ci95_low=mean-half,ci95_high=mean+half)


def report():
    identity=sha(RESULTS/'protocol_freeze.json');all_results=[];counts={}
    for phase in ('coarse','fine','review','final','ablation'):
        path=RESULTS/(phase+'_plan.json')
        if not path.exists():continue
        results=[success(c,identity) for c in read(path)]
        if any(r is None for r in results): raise RuntimeError('Cannot report incomplete phase '+phase)
        all_results+=results;counts[phase]=len(results)
    finals=[r for r in all_results if r['config']['stage']=='final']
    assert len(finals)==40
    summary=[];pairs=[]
    for method in METHODS:
        rows=[r for r in finals if r['config']['method']==method]
        assert sorted(r['config']['seed'] for r in rows)==PROTOCOL['final_seeds']
        for metric in ('accuracy','validation_loss','train_seconds','peak_allocated_bytes'):
            summary.append(dict(method=method,metric=metric,**stats([r[metric] for r in rows])))
    for a,b in (('lc','adam'),('lc','bc'),('hc','adam'),('hc','bc'),('hc','lc')):
        left={r['config']['seed']:r for r in finals if r['config']['method']==a}
        right={r['config']['seed']:r for r in finals if r['config']['method']==b}
        for metric in ('accuracy','validation_loss'):
            pairs.append(dict(comparison=a+' - '+b,metric=metric,
                **stats([left[s][metric]-right[s][metric] for s in PROTOCOL['final_seeds']])))
    diagnostics=[];curves=[]
    for r in finals:
        rows=[json.loads(line) for line in (output(r['config'])/'diagnostics.jsonl').read_text().splitlines()]
        assert len(rows)==250 and all(row['finite'] and row['noise_draws']==row['step'] for row in rows)
        for row in rows:
            curves.append(dict(method=r['config']['method'],seed=r['config']['seed'],**row))
        diagnostics.append(dict(method=r['config']['method'],seed=r['config']['seed'],
            mean_correction_norm=float(np.mean([x['correction_norm'] for x in rows])),
            mean_direction_cosine=float(np.mean([x['direction_cosine'] for x in rows])),
            mean_update_norm=float(np.mean([x['update_norm'] for x in rows])),finite=True))
    for r in all_results:
        expected=250 if r['config']['stage'] in ('final','ablation') else 225
        assert r['steps']==expected and 7.999<=r['actual_epsilon']<=8
    ablations=[]
    for m in ('lc','hc'):
        ab=[r for r in all_results if r['config']['stage']=='ablation' and r['config']['method']==m]
        if ab:
            assert len(ab)==10
            ablations.append(dict(method=m,kind='same smoothing, correction disabled; extra budget',
                                  **stats([r['accuracy'] for r in ab])))
    write_csv(RESULTS/'utility.csv',summary);write_csv(RESULTS/'paired_differences.csv',pairs)
    write_csv(RESULTS/'diagnostics.csv',diagnostics);write_csv(RESULTS/'training_curves.csv',curves)
    write_csv(RESULTS/'ablations.csv',ablations)
    write_csv(RESULTS/'runs.csv',[dict(method=r['config']['method'],stage=r['config']['stage'],seed=r['config']['seed'],
        accuracy=r['accuracy'],test_or_validation_loss=r['validation_loss'],seconds=r['seconds'],train_seconds=r['train_seconds'],
        peak_allocated_bytes=r['peak_allocated_bytes'],actual_epsilon=r['actual_epsilon']) for r in all_results])
    audit=dict(status='passed',actual_runs=counts,main_total=sum(v for k,v in counts.items() if k!='ablation'),
        additional_ablation_total=counts.get('ablation',0),duplicate_reuse=0,
        identities_and_artifacts_verified=True,all_final_seeds_paired=True,
        privacy=PROTOCOL['privacy'],theory=PROTOCOL['theory'],
        selection_actual_seed_counts=read(RESULTS/'final_freeze.json')['selection'])
    assert audit['main_total']==248
    save(RESULTS/'final_audit.json',audit)
    lines=['# Exp17: completed observations','',
        'Accuracy values are fractions; confidence intervals use Student-t across ten seeds. Paired intervals are unadjusted for multiplicity.','',
        '| Method | Accuracy mean | SD | 95% CI |','|---|---:|---:|---|']
    for row in summary:
        if row['metric']=='accuracy': lines.append(f"| {row['method']} | {row['mean']:.6f} | {row['std']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    lines+=['','## Paired accuracy differences','', '| Comparison | Mean difference | 95% CI |','|---|---:|---|']
    for row in pairs:
        if row['metric']=='accuracy': lines.append(f"| {row['comparison']} | {row['mean']:.6f} | [{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] |")
    lines+=['','## Audit and interpretation','',json.dumps(counts),PROTOCOL['privacy'],PROTOCOL['theory'],
        'Official test was opened only after all four final configs were frozen. Historical experiments have used this test set; this is not a newly blinded benchmark.',
        'Detailed validation repetition counts: final_freeze.json. Per-step losses, corrections, cosine and norm diagnostics: training_curves.csv. Runtime, memory and cross entropy: runs.csv/utility.csv.',
        'Calibration and environment: protocol_freeze.json. Monte Carlo bias/variance/MSE and measured slopes: monte_carlo.json.',
        'Ablations: '+('completed; see ablations.csv (20 additional trainings, excluded from main rankings).' if ablations else 'pending; no ablation conclusions yet.')]
    (RESULTS/'report.md').write_text('\n'.join(lines)+'\n')
    # Standard standalone figure artifacts, restricted to completed measurements.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for method in METHODS:
        values=np.array([[r['train_loss'] for r in curves if r['method']==method and r['seed']==s] for s in PROTOCOL['final_seeds']])
        ax.plot(np.arange(1,251),values.mean(0),label=method)
    ax.set(xlabel='Logical optimizer step',ylabel='Mean sampled training cross entropy');ax.legend();fig.tight_layout()
    fig.savefig(RESULTS/'training_loss.png',dpi=160);plt.close(fig)

if __name__=='__main__':report()
