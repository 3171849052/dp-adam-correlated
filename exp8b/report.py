"""Raw accuracies, sample statistics and common-seed Student-t intervals."""
import argparse
import csv
import json
import math
import numpy as np
from scipy.stats import t
from exp8b import RESULTS,ROOT
from exp8b.config import *
from exp8b.audit import audit_trial,audit_pairing


def statistics(values):
    a=np.asarray(values,dtype=float)
    assert len(a)==10 and np.isfinite(a).all()
    mean=float(a.mean());std=float(a.std(ddof=1));se=std/math.sqrt(10)
    radius=float(t.ppf(.975,9))*se
    return dict(raw_accuracy=a.tolist(),mean=mean,sample_std=std,standard_error=se,ci95=[mean-radius,mean+radius],n=10)


def write_csv(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def final_report():
    from exp8b.frozen import load_frozen
    frozen=load_frozen()
    records=[audit_trial(p.parent) for p in (RESULTS/'final/trials').glob('*/summary.json')]
    assert len(records)==70 and {(r['method'],r['seed']) for r in records}=={(m,s) for m in METHODS for s in FINAL_SEEDS}
    assert all(all(r[k]==frozen[r['method']][k] for k in ('lr','C','eps_scale')) for r in records)
    audit_pairing(records)
    lookup={(r['method'],r['seed']):r['accuracy'] for r in records}
    summary={m:statistics([lookup[m,s] for s in FINAL_SEEDS]) for m in METHODS}
    best_method=max(METHODS,key=lambda m:summary[m]['mean'])
    pairs=[(SGD,IID),(MOMENTUM,SGD),(BIAS,MOMENTUM),(SGD_SCALE,SGD),(MOMENTUM_SCALE,MOMENTUM),
           (BIAS_SCALE,BIAS),(BIAS_SCALE,MOMENTUM_SCALE),(best_method,IID)]
    effects=[]
    for i,(a,b) in enumerate(pairs):
        differences=[lookup[a,s]-lookup[b,s] for s in FINAL_SEEDS];st=statistics(differences)
        effects.append(dict(comparison='final best - IID' if i==7 else f'{a} - {b}',method_a=a,method_b=b,
                       mean_difference=st['mean'],sample_std=st['sample_std'],standard_error=st['standard_error'],
                       ci95_low=st['ci95'][0],ci95_high=st['ci95'][1],wins=sum(x>0 for x in differences),n=10,differences=differences))
    fields=('method','seed','lr','C','eps_scale','accuracy','result_dir')
    write_csv(RESULTS/'final_multiseed.csv',[{k:r[k] for k in fields} for r in sorted(records,key=lambda r:(r['seed'],list(METHODS).index(r['method'])))])
    save_json(RESULTS/'method_summary.json',summary);save_json(RESULTS/'paired_effects.json',effects)
    write_csv(RESULTS/'paired_effects.csv',[{k:v for k,v in e.items() if k!='differences'} for e in effects])
    lines=['# Exp8b final report','','70 full trials: 5 epochs, 310 logical steps, physical batch=1000, epsilon=8, delta=1e-5.',
           'Accuracy is evaluated on official SST-2 validation (872 examples). Search used only the held-out 5,349 train examples.',
           'Intervals use Student t(df=9), sample std(ddof=1); accuracy is a fraction. Wins exclude ties.',
           'The final-best comparison is descriptive, chosen by final mean, without multiple-comparison correction; it never changes frozen hyperparameters.',
           '', '| Method | Mean | Sample std | SE | 95% CI |','|---|---:|---:|---:|---|']
    lines += [f"| {m} | {s['mean']:.5f} | {s['sample_std']:.5f} | {s['standard_error']:.5f} | {s['ci95']} |" for m,s in summary.items()]
    lines += ['','## Raw accuracy in seed order 20261011–20261020','']
    lines += [f"- {m}: {s['raw_accuracy']}" for m,s in summary.items()]
    lines += ['','## Paired effects','','| Comparison | Mean | Sample std | 95% CI | Wins / 10 |','|---|---:|---:|---|---:|']
    lines += [f"| {e['comparison']} | {e['mean_difference']:.5f} | {e['sample_std']:.5f} | [{e['ci95_low']:.5f}, {e['ci95_high']:.5f}] | {e['wins']}/10 |" for e in effects]
    lines += ['','## Three workloads × two geometries','','| Workload | Standard mean | Scale mean | Paired Scale − Standard |','|---|---:|---:|---:|']
    for label,m in (('SGD',SGD),('Momentum',MOMENTUM),('Momentum-Bias',BIAS)):
        lines.append(f"| {label} | {summary[m]['mean']:.5f} | {summary[m+'-scale']['mean']:.5f} | {summary[m+'-scale']['mean']-summary[m]['mean']:.5f} |")
    lines += ['','## Frozen provenance','',f"Config SHA256: `{file_hash(RESULTS/'frozen_configs.json')}`",
              f"Manifest SHA256: `{file_hash(RESULTS/'frozen_manifest.json')}`",
              'All trials passed matrix, privacy, training pairing, finite checkpoint/Adam, pretrained, tokenizer, split, and frozen-hash audits.']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': final_report()
