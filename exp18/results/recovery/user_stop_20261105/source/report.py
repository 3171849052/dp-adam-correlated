import csv,numpy as np
from scipy.stats import t
from exp18 import RESULTS
from exp18.common import GROUPS,save

def stats(values):
    x=np.asarray(values);mean=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return dict(mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half,n=len(x))
def report(rows):
    assert len(rows)==30
    summaries=[];paired=[]
    for g,methods in GROUPS.items():
        by={m:{r['config']['seed']:r['metrics']['accuracy'] for r in rows if r['config']['method']==m} for m in methods}
        for m in methods:
            assert set(by[m])==set(range(20261111,20261116));summaries.append(dict(method=m,**stats(list(by[m].values()))))
        paired.append(dict(group=g,direction='B2LT minus IID',**stats([by[methods[1]][s]-by[methods[0]][s] for s in by[methods[0]]])))
    for name,data in [('formal_statistics',summaries),('paired_statistics',paired)]:
        with (RESULTS/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# Exp18 final report','', 'MNIST 54000/6000/10000, CNN from scratch, 3 epochs, batch 1000, FP32.', '', 'Each training run: epsilon=3, delta=1e-5, GDP without sampling amplification. This is NOT a composed epsilon=3 guarantee for tuning, validation or repeated runs. Validation is treated as public; raw loss and clipping diagnostics are internal audit artifacts.', '', 'Five-seed sample SD and Student-t 95% intervals; paired differences share initialization, order and innovation RNG seeds. No multiple-comparison correction. Matrix error describes the same-query reference, not a counterfactual adaptive trajectory.', '', '| Method | Mean | SD | 95% CI |','|---|---:|---:|---|']
    lines += [f"| {r['method']} | {r['mean']:.6f} | {r['sd']:.6f} | [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] |" for r in summaries]
    lines+=['','Paired B2LT − IID:']+[f"- {r['group']}: {r['mean']:.6f}, 95% CI [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]" for r in paired]
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n');save(RESULTS/'final_audit.json',dict(passed=True,formal_trials=30,statistics=summaries,paired=paired))
