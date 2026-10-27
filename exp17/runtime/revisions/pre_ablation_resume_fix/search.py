"""Frozen logarithmic designs and deterministic refinement/selection."""
import numpy as np
from exp17.common import PROTOCOL, METHODS, digest


def key(cfg): return digest({k:cfg[k] for k in ('method','lr','r') if k in cfg})
def config(method,lr,r=None):
    c=dict(method=method,lr=float(lr))
    if r is not None: c['r']=float(r)
    return c

def coarse():
    trials=[]
    for method in METHODS:
        lrs=np.geomspace(*PROTOCOL['lr_range'],12 if method=='adam' else 9)
        rs=[None] if method=='adam' else np.geomspace(*PROTOCOL['r_ranges'][method],4)
        seeds=PROTOCOL['search_seeds'] if method=='adam' else PROTOCOL['search_seeds'][:1]
        for lr in lrs:
            for r in rs:
                for seed in seeds: trials.append(dict(config(method,lr,r),stage='search',round='coarse',seed=seed))
    return trials

def rank(results,method):
    grouped={}
    for result in results:
        c=result['config']
        if c['method']==method: grouped.setdefault(key(c),[]).append(result)
    ranked=[]
    for k,group in grouped.items():
        c=group[0]['config'];seeds=[v['config']['seed'] for v in group]
        if len(set(seeds))!=len(seeds): raise ValueError('Repeated seed counted as independent')
        ranked.append(dict(config={z:c[z] for z in ('method','lr','r') if z in c},key=k,
            accuracy=float(np.mean([v['accuracy'] for v in group])),
            validation_loss=float(np.mean([v['validation_loss'] for v in group])),seeds=seeds,n=len(group)))
    return sorted(ranked,key=lambda r:(-r['accuracy'],r['validation_loss'],r['key']))

def neighborhood(value,coarse_values,n,exclude_center=False):
    values=np.log(np.asarray(coarse_values));x=np.log(value);i=int(np.argmin(abs(values-x)))
    lo=(values[i-1]+x)/2 if i>0 else values[0]
    hi=(values[i+1]+x)/2 if i+1<len(values) else values[-1]
    points=np.linspace(lo,hi,n+2)[1:-1]
    if exclude_center:
        for j,p in enumerate(points):
            if abs(p-x)<1e-12: points[j]=lo+(hi-lo)*.375
    assert len(set(points))==n
    return np.exp(points)

def fine(results):
    trials=[]
    for method in METHODS:
        best=rank(results,method)[0]['config']
        lrs=neighborhood(best['lr'],np.geomspace(*PROTOCOL['lr_range'],12 if method=='adam' else 9),4)
        rs=[None] if method=='adam' else neighborhood(best['r'],np.geomspace(*PROTOCOL['r_ranges'][method],4),3,True)
        seeds=PROTOCOL['search_seeds'] if method=='adam' else PROTOCOL['search_seeds'][:1]
        for lr in lrs:
            for r in rs:
                for seed in seeds: trials.append(dict(config(method,lr,r),stage='search',round='fine',seed=seed))
    existing={key(v['config']) for v in results}
    assert all(key(c) not in existing for c in trials)
    assert all(sum(c['method']==m for c in trials)==12 for m in METHODS)
    return trials
