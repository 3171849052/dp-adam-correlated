"""Conditional fixed-history risks; independent historical and current Gaussian draws."""
import csv
import os
import time
import numpy as np
import torch
from exp17b.common import RESULTS, PROTOCOL, save, read, sha
from exp17b.optimizer import directions, scalar_f


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run():
    identity=sha(RESULTS/'protocol_freeze.json')
    marker=RESULTS/'mc_complete.json'
    if marker.exists():
        done=read(marker)
        assert done['identity']==identity
        for name,value in done['hashes'].items(): assert sha(RESULTS/name)==value
        return
    torch.set_num_threads(2)
    spec=PROTOCOL['mc'];nu2=read(RESULTS/'protocol_freeze.json')['calibration']['search']['nu_squared'];nu=nu2**.5
    rows=[];paired=[];histories=[];oracle=[];start=time.monotonic()
    # Separate generators: no history draw can consume the current-noise stream.
    history_rng=np.random.default_rng(spec['seed']);current_rng=np.random.default_rng(spec['seed']+1)
    for path in spec['paths']:
        for t in spec['t']:
            for snr in spec['snr']:
                g=snr*nu
                for h in range(spec['histories']):
                    j=np.arange(1,t,dtype=float)
                    clean={'constant':np.full(t-1,g),'alternating':g*(-1.)**(t-j),
                           'varying':g*(.25+1.5*j/max(t-1,1))+.5*nu*np.sin(j/3),
                           'sparse':g*(j.astype(int)%5==0)}[path]
                    historical=clean+history_rng.normal(0,nu,t-1)
                    m=v=0.
                    for value in historical: m=.9*m+.1*value;v=.999*v+.001*value*value
                    scene=dict(path=path,t=t,snr=snr,history=h)
                    histories.append(dict(**scene,g=g,m=m,v=v,clean=clean.tolist(),private_history=historical.tolist()))
                    x=torch.from_numpy(g+current_rng.normal(0,nu,spec['samples']))
                    mt,vt=torch.tensor(m,dtype=torch.float64),torch.tensor(v,dtype=torch.float64)
                    B,D=.1/(1-.9**t),.001/(1-.999**t)
                    for r in spec['r']:
                        target=scalar_f(torch.tensor(g,dtype=torch.float64),.9*m/(1-.9**t),B,.999*v/(1-.999**t),D,r,nu2).item()
                        base=directions('smooth',x,mt,vt,t,r,nu2,0)[0]
                        base_err=(base-target).square();baseline=base_err.mean().item()
                        corrections={method:directions(method,x,mt,vt,t,r,nu2,1)[1] for method in (['gh'] if r==0 else ['lc','gh'])}
                        for method,c in corrections.items():
                            denominator=c.square().mean().item()
                            optimal=((base-target)*c).mean().item()/denominator if denominator else 0.
                            oracle.append(dict(**scene,r=r,method=method,alpha_unconstrained=optimal,alpha_clipped=float(np.clip(optimal,0,1))))
                        for method in ['smooth']+list(corrections):
                            for alpha in ([0.] if method=='smooth' else PROTOCOL['alphas']):
                                y=base if method=='smooth' else base-alpha*corrections[method]
                                err=(y-target).square();delta=err-base_err
                                bias=y.mean().item()-target;mse=err.mean().item()
                                se=delta.std(unbiased=True).item()/spec['samples']**.5
                                diff=delta.mean().item()
                                rows.append(dict(**scene,r=r,method=method,alpha=alpha,n=spec['samples'],target=target,
                                    bias=bias,bias2=bias*bias,variance=y.var(unbiased=False).item(),mse=mse,
                                    mse_ratio=mse/baseline,paired_mse_difference=diff,paired_se=se,
                                    paired_ci_low=diff-1.96*se,paired_ci_high=diff+1.96*se))
                        if 'lc' in corrections:
                            for alpha in PROTOCOL['alphas']:
                                delta=(base-alpha*corrections['gh']-target).square()-(base-alpha*corrections['lc']-target).square()
                                diff=delta.mean().item();se=delta.std().item()/spec['samples']**.5
                                paired.append(dict(**scene,r=r,alpha=alpha,comparison='gh-lc',difference=diff,se=se,ci_low=diff-1.96*se,ci_high=diff+1.96*se))
            save(RESULTS/'mc_progress.json',dict(path=path,t=t,rows=len(rows),seconds=time.monotonic()-start))
            print('Monte Carlo',path,t,len(rows),flush=True)
    write_csv(RESULTS/'mse_summary.csv',rows);write_csv(RESULTS/'mc_lc_gh_paired.csv',paired)
    write_csv(RESULTS/'mc_oracle_alpha.csv',oracle);save(RESULTS/'mc_histories.json',histories)
    keys=('path','t','snr','r','method','alpha');groups={}
    for row in rows: groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    aggregate=[dict(zip(keys,key),**{k:float(np.mean([v[k] for v in values])) for k in ('bias','bias2','variance','mse','mse_ratio')}) for key,values in groups.items()]
    write_csv(RESULTS/'mse_aggregate.csv',aggregate)
    best={}
    for row in rows:
        key=tuple(row[k] for k in ('path','t','snr','history','r','method'))
        if key not in best or row['mse']<best[key]['mse']: best[key]=row
    write_csv(RESULTS/'mc_best_alpha.csv',list(best.values()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,r in zip(axes,spec['r']):
        for method in ['lc','gh']:
            selected=[v for v in rows if v['r']==r and v['method']==method]
            if not selected:continue
            ax.plot(PROTOCOL['alphas'],[np.mean([v['mse'] for v in selected if v['alpha']==a]) for a in PROTOCOL['alphas']],marker='o',label=method)
        ax.set(title=f'r={r:.4g}',xlabel='alpha',ylabel='Mean MSE');ax.legend()
    fig.tight_layout();fig.savefig(RESULTS/'mse_alpha.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,r in zip(axes,spec['r']):
        for method in ['lc','gh']:
            for a in [.25,.5,1]:
                selected=[v for v in rows if v['r']==r and v['method']==method and v['alpha']==a]
                if selected:ax.plot(spec['snr'],[np.mean([v['mse_ratio'] for v in selected if v['snr']==s]) for s in spec['snr']],label=f'{method} {a}')
        ax.axhline(1,color='gray',linestyle='--');ax.set(title=f'r={r:.4g}',xlabel='|g| / nu',ylabel='MSE / Smooth MSE');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(RESULTS/'mse_snr.png');plt.close(fig)
    names=['mse_summary.csv','mse_aggregate.csv','mc_lc_gh_paired.csv','mc_oracle_alpha.csv','mc_best_alpha.csv','mc_histories.json','mse_alpha.png','mse_snr.png']
    save(marker,dict(identity=identity,rows=len(rows),seconds=time.monotonic()-start,hashes={n:sha(RESULTS/n) for n in names}))

if __name__=='__main__':run()
