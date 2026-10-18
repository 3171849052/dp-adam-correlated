"""Paired seed statistics in percentage points, with Student-t confidence intervals."""
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from exp14c import RESULTS
from exp14c.config import METHODS, save_json
from exp14c.support import historical, write_csv

def stats(values):
    x=np.asarray(values)*100;n=len(x);sd=float(x.std(ddof=1));mean=float(x.mean());half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    return dict(n=n,mean=mean,sample_sd=sd,ci95_low=mean-half,ci95_high=mean+half)

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    search=json.loads((RESULTS/'search_results.json').read_text());rows=json.loads((RESULTS/'formal_results.json').read_text())
    old=json.loads((historical/'final_results.json').read_text())
    summaries=[];paired=[]
    plots=RESULTS/'plots';plots.mkdir(exist_ok=True)
    text=['# Exp14c: MF + AdaM³ 联合学习率与裁剪阈值搜索',
          '120 次内部验证搜索与 80 次正式训练均完成。优化器使用未偏差修正的一阶矩平方更新二阶矩，β=(0.9,0.999)，eps=1e-8，权重衰减为零。',
          '准确率、标准差、区间及配对差均以百分点表示。区间采用跨 10 个种子的双侧 Student-t 95% 置信区间（df=9），未做多重比较校正。配对差为 AdaM³ − Exp9 Adam。',
          '每次训练分别校准 ε=8、δ=1e-5；不对搜索、选参或重复训练整体声称 ε=8。历史官方评价数据已被使用，本实验不是新的盲测。',
          '两组使用各自选定的超参数，因此比较反映各自调参协议下的整体表现，不能归因为优化器的孤立因果效应。']
    for task in ('cv','nlp'):
        text += [f'\n## {task.upper()}\n', '|机制|lr|C|AdaM³ 均值 ± SD|95% CI|Adam 均值|配对差|配对差 95% CI|','|---|---:|---:|---:|---|---:|---:|---|']
        taskstats=[]
        for method in METHODS:
            group=sorted([r for r in rows if r['task']==task and r['method']==method],key=lambda r:r['seed'])
            refs=[]
            for r in group:
                matched=[o for o in old if o['task']==task and o['method']==method and o['seed']==r['seed'] and o['epsilon']==8]
                assert len(matched)==1 and matched[0]['status']=='completed';o=matched[0];refs.append(o)
                paired.append(dict(task=task,method=method,seed=r['seed'],adam3_accuracy=100*r['accuracy'],adam_accuracy=100*o['accuracy'],difference_pp=100*(r['accuracy']-o['accuracy']),adam3_trial_id=r['trial_id'],exp9_trial_id=o['trial_id']))
            s=stats([r['accuracy'] for r in group]);baseline=stats([r['accuracy'] for r in refs]);delta=stats([r['accuracy']-o['accuracy'] for r,o in zip(group,refs)])
            summaries.append(dict(task=task,method=method,lr=group[0]['lr'],C=group[0]['C'],adam3=s,adam=baseline,paired_delta=delta));taskstats.append(summaries[-1])
            text.append(f"|{method}|{group[0]['lr']:g}|{group[0]['C']:g}|{s['mean']:.3f} ± {s['sample_sd']:.3f}|[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]|{baseline['mean']:.3f}|{delta['mean']:+.3f}|[{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}]|")
            selected=[r for r in search if r['task']==task and r['method']==method];lrs=sorted({r['lr'] for r in selected});cs=sorted({r['C'] for r in selected})
            z=np.array([[next(r['accuracy']*100 for r in selected if r['lr']==lr and r['C']==C) for lr in lrs] for C in cs])
            fig,ax=plt.subplots(figsize=(8,3));im=ax.imshow(z,aspect='auto');ax.set_xticks(range(5),[f'{lr:g}' for lr in lrs]);ax.set_yticks(range(3),[str(C) for C in cs]);ax.set(xlabel='Learning rate',ylabel='Clip C',title=f'{task} {method}: internal validation (%)');fig.colorbar(im,ax=ax)
            for i in range(3):
                for j in range(5): ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',color='white' if z[i,j]<np.median(z) else 'black')
            fig.tight_layout();fig.savefig(plots/f'{task}_{method}_heatmap.png');plt.close(fig)
            fig,axes=plt.subplots(2,3,figsize=(14,7))
            for r in group:
                epochs=pd.DataFrame(r['epochs']);d=RESULTS/r['result_dir'];mechanism=pd.read_csv(d/'mechanism_metrics.csv');opt=pd.read_json(d/'optimizer_metrics.jsonl',lines=True)
                for ax,x,y in [(axes[0,0],epochs.epoch,epochs.accuracy*100),(axes[0,1],epochs.epoch,epochs.train_loss),(axes[0,2],mechanism.step,mechanism.clip_fraction),(axes[1,0],opt.step,opt.update_norm),(axes[1,1],opt.step,opt.v_mean),(axes[1,2],opt.step,opt.vhat_mean)]: ax.plot(x,y,alpha=.5)
            for ax,title in zip(axes.flat,['Official accuracy (%)','Train loss','Clip fraction','Update norm','Mean v','Mean bias-corrected v']): ax.set_title(title);ax.set_xlabel('Epoch' if ax in axes[0,:2] else 'Step')
            fig.suptitle(f'{task} {method}: all 10 seeds');fig.tight_layout();fig.savefig(plots/f'{task}_{method}_diagnostics.png');plt.close(fig)
        best=max(taskstats,key=lambda s:s['adam3']['mean'])
        text += ['',f"AdaM³ 正式均值最高的机制为 {best['method']}（{best['adam3']['mean']:.3f}%）；这只是冻结配置后的描述性比较，不用于重新选参。"]
        for s in taskstats:
            d=s['paired_delta'];finding='区间全部高于零' if d['ci95_low']>0 else '区间全部低于零' if d['ci95_high']<0 else '区间包含零'
            text.append(f"- {s['method']}：相对 Adam 的配对差 {d['mean']:+.3f} pp，{finding}。")
        for method in METHODS:
            text += [f'![{method} 搜索](plots/{task}_{method}_heatmap.png)',f'![{method} 诊断](plots/{task}_{method}_diagnostics.png)']
    save_json(RESULTS/'statistics.json',summaries);write_csv(RESULTS/'paired_differences.csv',paired)
    write_csv(RESULTS/'statistics.csv',[dict(task=s['task'],method=s['method'],lr=s['lr'],C=s['C'],**{f'{group}_{k}':v for group in ('adam3','adam','paired_delta') for k,v in s[group].items()}) for s in summaries])
    text += ['\n## 审计与证据\n','见 `audit.json`、`frozen_manifest.json`、`search_ranking.csv`、`statistics.csv`、`paired_differences.csv` 和 `scheduler.jsonl`。严格配对检查包括初始化、数据顺序、增强/Dropout 随机轨迹及 MF 矩阵；噪声创新生成器同 seed+1，噪声幅度随各自 C 校准。',
             'CV 内部搜索 45,000/5,000，正式训练 50,000/10,000；NLP 内部搜索 62,000/5,349，正式训练使用同一 62,000 训练样本与官方 872 条评价样本。各次逐轮指标、裁剪比例、更新范数、二阶矩和检查点保存在对应 trial 目录。']
    (RESULTS/'final_report.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__': main()
