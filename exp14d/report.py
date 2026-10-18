"""Ten-seed summaries, paired Student-t intervals and reproducible figures."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from exp14d import ROOT, RESULTS
from exp14d.config import METHODS, FINAL_SEEDS, file_hash, save_json
from exp14d.support import historical_rows, write_csv
LABELS=('IID + Adam','MF-SGD + Scale + Adam','MF-SGD + Adam','MF-SGD + AdaM³','MF-Momentum + AdaM³','MF-Momentum-Bias + AdaM³')

def stats(values):
    a=np.asarray(values,dtype=float);n=len(a);s=float(a.std(ddof=1));m=float(a.mean());h=float(t.ppf(.975,n-1)*s/np.sqrt(n))
    return dict(n=n,mean=m,sample_std=s,ci95_low=m-h,ci95_high=m+h)

def directory(r):
    return Path(r['source_summary']).parent if r.get('source_summary') else RESULTS/r['result_dir']

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    b=json.loads((RESULTS/'B_formal_results.json').read_text());a=json.loads((RESULTS/'A_formal_results.json').read_text())
    exp9=historical_rows('exp9','final');exp14c=historical_rows('exp14c','final')
    original=[r for r in exp9 if r['method'] in METHODS[:3]]+[r for r in exp14c if r['method'] in METHODS[3:]]
    updated=[r for r in exp9 if r['method'] in METHODS[:3]]+[r for r in a if r['method'] in METHODS[3:]]
    sets={'original':original,'boundary_updated':updated,'C1':b};summaries=[];pairs=[];sources={};diagnostics=[]
    def group(rows,task,m):
        g=sorted([r for r in rows if r['task']==task and r['method']==m],key=lambda r:r['seed'])
        assert [r['seed'] for r in g]==list(FINAL_SEEDS)
        return g
    for name,rows in sets.items():
        for task in ('cv','nlp'):
            for m,label in zip(METHODS,LABELS):
                g=group(rows,task,m)
                summaries.append(dict(configuration=name,task=task,method=label,lr=g[0]['lr'],C=g[0]['C'],eps_scale=g[0].get('eps_scale'),**stats([r['accuracy'] for r in g])))
    seed_pairs=[]
    def paired(x,y,contrast,task,method):
        differences=[u['accuracy']-v['accuracy'] for u,v in zip(x,y)]
        seed_pairs.extend(dict(contrast=contrast,task=task,method=method,seed=u['seed'],left_accuracy=u['accuracy'],right_accuracy=v['accuracy'],difference=d) for u,v,d in zip(x,y,differences))
        return dict(contrast=contrast,task=task,method=method,**stats(differences))
    for task in ('cv','nlp'):
        for m,label in zip(METHODS[3:],LABELS[3:]):
            pairs.append(paired(group(updated,task,m),group(original,task,m),'boundary_updated minus original',task,label))
    for reference in ('original','boundary_updated'):
        for task in ('cv','nlp'):
            for m,label in zip(METHODS,LABELS):
                x=group(b,task,m);y=group(sets[reference],task,m)
                pairs.append(paired(x,y,'C1 minus '+reference,task,label))
    # Method contrasts remain descriptive: independently selected learning rates.
    for name,rows in sets.items():
        for task in ('cv','nlp'):
            for i,j in ((2,0),(1,2),(3,2),(4,3),(5,4)):
                x=group(rows,task,METHODS[i]);y=group(rows,task,METHODS[j])
                pairs.append(paired(x,y,name+': '+LABELS[i]+' minus '+LABELS[j],task,'method comparison'))
    unique={str(directory(r)):r for rows in sets.values() for r in rows}
    for path,r in unique.items():
        d=Path(path)
        for n in ('summary.json','config.json','metrics.csv','mechanism_metrics.csv','optimizer_metrics.jsonl'):
            if (d/n).exists():sources[str(d/n)]=file_hash(d/n)
        mechanisms=pd.read_csv(d/'mechanism_metrics.csv')
        row=dict(task=r['task'],method=r['method'],seed=r['seed'],source=path,clip_fraction=float(mechanisms.clip_fraction.mean()),lr=r['lr'])
        if (d/'optimizer_metrics.jsonl').exists():
            opt=pd.read_json(d/'optimizer_metrics.jsonl',lines=True)
            for k in ('update_norm','v_mean','vhat_mean'):row[k+'_sampled_mean']=float(opt[k].mean())
        else:row['optimizer_diagnostics']='not recorded historically; no retraining'
        diagnostics.append(row)
    write_csv(RESULTS/'paired_seed_differences.csv',seed_pairs)
    write_csv(RESULTS/'six_algorithm_comparison.csv',summaries);write_csv(RESULTS/'paired_differences.csv',pairs)
    write_csv(RESULTS/'diagnostic_summary.csv',diagnostics);save_json(RESULTS/'report_sources_sha256.json',sources)
    figs=RESULTS/'figures';figs.mkdir(exist_ok=True)
    for task in ('cv','nlp'):
        fig,ax=plt.subplots(figsize=(12,5))
        for k,name in enumerate(sets):
            rs=[r for r in summaries if r['task']==task and r['configuration']==name]
            x=np.arange(6)+(k-1)*.23
            ax.bar(x,[r['mean'] for r in rs],.23,label=name,yerr=[r['ci95_high']-r['mean'] for r in rs],capsize=3)
        ax.set_xticks(np.arange(6),LABELS,rotation=20,ha='right');ax.set_ylabel('Official accuracy (95% t CI)');ax.legend();fig.tight_layout();fig.savefig(figs/f'{task}_comparison.png');plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for m,label in zip(METHODS,LABELS):
            g=group(b,task,m)
            axes[0,0].plot(range(1,6),np.mean([[e['accuracy'] for e in r['epochs']] for r in g],axis=0),label=label)
            for ax,key in ((axes[0,1],'clip_fraction'),):
                frames=[pd.read_csv(directory(r)/'mechanism_metrics.csv') for r in g];ax.plot(frames[0]['step'],np.mean([f[key] for f in frames],axis=0),label=label)
            frames=[pd.read_json(directory(r)/'optimizer_metrics.jsonl',lines=True) for r in g]
            for ax,key in ((axes[0,2],'update_norm'),(axes[1,0],'v_mean'),(axes[1,1],'vhat_mean'),(axes[1,2],'lr')):
                ax.plot(frames[0]['step'],np.mean([f[key] for f in frames],axis=0),label=label)
                if key!='lr':ax.set_yscale('log')
        for ax,title in zip(axes.flat,('Official accuracy','Clipping fraction','Update norm','Second moment v','Bias-corrected v','Constant learning rate')):ax.set_title(title);ax.set_xlabel('Epoch' if ax is axes[0,0] else 'Logical step')
        axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(figs/f'{task}_C1_training.png');plt.close(fig)
    changes=json.loads((RESULTS/'A_config_changes.json').read_text())
    lines=['# Exp14d 最终报告','', '所有结果已完成；统计使用 10 个固定种子、样本标准差（ddof=1）和双侧 Student-t 95% 置信区间。配对差按同一种子计算，单位为准确率（0–1）。区间未作多重比较校正。','',
           '隐私 ε=8、δ=1e-5 仅针对单次训练；不宣称搜索、选择和重复训练合成后仍为 ε=8。历史官方评价集已经使用过，本实验不是新的盲测。所有选参只使用最后一轮内部验证准确率，依次以验证损失和固定 trial ID 打破并列。','',
           '普通 C=1 裁剪约束欧氏梯度范数；Scale 的 C=1 约束上一时刻 vhat 所定义的缩放梯度范数，噪声在该空间加入后逆缩放。两者几何不等价，因此不能把 Scale 差异归结为相同欧氏阈值下的收益。CV/NLP eps_scale 分别固定为 0.05/0.01。','',
           '训练严格复用 Exp9/Exp14c：CV 搜索 45,000 条、正式 50,000 条；NLP 搜索和正式均为历史固定的 62,000 条训练子集（其余 5,349 条为内部验证），每轮 62 步，并非把全部 67,349 条原始样本重新纳入训练。','', '## A：学习率边界补充','']
    for c in changes:
        if c['method'] in METHODS[3:]:lines.append(f"- {c['task']} {c['method']}: ({c['original']['lr']}, C={c['original']['C']}) → ({c['updated']['lr']}, C={c['updated']['C']}); {'重跑全部十种子' if c['changed'] else '引用 Exp14c 十种子结果'}。")
    lines+=['','边界搜索仅扩展预先指定的八个候选，不证明连续学习率空间的全局最优。合并排名及来源哈希见 A_search_ranking.csv、A_frozen_manifest.json。','', '## 六种算法：原始配置、边界更新、C=1','', '|任务|方法|配置|lr|C|均值|样本标准差|95% CI|','|---|---|---|---|---|---|---|---|']
    for r in summaries:lines.append(f"|{r['task']}|{r['method']}|{r['configuration']}|{r['lr']}|{r['C']}|{r['mean']:.5f}|{r['sample_std']:.5f}|[{r['ci95_low']:.5f}, {r['ci95_high']:.5f}]|")
    lines+=['','## 配对差与相对收益','']
    for r in pairs:
        direction='正向' if r['ci95_low']>0 else '负向' if r['ci95_high']<0 else '区间包含零'
        lines.append(f"- {r['task']} {r['method']}，{r['contrast']}: {r['mean']:+.5f} [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}]（{direction}）。")
    lines+=['','C=1 对照同时改变裁剪阈值及独立选择的学习率；算法对照同样使用各自最优学习率，因此属于完整训练配置比较，不能单独解释为某个机制的因果效应。MF、Scale、AdaM³ 的相对表现以上述配对差为依据。','',
            '诊断曲线见 figures/，裁剪比例来自每一步；更新范数、二阶矩和学习率记录前 20 步及之后每 10 步。历史未记录的优化器诊断明确标为缺失，不重跑补齐。所有新增训练保留检查点、训练曲线和噪声校准记录。','',
            '来源与审计：historical_sources_sha256.json、report_sources_sha256.json、source_provenance.json、scheduler.jsonl、failures.jsonl、final_audit.json。']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
