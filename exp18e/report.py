"""Five-seed descriptive statistics and seed-paired B²LT minus IID differences."""
import math
import numpy as np
from scipy.stats import t
from exp18e.common import BASE,ROOT,GROUPS,SEEDS,configs,outdir,completed,read,save,sha,now,write_csv


def stats(values):
    n=len(values)
    if not n:return dict(n=0,mean=None,sd=None,ci95_low=None,ci95_high=None)
    mean=float(np.mean(values));sd=float(np.std(values,ddof=1)) if n>1 else None
    half=float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-half if half is not None else None,
                ci95_high=mean+half if half is not None else None)

def report():
    rows=[];pairing={};results={};missing=[]
    for cfg in configs():
        result=completed(cfg);out=outdir(cfg)
        if result is None:missing.append(str(out.relative_to(BASE)));continue
        assert result['formal'] and result['steps']==310 and result['evaluation']=='official_validation'
        assert result['metrics']['examples']==872
        assert result['privacy']==read(BASE/'formal_privacy_audit.json')[cfg['method']]
        seed=cfg['seed']
        if seed in pairing:assert result['pairing']==pairing[seed],f'Unpaired seed {seed}'
        pairing[seed]=result['pairing'];results[cfg['method'],seed]=result
        rows.append(dict(method=cfg['method'],seed=seed,**{p:cfg[p] for p in ('lr','clip','eps')},
            **result['metrics'],steps=310,epsilon=result['privacy']['epsilon'],delta=1e-5,
            seconds=result['seconds'],train_seconds=result['train_seconds'],gpu_uuid=result['gpu_uuid'],
            peak_allocated_mib=result['peak_allocated_bytes']/1048576,
            peak_reserved_mib=result['peak_reserved_bytes']/1048576,directory=str(out.relative_to(BASE))))
    fields=['method','seed','lr','clip','eps','accuracy','loss','examples','steps','epsilon','delta',
            'seconds','train_seconds','gpu_uuid','peak_allocated_mib','peak_reserved_mib','directory']
    write_csv(BASE/'per_seed.csv',rows,fields)
    summaries=[]
    for methods in GROUPS.values():
        for m in methods:
            selected=[r for r in rows if r['method']==m]
            s=dict(method=m,completed_seeds=len(selected),expected_seeds=5,
                   status='complete' if len(selected)==5 else 'incomplete',epsilon=8,delta=1e-5)
            for metric in ('accuracy','loss','seconds','peak_reserved_mib'):
                s.update({f'{metric}_{k}':v for k,v in stats([r[metric] for r in selected]).items() if k!='n'})
            summaries.append(s)
    write_csv(BASE/'summary.csv',summaries,list(summaries[0]))
    pairs=[];paired_summary=[]
    for group,(iid,b2lt) in GROUPS.items():
        subset=[]
        for seed in SEEDS:
            if (iid,seed) not in results or (b2lt,seed) not in results:continue
            a,b=results[iid,seed],results[b2lt,seed]
            row=dict(group=group,seed=seed,iid_method=iid,b2lt_method=b2lt,
                accuracy_iid=a['metrics']['accuracy'],accuracy_b2lt=b['metrics']['accuracy'],
                accuracy_difference=b['metrics']['accuracy']-a['metrics']['accuracy'],
                loss_iid=a['metrics']['loss'],loss_b2lt=b['metrics']['loss'],
                loss_difference=b['metrics']['loss']-a['metrics']['loss'])
            pairs.append(row);subset.append(row)
        paired_summary.append(dict(group=group,accuracy=stats([r['accuracy_difference'] for r in subset]),
            loss=stats([r['loss_difference'] for r in subset])))
    write_csv(BASE/'paired_differences.csv',pairs,['group','seed','iid_method','b2lt_method','accuracy_iid',
        'accuracy_b2lt','accuracy_difference','loss_iid','loss_b2lt','loss_difference'])
    save(BASE/'paired_statistics.json',paired_summary)
    complete=len(rows)==30
    save(BASE/'integrity_audit.json',dict(complete=complete,expected=30,completed=len(rows),missing=missing,
        seeds=SEEDS,pairing_verified=True,artifacts_verified=True,time=now()))
    freeze=read(BASE/'selection_freeze.json');counts=read(BASE/'search/counts.json')
    lines=['# Exp18e 实验报告','',f'状态：{"已完成" if complete else "尚未完成"}，正式训练完成 {len(rows)}/30。',
        '',f'共同 STP-Ex2 参数：{freeze["parameters"]}。搜索训练共 {counts["planned_trainings"]} 次，全部采用内部验证集排序。',
        'SST-2 / BERT-Tiny 全参数微调；FP32，TF32关闭，长度128；5轮×62步=310步，批次1000，累积1，权重衰减0。',
        '62000训练、5349内部验证；872官方验证仅在参数冻结后用于正式训练最终评价。',
        '正式种子为20261111、20261112、20261113、20261114、20261115；每次从本地预训练参数重新初始化。',
        '', '|方法|完成种子|准确率均值 ± 样本SD|准确率95% CI|损失均值|', '|---|---:|---:|---:|---:|']
    for r in summaries:
        if r['completed_seeds']>1:
            acc=f"{r['accuracy_mean']:.6f} ± {r['accuracy_sd']:.6f}"
            ci=f"[{r['accuracy_ci95_low']:.6f}, {r['accuracy_ci95_high']:.6f}]"
        else:acc=ci='待完成'
        loss=f"{r['loss_mean']:.6f}" if r['completed_seeds'] else '待完成'
        lines.append(f"|{r['method']}|{r['completed_seeds']}/5|{acc}|{ci}|{loss}|")
    lines += ['', '样本标准差使用ddof=1；准确率及损失区间均为Student-t 95% CI。',
        'paired_differences.csv 保存每个种子的 B²LT − IID 准确率/损失差；paired_statistics.json 保存配对差的均值、样本SD和95% CI。',
        'Adam和Momentum两组的超参数不同，配对差不能只归因于噪声机制。',
        '每次训练独立重新校准ε=8、δ=1e-5；增删邻接、固定零槽、参与5次且间隔62步；无抽样放大。',
        '搜索与正式训练整体不具有联合ε=8保证；内部损失和裁剪诊断不是DP发布。',
        '矩阵复制自Exp18d并独立审计。各任务privacy.json保存该裁剪阈值的噪声标准差和矩阵敏感度。',
        'logs/gpu_telemetry.jsonl 保存四张GPU利用率/显存，逐任务summary.json及per_seed.csv保存训练时间和峰值显存。']
    (BASE/'final_report.md').write_text('\n'.join(lines)+'\n')
    return complete

if __name__=='__main__':report()
