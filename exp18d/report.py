"""Strict completeness audit and descriptive statistics; partial runs stay partial."""
import math
import numpy as np
from scipy.stats import t
from exp18d.common import (BASE, ROOT, GROUPS, SEEDS, configs, outdir, completed, read,
                           save, sha, now, write_csv)

PER_FIELDS = ['method','seed','lr','clip','eps','accuracy','loss','examples','steps','epsilon','delta','seconds','directory']
SUM_FIELDS = ['method','completed_seeds','expected_seeds','status','accuracy_mean','accuracy_sd',
              'accuracy_ci95_low','accuracy_ci95_high','loss_mean','loss_sd','epsilon','delta']


def report():
    expected = {outdir(c).name for c in configs()}
    actual = {p.name for p in (BASE / 'formal').glob('*') if p.is_dir()}
    assert actual <= expected, 'Unexpected formal trial directories'
    rows = []; missing = []; failures = []
    for cfg in configs():
        out = outdir(cfg); result = completed(cfg)
        if not result:
            missing.append(str(out.relative_to(BASE)))
            if (out / 'failure.json').exists(): failures.append(str(out.relative_to(BASE)))
            continue
        assert result['evaluation']=='official_validation' and result['metrics']['examples']==872
        assert result['privacy']==read(BASE / 'privacy_audit.json')['methods'][cfg['method']]
        rows.append(dict(method=cfg['method'],seed=cfg['seed'],lr=cfg['lr'],clip=cfg['clip'],eps=cfg['eps'],
            **result['metrics'],steps=result['steps'],epsilon=result['privacy']['epsilon'],delta=1e-5,
            seconds=result['seconds'],directory=str(out.relative_to(BASE))))
    summaries = []
    for methods in GROUPS.values():
        for method in methods:
            group = [r for r in rows if r['method']==method]; n = len(group)
            acc = np.array([r['accuracy'] for r in group]); loss = np.array([r['loss'] for r in group])
            mean = float(acc.mean()) if n else None
            sd = float(acc.std(ddof=1)) if n>1 else None
            ci = float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
            summaries.append(dict(method=method,completed_seeds=n,expected_seeds=10,
                status='complete' if n==10 else 'incomplete',accuracy_mean=mean,accuracy_sd=sd,
                accuracy_ci95_low=mean-ci if ci is not None else None,
                accuracy_ci95_high=mean+ci if ci is not None else None,
                loss_mean=float(loss.mean()) if n else None,loss_sd=float(loss.std(ddof=1)) if n>1 else None,
                epsilon=8,delta=1e-5))
    write_csv(BASE / 'per_seed.csv',rows,PER_FIELDS)
    write_csv(BASE / 'summary.csv',summaries,SUM_FIELDS)
    pairing = {}
    for seed in SEEDS:
        found = [read(outdir(cfg)/'pairing.json') for cfg in configs() if cfg['seed']==seed and completed(cfg)]
        assert not found or all(x==found[0] for x in found), f'Unpaired seed {seed}'
        pairing[str(seed)] = len(found)
    unchanged = all(sha(ROOT / p)==h for p,h in read(BASE / 'upstream_manifest.json').items())
    assert unchanged
    complete = len(rows)==60 and not missing and not failures
    save(BASE / 'integrity_audit.json',dict(complete=complete,expected=60,completed=len(rows),
        missing=missing,failed=failures,upstream_inputs_and_sources_unchanged=unchanged,
        pairing_counts=pairing,artifact_hashes_verified=True,time=now()))
    lines = ['# Exp18d（ε=8）实验报告','',f"状态：{'正式完成' if complete else '未完成；当前统计仅供进度查看'}；{len(rows)}/60 个正式训练完成。",'',
        'SST-2 / BERT-Tiny，全参数 FP32，长度128，5轮×62步，逻辑/物理批次1000。',
        '十个种子20261111–20261120；62000训练、5349内部验证、872官方验证，划分与Exp16e一致。',
        '最终官方验证准确率；标准差为样本SD（ddof=1），区间为Student-t 95% CI。','',
        '|方法|种子数|准确率均值 ± SD (%)|95% CI (%)|损失均值|','|---|---:|---:|---|---:|']
    for r in summaries:
        n = r['completed_seeds']
        stats = f"{100*r['accuracy_mean']:.3f} ± {100*r['accuracy_sd']:.3f}" if n>1 else '待完成'
        ci = f"[{100*r['accuracy_ci95_low']:.3f}, {100*r['accuracy_ci95_high']:.3f}]" if n>1 else '—'
        loss = f"{r['loss_mean']:.6f}" if n else '—'
        lines.append(f"|{r['method']}|{n}/10|{stats}|{ci}|{loss}|")
    lines += ['', '六种方法全部在本次统一协议下重新训练；历史Exp16e/16f表仅位于history/，不计入正式结果。',
        'Adam/Momentum按方法复用Exp16e已冻结NLP参数；STP-Ex2复用根目录Exp18c冻结的lr=0.08、clip=100、stability=0.005，属于MNIST到NLP的固定参数迁移。',
        '不同方法的历史最优超参数不同，差值不能解释为仅改变噪声机制的因果效果。',
        'STP-Ex2直接调用Exp18实现，d0=1、previous-d缩放、adaptive alpha；与Exp16 STP-EX明确区分。',
        'B²LT按Exp18统一定义新生成T=310、K=P=62、Q=5；矩阵逆、分解、正性、收敛与噪声递推均经审计/测试。',
        '每次完整训练使用Exp18 GDP会计，ε=8、δ=1e-5，add/remove zero-out邻接；每条训练记录固定参与5次、间隔62步，不使用采样放大。',
        '这是单次训练保证；60次训练及历史搜索的组合不具有联合ε=8保证。内部原始训练损失/裁剪诊断不是DP发布。',
        '详细参数、噪声标准差和矩阵散列见hyperparameters.json与privacy_audit.json；结果完整性见integrity_audit.json。']
    (BASE / 'final_report.md').write_text('\n'.join(lines)+'\n')
    return complete

if __name__ == '__main__': report()
