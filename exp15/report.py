"""Final ten-seed sample statistics and paired differences, with provenance."""
import itertools
import json
import numpy as np
from scipy.stats import t
from exp15 import RESULTS
from exp15.config import METHODS, FINAL_SEEDS, save_json, file_hash
from exp15.search import SELECTED, FROZEN, cfg_of
from exp15.support import write_csv, protected_snapshot


def stats(values):
    a=np.asarray(values,dtype=float)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('Need finite repeated measurements')
    mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    return dict(n=len(a),mean=mean,sample_std=sd,ci95_low=mean-half,ci95_high=mean+half)


def main():
    selected=json.loads(SELECTED.read_text());formal={};sources={}
    for c in selected['final_trials']:
        from exp15.config import Trial
        cfg=Trial(**c);path=cfg.output/'summary.json';r=json.loads(path.read_text())
        if r['status']!='completed' or r['completed_epochs']!=5:raise ValueError('Incomplete formal results')
        formal[(cfg.task,cfg.method,cfg.lr,cfg.clip,cfg.seed)]=r;sources[str(path)]=file_hash(path)
    grouped={};summary=[];cost=[]
    for w in selected['winners']:
        rows=[formal[(w['task'],w['method'],w['lr'],w['clip'],seed)] for seed in FINAL_SEEDS]
        key=(w['scheme'],w['task'],w['method']);grouped[key]=rows
        row={**w,**stats([r['accuracy'] for r in rows])}
        row.update({f'loss_{k}':v for k,v in stats([r['validation_loss'] for r in rows]).items()})
        row['metric_scope']='official final evaluation; selected validation metrics shown separately'
        summary.append(row)
    paired=[]
    for scheme,task in itertools.product(('A','B'),('cv','nlp')):
        for left,right in itertools.combinations(METHODS,2):
            a,b=grouped[(scheme,task,left)],grouped[(scheme,task,right)]
            if [r['seed'] for r in a]!=[r['seed'] for r in b]:raise ValueError('Unpaired seeds')
            for metric in ('accuracy','validation_loss'):
                paired.append(dict(scheme=scheme,task=task,left=left,right=right,metric=metric,
                    difference='right minus left',**stats([y[metric]-x[metric] for x,y in zip(a,b)])))
    for task,method in itertools.product(('cv','nlp'),METHODS):
        a,b=grouped[('A',task,method)],grouped[('B',task,method)]
        for metric in ('accuracy','validation_loss'):
            paired.append(dict(scheme='B_vs_A',task=task,left=f'A:{method}',right=f'B:{method}',metric=metric,
                difference='right minus left',**stats([y[metric]-x[metric] for x,y in zip(a,b)])))
    write_csv(RESULTS/'summary.csv',summary);write_csv(RESULTS/'paired_differences.csv',paired)
    all_rows=[]
    for stage in ('pilot','search','final'):
        rows=[json.loads(p.read_text()) for p in (RESULTS/stage).rglob('summary.json')]
        cost.append(dict(stage=stage,runs=len(rows),completed=sum(r['status']=='completed' for r in rows),
            failed=sum(r['status']!='completed' for r in rows),total_gpu_seconds=sum(r['seconds'] for r in rows),
            peak_allocated_bytes=max((r.get('peak_allocated_bytes',0) for r in rows),default=0)))
        for r in rows:
            if r['status']=='completed':
                cfg=cfg_of(r)
                for line in (cfg.output/'diagnostics.jsonl').read_text().splitlines():
                    d=json.loads(line);all_rows.append(dict(stage=stage,task=cfg.task,method=cfg.method,lr=cfg.lr,
                        clip=cfg.clip,seed=cfg.seed,trial_id=cfg.id,**d))
    for label,folder in [('phase1_smoke','smoke'),('preliminary_pilot_4step','preliminary_4step/pilot'),
                         ('phase1_before_protocol_revision','preliminary_4step/smoke')]:
        rows=[json.loads(p.read_text()) for p in (RESULTS/folder).rglob('summary.json')]
        cost.append(dict(stage=label,runs=len(rows),completed=sum(r['status']=='completed' for r in rows),
            failed=sum(r['status']!='completed' for r in rows),total_gpu_seconds=sum(r['seconds'] for r in rows),
            peak_allocated_bytes=max((r.get('peak_allocated_bytes',0) for r in rows),default=0)))
    write_csv(RESULTS/'cost.csv',cost);write_csv(RESULTS/'all_diagnostics.csv',all_rows)
    save_json(RESULTS/'statistics.json',dict(summary=summary,paired=paired,cost=cost))
    save_json(RESULTS/'report_sources.json',dict(grid_sha256=file_hash(FROZEN),selection_sha256=file_hash(SELECTED),sources=sources))
    unchanged=protected_snapshot()==json.loads((RESULTS/'protected_before.json').read_text())
    if not unchanged:raise ValueError('Historical files changed')
    save_json(RESULTS/'final_audit.json',dict(status='passed',historical_files_unchanged=True,formal_runs=len(formal)))
    lines=['# Exp15 最终报告','',
      '正式结果使用 Exp9 定义的十个种子，报告样本标准差、Student-t 双侧 95% 置信区间和同种子配对差；未作多重比较校正。',
      '选参仅使用内部验证集；下表十种子指标为沿用历史流程的官方最终评价。CV 最终训练使用全部 50,000 条训练数据，内部验证数据因此不能再作为独立最终验证集。NLP 最终训练沿用 62,000 条训练子集。',
      '历史官方评价集已使用过，不能称为新盲测。ε=8、δ=1e-5 仅针对单次模型训练，不包含搜索与重复运行的组合开销；未加噪诊断仅为本地私有审计记录。',
      '所有方法共用 Exp9 Momentum-Bias 的四频带滤波器，momentum 的真实工作负载不同，其滤波器未单独优化。C/R 的相同数值代表不同裁剪空间。','',
      '|方案|任务|方法|lr|C/R|准确率均值|样本标准差|95% CI|损失均值|','|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        lines.append(f"|{r['scheme']}|{r['task']}|{r['method']}|{r['lr']:g}|{r['clip']:g}|{r['mean']:.5f}|{r['sample_std']:.5f}|[{r['ci95_low']:.5f}, {r['ci95_high']:.5f}]|{r['loss_mean']:.5f}|")
    lines+=['',f"正式训练最大 200 次，配置完全相同后去重为 {len(formal)} 次。",
      '冻结网格和依据见 frozen_grid.json；全部候选及失败原因见 search_results.json；配对差见 paired_differences.csv；逐步诊断见 all_diagnostics.csv；成本见 cost.csv；源文件及配置、噪声矩阵哈希保存在每个 trial 目录。']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
