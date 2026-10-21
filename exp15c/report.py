"""Audited ten-seed statistics and matched historical comparisons."""
import csv
import itertools
from collections import defaultdict
import numpy as np
from scipy.stats import t
from exp15c import ROOT,RESULTS
from exp15c.config import Trial,FINAL_SEEDS,SEARCH_SEED,file_hash,save_json
from exp15c.support import read,write_csv,result_dir,protected_snapshot
from exp15c.scheduler import completed, non_finite, valid_metrics
from exp15c.search import cfg_of,load_frozen


def stats(values):
    a=np.asarray(values,dtype=float)
    if len(a)<2 or not np.isfinite(a).all(): raise ValueError('Need finite repeated measurements')
    mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    return dict(n=len(a),mean=mean,sample_std=sd,ci95_low=mean-half,ci95_high=mean+half)


def assert_paired(a,b):
    for key in ('task','seed','initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
        if a[key]!=b[key]: raise ValueError(f'Unpaired {key}')


def pair(a,b,scheme,task,left,right,scope):
    if [r['seed'] for r in a]!=list(FINAL_SEEDS) or [r['seed'] for r in b]!=list(FINAL_SEEDS): raise ValueError('Ten paired seeds required')
    for x,y in zip(a,b): assert_paired(x,y)
    return [dict(scheme=scheme,task=task,left=left,right=right,metric=k,comparison=scope,
                 difference='right minus left',**stats([y[k]-x[k] for x,y in zip(a,b)])) for k in ('accuracy','validation_loss')]


def artifact_dir(row):
    if row['method']=='momentum' and not row.get('reuse'):
        return ROOT/'exp15b/results'/row['stage']/row['replicate']/row['task']/row['trial_id']
    return result_dir(row)


def diagnostics(row):
    directory=artifact_dir(row)
    steps=[__import__('json').loads(line) for line in (directory/'diagnostics.jsonl').read_text().splitlines()]
    private=row['method'] not in ('adam-no-noise','momentum-no-noise')
    if not private: return None,None
    with np.load(directory/'matrices.npz') as f:
        d=f['coefficients'];n=len(f['strategy'])
    from exp15c.bandinvmf import momentum_workload,workload_error
    sigma=row['calibration']['innovation_std_sum']
    mse=.1**2*workload_error(momentum_workload(n),d)*sigma**2/1000**2
    return float(np.mean([r['clip_fraction'] for r in steps])),mse


def historical_rows(stage):
    path=ROOT/f'exp15b/results/{stage}_results.json'
    raw=read(path);rows=raw['rows'] if isinstance(raw,dict) else raw
    rows=[r for r in rows if r['method']=='momentum']
    for r in rows:
        if r['status']!='completed': raise ValueError('Historical Momentum result incomplete')
        directory=artifact_dir(r)
        for name,h in r['file_sha256'].items():
            if file_hash(directory/name)!=h: raise ValueError(f'Historical artifact changed: {directory/name}')
    return rows


def audit_trials(stage,values,rows):
    if (len(rows)!=len(values) or len({r['trial_id'] for r in rows})!=len(rows)
        or {Trial(**v).id for v in values}!={r['trial_id'] for r in rows}):
        raise ValueError('Missing planned training or duplicate result')
    exclusions=[]
    for row in rows:
        cfg=cfg_of(row)
        if row['status']=='non_finite':
            if stage not in ('search','review') or not non_finite(cfg):
                raise ValueError(f'Invalid exclusion in {stage}: {cfg.id}')
            if row!=read(cfg.output/'summary.json'): raise ValueError('Exclusion summary changed')
            exclusions.append(dict(stage=stage,task=cfg.task,method=cfg.method,lr=cfg.lr,clip=cfg.clip,
                seed=cfg.seed,trial_id=cfg.id,status='non_finite',reason=row['error'],
                source_summary=str((cfg.output/'summary.json').relative_to(ROOT))))
        else:
            if not completed(cfg) or not valid_metrics(row) or row['completed_epochs']!=5:
                raise ValueError('Incomplete or invalid result')
            if row['method']=='momentum-mf-native' and abs(row['actual_epsilon']-8.)>1e-7:
                raise ValueError('Wrong epsilon')
    return dict(planned=len(values),completed=len(rows)-len(exclusions),non_finite=len(exclusions)),exclusions


def main():
    frozen=load_frozen();selected=read(RESULTS/'selected_configs.json')
    search=read(RESULTS/'search_results.json');review=read(RESULTS/'review_results.json')['rows'];formal=read(RESULTS/'formal_results.json')
    plan=read(RESULTS/'review_plan.json')
    if selected['grid_sha256']!=file_hash(RESULTS/'extended_grid.json') or selected['review_results_sha256']!=file_hash(RESULTS/'review_results.json'):
        raise ValueError('Decision provenance changed')
    stage_counts={};exclusions=[]
    for stage,values,rows in (('search',list(frozen['candidates'].values()),search),('review',plan['trials'],review),('formal',selected['final_trials'],formal)):
        stage_counts[stage],excluded=audit_trials(stage,values,rows);exclusions.extend(excluded)
    # Keep every planned outcome in the result JSONs, but use only successes below.
    search=[r for r in search if valid_metrics(r)];review=[r for r in review if valid_metrics(r)]
    if any(r['official_evaluation_used'] for r in search+review): raise ValueError('Official evaluation used for selection')
    lookup={(r['task'],r['method'],r['lr'],r['clip'],r['seed']):r for r in search+review}
    for w in selected['winners']:
        for seed in (SEARCH_SEED,20261102,20261103):
            if (w['task'],w['method'],w['lr'],w['clip'],seed) not in lookup:
                raise ValueError('Selected configuration has an excluded or missing validation seed')
    historical={s:historical_rows(s) for s in ('search','review','formal')}
    old_selected=read(ROOT/'exp15b/results/selected_configs.json')
    summaries=[];pairs=[];groups={};cache={}
    for experiment,winners,rows in [('Exp15c',selected['winners'],formal),('Exp15b',old_selected['winners'],historical['formal'])]:
        for w in winners:
            if experiment=='Exp15b' and w['method']!='momentum': continue
            task=w['task'];scheme=w['scheme'];method=w['method']
            values=sorted([r for r in rows if (r['task'],r['method'],r['lr'],r['clip'])==(task,method,w['lr'],w['clip'])],key=lambda r:r['seed'])
            if [r['seed'] for r in values]!=list(FINAL_SEEDS): raise ValueError('Missing/duplicate formal seed')
            label=method if experiment=='Exp15c' else 'Exp15b-momentum-MF'
            groups[(scheme,task,label)]=values
            for r in values:
                key=(experiment,r['trial_id'])
                if key not in cache: cache[key]=diagnostics(r)
            private=method not in ('adam-no-noise','momentum-no-noise')
            accuracy=stats([r['accuracy'] for r in values]);loss=stats([r['validation_loss'] for r in values])
            summaries.append(dict(experiment=experiment,scheme=scheme,task=task,method=label,private=private,
                epsilon=8. if private else None,delta=1e-5 if private else None,lr=w['lr'],clip=w['clip'],
                accuracy_mean=accuracy['mean'],accuracy_std=accuracy['sample_std'],accuracy_ci95_low=accuracy['ci95_low'],accuracy_ci95_high=accuracy['ci95_high'],
                cross_entropy_mean=loss['mean'],cross_entropy_std=loss['sample_std'],n=10,
                clip_fraction_mean=float(np.mean([cache[(experiment,r['trial_id'])][0] for r in values])) if private else None,
                momentum_workload_MSE=float(np.mean([cache[(experiment,r['trial_id'])][1] for r in values])) if private else None))
    for scheme in ('A','B'):
        for task in ('cv','nlp'):
            names=['adam-no-noise','momentum-no-noise','momentum-mf-native','Exp15b-momentum-MF']
            for left,right in itertools.combinations(names,2):
                pairs+=pair(groups[(scheme,task,left)],groups[(scheme,task,right)],scheme,task,left,right,'independently_tuned_official_evaluation')
    controlled=[]
    for stage,new in [('search',search),('review',review),('formal',formal)]:
        old={(r['task'],r['lr'],r['clip'],r['seed']):r for r in historical[stage]}
        for r in new:
            if r['method']!='momentum-mf-native': continue
            key=(r['task'],r['lr'],r['clip'],r['seed']);o=old.get(key)
            if o is None: continue
            assert_paired(o,r)
            for metric in ('accuracy','validation_loss'):
                controlled.append(dict(stage=stage,task=r['task'],lr=r['lr'],clip=r['clip'],seed=r['seed'],metric=metric,
                    left='Exp15b-momentum-MF',right='momentum-mf-native',difference=r[metric]-o[metric],
                    evaluation_scope=r['evaluation_scope']))
    fixed=defaultdict(list)
    for r in controlled:
        if r['stage']=='formal': fixed[(r['task'],r['lr'],r['clip'],r['metric'])].append(r)
    for (task,lr,clip,metric),rows in fixed.items():
        if sorted(r['seed'] for r in rows)==list(FINAL_SEEDS):
            pairs.append(dict(scheme=f'fixed lr={lr}, C={clip}',task=task,left='Exp15b-momentum-MF',right='momentum-mf-native',metric=metric,
                comparison='fixed_hyperparameters_official_evaluation',difference='right minus left',**stats([r['difference'] for r in rows])))
    write_csv(RESULTS/'summary.csv',summaries);write_csv(RESULTS/'paired_differences.csv',pairs)
    write_csv(RESULTS/'controlled_differences.csv',controlled)
    save_json(RESULTS/'statistics.json',dict(summary=summaries,paired=pairs))
    lines=['# Exp15c 最终报告','',
        '无噪声 Adam/Momentum 是非私有实验。Momentum-MF 和历史 Exp15b Momentum+MF 为单次训练 ε=8、δ=10⁻⁵；固定五轮参与、add/remove zero-out、不使用采样放大。搜索、复核及重复训练未声明组合隐私保证，内部验证假定公开；原始裁剪诊断是私有审计产物。',
        '','选参只用内部验证：搜索前二，复核两个种子，按三个种子的平均准确率和交叉熵排序。官方评价集历史上已被使用，这不是新盲测。A 固定 C=1；B 联合搜索。无噪声两方案共享同一学习率搜索。','',
        '|实验|方案|任务|方法|LR|C|准确率均值 ± 样本标准差 (%)|交叉熵|裁剪比例|Momentum workload MSE|',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        clip='—' if r['clip'] is None else f"{r['clip']:g}"
        frac='—' if r['clip_fraction_mean'] is None else f"{r['clip_fraction_mean']:.4f}"
        mse='—' if r['momentum_workload_MSE'] is None else f"{r['momentum_workload_MSE']:.6g}"
        lines.append(f"|{r['experiment']}|{r['scheme']}|{r['task']}|{r['method']}|{r['lr']:g}|{clip}|{100*r['accuracy_mean']:.3f} ± {100*r['accuracy_std']:.3f}|{r['cross_entropy_mean']:.5f}|{frac}|{mse}|")
    lines+=['','以下差值均为右组减左组，95% CI 为十个相同种子配对差的 Student-t 区间。准确率用百分点。CI 包含零时不作明确优越结论。','',
        '|方案|任务|左组|右组|准确率差 (百分点)|配对 95% CI|','|---|---|---|---|---:|---:|']
    for r in pairs:
        if r['metric']=='accuracy':lines.append(f"|{r['scheme']}|{r['task']}|{r['left']}|{r['right']}|{100*r['mean']:.3f}|[{100*r['ci95_low']:.3f}, {100*r['ci95_high']:.3f}]|")
    lines+=['','主要问题的结果（B 方案；A 方案及完整配对结果见上表）：','']
    for task in ('cv','nlp'):
        adam=groups[('B',task,'adam-no-noise')];momentum=groups[('B',task,'momentum-no-noise')]
        native=groups[('B',task,'momentum-mf-native')];old=groups[('B',task,'Exp15b-momentum-MF')]
        difference=stats([m['accuracy']-a['accuracy'] for a,m in zip(adam,momentum)])
        winner='Momentum' if difference['mean']>0 else 'Adam' if difference['mean']<0 else '两者相同'
        evidence='配对区间包含零，差异尚不明确。' if difference['ci95_low']<=0<=difference['ci95_high'] else '配对区间不包含零。'
        lines.append(f"- {task} 无噪声：{winner} 的十种子平均准确率较高；Momentum−Adam={100*difference['mean']:.3f} 个百分点，95% CI [{100*difference['ci95_low']:.3f}, {100*difference['ci95_high']:.3f}]。{evidence}")
        difference=stats([n['accuracy']-o['accuracy'] for n,o in zip(native,old)])
        answer='超过' if difference['mean']>0 else '未超过'
        evidence='配对区间包含零，不能确认稳定改善。' if difference['ci95_low']<=0<=difference['ci95_high'] else '配对区间不包含零。'
        lines.append(f"- {task} 独立调参的 Momentum-Matched MF：平均准确率{answer}历史 Momentum+MF，差值={100*difference['mean']:.3f} 个百分点，95% CI [{100*difference['ci95_low']:.3f}, {100*difference['ci95_high']:.3f}]。{evidence}")
        for name,baseline in [('Adam',adam),('Momentum',momentum)]:
            gap=stats([a['accuracy']-n['accuracy'] for a,n in zip(baseline,native)])
            lines.append(f"- {task} 无噪声 {name}−DP Momentum-MF 的差距={100*gap['mean']:.3f} 个百分点，配对 95% CI [{100*gap['ci95_low']:.3f}, {100*gap['ci95_high']:.3f}]。")
        matching=[r for r in controlled if r['task']==task and r['stage']=='search' and r['metric']=='accuracy']
        if matching:
            improvements=sum(r['difference']>0 for r in matching)
            lines.append(f"- {task} 相同 LR/C 的单种子搜索对照：{len(matching)} 对中有 {improvements} 对提高准确率；差值范围 [{100*min(r['difference'] for r in matching):.3f}, {100*max(r['difference'] for r in matching):.3f}] 个百分点。这些配置并非独立重复，不据此计算十种子 CI。")
    lines+=['','历史 MF 仅从 Exp15b 读取，没有创建或重训历史方法。','',
        '任务验收：搜索/复核仅排除明确记录的非有限值，正式训练不排除任何种子。','',
        '|阶段|计划数|完成数|发散排除数|','|---|---:|---:|---:|']
    for stage,counts in stage_counts.items():
        lines.append(f"|{stage}|{counts['planned']}|{counts['completed']}|{counts['non_finite']}|")
    lines+=['','排除原因（原始 summary、traceback 和训练日志全部保留）：','']
    for row in exclusions:
        lines.append(f"- {row['stage']} {row['task']} {row['method']} LR={row['lr']:g}, C={row['clip']}, seed={row['seed']}：{row['reason']}；`{row['source_summary']}`。")
    with (RESULTS/'mf_comparison.csv').open() as f:
        for r in csv.DictReader(f):
            lines.append(f"- {r['task']} T={r['total_steps']}：在相同 Momentum workload 上，重新校准后的理论 MSE 降低 {100*float(r['relative_reduction']):.5f}%。")
    lines+=['','MSE 为累计动量噪声的每坐标均方误差，包含 (1−β)²、C²、噪声校准和 1000² 归一化，未乘学习率平方；参数轨迹误差应再乘 LR²。mf_comparison.csv 固定 C=1。优化目标省略的 (1−β) 常数不影响滤波器最优解。','',
        f'固定相同学习率和裁剪阈值的对照保存在 controlled_differences.csv（{len(controlled)//2} 对）。搜索为单种子内部验证，复核只覆盖双方均已复核的配置，不能称作十种子官方机制效应。',
        '如果双方选定的正式 LR/C 不同，固定超参数的十种子对照不存在；不额外训练历史方法。独立调参后的十种子比较仍列在配对表中。','',
        '完整交叉熵配对 CI 见 paired_differences.csv；最优超参数、十种子准确率/标准差/CI、交叉熵、裁剪比例和 MSE 见 summary.csv。']
    if protected_snapshot()!=read(RESULTS/'protected_before.json'):raise ValueError('Protected files changed')
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')
    save_json(RESULTS/'final_audit.json',dict(status='passed',protected_files_unchanged=True,
        stage_counts=stage_counts,excluded_candidates=exclusions,
        sources={str(p.relative_to(ROOT)):file_hash(p) for p in [ROOT/f'exp15b/results/{s}_results.json' for s in ('search','review','formal')]}))


if __name__=='__main__':main()
