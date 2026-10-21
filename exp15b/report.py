"""Audited paired statistics; controlled and independently tuned effects separated."""
import itertools
import json
from collections import defaultdict
import numpy as np
from scipy.stats import t
from exp15b import ROOT, RESULTS
from exp15b.config import METHODS, FINAL_SEEDS, file_hash, array_hash, save_json
from exp15b.search import SELECTED, FROZEN, cfg_of, load_frozen
from exp15b.scheduler import completed
from exp15b.support import write_csv, result_dir, protected_snapshot
from exp15b.audit import read


def stats(values):
    a=np.asarray(values,dtype=float)
    if len(a)<2 or not np.isfinite(a).all():raise ValueError('Need finite repeated measurements')
    mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    return dict(n=len(a),mean=mean,sample_std=sd,ci95_low=mean-half,ci95_high=mean+half)


def assert_paired(left,right):
    for key in ('task','seed','initialization_sha256','classifier_initialization_sha256',
                'train_order_sha256','rng_trace','noise_rng_start','noise_rng_end'):
        if left[key]!=right[key]:raise ValueError(f'Non-noise random streams are unpaired: {key}')


def pair_stats(rows,scheme,task,left,right,comparison):
    a,b=rows[left],rows[right]
    if [r['seed'] for r in a]!=list(FINAL_SEEDS) or [r['seed'] for r in b]!=list(FINAL_SEEDS):
        raise ValueError('Missing ten formal seeds')
    for x,y in zip(a,b):assert_paired(x,y)
    return [dict(scheme=scheme,task=task,left=left,right=right,metric=metric,
        comparison=comparison,difference='right minus left',
        **stats([y[metric]-x[metric] for x,y in zip(a,b)])) for metric in ('accuracy','validation_loss')]


def diagnostic_noise(row):
    path=result_dir(row)
    steps=[json.loads(line) for line in (path/'diagnostics.jsonl').read_text().splitlines()]
    with np.load(path/'matrices.npz') as matrices:
        d=matrices['coefficients'];A=matrices['actual_workload'];S=matrices['strategy']
    n=len(S);D=np.zeros((n,n))
    for lag,coef in enumerate(d):D+=np.diag(np.full(n-lag,coef),-lag)
    sigma=row['calibration']['innovation_std_sum']
    covariance=sigma**2*(D@D.T)
    geometry=RESULTS/'noise_covariances'/f'{array_hash(covariance)}.npz'
    geometry.parent.mkdir(exist_ok=True)
    if not geometry.exists():np.savez_compressed(geometry,covariance_sum=covariance,covariance_average=covariance/1000**2)
    return dict(stage=row['stage'],task=row['task'],method=row['method'],lr=row['lr'],clip=row['clip'],seed=row['seed'],
        trial_id=row['trial_id'],innovation_std_sum=sigma,
        marginal_std_sum_first=float(np.sqrt(covariance[0,0])),marginal_std_sum_last=float(np.sqrt(covariance[-1,-1])),
        mean_marginal_variance_sum=float(np.trace(covariance)/n),
        lag1_covariance_sum=float(np.diag(covariance,-1).mean()),
        covariance_sum_sha256=array_hash(covariance),covariance_file=str(geometry.relative_to(ROOT)),
        strategy_sha256=array_hash(S),sensitivity=row['calibration']['sensitivity'],
        workload_mean_squared_noise_average=float(np.sum((A@D)**2)*sigma**2/n/1000**2),
        clip_fraction_mean=float(np.mean([r['clip_fraction'] for r in steps])),
        update_norm_max=max(r['update_norm'] for r in steps),finite=True,
        epsilon=row['actual_epsilon'],source_summary=str((path/'summary.json').relative_to(ROOT)))


def controlled_comparisons(rows):
    lookup={(r['task'],r['stage'],r['lr'],r['clip'],r['seed'],r['method']):r
            for r in rows if r['status']=='completed' and r['method'] in ('momentum','momentum+iid')}
    pairs=[]
    for key,left in lookup.items():
        task,stage,lr,clip,seed,method=key
        if method!='momentum':continue
        right=lookup.get((task,stage,lr,clip,seed,'momentum+iid'))
        if right is None:continue
        assert_paired(left,right)
        for metric in ('accuracy','validation_loss'):
            pairs.append(dict(task=task,stage=stage,lr=lr,clip=clip,seed=seed,metric=metric,
                left='momentum',right='momentum+iid',difference=right[metric]-left[metric],
                evaluation_scope=left['evaluation_scope'],comparison='fixed_hyperparameters_mechanism_effect'))
    return pairs


def historical_tables():
    tables=[];sources={}
    for relative,label in [('exp15/results/formal_results.json','Exp15'),
                           ('exp14d/results/A_formal_results.json','Exp14d_A'),
                           ('exp14d/results/B_formal_results.json','Exp14d_B')]:
        path=ROOT/relative
        if path.exists():
            sources[relative]=file_hash(path)
            groups=defaultdict(list)
            for r in read(path):
                if r['status']=='completed' and r['seed'] in FINAL_SEEDS:
                    groups[(r['task'],r['method'],r['lr'],r.get('clip',r.get('C')),r.get('eps_scale'))].append(r)
            for (task,method,lr,clip,eps),rows in groups.items():
                dedup={r['seed']:r for r in rows}
                if len(dedup)==10:
                    tables.append(dict(experiment=label,task=task,method=method,lr=lr,clip=clip,eps_scale=eps,
                        **stats([r['accuracy'] for r in dedup.values()]),evidence='descriptive cross-experiment'))
    for task in ('cv','nlp'):
        groups=defaultdict(list)
        for path in (ROOT/f'exp9/results/final/{task}').glob('*/summary.json'):
            r=read(path)
            if r['status']=='completed' and r['method'] in ('dp-adam-momentum-bias-bandinvmf','dp-adam-momentum-bias-bandinvmf-scale'):
                groups[r['method']].append(r);sources[str(path.relative_to(ROOT))]=file_hash(path)
        for method,rows in groups.items():
            if {r['seed'] for r in rows}!=set(FINAL_SEEDS):raise ValueError('Historical Exp9 missing seeds')
            tables.append(dict(experiment='Exp9',task=task,method=method,lr=rows[0]['lr'],clip=rows[0].get('C',1),
                eps_scale=rows[0].get('eps_scale'),**stats([r['accuracy'] for r in rows]),evidence='descriptive cross-experiment'))
    return tables,sources


def boundary_flags(winner,data):
    ids=data['schemes'][winner['scheme']][f"{winner['task']}:{winner['method']}"]
    values=[data['candidates'][i] for i in ids]
    conditional=[v['lr'] for v in values if v['clip']==winner['clip']]
    clips=[v['clip'] for v in values]
    return dict(lr_boundary_at_selected_clip=winner['lr'] in (min(conditional),max(conditional)),
                clip_boundary=winner['scheme']=='B' and winner['clip'] in (min(clips),max(clips)),
                tested_lr_min=min(conditional),tested_lr_max=max(conditional),tested_clip_min=min(clips),tested_clip_max=max(clips))


def main():
    data=load_frozen();selected=read(SELECTED)
    search=read(RESULTS/'search_results.json');review=read(RESULTS/'review_results.json')
    formal=read(RESULTS/'formal_results.json')
    plan=read(RESULTS/'review_plan.json')
    if selected['grid_sha256']!=file_hash(FROZEN) or plan['grid_sha256']!=file_hash(FROZEN):
        raise ValueError('Report selection/grid mismatch')
    if selected['review_results_sha256']!=file_hash(RESULTS/'review_results.json') or plan['search_results_sha256']!=file_hash(RESULTS/'search_results.json'):
        raise ValueError('Selection source results changed')
    if (len(search)!=224 or {r['trial_id'] for r in search}!=set(data['candidates'])
        or {r['trial_id'] for r in review['rows']}!={cfg_of(c).id for c in plan['trials']}
        or len(review['rows'])!=len(plan['trials'])
        or {r['trial_id'] for r in formal}!={cfg_of(c).id for c in selected['final_trials']}
        or len(formal)!=len(selected['final_trials'])):
        raise ValueError('Missing planned training')
    for r in search+review['rows']+formal:
        if r['status']=='completed':
            if not completed(cfg_of(r)) or r['completed_epochs']!=5 or abs(r['actual_epsilon']-8)>1e-7:
                raise ValueError('Incomplete or unaudited training')
        elif r['status']!='non_finite' or r['stage']=='final':
            raise ValueError('Unexpected failed training')
    lookup={(r['task'],r['method'],r['lr'],r['clip'],r['seed']):r for r in formal}
    grouped={};summary=[];paired=[]
    for w in selected['winners']:
        rows=[lookup[(w['task'],w['method'],w['lr'],w['clip'],seed)] for seed in FINAL_SEEDS]
        grouped[(w['scheme'],w['task'],w['method'])]=rows
        summary.append(dict(scheme=w['scheme'],task=w['task'],method=w['method'],lr=w['lr'],clip=w['clip'],
            review_accuracy_mean=w['accuracy_mean'],review_accuracy_std=w['accuracy_std'],
            **stats([r['accuracy'] for r in rows]),**{f'loss_{k}':v for k,v in stats([r['validation_loss'] for r in rows]).items()},
            **boundary_flags(w,data)))
    for scheme,task in itertools.product(('A','B'),('cv','nlp')):
        rows={m:grouped[(scheme,task,m)] for m in METHODS}
        for left,right in itertools.combinations(METHODS,2):
            paired+=pair_stats(rows,scheme,task,left,right,'independently_tuned_ten_seed')
    for task,method in itertools.product(('cv','nlp'),METHODS):
        paired+=pair_stats({s:grouped[(s,task,method)] for s in ('A','B')},'B_vs_A',task,'A','B',f'search_scheme:{method}')
    controlled=controlled_comparisons(search+review['rows']+formal)
    noise=[diagnostic_noise(r) for r in search+review['rows']+formal if r['status']=='completed']
    historical,historical_sources=historical_tables()
    old_stats=read(ROOT/'exp15/results/statistics.json')
    ranks=[]
    for scheme,task in itertools.product(('A','B'),('cv','nlp')):
        new=sorted([r for r in summary if r['scheme']==scheme and r['task']==task],key=lambda r:(-r['mean'],r['method']))
        old=sorted([r for r in old_stats['summary'] if r['scheme']==scheme and r['task']==task],key=lambda r:(-r['mean'],r['method']))
        ranks.append(dict(scheme=scheme,task=task,exp15=[r['method'] for r in old],exp15b=[r['method'] for r in new],
            existing_five_changed=[r['method'] for r in new if r['method']!='momentum+iid']!=[r['method'] for r in old]))
    write_csv(RESULTS/'summary.csv',summary)
    write_csv(RESULTS/'paired_differences.csv',paired)
    write_csv(RESULTS/'controlled_differences.csv',controlled)
    write_csv(RESULTS/'noise_comparison.csv',noise)
    write_csv(RESULTS/'historical_comparison.csv',historical)
    save_json(RESULTS/'statistics.json',dict(summary=summary,paired=paired,rankings=ranks))
    unchanged=protected_snapshot()==read(RESULTS/'protected_before.json')
    if not unchanged:raise ValueError('Protected historical files changed')
    sources={str((result_dir(r)/'summary.json').relative_to(ROOT)):file_hash(result_dir(r)/'summary.json')
             for r in search+review['rows']+formal}
    save_json(RESULTS/'report_sources.json',dict(grid_sha256=file_hash(FROZEN),selection_sha256=file_hash(SELECTED),
        sources=sources,historical_descriptive_sources=historical_sources))
    save_json(RESULTS/'final_audit.json',dict(status='passed',historical_files_unchanged=True,
        formal_runs=len(formal),reused_formal=sum(bool(r.get('reuse')) for r in formal),search_candidates=len(search),
        review_runs=len(review['rows']),gpus=[0,1,2],single_training_privacy_only=True,
        controlled_comparison_count=len(controlled),nonfinite_candidates=sum(r['status']=='non_finite' for r in search+review['rows'])))
    lines=['# Exp15b 最终报告','',
        'ε=8、δ=10⁻⁵ 仅为单次训练预算，不包含搜索、多种子训练及选参的组合隐私。历史官方评价集已使用过，本次不是新盲测。',
        '选参使用内部验证；正式指标使用历史官方评价。十种子配对 Student-t 95% 区间未作多重比较校正，属于探索性统计证据。CV 正式训练包含原内部验证数据。',
        'Momentum 与 Momentum+IID 的更新公式完全相同且均无偏差修正。IID 用身份策略和完整五轮参与敏感度 √5 重新校准；相关噪声沿用四频带策略。', '',
        '|方案|任务|方法|lr|C/R|准确率|标准差|95% CI|交叉熵|',
        '|---|---|---|---|---|---|---|---|---|']
    for r in summary:
        lines.append(f"|{r['scheme']}|{r['task']}|{r['method']}|{r['lr']:g}|{r['clip']:g}|{r['mean']:.5f}|{r['sample_std']:.5f}|[{r['ci95_low']:.5f}, {r['ci95_high']:.5f}]|{r['loss_mean']:.5f}|")
    lines+=['','## 1. 相同超参数下的噪声机制效应','',
        'controlled_differences.csv 保存所有相同任务、lr、C、种子的 IID−相关噪声差。搜索/复核对照评价内部验证；只有两组正式选中相同配置时才有十种子官方受控对照，不能将单种子验证差当作十种子正式效应。']
    for task in ('cv','nlp'):
        group=[r for r in controlled if r['task']==task and r['stage']=='search' and r['metric']=='accuracy']
        differences=[r['difference'] for r in group]
        lines.append(f"{task}：{len(group)} 个匹配配置，IID−相关噪声验证准确率差范围 [{min(differences):.5f}, {max(differences):.5f}]；这一区间是配置范围，不是置信区间。")
    lines+=['','## 2. 各自调参后的效果与 Momentum 偏差修正','']
    for r in paired:
        if r['metric']=='accuracy' and r['scheme'] in ('A','B') and (r['left'],r['right']) in [('momentum','momentum+iid'),('momentum','momentum-bias')]:
            lines.append(f"{r['scheme']} / {r['task']}：{r['right']}−{r['left']} = {r['mean']:.5f}，95% CI [{r['ci95_low']:.5f}, {r['ci95_high']:.5f}]。")
    lines+=['Momentum 与 Momentum-Bias 的真实时间工作负载不同；主实验使用相同相关噪声滤波器，未针对未校正 Momentum 重新优化滤波器。',
        '', '## 3. STP-EX2 是否稳定改善','']
    for task in ('cv','nlp'):
        comparisons=[r for r in paired if r['task']==task and r['scheme'] in ('A','B') and r['metric']=='accuracy'
                     and 'stpex2' in (r['left'],r['right']) and any(m in (r['left'],r['right']) for m in ('stpex','stpex-adam3'))]
        stable=True
        for r in comparisons:
            direction=1 if r['right']=='stpex2' else -1
            mean=direction*r['mean'];lo=r['ci95_low'] if direction==1 else -r['ci95_high'];hi=r['ci95_high'] if direction==1 else -r['ci95_low']
            stable &= lo>0
            other=r['left'] if direction==1 else r['right']
            lines.append(f"{task}/{r['scheme']}：stpex2−{other} = {mean:.5f}，95% CI [{lo:.5f}, {hi:.5f}]。")
        lines.append(f"{task} 在两方案下相对于另外两种二阶矩均有正向配对区间支持：{'是' if stable else '否'}。这不构成所有任务与配置上的普遍优势。")
    lines+=['','## 4. 排名变化与历史基线','']
    for r in ranks:
        lines.append(f"{r['task']}/{r['scheme']}：Exp15 {' > '.join(r['exp15'])}；Exp15b {' > '.join(r['exp15b'])}；原五组排名是否变化：{'是' if r['existing_five_changed'] else '否'}。")
    for task in ('cv','nlp'):
        best=max((r for r in summary if r['task']==task),key=lambda r:r['mean'])
        for h in historical:
            if h['task']==task:
                lines.append(f"{task} 最佳 {best['scheme']}/{best['method']} 相对 {h['experiment']}/{h['method']} 准确率均值差 {best['mean']-h['mean']:.5f}（描述性跨实验差异）。")
    lines+=['','## 5. 搜索边界','']
    for r in summary:
        if r['lr_boundary_at_selected_clip'] or r['clip_boundary']:
            lines.append(f"{r['task']}/{r['scheme']}/{r['method']}：lr={r['lr']:g}、C/R={r['clip']:g}；条件学习率边界={r['lr_boundary_at_selected_clip']}，裁剪边界={r['clip_boundary']}，需要未来独立实验验证；本次不追加网格。")
    lines+=['','## 6. 统计证据及局限','',
        '各自调参后的方法比较、两种方案比较均有十种子配对差及区间，见 paired_differences.csv。区间跨零表示当前十种子不足以明确方向；区间不跨零为未经多重比较校正的探索性支持。固定超参数搜索差及跨实验基线均按其实际样本范围作描述，不自动视为严格单因素或新盲测证据。',
        '噪声标准差、边际协方差、完整协方差文件、裁剪比例、更新范数与实际 Momentum 工作负载效用见 noise_comparison.csv；非有限值及原因保留于各阶段结果。',
        f"本次合并 {len(search)} 个候选，追加复核 {len(review['rows'])} 次；正式配置去重后 {len(formal)} 次，其中复用 {sum(bool(r.get('reuse')) for r in formal)} 次。历史只读和结果哈希审计通过后才生成本报告。"]
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
