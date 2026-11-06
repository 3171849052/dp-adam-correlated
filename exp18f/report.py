"""Internal-only search reports, seed statistics, paired differences, and curves."""
import json
import math
from collections import defaultdict
import numpy as np
from scipy.stats import t
from exp18f.common import BASE,METHODS,PARAMS,REVIEW_SEEDS,read,save,write_csv,configs,outdir,terminal,now

def stats(values):
    n=len(values)
    if not n:return dict(n=0,mean=None,sd=None,ci95_low=None,ci95_high=None)
    mean=float(np.mean(values));sd=float(np.std(values,ddof=1)) if n>1 else None
    half=float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
    return dict(n=n,mean=mean,sd=sd,ci95_low=mean-half if half is not None else None,
        ci95_high=mean+half if half is not None else None)

def raw_comparison(review,raw):
    a,b=stats(review),stats(raw)
    result=dict(review=a,raw_search=b,difference=None,ci95_low=None,ci95_high=None,
        interpretation='Descriptive only: different seeds; raw optimum selected on search data; no unbiased superiority claim')
    if not review or not raw:return result
    difference=a['mean']-b['mean'];result['difference']=difference
    if len(review)>1 and len(raw)>1:
        v1=a['sd']**2/len(review);v2=b['sd']**2/len(raw);variance=v1+v2
        df=variance**2/(v1*v1/(len(review)-1)+v2*v2/(len(raw)-1)) if variance else None
        half=float(t.ppf(.975,df)*math.sqrt(variance)) if variance else 0.
        result.update(ci95_low=difference-half,ci95_high=difference+half)
    return result

def planned_records():
    from exp18f.search import stages
    rows=[];results={}
    for stage in stages():
        for c in configs(stage):
            r=terminal(c);directory=str(outdir(c).relative_to(BASE))
            if r:results[directory]=r
            rows.append(dict(stage=stage,candidate=c['candidate'],method=c['method'],seed=c['seed'],
                **{p:c[p] for p in PARAMS},status=r['status'] if r else 'pending',
                accuracy=r.get('metrics',{}).get('accuracy') if r else None,
                loss=r.get('metrics',{}).get('loss') if r else None,
                steps=r['steps'] if r else None,directory=directory))
    return rows,results

def make_plots(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped=defaultdict(list)
    for r in rows:
        if (BASE/r['directory']/'steps.jsonl').exists():grouped[r['candidate'],r['method']].append(r)
    for (candidate,method),group in grouped.items():
        fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
        for r in group:
            trace=[json.loads(l) for l in (BASE/r['directory']/'steps.jsonl').read_text().splitlines()]
            x=[s['step'] for s in trace]
            axes[0].plot(x,[s['train_loss'] for s in trace],alpha=.15,lw=.5)
            axes[0].plot(x,[s['train_loss_smooth10'] for s in trace],lw=1,label=str(r['seed']))
            axes[1].plot(x,[s['alpha'] for s in trace],lw=1)
            axes[2].plot(x,[s['clipped_fraction'] for s in trace],lw=1)
        for ax,label in zip(axes,['loss (raw + smooth10)','alpha','clipped fraction']):ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);axes[2].set_xlabel('logical step');fig.suptitle(f'{candidate} / {method}')
        fig.tight_layout();fig.savefig(BASE/'plots'/f'{candidate}_{method}.png',dpi=120);plt.close(fig)
    for axis in (*PARAMS,'rounded'):
        fig,ax=plt.subplots(figsize=(8,4))
        for method in METHODS:
            subset=[r for r in rows if r['method']==method and r['status']=='complete' and
                (r['candidate'].startswith(axis+'_') if axis!='rounded' else r['stage']=='rounded')]
            group=defaultdict(list)
            for r in subset:group[r['lr'] if axis=='rounded' else r[axis]].append(r['accuracy'])
            x=sorted(group);means=[stats(group[v]) for v in x]
            ax.errorbar(x,[s['mean'] for s in means],yerr=[(s['ci95_high']-s['mean']) if s['n']>1 else 0 for s in means],
                marker='o',label=method,capsize=3)
        ax.set_xlabel('lr' if axis=='rounded' else axis);ax.set_ylabel('internal final accuracy (95% t CI)')
        ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(BASE/'plots'/f'{axis}_search.png',dpi=150);plt.close(fig)

def report(plots=True):
    rows,results=planned_records()
    fields=['stage','candidate','method','seed',*PARAMS,'status','accuracy','loss','steps','directory']
    write_csv(BASE/'search/records.csv',rows,fields)
    for axis in PARAMS:write_csv(BASE/'search'/f'{axis}_results.csv',[r for r in rows if r['candidate'].startswith(axis+'_')],fields)
    write_csv(BASE/'search/rounded_results.csv',[r for r in rows if r['stage']=='rounded'],fields)
    counts=dict(planned=len(rows),complete=sum(r['status']=='complete' for r in rows),
        numerical_failures=sum(r['status']=='numerical_failure' for r in rows),
        pending=sum(r['status']=='pending' for r in rows),maximum=156,time=now())
    save(BASE/'search/counts.json',counts)
    grouped=defaultdict(list)
    for r in rows:grouped[r['stage'],r['candidate'],r['method']].append(r)
    summaries=[]
    for (stage,candidate,method),group in grouped.items():
        ok=[r for r in group if r['status']=='complete']
        summaries.append(dict(stage=stage,candidate=candidate,method=method,
            parameters={p:group[0][p] for p in PARAMS},expected_seeds=len(group),completed_seeds=len(ok),
            numerical_failures=sum(r['status']=='numerical_failure' for r in group),
            accuracy=stats([r['accuracy'] for r in ok]),loss=stats([r['loss'] for r in ok])))
    save(BASE/'search/statistics.json',summaries)
    # Each seed is one observational unit for method averages and paired differences.
    pairs=[];pooled=defaultdict(list);pair_groups=defaultdict(list)
    by_key={(r['stage'],r['candidate'],r['method'],r['seed']):r for r in rows if r['status']=='complete'}
    for r in rows:
        if r['status']!='complete' or r['method']!=METHODS[0]:continue
        b=by_key.get((r['stage'],r['candidate'],METHODS[1],r['seed']))
        if not b:continue
        assert results[r['directory']]['pairing']==results[b['directory']]['pairing']
        row=dict(stage=r['stage'],candidate=r['candidate'],seed=r['seed'],
            accuracy_difference=b['accuracy']-r['accuracy'],loss_difference=b['loss']-r['loss'])
        pairs.append(row);pair_groups[r['stage'],r['candidate']].append(row)
        pooled[r['candidate']].append((r['accuracy']+b['accuracy'])/2)
    write_csv(BASE/'search/paired_differences.csv',pairs,['stage','candidate','seed','accuracy_difference','loss_difference'])
    save(BASE/'search/paired_statistics.json',[dict(stage=s,candidate=c,
        accuracy=stats([r['accuracy_difference'] for r in group]),loss=stats([r['loss_difference'] for r in group]))
        for (s,c),group in pair_groups.items()])
    save(BASE/'search/pooled_seed_statistics.json',{c:stats(v) for c,v in pooled.items()})
    turn=read(BASE/'turning_point.json');freeze=read(BASE/'rounded_freeze.json')
    comparisons=[]
    if turn.get('raw_best'):
        raw=turn['raw_best']['candidate']
        for candidate in sorted({r['candidate'] for r in rows if r['stage']=='rounded'}):
            comparisons.append(dict(candidate=candidate,raw_candidate=raw,
                pooled_accuracy=raw_comparison(pooled[candidate],pooled[raw]),
                per_method={m:raw_comparison([r['accuracy'] for r in rows if r['candidate']==candidate and r['method']==m and r['status']=='complete'],
                    [r['accuracy'] for r in rows if r['candidate']==raw and r['method']==m and r['status']=='complete']) for m in METHODS}))
    save(BASE/'search/rounded_vs_raw.json',comparisons)
    if plots and any(r['status']!='pending' for r in rows):make_plots(rows)
    state=read(BASE/'pipeline_status.json') if (BASE/'pipeline_status.json').exists() else dict(status='not_started',stage='acceptance')
    lines=['# Exp18f 实验报告','',f"更新时间：{now()}；流水线状态：{state['status']}；阶段：{state['stage']}。",
        f"计划 {counts['planned']} 次，完成 {counts['complete']} 次，数值失败 {counts['numerical_failures']} 次；总上限156次（另有6次3步冒烟测试）。",
        '', 'SST-2 / BERT-Tiny全参数微调；5×62=310步；逻辑和物理批次1000；长度128；FP32，TF32关闭。',
        '直接读取Exp18e已有的62000训练/5349内部验证分词数据和固定划分；模型从cache/bert-tiny加载；全程离线，不读取官方验证集。',
        '第一阶段27个轴条目（基准在三轴各保留一次）×2方法×2种子=108次；第二阶段至多3轮×6=18次；整齐参数3组×2方法×5全新种子=30次。',
        '两方法使用相同超参数、初始化种子、排列及dropout顺序；每轮重复同一固定排列，无DDP。',
        '', '## 转折点与冻结', '',f"学习率峰值：{turn.get('performance_peak',{}).get('parameters') if turn.get('performance_peak') else '待搜索完成'}。",
        f"共同稳定性边界：{turn.get('common','待搜索完成')}。",
        '数值失败指NaN/Inf损失、梯度、优化器状态或模型参数；该任务终止，其他任务继续。有限值但准确率比基准低超过5个百分点或验证损失高于max(1.0,1.5×基准)标记为预测失稳；alpha衰减不判失败。',
        '基准若无成功训练，预测失稳标签仅使用协议预先固定的绝对阈值；数值失稳始终单独记录。',
        '出现失稳后再次稳定即视为非单调，改做局部几何中点加密，报告所有过渡区间；没有观测到两侧时保留删失结论。',
        f"冻结状态：{freeze['status']}；最终共同参数：{freeze.get('parameters')}。",
        '冻结只依据20261131–20261135五个新种子两方法的最终内部验证准确率均值，其次损失；不采用搜索阶段单次最高值。',
        '', '## 五种子整齐参数复核', '', '|候选|方法|完成/5|准确率均值|样本SD|95% t CI|', '|---|---|---:|---:|---:|---|']
    for s in summaries:
        if s['stage']!='rounded':continue
        a=s['accuracy'];fmt=lambda v:'待完成' if v is None else f'{v:.6f}'
        lines.append(f"|{s['candidate']}|{s['method']}|{s['completed_seeds']}/5|{fmt(a['mean'])}|{fmt(a['sd'])}|[{fmt(a['ci95_low'])}, {fmt(a['ci95_high'])}]|")
    lines += ['', '### 整齐参数与原始搜索最优的差异', '', '|复核候选|方法|复核减原始最优准确率|95% Welch CI|', '|---|---|---:|---|']
    fmt=lambda v:'待完成' if v is None else f'{v:.6f}'
    for comparison in comparisons:
        for method,value in comparison['per_method'].items():
            lines.append(f"|{comparison['candidate']}|{method}|{fmt(value['difference'])}|[{fmt(value['ci95_low'])}, {fmt(value['ci95_high'])}]|")
    if turn.get('raw_best'):
        lines += ['', f"原始共同搜索最优：{turn['raw_best']['parameters']}，候选ID={turn['raw_best']['candidate']}。"]
    for method,peak in turn.get('performance_peak_by_method',{}).items():
        lines += ['', f"{method} 的学习率性能峰值：{peak}；预测与数值联合边界：{turn['per_method'][method]}；纯数值边界：{turn['numerical_only']['per_method'][method]}。"]
    if plots and any(r['status']!='pending' for r in rows):
        lines += ['', '![学习率扫描](plots/lr_search.png)', '![裁剪阈值扫描](plots/clip_search.png)',
            '![稳定系数扫描](plots/eps_search.png)', '![整齐参数复核](plots/rounded_search.png)']
    lines += ['', '样本SD使用ddof=1，95%区间使用Student-t；不足五种子结果标记为未完成。方法间差采用逐种子配对差；双方法总体均值以五个种子为五个观测单位。',
        '整齐参数与原始搜索最优的差值及Welch区间保存于search/rounded_vs_raw.json。因种子不同且原始最优经过搜索选择，该比较仅作描述，不作为无偏优越性证据。',
        '', '## 输出与隐私', '', 'search/{lr,clip,eps}_results.csv为完整轴结果；search/statistics.json包含准确率和损失的均值、样本SD及95% CI；search/rounded_results.csv为复核明细。',
        'plots/下保存各候选两方法的逐种子损失（原始与10步平滑）、alpha和裁剪比例曲线，以及各轴与整齐参数准确率图；每个任务curve.json保存逐轮内部验证准确率和损失。',
        '每次训练独立按实际clip和对应矩阵重新校准ε=8、δ=1e-5；固定零槽增删邻接，每样本参与5次、间隔62步；不使用抽样放大。',
        'eps是优化器稳定系数，独立于隐私epsilon。搜索和复核的整体组合不具有ε=8保证；内部原始诊断不是DP发布。',
        '三张GPU使用UUID隔离，动态显存准入；每卡至多2进程，全局至多6进程。logs/保存阶段状态、调度事件和GPU遥测；launch.json保存后台PID和启动命令。',
        '基础设施故障停止当前阶段、终止同阶段工作进程并保留现场，不做静默重试。']
    (BASE/'final_report.md').write_text('\n'.join(lines)+'\n')
    return counts

if __name__=='__main__':report()
