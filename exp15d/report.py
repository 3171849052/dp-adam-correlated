"""Ten-seed statistics and paired historical comparisons, without test-driven tuning."""
import itertools
import json
import csv
from collections import defaultdict
import numpy as np
from scipy.stats import t
from exp15d import ROOT, RESULTS
from exp15d.config import METHODS, FINAL_SEEDS, inherited_eps, save_json, file_hash
from exp15d.search import read, cfg_of, load_frozen, FROZEN, SELECTED, REVIEW
from exp15d.scheduler import completed
from exp15d.audit import check_protected


def stats(values):
    a=np.asarray(values,dtype=float)
    if len(a)<2 or not np.isfinite(a).all(): raise ValueError('Finite repeated measurements required')
    mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    return dict(n=len(a),mean=mean,sample_std=sd,ci95_low=mean-half,ci95_high=mean+half)


def write_csv(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def directory(row,historical=False):
    if row.get('reuse'): return ROOT/row['reuse']['source_directory']
    if historical: return ROOT/'exp15b/results'/row['stage']/row['replicate']/row['task']/row['trial_id']
    return cfg_of(row).output


def assert_paired(left,right):
    for key in ('task','seed','initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','noise_rng_start','noise_rng_end'):
        if left[key]!=right[key]: raise ValueError(f'Unpaired random streams: {key}')


def paired(left_rows,right_rows,**labels):
    a={r['seed']:r for r in left_rows};b={r['seed']:r for r in right_rows}
    if set(a)!=set(FINAL_SEEDS) or set(b)!=set(FINAL_SEEDS): raise ValueError('Missing ten formal seeds')
    for seed in FINAL_SEEDS: assert_paired(a[seed],b[seed])
    return [dict(**labels,metric=metric,direction='right minus left',**stats([b[s][metric]-a[s][metric] for s in FINAL_SEEDS])) for metric in ('accuracy','validation_loss')]


def sensitivity(rows,stage1_ids):
    output=[]
    for r in rows:
        row={k:r[k] for k in ('task','method','lr','clip','eps_scale','stage','seed','trial_id','status')}
        row.update(search_scope='fixed_lr_R_screening' if r['trial_id'] in stage1_ids else 'joint_lr_R_eps',accuracy=r.get('accuracy'),cross_entropy=r.get('validation_loss'))
        steps=[json.loads(s) for s in (directory(r)/'diagnostics.jsonl').read_text().splitlines()] if (directory(r)/'diagnostics.jsonl').exists() else []
        if steps:
            row.update(clip_fraction_mean=float(np.mean([s['clip_fraction'] for s in steps])),
                clip_fraction_max=max(s['clip_fraction'] for s in steps),
                clip_saturation_step_fraction=float(np.mean([s['clip_fraction']>=.99 for s in steps])),
                denominator_mean=float(np.mean([s['denominator_mean'] for s in steps])),
                denominator_min=min(s['denominator_min'] for s in steps),denominator_max=max(s['denominator_max'] for s in steps),
                update_norm_max=max(s['update_norm'] for s in steps),v_mean=float(np.mean([s['v_mean'] for s in steps])),v_max=max(s['v_max'] for s in steps))
        output.append(row)
    return output


def curves(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=RESULTS/'curves';out.mkdir(exist_ok=True)
    for task,method in itertools.product(('cv','nlp'),METHODS):
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for scope,label in [('fixed_lr_R_screening','Fixed lr/R'),('joint_lr_R_eps','Joint search (validation-best per eps)')]:
            group=[r for r in rows if r['task']==task and r['method']==method and r['status']=='completed' and r['search_scope']==scope]
            # Include the reused screening points in the joint candidate set.
            if scope=='joint_lr_R_eps':
                plan=read(RESULTS/'joint_plan.json');ids={cfg_of(c).id for c in plan['trials']}
                group=[r for r in rows if r['task']==task and r['method']==method and r['status']=='completed' and r['trial_id'] in ids]
            best={}
            for r in sorted(group,key=lambda r:(-r['accuracy'],r['cross_entropy'],r['trial_id'])): best.setdefault(r['eps_scale'],r)
            values=[best[e] for e in sorted(best)]
            for ax,key,title in zip(axes,('accuracy','cross_entropy','clip_fraction_mean'),('Accuracy','Cross entropy','Clipping fraction')):
                ax.plot([r['eps_scale'] for r in values],[r[key] for r in values],marker='o',label=label)
                ax.set_xscale('log');ax.set_xlabel('eps_scale');ax.set_title(title);ax.grid(alpha=.3)
        axes[0].legend(fontsize=7);fig.suptitle(f'{task} / {method}: internal validation, seed 20261101')
        fig.tight_layout();fig.savefig(out/f'{task}_{method}.png',dpi=160);plt.close(fig)


def main():
    load_frozen();selected=read(SELECTED);plan=read(REVIEW)
    search=read(RESULTS/'search_results.json');review=read(RESULTS/'review_results.json')['rows'];formal=read(RESULTS/'formal_results.json')
    if selected['review_results_sha256']!=file_hash(RESULTS/'review_results.json') or plan['search_results_sha256']!=file_hash(RESULTS/'search_results.json'): raise ValueError('Selection artifacts changed')
    joint=read(RESULTS/'joint_plan.json');first=read(RESULTS/'stage1_results.json')
    if joint['stage1_results_sha256']!=file_hash(RESULTS/'stage1_results.json'): raise ValueError('Screening artifacts changed')
    expected_search={cfg_of(c).id for c in read(FROZEN)['stage1']+joint['trials']}
    for rows,ids in [(search,expected_search),(review,{cfg_of(c).id for c in plan['trials']}),(formal,{cfg_of(c).id for c in selected['final_trials']})]:
        if len(rows)!=len(ids) or {r['trial_id'] for r in rows}!=ids: raise ValueError('Missing/duplicate planned runs')
        for r in rows:
            if r['status']=='completed':
                if not completed(cfg_of(r)) or r['completed_epochs']!=5 or abs(r['actual_epsilon']-8)>1e-7: raise ValueError('Unaudited training')
            elif r['status']!='non_finite' or r['stage']=='final': raise ValueError('Unexpected failure')
    # Historical baseline stays immutable; dereference Exp15b's verified reuse pointers.
    historical=read(ROOT/'exp15b/results/formal_results.json');old_selected=read(ROOT/'exp15b/results/selected_configs.json')
    for r in historical:
        if r['method'] not in METHODS+('momentum',):continue
        for name,h in r['file_sha256'].items():
            if file_hash(directory(r,True)/name)!=h:raise ValueError('Historical artifact changed')
    groups={};old_groups={};summary=[];differences=[]
    def group_for(rows,w):
        group=sorted([r for r in rows if r['task']==w['task'] and r['method']==w['method'] and r['lr']==w['lr'] and r['clip']==w['clip'] and (r.get('eps_scale',inherited_eps(w['task']))==w.get('eps_scale',inherited_eps(w['task'])))],key=lambda r:r['seed'])
        if [r['seed'] for r in group]!=list(FINAL_SEEDS):raise ValueError('Incomplete formal group')
        return group
    for w in old_selected['winners']:
        if w['method'] in METHODS+('momentum',):old_groups[(w['scheme'],w['task'],w['method'])]=group_for(historical,w)
    for w in selected['winners']:
        key=(w['scheme'],w['task'],w['method']);group=group_for(formal,w);groups[key]=group
        labels={k:w[k] for k in ('scheme','task','method','lr','clip','eps_scale')}
        summary.append(dict(**labels,experiment='Exp15d',**stats([r['accuracy'] for r in group]),**{f'loss_{k}':v for k,v in stats([r['validation_loss'] for r in group]).items()}))
        for baseline in (w['method'],'momentum'):
            differences+=paired(old_groups[(w['scheme'],w['task'],baseline)],group,scheme=w['scheme'],task=w['task'],left=f'Exp15b/{baseline}',right=f'Exp15d/{w["method"]}',comparison='independently_tuned_historical_baseline')
    for key,group in groups.items():
        scheme,task,method=key;baseline=old_groups[key]
        if all(group[0][k]==baseline[0][k] for k in ('lr','clip')):
            differences+=paired(baseline,group,scheme=scheme,task=task,left=f'Exp15b/{method}',right=f'Exp15d/{method}',comparison='fixed_lr_R_eps_only_historical')
    for (scheme,task,method),group in old_groups.items():
        summary.append(dict(scheme=scheme,task=task,method=method,lr=group[0]['lr'],clip=group[0]['clip'],eps_scale=inherited_eps(task) if method in METHODS else None,experiment='Exp15b',**stats([r['accuracy'] for r in group]),**{f'loss_{k}':v for k,v in stats([r['validation_loss'] for r in group]).items()}))
    for scheme,task in itertools.product(('A','B'),('cv','nlp')):
        for left,right in itertools.combinations(METHODS,2):differences+=paired(groups[(scheme,task,left)],groups[(scheme,task,right)],scheme=scheme,task=task,left=left,right=right,comparison='independently_tuned_Exp15d')
    controlled=[];lookup={(r['task'],r['method'],r['lr'],r['clip'],r['stage'],r['seed'],r['eps_scale']):r for r in search+review+formal if r['status']=='completed'}
    for key,r in lookup.items():
        historical_eps=inherited_eps(r['task']);base=lookup.get(key[:-1]+(historical_eps,))
        if base is None or r['eps_scale']==historical_eps: continue
        assert_paired(base,r)
        for metric in ('accuracy','validation_loss'):
            controlled.append(dict(**{k:r[k] for k in ('task','method','lr','clip','stage','seed','eps_scale')},baseline_eps=historical_eps,metric=metric,difference=r[metric]-base[metric],comparison='fixed_lr_R_eps_only',evaluation_scope=r['evaluation_scope']))
    for key,group in groups.items():
        baseline=old_groups[key]
        if not all(group[0][k]==baseline[0][k] for k in ('lr','clip')):continue
        for base,r in zip(baseline,group):
            assert_paired(base,r)
            for metric in ('accuracy','validation_loss'):
                controlled.append(dict(**{k:r[k] for k in ('task','method','lr','clip','stage','seed','eps_scale')},baseline_eps=inherited_eps(r['task']),metric=metric,difference=r[metric]-base[metric],comparison='fixed_lr_R_eps_only_historical',evaluation_scope=r['evaluation_scope']))
    diag=sensitivity(search,{r['trial_id'] for r in first});curves(diag)
    write_csv(RESULTS/'summary.csv',summary);write_csv(RESULTS/'paired_differences.csv',differences)
    write_csv(RESULTS/'controlled_differences.csv',controlled,['task','method','lr','clip','stage','seed','eps_scale','baseline_eps','metric','difference','comparison','evaluation_scope'])
    write_csv(RESULTS/'eps_sensitivity.csv',diag)
    rankings=[]
    for scheme,task in itertools.product(('A','B'),('cv','nlp')):
        rankings.append(dict(scheme=scheme,task=task,**{exp:[r['method'] for r in sorted([r for r in summary if r['experiment']==exp and r['scheme']==scheme and r['task']==task and r['method'] in METHODS],key=lambda r:(-r['mean'],r['loss_mean']))] for exp in ('Exp15b','Exp15d')}))
    save_json(RESULTS/'statistics.json',dict(summary=summary,paired=differences,rankings=rankings))
    lines=['# Exp15d 最终报告','',
        '历史官方评价集已使用过，本次不是新盲测。选参仅使用内部验证；未根据官方结果修改网格。',
        '这是固定 lr/R 的 13 点 eps 筛选，再对保留 eps 局部联合优化，不是全局三维穷举。方案 A 固定 R=1；方案 B 联合搜索 lr/R/eps。',
        '优化器、二阶矩、Momentum-Bias 四频带 MF 和随机流协议沿用 Exp15b。历史训练全部只读，本实验重新训练所有候选。',
        'ε=8、δ=1e-5 仅是单次完整五轮训练预算，无采样放大，不包括搜索及多次训练的组合预算。原始训练诊断仅作私有审计。',
        '十种子样本标准差与同种子配对 Student-t 95% CI 未作多重比较校正。固定超参数差与独立调参差分开报告。','',
        '|实验|方案|任务|方法|eps|lr|R|准确率均值 ± 样本SD|交叉熵均值 ± 样本SD|',
        '|---|---|---|---|---|---|---|---|---|']
    for r in summary:lines.append(f"|{r['experiment']}|{r['scheme']}|{r['task']}|{r['method']}|{r['eps_scale']}|{r['lr']}|{r['clip']}|{r['mean']:.6f} ± {r['sample_std']:.6f}|{r['loss_mean']:.6f} ± {r['loss_sample_std']:.6f}|")
    lines+=['','## 最佳 eps、相对历史固定 eps 与 Momentum+MF 的改善','', '以下均为独立调参后的正式十种子差，正值表示右侧较高（交叉熵越低越好）。']
    for r in differences:
        if r['comparison']=='independently_tuned_historical_baseline':lines.append(f"{r['scheme']}/{r['task']} {r['right']} − {r['left']}，{r['metric']}：{r['mean']:.6f}，配对 95% CI [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]。")
    for r in differences:
        if r['metric']=='accuracy' and r['left']=='Exp15b/momentum':
            lines.append(f"{r['scheme']}/{r['task']}/{r['right']} 是否超过历史 Momentum+MF：均值={'是' if r['mean']>0 else '否'}；配对95%区间下界>0={'是' if r['ci95_low']>0 else '否'}。")
    lines+=['','## 单独改变 eps 的受控效果','', 'controlled_differences.csv 记录同 lr/R、同种子、同训练阶段的 eps−历史 eps 差；搜索/复核是内部验证，不能视为十种子正式单因素证据。']
    for task,method in itertools.product(('cv','nlp'),METHODS):
        group=[r for r in controlled if r['task']==task and r['method']==method and r['stage']=='search' and r['metric']=='accuracy' and r['lr']==(.5 if task=='cv' else .20 if method=='stpex2' else .15) and r['clip']==(3. if task=='cv' else 20.)]
        lines.append(f"{task}/{method} 固定 lr/R 的验证准确率差范围：[{min(r['difference'] for r in group):.6f}, {max(r['difference'] for r in group):.6f}]（配置范围，非 CI）。" if group else f'{task}/{method} 无有效受控对；见 non_finite 状态。')
    for r in differences:
        if r['comparison']=='fixed_lr_R_eps_only_historical':lines.append(f"正式固定 lr/R 对照 {r['scheme']}/{r['task']}/{r['right']}，{r['metric']} 差 {r['mean']:.6f}，同种子配对95% CI [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]。")
    lines+=['','## 联合调整后的排名','']
    for r in rankings:lines.append(f"{r['scheme']}/{r['task']}：Exp15b {' > '.join(r['Exp15b'])}；Exp15d {' > '.join(r['Exp15d'])}；变化={r['Exp15b']!=r['Exp15d']}。")
    lines+=['','## 极小 eps、裁剪与优化稳定性','', 'eps_sensitivity.csv 提供所有搜索候选的裁剪饱和步比例（逐步裁剪比例≥99%）、分母极值、最大更新范数、二阶矩和 non_finite 状态。以下是极小 eps（≤1e-6）的实际诊断，较大更新和饱和不能单独证明因果。']
    for r in diag:
        if r['eps_scale']<=1e-6:lines.append(f"{r['task']}/{r['method']} eps={r['eps_scale']:g}, lr={r['lr']:g}, R={r['clip']:g}：status={r['status']}, clip={r.get('clip_fraction_mean')}, saturation={r.get('clip_saturation_step_fraction')}, d_min={r.get('denominator_min')}, max_update={r.get('update_norm_max')}。")
    lines+=['','固定 lr/R 和联合调参的 eps—准确率、交叉熵、裁剪比例曲线见 curves/。联合曲线按每个保留 eps 的单种子验证最优 lr/R 展示，正式 eps 由额外两种子复核选定。',
        f'搜索 {len(search)}，复核 {len(review)}，正式去重 {len(formal)}；非有限搜索/复核候选 {sum(r["status"]=="non_finite" for r in search+review)}。']
    check_protected()
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')
    outputs=['search_results.json','review_results.json','formal_results.json','summary.csv','paired_differences.csv','controlled_differences.csv','eps_sensitivity.csv','final_report.md']
    save_json(RESULTS/'final_audit.json',dict(status='passed',protected_files_unchanged=True,search_count=len(search),review_count=len(review),formal_count=len(formal),
        gpus=[0,1,2,3],historical_reuse=False,official_evaluation_previously_used=True,new_blind_test=False,
        search_plan_sha256=file_hash(FROZEN),selected_sha256=file_hash(SELECTED),outputs_sha256={s:file_hash(RESULTS/s) for s in outputs}))


if __name__=='__main__':main()
