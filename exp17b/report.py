"""Stage-complete tables and paired Student-t uncertainty, no partial rankings."""
import csv
import itertools
import json
import numpy as np
from scipy.stats import t as student_t
from exp17b.common import RESULTS, R, read, output
from exp17b.monte_carlo import write_csv


def interval(values):
    x=np.asarray(values,dtype=float);mean=float(x.mean());sd=float(x.std(ddof=1))
    width=float(student_t.ppf(.975,len(x)-1)*sd/len(x)**.5)
    return dict(n=len(x),mean=mean,sd=sd,ci_low=mean-width,ci_high=mean+width)


def report():
    text=['# Exp17b 实验报告','', '仅对已完整完成的阶段生成比较；未完成阶段不产生排名。',
          '置信区间采用配对种子 Student-t，未作多重比较校正。准确率同时列出比例和百分比，差值列出百分点。',
          '每次训练独立核算 ε；搜索、重复运行及模型选择不具有联合 ε=8 保证。原始诊断和检查点属于私有审计材料。',
          '验证集假定公开。官方测试集在本实验冻结配置后读取，但已有历史实验接触。','']
    diagnostics=[];paired=[];utility=[];stage_rows={}
    names={'B':'screening_results.csv','C':'validation_results.csv','C2':'smoothing_ablation.csv','D':'final_results.csv'}
    for stage,name in names.items():
        path=RESULTS/(stage+'_results.json')
        if not path.exists():
            text += [f'## 阶段 {stage}：未完成（等待或进行中）','']
            continue
        results=read(path);stage_rows[stage]=results;rows=[]
        for s in results:
            cfg=s['config'];ds=[json.loads(line) for line in (output(cfg)/'diagnostics.jsonl').read_text().splitlines()]
            for d in ds: diagnostics.append(dict(stage=stage,method=cfg['method'],alpha=cfg['alpha'],r=cfg['r'],seed=cfg['seed'],**d))
            row=dict(**cfg,accuracy=s['accuracy'],accuracy_percent=100*s['accuracy'],cross_entropy=s['validation_loss'],
                     epsilon=s['actual_epsilon'],seconds=s['seconds'],peak_allocated_bytes=s['peak_allocated_bytes'])
            for key in ['train_loss','correction_norm','update_norm','direction_norm','direction_cosine','relative_update_scale','correction_parallel','correction_orthogonal']:
                row['mean_'+key]=float(np.mean([d[key] for d in ds if d[key] is not None]))
            row['first_update_norm']=ds[0]['update_norm'];row['first_relative_update_scale']=ds[0]['relative_update_scale']
            rows.append(row)
        write_csv(RESULTS/name,rows)
        groups={}
        for row in rows:groups.setdefault((row['method'],row['alpha'],row['r']),[]).append(row)
        text += [f'## 阶段 {stage}：完成 ({len(rows)} 次)','', '| 方法 | α | r | 准确率 % (SD) | CE (SD) |','|---|---:|---:|---:|---:|']
        for (m,a,r),values in groups.items():
            ac=interval([x['accuracy_percent'] for x in values]);ce=interval([x['cross_entropy'] for x in values])
            text.append(f"| {m} | {a} | {r:.5g} | {ac['mean']:.4f} ({ac['sd']:.4f}) | {ce['mean']:.6f} ({ce['sd']:.6f}) |")
            for metric in ['accuracy','accuracy_percent','cross_entropy','mean_train_loss','mean_correction_norm','mean_update_norm','mean_direction_cosine','seconds','peak_allocated_bytes','epsilon']:
                utility.append(dict(stage=stage,method=m,alpha=a,r=r,metric=metric,**interval([x[metric] for x in values])))
        text += ['','同 r 配对比较（CE 负值表示前者更好）：','', '| 比较 | r | Δ CE [95% CI] | Δ准确率百分点 [95% CI] |','|---|---:|---:|---:|']
        notes=[]
        for left,right in itertools.combinations(groups,2):
            if left[2]!=right[2]:continue
            # Smooth is always the subtracted baseline; GH-LC is the cross-method contrast.
            if left[0]=='smooth' or (left[0]=='lc' and right[0]=='gh'):left,right=right,left
            a={v['seed']:v for v in groups[left]};b={v['seed']:v for v in groups[right]};assert a.keys()==b.keys()
            stats={}
            for metric in ['cross_entropy','accuracy','accuracy_percent']:
                stats[metric]=interval([a[s][metric]-b[s][metric] for s in sorted(a)])
                paired.append(dict(stage=stage,left=f'{left[0]}:{left[1]}',right=f'{right[0]}:{right[1]}',r=left[2],metric=metric,**stats[metric]))
            ce,ac=stats['cross_entropy'],stats['accuracy_percent']
            text.append(f"| {left[0]} α={left[1]} − {right[0]} α={right[1]} | {left[2]:.5g} | {ce['mean']:.6f} [{ce['ci_low']:.6f}, {ce['ci_high']:.6f}] | {ac['mean']:.4f} [{ac['ci_low']:.4f}, {ac['ci_high']:.4f}] |")
            if right[0]=='smooth':
                notes.append(f"{left[0]} α={left[1]}："+('平均验证/测试 CE 未优于匹配 Smooth。' if ce['mean']>=0 else '平均 CE 改善；'+('区间仍包含零。' if ce['ci_high']>=0 else '95% 配对区间低于零。')))
                if ac['mean']<0: notes.append(f"{left[0]} α={left[1]} 平均准确率低于同 r Smooth {abs(ac['mean']):.4f} 个百分点。")
        text += ['',*notes,'']
    if diagnostics:
        write_csv(RESULTS/'diagnostics.csv',diagnostics);write_csv(RESULTS/'paired_differences.csv',paired);write_csv(RESULTS/'utility_summary.csv',utility)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,metric in zip(axes.flat,['train_loss','update_norm','correction_norm','relative_update_scale']):
            keys=sorted(set((d['stage'],d['method'],d['alpha'],d['r']) for d in diagnostics))
            for key in keys:
                subset=[d for d in diagnostics if (d['stage'],d['method'],d['alpha'],d['r'])==key]
                steps=sorted(set(d['step'] for d in subset))
                ax.plot(steps,[np.mean([d[metric] for d in subset if d['step']==s and d[metric] is not None]) for s in steps],label=str(key))
            ax.set(xlabel='step',ylabel=metric)
        axes[0,0].legend(fontsize=5);fig.tight_layout();fig.savefig(RESULTS/'training_curves.png');plt.close(fig)
    text += ['## 研究问题','']
    if (RESULTS/'mc_complete.json').exists():
        with (RESULTS/'mse_summary.csv').open() as f:mc=list(csv.DictReader(f))
        text += ['A 已完成：固定状态的 Bias、Bias²、Variance、MSE、配对标准误见 mse_summary.csv；LC−GH 比较见 mc_lc_gh_paired.csv。',
                 '每个场景的离线最优 α 见 mc_best_alpha.csv 和 mc_oracle_alpha.csv；oracle 使用已知 g，未用于训练。']
        for method in ['lc','gh']:
            means={a:np.mean([float(v['mse']) for v in mc if v['method']==method and float(v['alpha'])==a]) for a in [0.,.25,.5,1.]}
            text.append(f"{method} 在其有效扫描域上平均 MSE：{means}；该等权汇总不是普遍最优性结论。LC 排除 r=0，不能直接与 GH 全域均值比较。")
    else:text.append('A 未完成：尚不能给出 MSE 结论。')
    text += ['', '1. **部分修正是否优于完整修正？** B 表比较所有 α 的训练效用；MC 分场景表比较 MSE。'+('选定 α：'+str(read(RESULTS/'selected_alphas.json')) if (RESULTS/'selected_alphas.json').exists() else '训练筛选尚未完成。'),
             '2. **GH 是否优于 LC？** MC 的偏差、方差、MSE 必须分别判断；C、D 的 GH−LC 配对结果检验实际效用。'+('完整正式结果见 D 表。' if 'D' in stage_rows else '正式测试尚未完成。'),
             '3. **曲率修正是否有用？** 以上只用同 r、同学习率、配对种子的 Smooth 对照判断；CE 不改善时明确记录负面结果，准确率是并列结果而非替代筛选指标。',
             '4. **额外平滑是否必须？** C2 比较 r=0、1 的同 r 差值；与主 r 的选择比较仅用三个共同种子。']
    if 'C2' in stage_rows:
        for r in [0.,1.,R]:
            selected=[d for d in diagnostics if d['stage'] in ['C','C2'] and d['r']==r and d['seed'] in range(20261104,20261107) and d['step']==1]
            for m in ['smooth','gh']:
                ds=[d for d in selected if d['method']==m]
                text.append(f"r={r}, {m}, t=1：平均更新范数 {np.mean([d['update_norm'] for d in ds]):.6g}；相对尺度 {np.mean([d['relative_update_scale'] for d in ds]):.6g}；首步训练损失 {np.mean([d['train_loss'] for d in ds]):.6g}。后续损失证据见 diagnostics.csv；未引入 warmup 或额外裁剪。")
    else:text.append('C2 尚未完成，不能判断无额外平滑的实际稳定性或收益。')
    text += ['5. **MSE 改善是否转化为优化收益？** 固定状态 MSE 衡量单步估计误差；训练会改变梯度和历史状态。更新范数、相对尺度、方向余弦及平行/正交修正见 diagnostics.csv。即使 MSE 下降，放大有效步长、改变更新方向或历史反馈仍可能降低训练效用；这些是机制解释，不能当作因果验证。',
             '训练效用的结论以完整 C/D 同参数配对结果为准，不把固定路径结论直接推广为完整自适应训练定理。']
    if 'B' in stage_rows:
        text += ['', '筛选阶段的直接判断（仅短程验证，不是正式测试结论）：']
        for method in ['lc','gh']:
            losses={a:float(np.mean([x['validation_loss'] for x in stage_rows['B'] if x['config']['method']==method and x['config']['alpha']==a])) for a in [.25,.5,1.]}
            partial=min([.25,.5],key=lambda a:losses[a])
            diff=losses[partial]-losses[1.]
            text.append(f'{method} 最佳部分 α={partial} 相对完整 α=1 的平均验证 CE 差为 {diff:+.6f}；'+('部分修正更好。' if diff<0 else '没有观察到部分修正优于完整修正。'))
    for stage in ['C','D']:
        comparisons=[p for p in paired if p['stage']==stage and p['left'].startswith('gh:') and p['right'].startswith('lc:') and p['metric']=='cross_entropy']
        for p in comparisons:
            text.append(f"{stage} 的 GH−LC 平均 CE 差 {p['mean']:+.6f}，95% CI [{p['ci_low']:+.6f}, {p['ci_high']:+.6f}]；"+('GH 的平均 CE 更好。' if p['mean']<0 else '未观察到 GH 的平均 CE 优于 LC。'))
    if (RESULTS/'final_freeze.json').exists():
        decision=read(RESULTS/'final_freeze.json')['decision']
        text.append('额外平滑选择：'+str(decision)+'。这是验证选择结果，不代表所有设置均需要或均不需要平滑。')
    (RESULTS/'report.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':report()
