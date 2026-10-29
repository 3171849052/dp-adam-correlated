"""Ten-seed paired statistics; tuned results and controlled comparisons stay separate."""
from itertools import combinations
import numpy as np
from scipy.stats import t
from exp16c import RESULTS,ROOT
from exp16c.config import *
from exp16c.support import read,write_csv,code_hashes
from exp16c.search import config
from exp16c.scheduler import verify
from exp16c.history import verify_path

def stats(values):
    x=np.asarray(values,dtype=float);mean=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return dict(mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half)
def paired(a,b):
    for key in ('seed','initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
        if a[key]!=b[key]:raise ValueError(f'Unpaired random streams: {key}')
def report():
    from exp16c.stage1 import verify_acceptance,verify_readonly
    verify_acceptance();rows=read(RESULTS/'formal_results.json');freeze=read(RESULTS/'final_freeze.json')
    if len(rows)!=10*len(METHODS) or any(r['status']!='completed' for r in rows):raise ValueError('Formal training incomplete')
    for r in rows:
        if r.get('historical'):verify_path(ROOT/r['source'])
        else:verify(config(r))
        if any(r[k]!=freeze['winners']['cv'][r['method']][k] for k in ('lr','clip','eps_scale')):raise ValueError('Formal configuration mismatch')
    groups={m:sorted([r for r in rows if r['method']==m],key=lambda r:r['seed']) for m in METHODS}
    summaries=[];per_seed=[];pairs=[];pair_seeds=[]
    for m,runs in groups.items():
        if [r['seed'] for r in runs]!=list(FINAL_SEEDS):raise ValueError('Missing formal seed')
        c=freeze['winners']['cv'][m];out=dict(method=m,lr=c['lr'],clip=c['clip'],eps_scale=c['eps_scale'],n=10)
        for metric,scale in [('accuracy',100),('validation_loss',1),('seconds',1),('train_seconds',1),('peak_allocated_bytes',1),('peak_reserved_bytes',1)]:
            out.update({f'{metric}_{k}':v for k,v in stats([scale*r[metric] for r in runs]).items()})
        out['clip_fraction_mean']=float(np.mean([np.mean([e['clip_fraction'] for e in r['epochs']]) for r in runs]));summaries.append(out)
        for r in runs:
            per_seed.append(dict(method=m,seed=r['seed'],lr=r['lr'],clip=r['clip'],eps_scale=r['eps_scale'],accuracy_percent=100*r['accuracy'],loss=r['validation_loss'],clip_fraction=float(np.mean([e['clip_fraction'] for e in r['epochs']])),last_epoch_clip_fraction=r['epochs'][-1]['clip_fraction'],epsilon=r['actual_epsilon'],seconds=r['seconds'],peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes'],historical=r.get('historical',False),trial_id=r['trial_id']))
    for left,right in combinations(METHODS,2):
        for a,b in zip(groups[left],groups[right]):
            paired(a,b);pair_seeds.append(dict(left=left,right=right,seed=a['seed'],accuracy_difference_pp=100*(b['accuracy']-a['accuracy']),loss_difference=b['validation_loss']-a['validation_loss']))
        for metric,scale in [('accuracy',100),('validation_loss',1)]:
            pairs.append(dict(left=left,right=right,direction='right-minus-left',metric=metric,units='percentage_points' if metric=='accuracy' else 'cross_entropy',n=10,**stats([scale*(b[metric]-a[metric]) for a,b in zip(groups[left],groups[right])])))
    write_csv(RESULTS/'summary.csv',summaries);write_csv(RESULTS/'formal_per_seed.csv',per_seed)
    write_csv(RESULTS/'paired_differences.csv',pairs);write_csv(RESULTS/'paired_per_seed.csv',pair_seeds)
    write_csv(RESULTS/'momentum_comparison.csv',[p for p in pairs if p['left'] in MOMENTUM_METHODS and p['right'] in MOMENTUM_METHODS])
    controlled=[];internal=read(RESULTS/'search_results.json')+read(RESULTS/'review_results.json')
    for left,right in combinations(MOMENTUM_METHODS,2):
        a={(r['lr'],r['clip'],r['seed']):r for r in internal if r['method']==left and r['status']=='completed'}
        b={(r['lr'],r['clip'],r['seed']):r for r in internal if r['method']==right and r['status']=='completed'}
        for key in sorted(a.keys() & b.keys()):
            paired(a[key],b[key]);controlled.append(dict(left=left,right=right,lr=key[0],clip=key[1],seed=key[2],accuracy_difference_pp=100*(b[key]['accuracy']-a[key]['accuracy']),loss_difference=b[key]['validation_loss']-a[key]['validation_loss'],evaluation='internal',note='matched hyperparameters; no ten-seed causal inference'))
    write_csv(RESULTS/'controlled_momentum.csv',controlled)
    matrices=read(RESULTS/'matrix_audit.json');write_csv(RESULTS/'matrix_errors.csv',matrices)
    workload_errors=[]
    from exp16c.matrices import load,workload
    from exp16c.privacy import calibrate
    for m in MOMENTUM_METHODS:
        cfg= config(groups[m][0]);a=load(cfg.workload,cfg.mechanism,250);cal=calibrate(a['C'],cfg.protocol());W=workload('momentum',250)
        # This W already contains (1-beta); do not multiply by .1 again.
        mse=float(cal['innovation_std_sum']**2*np.sum((W@a['D'])**2)/250/1e6)
        workload_errors.append(dict(method=m,clip=cfg.clip,lr=cfg.lr,momentum_noise_mse_per_coordinate=mse,parameter_noise_mse_per_coordinate=cfg.lr**2*mse,note='linear noise workload only; not nonlinear training trajectory error'))
    write_csv(RESULTS/'momentum_workload_errors.csv',workload_errors)
    counts={};failures=[]
    for stage in ('search','review','formal'):
        states=read(RESULTS/f'{stage}_status.json');merged=read(RESULTS/f'{stage}_results.json')
        counts[stage]=dict(new_planned=len(states),new_completed=sum(r['status']=='completed' for r in states.values()),historical_completed=sum(r.get('historical',False) for r in merged),new_non_finite=sum(r['status']=='non_finite' for r in states.values()))
        failures.extend(dict(stage=stage,trial_id=k,**v) for k,v in states.items() if v['status'] not in ('completed','running','pending'))
    verify_readonly()
    audit=dict(status='completed',formal_completed=len(rows),counts=counts,failures=failures,paired_randomness_verified=True,readonly_verified=True,code_sha256=code_hashes(),actual_epsilon_range=[min(r['actual_epsilon'] for r in rows),max(r['actual_epsilon'] for r in rows)],matrix_audit=matrices,selection=read(RESULTS/'selection_evidence.json'))
    save_json(RESULTS/'final_audit.json',audit)
    lines=['# exp16c 最终报告','', '状态：正式训练及审计已完成。仅 CV / CIFAR-100 + ViT-Tiny。', '', '各方法分别调参；下表是调参后性能比较，不是固定超参数下噪声机制的因果效应。历史测试集已被使用，不声称新盲测。', '', '|方法|lr|clip|eps_scale|准确率均值 ± SD (%)|95% t CI|', '|---|---:|---:|---:|---:|---|']
    for r in summaries:lines.append(f"|{r['method']}|{r['lr']}|{r['clip']}|{r['eps_scale']}|{r['accuracy_mean']:.3f} ± {r['accuracy_sd']:.3f}|[{r['accuracy_ci95_low']:.3f}, {r['accuracy_ci95_high']:.3f}]|")
    lines += ['', '## Momentum 三方法配对差（右减左，百分点）','']
    for p in pairs:
        if p['left'] in MOMENTUM_METHODS and p['right'] in MOMENTUM_METHODS and p['metric']=='accuracy':lines.append(f"- {p['right']} − {p['left']}: {p['mean']:.3f}, 95% CI [{p['ci95_low']:.3f}, {p['ci95_high']:.3f}]")
    lines += ['', '选参证据见 `selection_evidence.json` 和 `top2.json`：搜索种子前二，追加 20261102/03，按三种子内部准确率均值、损失选优；两个 Adam 扩展保留原三种子最优作为基准。', '', '`formal_per_seed.csv` 列出十种子的准确率、损失、五轮平均及末轮裁剪比例、耗时、显存和实际预算；`paired_differences.csv` 与 `paired_per_seed.csv` 给出所有方法同种子配对差及 t95% CI。区间未作多重比较校正。', '', '`controlled_momentum.csv` 是共同 lr/clip/种子下的内部验证比较（搜索均有，复核仅取交集），不替代十种子正式受控实验。`momentum_comparison.csv` 是各自调参后的正式比较。', '', '`matrix_errors.csv` 给出矩阵逆误差、分解误差、敏感度、目标函数和预算。`momentum_workload_errors.csv` 在共同实际 Momentum workload 下计算各机制噪声误差，已含 (1−β) 因子、clip²/1000²；参数误差另乘 lr²。这些线性噪声诊断不是训练误差。', '', '单次 DP 训练 ε=8、δ=1e-5，固定五轮参与，GDP 校准，无采样放大；没有搜索、复核与重复运行的组合隐私声明。内部验证假设公开，裁剪诊断为内部审计产物。', '', '完成数量：', '```json', __import__('json').dumps(counts,indent=2), '```', f'失败/数值排除记录：{len(failures)}，详见 final_audit.json。历史正式结果仅在配置完全相同时复用。所有历史来源只读校验通过。']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':report()
