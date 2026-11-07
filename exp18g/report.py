"""Seven official algorithms, five paired seeds, and separate internal monitoring."""
import csv
import math
import numpy as np
import torch
from scipy.stats import t
from exp18g.common import *

LABELS=['Adam-IID','Adam-B²LT-SGD','Momentum-IID','Momentum-B²LT-Momentum','STP-Ex2-IID','STP-Ex2-B²LT-Momentum-Bias','STP-B²LT']

def stats(values):
    assert len(values)==5
    mean=float(np.mean(values));sd=float(np.std(values,ddof=1));half=float(t.ppf(.975,4)*sd/math.sqrt(5))
    return dict(mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half)

def report():
    verify_seal()
    old=list(csv.DictReader((ROOT/'exp18f/final_official/per_seed.csv').open()))
    methods=list(dict.fromkeys(r['method'] for r in old))+[METHOD]
    labels=dict(zip(methods,LABELS));rows=[];pairings={};sources={}
    for r in old:
        out=ROOT/r['directory'];result=read(out/'summary.json')
        assert result['formal'] and result['steps']==310 and result['evaluation']=='official_validation'
        assert result['metrics']['examples']==872
        seed=int(r['seed']);pair=result['pairing']
        if seed in pairings:assert pairings[seed]==pair
        pairings[seed]=pair
        sources[str(out.relative_to(ROOT))]=sha(out/'summary.json')
        stp=r['method'].startswith('stpex2')
        rows.append(dict(method=r['method'],seed=seed,lr=float(r['lr']),clip=float(r['clip']),
            eps_scale='',optimizer_eps=float(r['eps']),beta1=.9,beta2=.999,weight_decay=0,
            accuracy=float(r['accuracy']),loss=float(r['loss']),steps=310,evaluation='official_validation',
            examples=872,source='Exp18f final official reused',directory=r['directory']))
    internals=[];gpus=set()
    for c in configs():
        result=completed(c);assert result is not None
        out=outdir(c);assert result['evaluation']=='official_validation' and result['metrics']['examples']==872
        assert result['pairing']==pairings[c['seed']]
        assert result['privacy']==read(BASE/'acceptance/privacy_audit.json')['per_seed'][str(c['seed'])]
        curve=read(out/'curve.json');assert [p['step'] for p in curve]==[62,124,186,248,310]
        assert all(p['internal']['examples']==5349 for p in curve)
        ckpt=torch.load(out/'checkpoint.pt',weights_only=True,map_location='cpu')
        assert ckpt['steps']==310 and ckpt['config']==c and ckpt['pairing']==result['pairing']
        assert all(int(state['step'])==310 for state in ckpt['optimizer']['state'].values())
        assert all(torch.isfinite(p).all() for p in ckpt['model'].values())
        del ckpt
        gpus.add(result['gpu_uuid']);sources[str(out.relative_to(ROOT))]=sha(out/'summary.json')
        rows.append(dict(method=METHOD,seed=c['seed'],lr=c['lr'],clip=c['clip'],eps_scale=c['eps_scale'],
            optimizer_eps=c['adam_eps'],beta1=.9,beta2=.999,weight_decay=0,**result['metrics'],steps=310,
            evaluation='official_validation',source='Exp18g new formal',directory=str(out.relative_to(ROOT))))
        internals.append(dict(seed=c['seed'],evaluation='internal_validation',**result['internal_metrics']))
    assert len(rows)==35 and len(gpus)==4
    write_csv(BASE/'results/per_seed.csv',rows,list(rows[0]))
    write_csv(BASE/'results/internal_monitoring.csv',internals,list(internals[0]))
    summaries=[]
    for m in methods:
        selected=sorted([r for r in rows if r['method']==m],key=lambda r:r['seed'])
        assert [r['seed'] for r in selected]==SEEDS
        s=dict(method=m,label=labels[m],completed_seeds=5,**{k:selected[0][k] for k in ('lr','clip','eps_scale','optimizer_eps','beta1','beta2','weight_decay')})
        s.update({f"accuracy_seed_{r['seed']}":r['accuracy'] for r in selected})
        for metric in ('accuracy','loss'):s.update({f'{metric}_{k}':v for k,v in stats([r[metric] for r in selected]).items()})
        summaries.append(s)
    write_csv(BASE/'results/seven_algorithm_summary.csv',summaries,list(summaries[0]))
    write_csv(BASE/'results/summary.csv',summaries[-1:],list(summaries[-1]))
    new={r['seed']:r for r in rows if r['method']==METHOD};pairs=[]
    for m in methods[:-1]:
        ds=[dict(seed=r['seed'],accuracy_difference=new[r['seed']]['accuracy']-r['accuracy'],
            loss_difference=new[r['seed']]['loss']-r['loss']) for r in rows if r['method']==m]
        pairs.append(dict(reference=m,direction='STP-B2LT minus reference',per_seed=ds,
            accuracy=stats([d['accuracy_difference'] for d in ds]),loss=stats([d['loss_difference'] for d in ds])))
    save(BASE/'results/paired_statistics.json',dict(seeds=SEEDS,units='accuracy fraction; cross entropy',
        confidence='two-sided Student-t(df=4); descriptive intervals; no multiplicity correction',comparisons=pairs))
    matrix=read(BASE/'acceptance/matrix_audit.json');launch=read(BASE/'launch.json')
    lines=['# Exp18g STP-B²LT 正式实验报告','',
        '已完成5/5次正式训练；没有超参数搜索。每次均完整5轮×62步=310步，官方验证仅评价保存后的第310步最终模型。',
        '固定 lr=0.01、C=500、eps_scale=0.001；Adam β₁=0.9、β₂=0.999、eps=1e-8、weight decay=0。',
        '严格复用Exp16a的缩放、裁剪和恢复实现，使用标准torch.optim.Adam；无STP-Ex2更新或自适应α。',
        '本地SST-2/BERT-Tiny全参数微调；长度128、FP32、TF32关闭，逻辑及物理批次1000、累积1；离线读取缓存。',
        '种子为20261111、20261112、20261113、20261114、20261115；初始化及训练排列哈希与Exp18f逐种子一致，Dropout每步seed+100000+step。','',
        '## 七算法官方验证（872条）','',
        '以下准确率、SD和CI均为百分数。SD使用ddof=1；95%CI使用五种子的Student-t(df=4)。优化器eps列在STP-Ex2中表示其稳定系数；STP-B²LT单列STP的eps_scale。','',
        '|算法|lr|C|eps_scale|优化器eps|20261111|20261112|20261113|20261114|20261115|均值|样本SD|95%CI|平均交叉熵|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for s in summaries:
        cells='|'.join(f"{100*s[f'accuracy_seed_{seed}']:.3f}" for seed in SEEDS)
        scale=s['eps_scale'] if s['eps_scale']!='' else '—'
        lines.append(f"|{s['label']}|{s['lr']:g}|{s['clip']:g}|{scale}|{s['optimizer_eps']:g}|{cells}|{100*s['accuracy_mean']:.3f}|{100*s['accuracy_sd']:.3f}|[{100*s['accuracy_ci95_low']:.3f}, {100*s['accuracy_ci95_high']:.3f}]|{s['loss_mean']:.6f}|")
    lines += ['', '前六算法完全复用Exp18f最终官方结果；其Adam/Momentum行原始来源为Exp18e。未重新训练这些算法。',
        '', '## STP-B²LT 减其他算法的配对差异','',
        '单位：百分点；描述性95%配对CI，未做多重比较校正。不同算法超参数不同，差异不能仅归因于噪声机制。','',
        '|参考算法|平均差|95%配对CI|','|---|---:|---|']
    for p in pairs:
        s=p['accuracy'];lines.append(f"|{labels[p['reference']]}|{100*s['mean']:+.3f}|[{100*s['ci95_low']:+.3f}, {100*s['ci95_high']:+.3f}]|")
    lines += ['', '## 内部验证（5349条，仅监控）','',
        '与上述官方验证分开记录，不用于早停、选择模型或调参。每轮结果见各seed目录curve.json；末轮结果如下。','',
        '|种子|内部准确率(%)|内部交叉熵|','|---|---:|---:|']
    for r in internals:lines.append(f"|{r['seed']}|{100*r['accuracy']:.3f}|{r['loss']:.6f}|")
    a=calibration(configs()[0])
    lines += ['', '## 验收与运行','',
        '数学测试、矩阵审计、隐私校准、四卡真实冒烟测试全部通过。矩阵使用Exp18d的SGD工作负载，T=310、K=P=62、Q=5。',
        f"矩阵SHA256：`{matrix['sha256']}`。求解器收敛；终点一阶投影梯度={matrix['endpoint_projected_gradient_inf']:.12g}。",
        f"沿用Exp18d的选择规则：保存轨迹中未惩罚误差最小的矩阵；它与带障碍项目标的收敛终点不同。所选矩阵重算梯度={matrix['selected_projected_gradient_inf']:.12g}；一阶通过指求解器终点。",
        f"每次训练独立ε=8、δ=1e-5，无抽样放大；敏感度={a['sensitivity']:.12g}，C=500，噪声和标准差={a['sigma_sum']:.12g}，批平均标准差={a['sigma_average']:.12g}。",
        '五次训练的组合隐私预算不宣称仍为ε=8；逐训练审计见formal/seed*/privacy.json。原始内部监控量不是DP发布。',
        f"后台启动器PID={launch['pid']}；nohup启动命令见launch.json；控制器PID见pipeline_status.json；完整总日志logs/pipeline.log，工作PID和命令见logs/formal_events.jsonl。",
        '每卡最多一个进程，最多4并行，无DDP；第5种子自动使用空闲卡。全部训练均独立初始化优化器状态。']
    for gpu,uuid in read(BASE/'scheduling.json')['gpu_uuids'].items():lines.append(f'- GPU {gpu}: RTX 3080 Ti (12288 MiB), `{uuid}`')
    lines += ['', '验收证据位于acceptance/；逐步日志、检查点、官方结果及完整性哈希位于formal/seed*/。',
        'input_manifest.json封存只读输入，code_manifest.json封存执行代码。最终检查确认输入与封存哈希一致。']
    (BASE/'results/final_report.md').write_text('\n'.join(lines)+'\n')
    save(BASE/'results/completion_verification.json',dict(passed=True,formal_completed=5,steps_each=310,
        official_examples=872,internal_examples=5349,seven_algorithms=7,paired_comparisons=6,
        four_gpus_used=True,all_seeds_paired=True,source_hashes=sources,input_integrity=True,time=now(),
        results_sha256={p.name:sha(p) for p in (BASE/'results').iterdir() if p.name!='completion_verification.json'}))
    print('REPORT COMPLETE: 5 new runs, 30 historical results, 7 algorithms',flush=True)
if __name__=='__main__':report()
