"""Ten-seed statistics, separate internal controls and historical Adam comparison."""
from itertools import combinations
import numpy as np
from scipy.stats import t
from exp16d import RESULTS,ROOT
from exp16d.config import *
from exp16d.support import read,write_csv,code_hashes
from exp16d.search import config
from exp16d.scheduler import verify

def stats(values):
    x=np.asarray(values,dtype=float);mean=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
    return dict(mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half)
def paired(a,b):
    for k in ('seed','initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace'):
        if a[k]!=b[k]:raise ValueError(f'Unpaired random streams: {k}')
def differences(groups,methods,check=True):
    out=[]
    for left,right in combinations(methods,2):
        a,b=groups[left],groups[right]
        assert [r['seed'] for r in a]==[r['seed'] for r in b]
        if check:
            for x,y in zip(a,b):paired(x,y)
        for metric,scale in [('accuracy',100),('validation_loss',1)]:
            out.append(dict(left=left,right=right,direction='right-minus-left',metric=metric,n=len(a),**stats([scale*(y[metric]-x[metric]) for x,y in zip(a,b)])))
    return out
def report():
    from exp16d.stage1 import verify_acceptance
    verify_acceptance(gpu=True)
    rows=read(RESULTS/'formal_results.json');controls=read(RESULTS/'controlled_results.json');freeze=read(RESULTS/'final_freeze.json')
    assert len(rows)==40 and len(controls)==9
    search=read(RESULTS/'search_results.json');review=read(RESULTS/'review_results.json')
    assert len(search)==20 and len(review)==16
    for r in search+review:
        verify(config(r))
        assert not r['official_evaluation_used']
    from exp16d.search import top_two,select
    selected,evidence=select(top_two(search),search+review)
    assert selected==freeze['winners'] and evidence==read(RESULTS/'selection_evidence.json')
    for r in rows+controls:verify(config(r))
    groups={m:sorted([r for r in rows if r['method']==m],key=lambda r:r['seed']) for m in METHODS}
    summaries=[]
    for m,rs in groups.items():
        assert [r['seed'] for r in rs]==list(FINAL_SEEDS)
        c=freeze['winners']['cv'][m]
        assert all(all(r[k]==c[k] for k in ('lr','clip','eps_scale')) for r in rs)
        out=dict(method=m,lr=c['lr'],clip=c['clip'],n=10)
        for metric,scale in [('accuracy',100),('validation_loss',1),('seconds',1),('train_seconds',1),('peak_allocated_bytes',1),('peak_reserved_bytes',1)]:
            out.update({f'{metric}_{k}':v for k,v in stats([scale*r[metric] for r in rs]).items()})
        out.update(clip_fraction_mean=float(np.mean([e['clip_fraction'] for r in rs for e in r['epochs']])),epsilon=8.,delta=1e-5,sensitivity=rs[0]['calibration']['sensitivity'],innovation_std_sum=rs[0]['calibration']['innovation_std_sum'])
        summaries.append(out)
    write_csv(RESULTS/'summary.csv',summaries)
    pairs=differences(groups,METHODS);write_csv(RESULTS/'paired_differences.csv',pairs)
    cg={m:sorted([r for r in controls if r['method']==m],key=lambda r:r['seed']) for m in METHODS[1:]}
    for rs in cg.values():
        assert [r['seed'] for r in rs]==[SEARCH_SEED,*RECHECK_SEEDS]
        assert all(r['lr']==.0004 and r['clip']==20 and not r['official_evaluation_used'] for r in rs)
    cp=differences(cg,METHODS[1:]);write_csv(RESULTS/'controlled_differences.csv',cp)
    write_csv(RESULTS/'controlled_summary.csv',[dict(method=m,lr=.0004,clip=20,n=3,**stats([100*r['accuracy'] for r in rs])) for m,rs in cg.items()])
    historical=[];sources={}
    for m in METHODS[1:]:
        old=m.replace('adam3-','adam-',1);rs=[]
        for p in (ROOT/'exp16a/results/final/cv').glob('*/summary.json'):
            r=read(p)
            if r['method']!=old:continue
            assert r['status']=='completed' and r['optimizer_steps']==250
            for name,h in r['file_sha256'].items():assert file_hash(p.parent/name)==h
            sources[str(p.relative_to(ROOT))]=file_hash(p);rs.append(r)
        rs.sort(key=lambda r:r['seed']);assert [r['seed'] for r in rs]==list(FINAL_SEEDS)
        for metric,scale in [('accuracy',100),('validation_loss',1)]:
            historical.append(dict(adam3_method=m,adam_method=old,adam3_lr=groups[m][0]['lr'],adam3_clip=groups[m][0]['clip'],adam_lr=rs[0]['lr'],adam_clip=rs[0]['clip'],metric=metric,n=10,adam_mean=float(np.mean([scale*r[metric] for r in rs])),adam3_mean=float(np.mean([scale*r[metric] for r in groups[m]])),**stats([scale*(a[metric]-b[metric]) for a,b in zip(groups[m],rs)])))
    write_csv(RESULTS/'adam_comparison.csv',historical)
    write_csv(RESULTS/'privacy_calibration.csv',[dict(method=r['method'],stage=r['stage'],seed=r['seed'],clip=r['clip'],epsilon=r['actual_epsilon'],**r['calibration']) for r in rows])
    lines=['# exp16d 最终报告','','状态：40 次正式训练完成。CIFAR-100 / ViT-Tiny，FP32，5 epochs。','','|方法|lr|clip|准确率均值 ± SD (%)|95% t CI|loss|裁剪比例|秒|峰值显存 GiB|','|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"|{r['method']}|{r['lr']}|{r['clip']}|{r['accuracy_mean']:.3f} ± {r['accuracy_sd']:.3f}|[{r['accuracy_ci95_low']:.3f}, {r['accuracy_ci95_high']:.3f}]|{r['validation_loss_mean']:.4f}|{r['clip_fraction_mean']:.4f}|{r['seconds_mean']:.1f}|{r['peak_reserved_bytes_mean']/2**30:.2f}|")
    lines+=['','## B²LT workload 正式比较（各自调参）']
    for p in pairs:
        if p['left'] in METHODS[1:] and p['right'] in METHODS[1:] and p['metric']=='accuracy':lines.append(f"- {p['right']} − {p['left']}: {p['mean']:.3f} pp，95% CI [{p['ci95_low']:.3f}, {p['ci95_high']:.3f}]。")
    lines+=['','## 公共配置内部验证（三种子，lr=0.0004 / clip=20）']
    for p in cp:
        if p['metric']=='accuracy':lines.append(f"- {p['right']} − {p['left']}: {p['mean']:.3f} pp，95% CI [{p['ci95_low']:.3f}, {p['ci95_high']:.3f}]。")
    lines+=['三种子公共配置不作为十种子正式结论。','','## 与 exp16a 标准 Adam 对照']
    for r in historical:
        if r['metric']=='accuracy':lines.append(f"- {r['adam3_method']}: AdaM³ {r['adam3_mean']:.3f}% vs Adam {r['adam_mean']:.3f}%，同种子差均值 {r['mean']:.3f} pp。Adam lr={r['adam_lr']}, clip={r['adam_clip']}。")
    lines+=['','不同超参数的比较不能直接解释为优化器或 workload 本身的因果效应。历史官方测试集已使用，不声称新盲测。区间是种子间 Student-t 区间，未作多重比较校正。','', '单次训练 ε=8、δ=1e-5，固定五轮参与，无采样放大，GDP 校准；每个 clip 独立校准。全部搜索与重复训练不具有联合 ε=8 保证。内部验证及训练诊断按内部研究数据处理。','', '隐私数值见 privacy_calibration.csv；矩阵校验见 matrix_audit.json；选参证据见 selection_evidence.json；逐 trial 日志、checkpoint、耗时及显存保存在对应目录。']
    (RESULTS/'final_report.md').write_text('\n'.join(lines)+'\n')
    save_json(RESULTS/'final_audit.json',dict(status='completed',formal_completed=40,controlled_completed=9,search_completed=20,review_completed=16,unique_full_trainings=len({r['trial_id'] for r in search+review+controls+rows}),paired_randomness_verified=True,code_sha256=code_hashes(),historical_sources_sha256=sources,epsilon_range=[min(r['actual_epsilon'] for r in rows),max(r['actual_epsilon'] for r in rows)],report_sha256={n:file_hash(RESULTS/n) for n in ('search_results.json','selection_evidence.json','final_freeze.json','summary.csv','paired_differences.csv','controlled_differences.csv','final_report.md')}))
