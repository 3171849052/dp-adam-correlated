"""Merge new official STP results and unchanged Exp18e official baselines."""
import csv
import itertools
import math
import numpy as np
import torch
from scipy.stats import t
from exp18f.final_official.common import (BASE, ROOT, SEEDS, METHODS, BASELINES, configs,
    completed, outdir, read, save, sha, write_csv, now, verify_seal, verify_freeze)

LABELS = dict(zip(BASELINES+METHODS, ['Adam-IID','Adam-B²LT-SGD','Momentum-IID',
    'Momentum-B²LT-Momentum','STP-Ex2-IID','STP-Ex2-B²LT-Momentum-Bias']))


def stats(values):
    assert len(values) == 5
    mean = float(np.mean(values)); sd = float(np.std(values, ddof=1))
    half = float(t.ppf(.975, 4)*sd/math.sqrt(5))
    return dict(n=5, mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half)


def report():
    verify_seal(); verify_freeze()
    results = {}; rows = []; pairings = {}; sources = {}
    old_rows = {(r['method'], int(r['seed'])): r for r in csv.DictReader((ROOT/'exp18e/per_seed.csv').open())}
    for method in BASELINES+METHODS:
        for seed in SEEDS:
            if method in BASELINES:
                out = ROOT/f'exp18e/formal/{method}_seed{seed}'
                for name, h in read(out/'complete.json').items(): assert sha(out/name) == h
                result = read(out/'summary.json')
                assert float(old_rows[method,seed]['accuracy']) == result['metrics']['accuracy']
                assert float(old_rows[method,seed]['loss']) == result['metrics']['loss']
            else:
                cfg = next(c for c in configs() if c['method']==method and c['seed']==seed)
                result = completed(cfg); assert result is not None, f'Incomplete: {cfg}'
                out = outdir(cfg)
                ckpt = torch.load(out/'checkpoint.pt', weights_only=True, map_location='cpu')
                assert ckpt['config'] == cfg and ckpt['optimizer']['t'] == 310
                assert ckpt['pairing'] == result['pairing'] and ckpt['privacy'] == result['privacy']
                assert all(torch.isfinite(p).all() for p in ckpt['model'].values())
                assert len(read(out/'curve.json')) == 5
                assert result['privacy'] == read(BASE/'privacy_audit.json')[method]
                del ckpt
            assert result['status']=='complete' and result['formal'] and result['steps']==310
            assert result['evaluation']=='official_validation' and result['metrics']['examples']==872
            cfg = result['config']; assert cfg['method']==method and cfg['seed']==seed
            assert {k: cfg[k] for k in ('lr','clip','eps')} == {
                k: read(BASE/'hyperparameters.json')['methods'][method][k] for k in ('lr','clip','eps')}
            if seed in pairings: assert pairings[seed] == result['pairing'], f'Unpaired: {method}, {seed}'
            pairings[seed] = result['pairing']; results[method,seed] = result
            source = str(out.relative_to(ROOT)); sources[source] = sha(out/'summary.json')
            rows.append(dict(method=method, seed=seed, **{k: cfg[k] for k in ('lr','clip','eps')},
                beta1=.9, beta2=.999, **result['metrics'], steps=310, evaluation='official_validation',
                source='Exp18e reused' if method in BASELINES else 'Exp18f new formal', directory=source,
                gpu_uuid=result['gpu_uuid'], seconds=result['seconds'],
                peak_reserved_mib=result['peak_reserved_bytes']/1048576))
    write_csv(BASE/'per_seed.csv', rows, list(rows[0]))
    summaries = []
    for method in BASELINES+METHODS:
        selected = [r for r in rows if r['method']==method]
        s = dict(method=method, label=LABELS[method], source=selected[0]['source'], completed_seeds=5,
            **read(BASE/'hyperparameters.json')['methods'][method])
        s.update({f'accuracy_seed_{r["seed"]}': r['accuracy'] for r in selected})
        for metric in ('accuracy','loss'):
            s.update({f'{metric}_{k}': v for k,v in stats([r[metric] for r in selected]).items() if k!='n'})
        summaries.append(s)
    write_csv(BASE/'summary.csv', summaries, list(summaries[0]))
    differences = []; paired = []
    for a,b in itertools.combinations(BASELINES+METHODS, 2):
        ds = []
        for seed in SEEDS:
            ra, rb = results[a,seed]['metrics'], results[b,seed]['metrics']
            d = dict(method_a=a, method_b=b, seed=seed,
                accuracy_difference=rb['accuracy']-ra['accuracy'], loss_difference=rb['loss']-ra['loss'])
            differences.append(d); ds.append(d)
        paired.append(dict(method_a=a, method_b=b, direction='method_b - method_a',
            accuracy=stats([d['accuracy_difference'] for d in ds]), loss=stats([d['loss_difference'] for d in ds])))
    write_csv(BASE/'paired_differences.csv', differences, list(differences[0]))
    save(BASE/'paired_statistics.json', dict(seeds=SEEDS, units='accuracy fraction; loss cross entropy',
        confidence='two-sided Student-t(df=4), descriptive, no multiplicity adjustment', all_pairs=paired))
    launch = read(BASE/'launch.json'); scheduling = read(BASE/'scheduling.json')
    save(BASE/'integrity_audit.json', dict(complete=True, new_formal_completed=10, baseline_reused=20,
        combined_official_results=30, complete_steps_per_run=310, official_examples_per_run=872,
        all_six_methods_seed_paired=True, checkpoint_and_log_hashes_verified=True,
        original_inputs_unchanged=True, result_sources_sha256=sources, time=now()))
    lines = ['# Exp18f STP-Ex2 最终 NLP 正式实验', '',
        f'状态：已完成。新训练 10/10，复用 Exp18e 官方结果 20 条；更新时间 {now()}。',
        '两种 STP-Ex2 统一使用 lr=0.025、C=100、稳定系数 eps=0.001、β₁=0.9、β₂=0.999。',
        'SST-2 / 本地 BERT-Tiny 全参数微调；FP32、关闭 TF32，长度128，批次1000，累积1，权重衰减0。',
        '固定62000训练/5349内部验证划分，完整5轮×62步=310步。逐轮内部验证仅监控；无模型选择或提前停止。',
        '872条官方验证仅评价第310步最终模型。所有模型和数据来自本地 cache/ 与 data/，全程离线。',
        '种子顺序：20261111、20261112、20261113、20261114、20261115；六种方法逐种子初始化、训练排列和dropout种子核对一致。',
        '', '## 六种算法官方验证总表', '',
        '准确率、样本SD和置信区间均以百分数表示；SD使用ddof=1，95% CI使用五种子的Student-t(df=4)。',
        '', '|算法|最优冻结参数 (lr, C, eps)|20261111|20261112|20261113|20261114|20261115|均值|样本SD|95% CI|平均验证损失|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        cells = '|'.join(f"{100*s[f'accuracy_seed_{seed}']:.3f}" for seed in SEEDS)
        lines.append(f"|{s['label']}|({s['lr']:g}, {s['clip']:g}, {s['eps']:g})|{cells}|{100*s['accuracy_mean']:.3f}|{100*s['accuracy_sd']:.3f}|[{100*s['accuracy_ci95_low']:.3f}, {100*s['accuracy_ci95_high']:.3f}]|{s['loss_mean']:.6f}|")
    lines += ['', '前四行完全复用 exp18e/formal/ 和 exp18e/per_seed.csv 的正式结果；后两行为本次新训练及官方评价。',
        'Exp18f 搜索中的83.746%和79.787%是内部验证成绩，本表未将其用作官方成绩。',
        '', '## 配对差异', '', '以下均为后者减前者，单位为百分点；全部15组比较及损失差保存在 paired_statistics.json。',
        '这些区间为描述性逐比较区间，未进行多重比较校正；Adam和Momentum的两种噪声方法超参数不同，不能仅归因于噪声机制。',
        '', '|前者|后者|平均准确率差 (pp)|95% 配对CI (pp)|', '|---|---|---:|---:|']
    for p in paired:
        s = p['accuracy']
        lines.append(f"|{LABELS[p['method_a']]}|{LABELS[p['method_b']]}|{100*s['mean']:+.3f}|[{100*s['ci95_low']:+.3f}, {100*s['ci95_high']:+.3f}]|")
    lines += ['', '## 运行与审计', '', f"后台调度PID：{launch['pid']}；总日志：logs/pipeline.log。正式工作进程PID见 logs/formal_events.jsonl。",
        '冒烟测试：三卡各运行两方法3步（另计6次），检查完整日志、有限参数、配对初始化/排列和显存；正式训练仅10次。']
    for gpu, uuid in scheduling['gpu_uuids'].items(): lines.append(f'- GPU {gpu}: NVIDIA GeForce RTX 3080 Ti，`{uuid}`。')
    peaks = [r['peak_reserved_mib'] for r in rows if r['method'] in METHODS]
    lines += ['', f'正式训练最大单进程峰值预留显存：{max(peaks):.0f} MiB；每卡最多2进程，UUID隔离，无DDP。',
        '每次训练的 config.json、pairing.json、privacy.json、steps.jsonl、curve.json、worker.log、checkpoint.pt、summary.json 和 complete.json 位于 formal/<method>_seed<seed>/。',
        'protocol.json、hyperparameters.json、data_audit.json、matrix_audit.json、privacy_audit.json、smoke_verification.json 和 integrity_audit.json 保存协议和审计证据。',
        'input_manifest.json 与 code_manifest.json 固定原输入及本次执行代码；最终审计确认原Exp18e与Exp18f搜索/冻结输入未被修改。',
        '', '每次训练独立校准隐私 ε=8、δ=1e-5；增删固定零槽邻接，参与5次、间隔62步，无抽样放大。',
        '沿用Exp18e的B²LT矩阵(T=310,K=P=62,Q=5)；C和D的逆关系、工作负载重构及敏感度独立核验。',
        'IID噪声和标准差134.21530075084846，B²LT为229.321310727543；对应批平均标准差为0.13421530075084845和0.229321310727543。',
        '稳定系数eps与隐私epsilon不同；多次训练、搜索和正式实验的组合不具备联合ε=8保证；原始内部监控量不是DP发布。',
        '', '复现：使用 Conda curve 环境，在仓库根目录运行 `python -B -m exp18f.final_official.start_background`（新输出目录首次启动），或运行 `python -B -m exp18f.final_official.report` 重新核验汇总。']
    (BASE/'final_report.md').write_text('\n'.join(lines)+'\n')
    print('REPORT COMPLETE: 10 new formal runs, 20 reused baseline results, 15 paired comparisons', flush=True)


if __name__ == '__main__': report()
