"""Auditable hashes, pairing, GDP, and reports. No official validation access."""
import json
from exp8a import BASE, RESULTS, ROOT
from exp8a.config import save_json, file_hash, MODEL_PATH, Config


def source_hashes():
    return {str(p.relative_to(BASE)):file_hash(p) for p in sorted(BASE.rglob('*.py'))}


def verify_stage1():
    result=json.loads((RESULTS/'stage1.json').read_text())
    assert result['status']=='passed'
    assert result['source_sha256']==source_hashes(), 'Source changed since Stage 1; rerun Stage 1'
    assert result['split_sha256']==file_hash(RESULTS/'split.npz')
    assert result['checkpoint_sha256']==file_hash(MODEL_PATH/'pytorch_model.bin')
    assert json.loads((RESULTS/'clipping_correctness.json').read_text())['status']=='passed'
    for row in result['gpu_smoke']:
        assert row['status']=='completed' and row['optimizer_steps']==row['noise_draws']==1


def audit_runs(rows):
    assert len(rows) in (3,9)
    keys=('initialization_sha256','classifier_initialization_sha256','checkpoint_sha256','split_sha256')
    assert all(r['geometry']=='scale' for r in rows)
    for key in ('physical_batch_size','lr','C','eps_scale'):
        assert len({r[key] for r in rows})==1, f'Paired runs differ in {key}'
    for seed in {r['seed'] for r in rows}:
        group=[r for r in rows if r['seed']==seed]
        assert len(group)==3 and {r['max_length'] for r in group}=={32,64,128}
        assert all(len({r[k] for r in group})==1 for k in keys)
        for r in group:
            assert r['optimizer_steps']==r['noise_draws']==310
            assert r['full_run'] and not r['official_validation_used']
            assert abs(r['epsilon']-8)<1e-7 and r['examples']==5349
    save_json(RESULTS/'paired_audit.json',dict(status='passed',full_runs=len(rows),keys=list(keys),
                                            participation='each fixed training row appears at j,j+62,...,j+248'))


def table(rows):
    lines=['| 裁剪 | 长度 | Physical batch | 状态 | samples/s | peak allocated GiB | peak reserved GiB |',
           '|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['geometry']} | {r['max_length']} | {r['physical_batch_size']} | {r['status']} | "
                     f"{r.get('samples_per_second',0):.2f} | {r['peak_allocated_bytes']/2**30:.3f} | {r['peak_reserved_bytes']/2**30:.3f} |")
    return '\n'.join(lines)


def report(selected,batch_rows=(),length_rows=(),runs=(),final_rows=()):
    stage1=selected['status']=='stage1_only'
    text='# Exp8a 实验报告\n\n'
    text+='Stage 1 已验证；Stage 2 尚未运行，最终参数尚未选择。\n\n' if stage1 else f"最终推荐 physical_batch_size={selected['physical_batch_size']}，max_length={selected['max_length']}。\n\n"
    if not stage1 and selected.get('hyperparameters_converged') is False:
        text+='本轮没有找到正常收敛的超参数：9个完整训练的Accuracy均等于多数类基线，validation loss随epoch上升。因此长度64是按协议的证据不足回退，不能据此宣称它的Accuracy最佳；最终physical batch的吞吐选择有效，冻结的lr/C/eps_scale不得作为已验证收敛的Exp8b超参数。\n\n'
    text+='固定协议：BERT-Tiny 全参数普通二分类，CE，62,000 DP train / 5,349 search validation；官方 validation 从未下载或用于选择。1000 logical batch，5 epochs，310 steps；GDP ε=8、δ=1e-5，add/remove zero-out，无 sampling amplification，k=5、spacing=62。Adam (.9,.999,1e-8)，4 bands。\n\n'
    text+='随机规则：固定分层 split 和排列，跨 epoch/seed/裁剪/长度重用；paired seed 初始化，noise seed+1 独立生成器。FP32，dropout=0，保证 physical partition 不改变 dropout 随机样本。所有 BERT/分类头参数可训练。\n\n'
    text+='Embedding 优化：按 (sample, token) 聚合 backprops，重复 token 先求和再求范数；padding 导数置零；Scale 在聚合后逐坐标乘冻结 scale。CPU/CUDA 显式逐样本 autograd 验证了 embedding 范数、全 BERT 范数、裁剪系数及累积梯度。避免生成 B×V×H 的密集逐样本梯度；没有改变数学机制。\n\n'
    text+='Embedding张量大小的理论估算（非未优化GPU实测）：原Scale的B×30522×128 FP32逐样本梯度在physical100/250时为约1.455/3.639 GiB，乘scale的临时张量可能再占同等大小。优化后完整step显存见实测表。\n\n'
    text+='Momentum-Bias 使用完整非 Toeplitz bias-corrected 累计 workload，在 T=310 下以固定参与敏感度平方 × 全矩阵平均误差重新优化，未复用 T=250 系数。Scale 采用前一已完成 Adam step 的 vhat 冻结 scale；先累积剪裁梯度，在 scaled sum 空间加一次相关噪声，再 inverse scale 和除以1000。\n\n'
    stats=json.loads((RESULTS/'token_lengths.json').read_text())
    text+='Token 长度（含 [CLS]/[SEP]）实测：`'+json.dumps(stats['percentiles'])+'`；截断比例：`'+json.dumps(stats['truncation_ratio'])+'`。\n\n'
    if stage1:
        text+='GPU smoke 实测（每种机制仅一个完整 logical step，未预热，不能据此选择最优 batch）：\n\n'+table(batch_rows)+'\n\n'
        text+='Standard/Scale 搜索验证 Accuracy/loss：尚未测量；未做完整 DP 训练。表中吞吐仅是 smoke 实测，没有 5-epoch 运行耗时预估。\n\n'
    else:
        pilots=json.loads((RESULTS/'pilots.json').read_text())
        text+='Length64短pilot实测（每个62 steps，共'+str(pilots['count'])+'次）：\n\n| lr | C | eps_scale | Accuracy | loss |\n|---:|---:|---:|---:|---:|\n'
        for r in pilots.get('results',[]): text+=f"| {r['lr']} | {r['C']} | {r['eps_scale']} | {r['accuracy']:.6f} | {r['loss']:.6f} |\n"
        text+='\n'
        text+='Physical batch 搜索实测（1 warmup + 3 完整测量 steps，CUDA 同步计时）：\n\n'+table(batch_rows)+'\n\n'
        text+='长度 benchmark 实测：\n\n'+table(length_rows)+'\n\n'
        text+='完整 Momentum-Bias + Scale 训练实测：\n\n| 长度 | seed | Accuracy | loss | 秒 |\n|---|---:|---:|---:|---:|\n'
        for r in runs: text+=f"| {r['max_length']} | {r['seed']} | {r['accuracy']:.6f} | {r['loss']:.6f} | {r['seconds']:.1f} |\n"
        text+='\nStandard 完整训练 Accuracy 未测量（协议仅要求 Scale 长度 full runs）。所有 benchmark 的 Standard/Scale 吞吐和显存均为实测。\n\n'
        text+='选定长度后的 physical batch 复测：\n\n'+table(final_rows)+'\n\n'
        text+='选择依据：`'+json.dumps(selected['selection_basis'],ensure_ascii=False)+'`。\n\n'
    text+='吞吐统计覆盖 forward、两遍 backward、剪裁、logical accumulation、相关噪声和 Adam 更新及必要 non-finite 检查。reserved/allocated 为 PyTorch allocator 峰值，不包含驱动 context；选择以两者较大值保留至少20%显存，接近5%时选显存较小配置。\n\n'
    text+='结果仅对 DP 训练机制按所述预算校准。多个 pilot/full run 的搜索总隐私消耗不等于单次 ε=8；本实验用公开 SST-2，未声称整个调参过程 ε=8。\n'
    (RESULTS/'exp8a_report.md').write_text(text)
