"""Render phase-one measurements without rerunning training or choosing parameters."""
import json
import re
from exp15 import RESULTS
from exp15.config import Trial, FINAL_SEEDS, save_json, file_hash
from exp15.smoke import jobs


def main():
    accepted=json.loads((RESULTS/'phase1_acceptance.json').read_text())
    if accepted['status']!='passed':raise ValueError('Acceptance has not passed')
    log=(RESULTS/'tests.log').read_text();match=re.search(r'(\d+) passed',log)
    if not match or re.search(r'\d+ failed',log):raise ValueError('Missing passing tests')
    save_json(RESULTS/'test_results.json',dict(status='passed',passed=int(match[1]),
        command='/home/longt29/miniconda3/envs/curve/bin/python -B -m pytest exp15/tests -q -p no:cacheprovider',
        log_sha256=file_hash(RESULTS/'tests.log')))
    rows=[json.loads((c.output/'summary.json').read_text()) for c in jobs()]
    lines=['# Exp15 阶段一验收报告','',
      f"状态：通过。{match[1]} 项数值/接口测试、{len(rows)} 次真实模型两步运行完成；验收不执行正式搜索或官方评价。",'',
      '测试覆盖五优化器手算公式及二阶矩输入、上一步分母顺序、动量偏差修正区别、状态恢复、重复词/padding 的逐样本显式梯度、逻辑批累积与单次加噪、非有限值原子拒绝、矩阵逆关系、真实动量工作负载、FIR 协方差、不同阈值的完整 ε=8 校准、前缀预算、资源哈希和全部 20 个任务/正式种子的数据顺序。搜索接口另测候选预算、重复训练复用、冻结哈希防修改、候选授权、失败候选排除和调度事件。','',
      'CV 搜索 225 步、正式 250 步及 NLP 310 步均使用完整矩阵校准，仅执行前两步；不把两步预算记为 ε=8。每次验收只评价 64 个内部验证样本，不能据此比较方法优劣。','',
      '|任务|方法|计划长度|重复|步数|损失|裁剪比例|更新范数|前缀 ε|峰值显存 GB|耗时秒|',
      '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        cfg=Trial(**{k:r[k] for k in Trial.__dataclass_fields__})
        d=json.loads((cfg.output/'diagnostics.jsonl').read_text().splitlines()[-1])
        lines.append(f"|{r['task']}|{r['method']}|{r['planned_total_steps']}|{r['replicate']}|{r['optimizer_steps']}|{d['train_loss']:.5f}|{d['clip_fraction']:.4f}|{d['update_norm']:.5g}|{r['actual_epsilon']:.5f}|{r['peak_allocated_bytes']/1e9:.3f}|{r['seconds']:.2f}|")
    lines+=['','CV/NLP 的 stpex2 复跑逐张量完全一致、评价指标完全一致；五种方法配对模型初始化、训练顺序、CV 增强/NLP dropout 和独立创新生成器状态。',
      f'正式种子从 Exp9 源文件读取：{list(FINAL_SEEDS)}。搜索、摸底和验收各用独立种子。',
      '来源哈希见 copied_sources.json；验收和资源哈希见 phase1_acceptance.json；测试日志见 tests.log；命令/PID/GPU 见 smoke_status.json；每个 trial 保存矩阵、顺序、逐步诊断和模型/优化器检查点。',
      '隐私范围和工作负载说明见 ../privacy_report.md，历史差异见 ../differences.md，复现命令见 ../README.md。早期四步摸底和协议修订单独保留，不计作正式搜索。',
      '阶段二在本平台验收通过后继续执行；最新状态见 pipeline_status.json。']
    (RESULTS/'phase1_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
