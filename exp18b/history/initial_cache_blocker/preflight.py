"""Offline, CPU-only input audit. Never downloads or starts training."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from safetensors import safe_open

from exp18.model import CNN

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "exp18b"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(name, value):
    (BASE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    # Meta tensors inspect Exp18's actual structure without initializing weights.
    with torch.device("meta"):
        expected = {key: list(value.shape) for key, value in CNN().state_dict().items()}

    inventory = []
    for path in sorted((ROOT / "cache").rglob("*")):
        if path.is_file():
            inventory.append(dict(path=str(path), resolved_path=str(path.resolve()),
                                  bytes=path.stat().st_size, sha256=sha(path)))
    save("cache_inventory.json", inventory)

    candidates = []
    seen = set()
    for item in inventory:
        path = Path(item["path"])
        if path.suffix not in {".bin", ".pt", ".pth", ".ckpt", ".safetensors"}:
            continue
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as handle:
                shapes = {key: list(handle.get_slice(key).get_shape()) for key in handle.keys()}
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            for wrapper in ("state_dict", "model_state_dict", "model"):
                if wrapper in state and isinstance(state[wrapper], dict):
                    state = state[wrapper]
                    break
            shapes = {key: list(value.shape) for key, value in state.items()
                      if isinstance(value, torch.Tensor)}
        missing = sorted(set(expected) - set(shapes))
        unexpected = sorted(set(shapes) - set(expected))
        mismatches = {key: dict(expected=expected[key], actual=shapes[key])
                      for key in expected.keys() & shapes.keys() if expected[key] != shapes[key]}
        candidates.append(dict(**item, tensor_count=len(shapes), tensors=shapes,
                               missing_keys=missing, unexpected_keys=unexpected,
                               shape_mismatches=mismatches,
                               compatible=not (missing or unexpected or mismatches)))
    save("checkpoint_audit.json", dict(expected_exp18_state_dict=expected,
                                        candidates=candidates))

    reference = json.loads((ROOT / "exp18/results/assets.json").read_text())
    assets = []
    for name in ("train.pt", "split.npz", "test.pt"):
        path = ROOT / "data" / name
        actual = sha(path)
        assert actual == reference[str(path.relative_to(ROOT))], f"Exp18 asset differs: {path}"
        assets.append(dict(path=str(path), sha256=actual, matches_exp18=True))
    for path in sorted((ROOT / "data/MNIST/raw").iterdir()):
        if path.is_file():
            assets.append(dict(path=str(path), sha256=sha(path)))
    with np.load(ROOT / "data/split.npz") as split:
        train, valid = split["train"], split["valid"]
        assert train.shape == (54000,) and valid.shape == (6000,)
        assert np.array_equal(np.sort(np.concatenate((train, valid))), np.arange(60000))
    tensor_info = {}
    for name, count in (("train", 60000), ("test", 10000)):
        data = torch.load(ROOT / f"data/{name}.pt", map_location="cpu", weights_only=True)
        images, labels = data["images"], data["labels"]
        assert images.shape == (count, 1, 28, 28) and images.dtype == torch.float32
        assert labels.shape == (count,) and labels.dtype == torch.int64
        assert torch.isfinite(images).all() and labels.min() == 0 and labels.max() == 9
        tensor_info[name] = {key: dict(shape=list(value.shape), dtype=str(value.dtype))
                             for key, value in data.items()}
    save("data_audit.json", dict(status="passed", assets=assets, tensors=tensor_info,
                                 split_sizes=[54000, 6000, 10000],
                                 downloads=False, test_evaluation=False))

    compatible = [item["path"] for item in candidates if item["compatible"]]
    blocked = not compatible
    status = dict(timestamp=timestamp, status="blocked" if blocked else "inputs_checked",
                  reason="No compatible Exp18 CNN checkpoint in cache/" if blocked else None,
                  python=sys.executable, torch_version=torch.__version__,
                  data="passed", cache_checkpoint="failed" if blocked else "passed",
                  compatible_checkpoints=compatible, loaded_checkpoint=None,
                  smoke_tests="not_run", privacy_calibration="not_run",
                  b2lt_matrix_validation="not_run", gpu_scheduling_validation="not_run",
                  protocol_frozen=False, launched=False, background_pid=None,
                  training_count=0, official_test_evaluations=0,
                  intended_physical_gpus=[1, 2, 3], actual_gpu_assignment=[],
                  random_initialization_fallback=False)
    save("status.json", status)
    save("preflight.json", status)
    save("audit_manifest.json", {str(path.relative_to(ROOT)): sha(path) for path in
         [BASE / "preflight.py", ROOT / "exp18/model.py", ROOT / "exp18/protocol.json",
          ROOT / "exp18/results/assets.json", BASE / "cache_inventory.json",
          BASE / "checkpoint_audit.json", BASE / "data_audit.json"]})
    if blocked:
        (BASE / "final_report.md").write_text(
            "# Exp18b 输入验收：阻塞，未启动\n\n"
            f"检查时间：{timestamp}。环境：`{sys.executable}`。\n\n"
            "MNIST 数据检查通过，train.pt、test.pt、split.npz 的 SHA256 与 Exp18 完全一致；"
            "划分为 54000/6000/10000。未下载数据，未进行测试集评价。"
            "实际绝对路径、SHA256、形状和类型见 data_audit.json。\n\n"
            "cache 中只有 BERT-tiny 和 ViT 两份模型权重（ViT snapshot 是 blob 的符号链接）。"
            "BERT 有 49 个张量，ViT 有 152 个张量，均缺少 Exp18 CNN 的全部六个参数键。"
            "ViT 的 patch_embed.proj.weight 为 [192,3,16,16]，分类头为 [1000,192]；"
            "Exp18 首层为 [8,1,5,5]，分类头为 [10,784]。"
            "全部实际键名、维度、路径和 SHA256 见 checkpoint_audit.json；"
            "cache 全文件清单见 cache_inventory.json。\n\n"
            "按用户要求在权重不兼容时停止。未实现或启动后续训练流水线，"
            "未运行六方法 smoke test、隐私校准、矩阵验收或三卡并行验收，"
            "未冻结训练协议。实际加载的 CNN checkpoint：无。"
            "后台 PID：无。GPU 分配：无。完成训练：0/90。"
            "未生成虚构的指标、检查点、GPU 遥测或通过记录。\n\n"
            "Exp18 使用随机初始化；本任务要求统一使用兼容 cache 权重，"
            "因此不能回退到 Exp18 的随机初始化，也不能替换为 ViT/BERT。"
            "恢复需要在 cache/ 提供与 Exp18 CNN 完全兼容的真实 checkpoint，"
            "再复查输入并继续实现和验收。现有脚本仅进行离线输入审计，不启动训练。\n\n"
            "复查命令：`/home/longt29/miniconda3/envs/curve/bin/python -B -m exp18b.preflight`"
            "（缺少兼容权重时退出码为 2）。执行日志：preflight.log。\n\n"
            "隐私协议说明：要求每次完整训练 ε=3、δ=1e-5、GDP 无采样放大；"
            "90 次训练的组合隐私预算不能声明为 ε=3。当前没有执行训练，"
            "也未声称完成噪声标准差校准。\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
